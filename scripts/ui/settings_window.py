import os
import re
import sys
import copy
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser

from scripts.core.config_manager import (
    make_camera_entry,
    make_default_camera_modules,
    ConfigManager,
)
from scripts.core.hardware_info import install_ort_package, get_library_versions
from scripts.engine.camera_manager import probe_stream_info
from scripts.ui.helpers import (
    WindowGeometryTracker,
    ScrollableFrame,
    HelpButton,
    close_active_help_popover,
)

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

EVT_TREE_SELECT = "<<TreeviewSelect>>"

AI_BACKEND_CHOICES = [
    ("cpu", "CPU (CPUExecutionProvider — типово, максимальна сумісність)"),
    ("auto", "AUTO (автовибір найкращого доступного прискорювача)"),
    ("directml", "DirectML GPU (DX12 — працює на NVIDIA / AMD / Intel)"),
    ("cuda", "NVIDIA CUDA GPU (найвища швидкість на RTX/GeForce)"),
    ("openvino", "Intel OpenVINO (процесор / iGPU Intel)"),
]

HW_DECODE_CHOICES = [
    ("cpu", "cpu — Стандартний декодер CPU (найнадійніший)"),
    ("auto", "auto — Автовибір апаратного декодера GPU (fallback на CPU)"),
    ("d3d11va", "d3d11va — Windows Direct3D 11 GPU декодер"),
    ("cuda", "cuda — NVIDIA NVDEC GPU декодер"),
    ("qsv", "qsv — Intel QuickSync декодер"),
]

RESIZE_CHOICES = [
    ("linear", "linear / area — Оптимальний баланс швидкості та чіткості"),
    ("nearest", "nearest — Мінімум навантаження на CPU"),
    ("lanczos", "lanczos — Максимальна різкість (більше навантаження CPU)"),
]


def _clamp_int(val, default, min_v, max_v):
    try:
        v = int(round(float(str(val).replace(",", ".").strip())))
        return max(int(min_v), min(int(max_v), v))
    except Exception:
        return int(default)


def _clamp_float(val, default, min_v, max_v):
    try:
        v = float(str(val).replace(",", ".").strip())
        return max(float(min_v), min(float(max_v), v))
    except Exception:
        return float(default)


def _matches_condition(cond_dict: dict, values_map: dict) -> bool:
    if not isinstance(cond_dict, dict) or not cond_dict:
        return False
    for k, expected in cond_dict.items():
        actual = values_map.get(k)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


class SettingsWindow:
    """
    Незалежне вікно налаштувань із деревом розділів зліва, динамічними картками модулів
    із ModuleRegistry, телеметрією заліза, станом бібліотек та довідкою '?' біля кожного параметра.
    """
    def __init__(
        self,
        master,
        cfg_mgr: ConfigManager,
        hw_monitor,
        camera_manager,
        registry,
        on_apply_cb=None,
        on_close_cb=None,
        on_refresh_modules_cb=None,
        on_restart_app_cb=None,
    ):
        self.master = master
        self.cfg_mgr = cfg_mgr
        self.hw_monitor = hw_monitor
        self.camera_manager = camera_manager
        self.registry = registry
        self.on_apply_cb = on_apply_cb
        self.on_close_cb = on_close_cb
        self.on_refresh_modules_cb = on_refresh_modules_cb
        self.on_restart_app_cb = on_restart_app_cb

        self.draft = self.cfg_mgr.get_snapshot()
        self._is_closing = False
        self._current_node_id = "general"
        self._page_committers = []
        self._telemetry_after_id = None
        self._telemetry_labels = {}
        self._hw_widget_vars = {}
        self._caps_container = None
        self._caps_summary_lbl = None
        self._ort_install_lbl = None
        self._ort_installing = False
        self._last_caps = None
        self._face_thumbs_cache = []
        self._tree_syncing = False
        self._ui_queue = queue.SimpleQueue()
        self._ui_poll_after_id = None

        self.win = tk.Toplevel(master)
        self.win.title("Налаштування системи — ANPR Multi-Cam NVR")
        self.win.configure(bg="#14141b")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self._build_ui()

        self.geom_tracker = WindowGeometryTracker(
            self.win,
            get_geom_cb=lambda: self.cfg_mgr.get_window_geometry("settings"),
            set_geom_cb=lambda g: self.cfg_mgr.set_window_geometry("settings", g),
            min_w=860,
            min_h=560,
            default_geom={"left": 100, "top": 60, "width": 1080, "height": 740, "maximized": False},
        )

        self._drain_ui_queue()
        self._rebuild_nav_tree(select_id="general")
        self._start_async_caps_probe(force=False)

    def _drain_ui_queue(self):
        if self._is_closing:
            return
        while True:
            try:
                cb, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cb(*args)
            except Exception:
                pass
        if not self._is_closing:
            try:
                if self.win.winfo_exists():
                    self._ui_poll_after_id = self.win.after(25, self._drain_ui_queue)
            except Exception:
                pass

    def focus(self):
        try:
            if str(self.win.state()) == "iconic":
                self.win.deiconify()
            self.win.lift()
            self.win.focus_force()
        except Exception:
            pass

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "SettingsNav.Treeview",
            background="#1a1c26",
            foreground="#e2e6f3",
            fieldbackground="#1a1c26",
            font=("Segoe UI", 10),
            rowheight=30,
            borderwidth=0,
        )
        style.map(
            "SettingsNav.Treeview",
            background=[("selected", "#007acc")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "SettingsDark.TCombobox",
            fieldbackground="#232635",
            background="#2c3044",
            foreground="#ffffff",
            arrowcolor="#00e5ff",
        )
        style.configure("SettingsDark.Horizontal.TProgressbar", troughcolor="#252836", background="#00e5ff")

        footer = tk.Frame(self.win, bg="#1b1d27", padx=14, pady=10, highlightthickness=1, highlightbackground="#2b2e3d")
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.footer_status_lbl = tk.Label(
            footer,
            text="Редагування конфігурації (зміни набудуть чинності після натискання 'Застосувати')",
            fg="#9ba3b8",
            bg="#1b1d27",
            font=("Segoe UI", 9),
        )
        self.footer_status_lbl.pack(side=tk.LEFT)

        tk.Button(
            footer,
            text="Скасувати",
            command=self.close,
            bg="#353849",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            footer,
            text="✅ Зберегти і закрити",
            command=self._save_and_close,
            bg="#28a745",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=6)

        tk.Button(
            footer,
            text="💾 Застосувати",
            command=self._apply_changes,
            bg="#007acc",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=6)

        body = tk.Frame(self.win, bg="#14141b")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav_frame = tk.Frame(body, bg="#1a1c26", width=260, highlightthickness=1, highlightbackground="#282b3a")
        nav_frame.pack(side=tk.LEFT, fill=tk.Y)
        nav_frame.pack_propagate(False)

        tk.Label(
            nav_frame,
            text="РОЗДІЛИ НАЛАШТУВАНЬ",
            fg="#00e5ff",
            bg="#202331",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            padx=12,
            pady=10,
        ).pack(fill=tk.X)

        self.nav_tree = ttk.Treeview(nav_frame, style="SettingsNav.Treeview", show="tree", selectmode="browse")
        self.nav_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=6)
        self.nav_tree.bind(EVT_TREE_SELECT, self._on_tree_select)

        self.right_container = tk.Frame(body, bg="#161720")
        self.right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_panel = ScrollableFrame(self.right_container, bg="#161720")
        self.scroll_panel.pack(fill=tk.BOTH, expand=True)
        self.content_frame = self.scroll_panel.inner

    def _has_face_db_capability(self) -> bool:
        return bool(self.registry is not None and self.registry.list_by_capability("face_db"))

    def _rebuild_nav_tree(self, select_id=None):
        self._tree_syncing = True
        try:
            self.nav_tree.delete(*self.nav_tree.get_children())
            self.nav_tree.insert("", "end", iid="general", text="  ⚙ Загальні")
            self.nav_tree.insert("", "end", iid="grid", text="  ▦ Сітка камер")
            self.nav_tree.insert("", "end", iid="hardware", text="  ⚡ Залізо та графіка")
            self.nav_tree.insert("", "end", iid="modules", text="  🧩 Модулі та бібліотеки")
            if self._has_face_db_capability():
                self.nav_tree.insert("", "end", iid="faces_db", text="  👥 База облич (faces.txt)")
            cams_node = self.nav_tree.insert("", "end", iid="cameras_root", text="  📹 Камери", open=True)

            for idx, cam in enumerate(self.draft.get("cameras") or []):
                uid = cam["uuid"]
                name = str(cam.get("name") or f"Камера {idx + 1}")
                prefix = "●" if cam.get("enabled", True) else "⏸"
                suffix = "" if cam.get("enabled", True) else " (вимк.)"
                self.nav_tree.insert(cams_node, "end", iid=f"cam:{uid}", text=f"  {prefix} {name}{suffix}")

            target = select_id or self._current_node_id
            if target and self.nav_tree.exists(target):
                self.nav_tree.selection_set(target)
                self.nav_tree.focus(target)
                self.nav_tree.see(target)
            elif self.nav_tree.exists("general"):
                self.nav_tree.selection_set("general")
                target = "general"
        finally:
            self._tree_syncing = False

        if target:
            self._show_page(target)

    def _on_tree_select(self, event=None):
        if self._tree_syncing:
            return
        sel = self.nav_tree.selection()
        if not sel:
            return
        node_id = sel[0]
        if node_id != self._current_node_id:
            self._commit_current_page()
            self._show_page(node_id)

    def _commit_current_page(self):
        for fn in list(self._page_committers):
            try:
                fn()
            except Exception:
                pass

    def _clear_content(self):
        close_active_help_popover()
        self._cancel_telemetry_timer()
        self._page_committers.clear()
        self._telemetry_labels.clear()
        self._hw_widget_vars.clear()
        self._caps_container = None
        self._caps_summary_lbl = None
        self._ort_install_lbl = None
        for child in list(self.content_frame.winfo_children()):
            child.destroy()
        try:
            self.scroll_panel.canvas.yview_moveto(0.0)
        except Exception:
            pass

    def _show_page(self, node_id: str):
        self._current_node_id = node_id
        self._clear_content()

        pad = tk.Frame(self.content_frame, bg="#161720", padx=20, pady=16)
        pad.pack(fill=tk.BOTH, expand=True)

        if node_id == "general":
            self._build_page_general(pad)
        elif node_id == "grid":
            self._build_page_grid(pad)
        elif node_id == "hardware":
            self._build_page_hardware(pad)
        elif node_id == "modules":
            self._build_page_modules(pad)
        elif node_id == "faces_db":
            self._build_page_faces_db(pad)
        elif node_id == "cameras_root":
            self._build_page_cameras_root(pad)
        elif node_id.startswith("cam:"):
            uid = node_id.split("cam:", 1)[1]
            self._build_page_single_camera(pad, uid)

    def on_modules_reloaded(self):
        """Оновлює чернетку під нові модулі без втрати поточних правок користувача та перемальовує дерево."""
        if self._is_closing:
            return
        try:
            self._commit_current_page()
            reg_mods = make_default_camera_modules(registry=self.registry)
            tpl_mods = self.draft.setdefault("default_camera_modules", {})
            for mid, def_cfg in reg_mods.items():
                sec = tpl_mods.setdefault(mid, {})
                for k, v in def_cfg.items():
                    sec.setdefault(k, copy.deepcopy(v))

            for cam in self.draft.get("cameras") or []:
                cmods = cam.setdefault("modules", {})
                for mid, def_cfg in reg_mods.items():
                    sec = cmods.setdefault(mid, {})
                    for k, v in def_cfg.items():
                        sec.setdefault(k, copy.deepcopy(v))

            target = self._current_node_id
            if target == "faces_db" and not self._has_face_db_capability():
                target = "modules"
            self._rebuild_nav_tree(select_id=target)
        except Exception as e:
            print(f"[SettingsWindow on_modules_reloaded]: {e}")

    # ------------------------------------------------------------------
    # Допоміжні конструктори карток та полів із довідкою '?'
    # ------------------------------------------------------------------
    @staticmethod
    def _section_header(parent, title: str, subtitle: str = ""):
        tk.Label(
            parent, text=title, fg="#00e5ff", bg="#161720",
            font=("Segoe UI", 14, "bold"), anchor=tk.W
        ).pack(fill=tk.X, pady=(0, 2))
        if subtitle:
            tk.Label(
                parent, text=subtitle, fg="#9ba3b8", bg="#161720",
                font=("Segoe UI", 9), anchor=tk.W, wraplength=740, justify=tk.LEFT
            ).pack(fill=tk.X, pady=(0, 12))
        else:
            tk.Frame(parent, bg="#161720", height=8).pack()

    @staticmethod
    def _make_card(parent, title: str, subtitle: str = "", bg: str = "#1c1e2a"):
        card = tk.Frame(parent, bg=bg, padx=14, pady=12, highlightthickness=1, highlightbackground="#2b2e3d")
        card.pack(fill=tk.X, pady=(0, 12))
        if title:
            tk.Label(
                card, text=title, fg="#ffffff", bg=bg,
                font=("Segoe UI", 10, "bold"), anchor=tk.W
            ).pack(fill=tk.X, pady=(0, 2 if subtitle else 8))
        if subtitle:
            tk.Label(
                card, text=subtitle, fg="#8d95ab", bg=bg,
                font=("Segoe UI", 8), anchor=tk.W, wraplength=720, justify=tk.LEFT
            ).pack(fill=tk.X, pady=(0, 8))
        return card

    def _add_row(self, card, label: str, help_key: str, widget_fn, label_w: int = 28, help_text: str = ""):
        bg = str(card.cget("bg"))
        row = tk.Frame(card, bg=bg)
        row.pack(fill=tk.X, pady=4)

        lbl_box = tk.Frame(row, bg=bg, width=label_w * 8)
        lbl_box.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(
            lbl_box, text=label, fg="#d5dae8", bg=bg, font=("Segoe UI", 9),
            anchor=tk.W, width=label_w
        ).pack(side=tk.LEFT)

        ctrl_box = tk.Frame(row, bg=bg)
        ctrl_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 6))
        ctrl = widget_fn(ctrl_box, bg)

        resolved_help = str(help_text or "").strip()
        if not resolved_help and help_key:
            resolved_help = ConfigManager.get_help_text(help_key, registry=self.registry)

        if resolved_help:
            HelpButton(
                row,
                help_text=resolved_help,
                title=label.rstrip(":"),
                bg=bg,
            ).pack(side=tk.RIGHT, padx=(4, 0))
        return row, ctrl

    # ------------------------------------------------------------------
    # Сторінка 1: Загальні
    # ------------------------------------------------------------------
    def _build_page_general(self, parent):
        self._section_header(
            parent,
            "⚙ Загальні налаштування",
            "Кореневий каталог архіву, автозапуск з Windows, режим кіоску та управління вікнами.",
        )
        gen = self.draft.setdefault("general", {})
        card = self._make_card(parent, "Основні параметри")

        arch_var = tk.StringVar(value=str(gen.get("archive_root", "")))
        auto_var = tk.BooleanVar(value=bool(gen.get("autostart_os_and_play", False)))
        kiosk_var = tk.BooleanVar(value=bool(gen.get("kiosk_mode", False)))

        def _commit():
            gen["archive_root"] = arch_var.get().strip() or self.cfg_mgr.get_archive_root()
            gen["autostart_os_and_play"] = bool(auto_var.get())
            gen["kiosk_mode"] = bool(kiosk_var.get())

        self._page_committers.append(_commit)

        def _w_archive(box, bg):
            e = tk.Entry(box, textvariable=arch_var, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9))
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 6))

            def _browse():
                chosen = filedialog.askdirectory(parent=self.win, initialdir=arch_var.get() or self.cfg_mgr.base_dir, title="Оберіть папку архіву")
                if chosen:
                    arch_var.set(os.path.normpath(chosen))
                    _commit()

            def _open_dir():
                p = arch_var.get().strip()
                if p:
                    os.makedirs(p, exist_ok=True)
                    try:
                        os.startfile(p)
                    except Exception:
                        pass

            tk.Button(box, text="📁 Огляд...", command=_browse, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
            tk.Button(box, text="Відкрити", command=_open_dir, bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT)
            return e

        self._add_row(card, "Коренева папка архіву:", "general.archive_root", _w_archive)

        def _w_check(var, text):
            def _fn(box, bg):
                cb = tk.Checkbutton(
                    box, text=text, variable=var, command=_commit,
                    bg=bg, fg="#ffffff", selectcolor="#252836",
                    activebackground=bg, activeforeground="#ffffff", font=("Segoe UI", 9)
                )
                cb.pack(side=tk.LEFT)
                return cb
            return _fn

        self._add_row(
            card,
            "Автозапуск з Windows:",
            "general.autostart_os_and_play",
            _w_check(auto_var, "Додати в автозапуск Windows та автоматично запускати камери"),
        )
        self._add_row(
            card,
            "Режим кіоску:",
            "general.kiosk_mode",
            _w_check(kiosk_var, "Блокувати стандартне закриття (вихід лише Ctrl+Alt+Shift+Q)"),
        )

        win_card = self._make_card(parent, "Позиції вікон на моніторах")
        tk.Label(
            win_card,
            text=(
                "Усі вікна (головна сітка, незалежні вікна камер, вікна кропу й налаштування) є вільними:\n"
                "просто перетягуйте їх на будь-який монітор — координати й стан розгортання зберігаються автоматично."
            ),
            fg="#9ba3b8", bg=str(win_card.cget("bg")), font=("Segoe UI", 9), justify=tk.LEFT, anchor=tk.W
        ).pack(fill=tk.X, pady=(0, 8))

        def _reset_window_positions():
            self.draft["windows"] = {
                "grid": {"left": 40, "top": 30, "width": 1280, "height": 760, "maximized": False},
                "settings": {"left": 100, "top": 60, "width": 1080, "height": 740, "maximized": False},
            }
            for idx, cam in enumerate(self.draft.get("cameras") or []):
                cam["window"] = {"left": 80 + idx * 30, "top": 60 + idx * 30, "width": 1160, "height": 720, "maximized": False}
                cam["roi_window"] = {"left": 60, "top": 50, "width": 520, "height": 360, "maximized": False}
            self.footer_status_lbl.configure(text="Позиції вікон скинуто у чернетці — натисніть 'Застосувати'.", fg="#00e5ff")

        tk.Button(
            win_card,
            text="📍 Скинути збережені позиції всіх вікон на головний монітор",
            command=_reset_window_positions,
            bg="#2e3142", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=6, cursor="hand2"
        ).pack(anchor=tk.W)

    # ------------------------------------------------------------------
    # Сторінка 2: Сітка камер
    # ------------------------------------------------------------------
    def _build_page_grid(self, parent):
        self._section_header(
            parent,
            "▦ Налаштування сітки камер",
            "У загальній сітці завжди використовується стиснутий субпотік (sub_source) для економії CPU/GPU.",
        )
        grid = self.draft.setdefault("grid", {})
        card = self._make_card(parent, "Макет та оформлення плиток")

        mode_var = tk.StringVar(value=str(grid.get("mode", "auto")))
        rows_var = tk.StringVar(value=str(grid.get("rows", 2)))
        cols_var = tk.StringVar(value=str(grid.get("cols", 2)))
        bw_var = tk.StringVar(value=str(grid.get("border_width", 2)))
        gap_var = tk.StringVar(value=str(grid.get("tile_gap", 4)))
        bcol_var = tk.StringVar(value=str(grid.get("border_color", "#00e5ff")))
        acol_var = tk.StringVar(value=str(grid.get("active_border_color", "#00ff88")))
        ocol_var = tk.StringVar(value=str(grid.get("offline_border_color", "#ff4444")))
        lbl_var = tk.BooleanVar(value=bool(grid.get("show_labels", True)))
        st_var = tk.BooleanVar(value=bool(grid.get("show_status", True)))
        fps_var = tk.BooleanVar(value=bool(grid.get("show_fps", True)))

        rows_spin_holder = []
        cols_spin_holder = []

        def _commit():
            m = mode_var.get().strip().lower()
            grid["mode"] = "manual" if m == "manual" else "auto"
            grid["rows"] = _clamp_int(rows_var.get(), 2, 1, 8)
            grid["cols"] = _clamp_int(cols_var.get(), 2, 1, 8)
            grid["border_width"] = _clamp_int(bw_var.get(), 2, 1, 10)
            grid["tile_gap"] = _clamp_int(gap_var.get(), 4, 0, 20)
            grid["border_color"] = bcol_var.get().strip() or "#00e5ff"
            grid["active_border_color"] = acol_var.get().strip() or "#00ff88"
            grid["offline_border_color"] = ocol_var.get().strip() or "#ff4444"
            grid["show_labels"] = bool(lbl_var.get())
            grid["show_status"] = bool(st_var.get())
            grid["show_fps"] = bool(fps_var.get())

        self._page_committers.append(_commit)

        def _sync_manual_state(*_):
            is_man = (mode_var.get() == "manual")
            st = tk.NORMAL if is_man else tk.DISABLED
            for w in rows_spin_holder + cols_spin_holder:
                try:
                    w.configure(state=st)
                except Exception:
                    pass
            _commit()

        def _w_mode(box, bg):
            cb = ttk.Combobox(box, textvariable=mode_var, values=["auto", "manual"], state="readonly", width=16)
            cb.pack(side=tk.LEFT)
            cb.bind("<<ComboboxSelected>>", _sync_manual_state)
            tk.Label(box, text="  (auto = під кількість камер, manual = рядки × колонки)", fg="#9ba3b8", bg=bg, font=("Segoe UI", 8)).pack(side=tk.LEFT)
            return cb

        self._add_row(card, "Режим побудови сітки:", "grid.mode", _w_mode)

        def _w_spin(var, from_, to_, holder):
            def _fn(box, bg):
                sp = tk.Spinbox(
                    box, from_=from_, to=to_, textvariable=var, width=8,
                    bg="#252836", fg="#ffffff", buttonbackground="#2f3346", relief=tk.FLAT,
                    font=("Segoe UI", 9), command=_commit
                )
                sp.pack(side=tk.LEFT)
                holder.append(sp)
                return sp
            return _fn

        self._add_row(card, "Рядків (manual):", "grid.rows", _w_spin(rows_var, 1, 8, rows_spin_holder))
        self._add_row(card, "Колонок (manual):", "grid.cols", _w_spin(cols_var, 1, 8, cols_spin_holder))
        _sync_manual_state()

        def _w_color(var):
            def _fn(box, bg):
                e = tk.Entry(box, textvariable=var, width=12, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Consolas", 9))
                e.pack(side=tk.LEFT, ipady=3, padx=(0, 8))
                swatch = tk.Button(box, text="  Обрати колір  ", relief=tk.FLAT, cursor="hand2", font=("Segoe UI", 8, "bold"))

                def _refresh_swatch(*_):
                    col = var.get().strip()
                    if re.match(r"^#[0-9a-fA-F]{6}$", col):
                        swatch.configure(bg=col, fg="#000000")

                def _pick():
                    initial = var.get().strip() if re.match(r"^#[0-9a-fA-F]{6}$", var.get().strip()) else "#00e5ff"
                    _, hex_c = colorchooser.askcolor(color=initial, parent=self.win, title="Вибір кольору рамки")
                    if hex_c:
                        var.set(hex_c)
                        _commit()

                swatch.configure(command=_pick)
                swatch.pack(side=tk.LEFT)
                var.trace_add("write", _refresh_swatch)
                _refresh_swatch()
                return e
            return _fn

        self._add_row(card, "Колір рамки плитки:", "grid.border_color", _w_color(bcol_var))
        self._add_row(card, "Колір активного вікна:", "grid.active_border_color", _w_color(acol_var))
        self._add_row(card, "Колір втрати зв'язку:", "grid.offline_border_color", _w_color(ocol_var))
        self._add_row(card, "Товщина рамки (px):", "grid.border_width", _w_spin(bw_var, 1, 10, []))
        self._add_row(card, "Проміжок між плитками (px):", "grid.tile_gap", _w_spin(gap_var, 0, 20, []))

        def _w_chk(var, text):
            def _fn(box, bg):
                cb = tk.Checkbutton(box, text=text, variable=var, command=_commit, bg=bg, fg="#ffffff", selectcolor="#252836", activebackground=bg, activeforeground="#ffffff", font=("Segoe UI", 9))
                cb.pack(side=tk.LEFT)
                return cb
            return _fn

        self._add_row(card, "Підпис камери:", "grid.show_labels", _w_chk(lbl_var, "Показувати назву камери на плитці"))
        self._add_row(card, "Індикатори статусу:", "grid.show_status", _w_chk(st_var, "Показувати ● REC (кодек) та активні модулі"))
        self._add_row(card, "Частота кадрів:", "grid.show_fps", _w_chk(fps_var, "Показувати поточний FPS субпотоку на плитці"))

    # ------------------------------------------------------------------
    # Сторінка 3: Залізо та графіка
    # ------------------------------------------------------------------
    def _build_page_hardware(self, parent):
        self._section_header(
            parent,
            "⚡ Залізо, ресурси та графічне прискорення",
            "Жива телеметрія процесора/ОЗП/відеокарт/дисків, перевірка підтримки CUDA/DirectML/OpenVINO/NVENC/copy та налаштування продуктивності.",
        )
        hw = self.draft.setdefault("hardware", {})

        tel_card = self._make_card(parent, "Апаратні ресурси цього комп'ютера (оновлюється в реальному часі)")
        tel_bg = str(tel_card.cget("bg"))

        lbl_cpu = tk.Label(tel_card, text="CPU: сканування...", fg="#e6eaf5", bg=tel_bg, font=("Segoe UI", 9, "bold"), anchor=tk.W)
        lbl_cpu.pack(fill=tk.X, pady=(0, 2))
        pb_cpu = ttk.Progressbar(tel_card, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
        pb_cpu.pack(fill=tk.X, pady=(0, 8))

        lbl_ram = tk.Label(tel_card, text="RAM: сканування...", fg="#e6eaf5", bg=tel_bg, font=("Segoe UI", 9, "bold"), anchor=tk.W)
        lbl_ram.pack(fill=tk.X, pady=(0, 2))
        pb_ram = ttk.Progressbar(tel_card, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
        pb_ram.pack(fill=tk.X, pady=(0, 8))

        lbl_gpu = tk.Label(tel_card, text="Відеокарти (GPU): сканування...", fg="#9ee8ff", bg=tel_bg, font=("Segoe UI", 9), anchor=tk.W, justify=tk.LEFT, wraplength=740)
        lbl_gpu.pack(fill=tk.X, pady=(0, 6))

        disks_frame = tk.Frame(tel_card, bg=tel_bg)
        disks_frame.pack(fill=tk.X, pady=(2, 0))

        self._telemetry_labels = {
            "cpu_lbl": lbl_cpu,
            "cpu_pb": pb_cpu,
            "ram_lbl": lbl_ram,
            "ram_pb": pb_ram,
            "gpu_lbl": lbl_gpu,
            "disks_frame": disks_frame,
        }
        self._tick_telemetry()

        cap_card = self._make_card(parent, "Підтримка графічного прискорення (AI ONNX, FFmpeg кодування/декодування, OpenCL)")
        cap_bg = str(cap_card.cget("bg"))

        cap_top = tk.Frame(cap_card, bg=cap_bg)
        cap_top.pack(fill=tk.X, pady=(0, 8))

        self._caps_summary_lbl = tk.Label(
            cap_top, text="Триває перевірка ONNX Runtime / GPU / FFmpeg...",
            fg="#ffcc00", bg=cap_bg, font=("Segoe UI", 9, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=540
        )
        self._caps_summary_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            cap_top, text="🔄 Оновити перевірку",
            command=lambda: self._start_async_caps_probe(force=True),
            bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            cap_top, text="✨ Найкраще для заліза",
            command=self._apply_recommended_hw_profile,
            bg="#1f4d3a", fg="#75e08b", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.RIGHT, padx=(6, 0))

        self._caps_container = tk.Frame(cap_card, bg=cap_bg)
        self._caps_container.pack(fill=tk.X)
        if self._last_caps is not None:
            self._render_caps_matrix(self._last_caps)

        cfg_card = self._make_card(parent, "Вибір процесора/графіки для AI та відображення")

        ai_var = tk.StringVar(value=self._choice_label_for(AI_BACKEND_CHOICES, str(hw.get("ai_backend", "cpu"))))
        thr_var = tk.StringVar(value=str(hw.get("ai_threads", 2)))
        aifps_var = tk.StringVar(value=str(hw.get("ai_max_fps_per_cam", 8.0)))
        dec_var = tk.StringVar(value=self._choice_label_for(HW_DECODE_CHOICES, str(hw.get("video_decode_mode", "cpu"))))
        gfps_var = tk.StringVar(value=str(hw.get("display_grid_fps", 12.0)))
        dfps_var = tk.StringVar(value=str(hw.get("display_detail_fps", 25.0)))
        resz_var = tk.StringVar(value=self._choice_label_for(RESIZE_CHOICES, str(hw.get("display_resize_method", "linear"))))
        ocl_var = tk.BooleanVar(value=bool(hw.get("use_opencv_opencl", False)))

        self._hw_widget_vars = {
            "ai_backend": ai_var,
            "ai_threads": thr_var,
            "video_decode_mode": dec_var,
            "display_resize_method": resz_var,
        }

        def _commit():
            hw["ai_backend"] = self._choice_key_from(AI_BACKEND_CHOICES, ai_var.get(), "cpu")
            hw["ai_threads"] = _clamp_int(thr_var.get(), 2, 1, 8)
            hw["ai_max_fps_per_cam"] = _clamp_float(aifps_var.get(), 8.0, 1.0, 30.0)
            hw["video_decode_mode"] = self._choice_key_from(HW_DECODE_CHOICES, dec_var.get(), "cpu")
            hw["display_grid_fps"] = _clamp_float(gfps_var.get(), 12.0, 3.0, 30.0)
            hw["display_detail_fps"] = _clamp_float(dfps_var.get(), 25.0, 5.0, 60.0)
            hw["display_resize_method"] = self._choice_key_from(RESIZE_CHOICES, resz_var.get(), "linear")
            hw["use_opencv_opencl"] = bool(ocl_var.get())

        self._page_committers.append(_commit)

        def _w_combo(var, choices):
            def _fn(box, bg):
                cb = ttk.Combobox(box, textvariable=var, values=[lbl for _, lbl in choices], state="readonly", width=52)
                cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                cb.bind("<<ComboboxSelected>>", lambda e: _commit())
                return cb
            return _fn

        def _w_spin(var, fr, to_, inc=1):
            def _fn(box, bg):
                sp = tk.Spinbox(box, from_=fr, to=to_, increment=inc, textvariable=var, width=10, bg="#252836", fg="#ffffff", buttonbackground="#2f3346", relief=tk.FLAT, font=("Segoe UI", 9), command=_commit)
                sp.pack(side=tk.LEFT)
                return sp
            return _fn

        self._add_row(cfg_card, "Бекенд нейромереж (AI):", "hardware.ai_backend", _w_combo(ai_var, AI_BACKEND_CHOICES))
        self._add_row(cfg_card, "Потоки ONNX / AI-воркери:", "hardware.ai_threads", _w_spin(thr_var, 1, 8))
        self._add_row(cfg_card, "Макс. AI FPS на камеру:", "hardware.ai_max_fps_per_cam", _w_spin(aifps_var, 1, 30))
        self._add_row(cfg_card, "Декодування потоків:", "hardware.video_decode_mode", _w_combo(dec_var, HW_DECODE_CHOICES))
        self._add_row(cfg_card, "Оновлення сітки (FPS):", "hardware.display_grid_fps", _w_spin(gfps_var, 3, 30))
        self._add_row(cfg_card, "Оновлення 1 камери (FPS):", "hardware.display_detail_fps", _w_spin(dfps_var, 5, 60))
        self._add_row(cfg_card, "Якість ресайзу на екрані:", "hardware.display_resize_method", _w_combo(resz_var, RESIZE_CHOICES))

        def _w_ocl(box, bg):
            cb = tk.Checkbutton(box, text="Увімкнути OpenCV OpenCL (GPU UMat) для обробки зображень", variable=ocl_var, command=_commit, bg=bg, fg="#ffffff", selectcolor="#252836", activebackground=bg, activeforeground="#ffffff", font=("Segoe UI", 9))
            cb.pack(side=tk.LEFT)
            return cb

        self._add_row(cfg_card, "OpenCV OpenCL:", "hardware.use_opencv_opencl", _w_ocl)

        pkg_card = self._make_card(parent, "Бібліотека ONNX Runtime у середовищі (.venv)")
        pkg_bg = str(pkg_card.cget("bg"))
        tk.Label(
            pkg_card,
            text=(
                "За замовчуванням встановлено CPU-версію ('onnxruntime'). Щоб задіяти відеокарту (DirectML або CUDA), "
                "можна перемкнути пакет нижче або через 1_SETUP.bat (CPU/GPU-пакети взаємозамінні в Python)."
            ),
            fg="#9ba3b8", bg=pkg_bg, font=("Segoe UI", 9), justify=tk.LEFT, wraplength=740
        ).pack(fill=tk.X, pady=(0, 8))

        btn_row = tk.Frame(pkg_card, bg=pkg_bg)
        btn_row.pack(fill=tk.X)

        def _install_pkg(pkg_name):
            if self._ort_installing:
                return
            if not messagebox.askyesno(
                "Перемикання ONNX Runtime",
                f"Встановити '{pkg_name}' у поточне Python-оточення (.venv)?\nЦе видалить попередній варіант onnxruntime* та завантажить вибраний з PyPI.",
                parent=self.win,
            ):
                return
            self._ort_installing = True
            self._ort_install_lbl.configure(text=f"Підготовка встановлення {pkg_name}...", fg="#ffcc00")

            def _worker():
                ok, msg = install_ort_package(
                    pkg_name,
                    progress_cb=lambda txt: self._safe_set_label(self._ort_install_lbl, txt, "#ffcc00"),
                )
                if ok:
                    if self.registry is not None:
                        try:
                            self.registry.refresh_libraries_capabilities()
                        except Exception:
                            pass
                    try:
                        self.hw_monitor.refresh_all(force=True)
                    except Exception:
                        pass

                def _done():
                    self._ort_installing = False
                    if self._is_closing:
                        return
                    self._safe_set_label(self._ort_install_lbl, msg, "#75e08b" if ok else "#ff5555")
                    if ok:
                        try:
                            self.camera_manager.inference_pool.invalidate_all()
                        except Exception:
                            pass
                        self._start_async_caps_probe(force=True)

                self._ui_queue.put((_done, ()))

            threading.Thread(target=_worker, daemon=True).start()

        state_pkg = tk.DISABLED if getattr(sys, "frozen", False) else tk.NORMAL
        tk.Button(
            btn_row, text="CPU (onnxruntime)", state=state_pkg,
            command=lambda: _install_pkg("onnxruntime"),
            bg="#2e3142", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=5, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_row, text="DirectML GPU (DX12 — будь-яка GPU)", state=state_pkg,
            command=lambda: _install_pkg("onnxruntime-directml"),
            bg="#233b56", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=5, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_row, text="NVIDIA CUDA (onnxruntime-gpu)", state=state_pkg,
            command=lambda: _install_pkg("onnxruntime-gpu"),
            bg="#1e442c", fg="#75e08b", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=5, cursor="hand2"
        ).pack(side=tk.LEFT)

        init_pkg_note = "Працює всередині EXE — для зміни бібліотеки скористайтесь 1_SETUP.bat." if getattr(sys, "frozen", False) else ""
        self._ort_install_lbl = tk.Label(pkg_card, text=init_pkg_note, fg="#9ba3b8", bg=pkg_bg, font=("Segoe UI", 8), anchor=tk.W, wraplength=740, justify=tk.LEFT)
        self._ort_install_lbl.pack(fill=tk.X, pady=(6, 0))

    def _safe_set_label(self, lbl, text, color):
        def _upd():
            if not self._is_closing and lbl is not None and lbl.winfo_exists():
                lbl.configure(text=text, fg=color)
        self._ui_queue.put((_upd, ()))

    def _tick_telemetry(self):
        self._cancel_telemetry_timer()
        if self._is_closing or self._current_node_id != "hardware" or not self._telemetry_labels:
            return
        try:
            m = self.hw_monitor.get_live_metrics()
            cpu = m["cpu"]
            ram = m["ram"]
            gpus = m["gpus"]
            disks = m["disks"]

            cores_txt = f"{cpu['physical_cores']} ядер / {cpu['logical_threads']} потоків" if cpu["physical_cores"] else f"{cpu['logical_threads']} потоків"
            freq_txt = f" @ {int(cpu['freq_mhz'])} МГц" if cpu.get("freq_mhz") else ""
            self._telemetry_labels["cpu_lbl"].configure(
                text=f"Процесор (CPU): {cpu['name']} ({cores_txt}{freq_txt})  —  Загальне: {cpu['usage_pct']:.1f}% | Програма: {cpu['process_pct']:.1f}%"
            )
            self._telemetry_labels["cpu_pb"]["value"] = min(100, max(0, cpu["usage_pct"]))

            self._telemetry_labels["ram_lbl"].configure(
                text=f"Оперативна пам'ять (RAM): зайнято {ram['used_gb']:.2f} / {ram['total_gb']:.2f} ГБ  (Вільно: {ram['free_gb']:.2f} ГБ, {ram['percent']:.0f}% | Програма: {ram['process_mb']:.0f} МБ)"
            )
            self._telemetry_labels["ram_pb"]["value"] = min(100, max(0, ram["percent"]))

            if gpus:
                g_lines = []
                for g in gpus:
                    if g.get("live_metrics"):
                        enc_s = f", ENC {g['enc_util_pct']:.0f}%" if g.get("enc_util_pct") is not None else ""
                        dec_s = f", DEC {g['dec_util_pct']:.0f}%" if g.get("dec_util_pct") is not None else ""
                        gpu_s = f"{g['gpu_util_pct']:.0f}%" if g.get("gpu_util_pct") is not None else "—"
                        g_lines.append(
                            f"• [{g['vendor']}] {g['name']} (Драйвер {g['driver']}): "
                            f"VRAM {g['vram_used_mb']:.0f}/{g['vram_total_mb']:.0f} МБ (Вільно {g['vram_free_mb']:.0f} МБ), "
                            f"GPU {gpu_s}{enc_s}{dec_s}"
                        )
                    else:
                        vram_s = f", Адаптер RAM: {g['vram_total_mb']:.0f} МБ" if g.get("vram_total_mb") else ""
                        g_lines.append(f"• [{g['vendor']}] {g['name']} (Драйвер {g['driver']}{vram_s})")
                self._telemetry_labels["gpu_lbl"].configure(text="Відеокарти (GPU):\n" + "\n".join(g_lines))
            else:
                self._telemetry_labels["gpu_lbl"].configure(text="Відеокарти (GPU): дискретних або WMI GPU не виявлено.")

            df = self._telemetry_labels["disks_frame"]
            bg = str(df.cget("bg"))
            for ch in list(df.winfo_children()):
                ch.destroy()
            tk.Label(df, text="Накопичувачі та місце під архів:", fg="#ffffff", bg=bg, font=("Segoe UI", 9, "bold"), anchor=tk.W).pack(fill=tk.X, pady=(2, 4))
            cur_arch = os.path.abspath(self.draft.get("general", {}).get("archive_root") or self.cfg_mgr.get_archive_root())

            for d in disks:
                row = tk.Frame(df, bg="#222432", padx=8, pady=4)
                row.pack(fill=tk.X, pady=2)
                mount = d["mount"]
                is_arch_drive = cur_arch.lower().startswith(mount.lower())
                badge_col = "#00e5ff" if "SSD" in d["media"].upper() else "#ffcc00"
                tk.Label(row, text=f"{mount}", fg="#ffffff", bg="#222432", font=("Consolas", 9, "bold"), width=6, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(row, text=f"[{d['media']}]", fg=badge_col, bg="#222432", font=("Segoe UI", 8, "bold"), width=11, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(
                    row,
                    text=f"Вільно: {d['free_gb']:.1f} / {d['total_gb']:.1f} ГБ  ({d['percent']:.0f}% зайнято) {d['fs']}",
                    fg="#d5dae8", bg="#222432", font=("Segoe UI", 9), anchor=tk.W
                ).pack(side=tk.LEFT, fill=tk.X, expand=True)

                def _set_drive_arch(m_path=mount):
                    new_path = os.path.join(m_path, "ANPR_Archive")
                    self.draft.setdefault("general", {})["archive_root"] = new_path
                    self.footer_status_lbl.configure(text=f"Новий архів: {new_path} (натисніть 'Застосувати')", fg="#00e5ff")
                    self._tick_telemetry()

                btn_txt = "★ Поточний архів" if is_arch_drive else "Обрати під архів"
                tk.Button(
                    row, text=btn_txt, command=_set_drive_arch,
                    bg="#1e442c" if is_arch_drive else "#2e3142",
                    fg="#75e08b" if is_arch_drive else "#00e5ff",
                    relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=2, cursor="hand2"
                ).pack(side=tk.RIGHT)
        except Exception:
            pass

        if not self._is_closing and self._current_node_id == "hardware":
            self._telemetry_after_id = self.win.after(1500, self._tick_telemetry)

    def _cancel_telemetry_timer(self):
        if self._telemetry_after_id is not None:
            try:
                self.win.after_cancel(self._telemetry_after_id)
            except Exception:
                pass
            self._telemetry_after_id = None

    def _start_async_caps_probe(self, force: bool = False):
        if self._caps_summary_lbl is not None:
            try:
                self._caps_summary_lbl.configure(text="Тестування CUDA / DirectML / OpenVINO / NVENC / FFmpeg...", fg="#ffcc00")
            except Exception:
                pass

        def _bg():
            caps = self.hw_monitor.probe_acceleration_capabilities(force=force)
            self._last_caps = caps

            def _apply():
                if self._is_closing:
                    return
                if self._current_node_id == "hardware" and self._caps_container is not None:
                    self._render_caps_matrix(caps)

            self._ui_queue.put((_apply, ()))

        threading.Thread(target=_bg, daemon=True).start()

    def _render_caps_matrix(self, caps: dict):
        if self._caps_container is None:
            return
        bg = str(self._caps_container.cget("bg"))
        for ch in list(self._caps_container.winfo_children()):
            ch.destroy()

        rec = caps.get("recommendations") or {}
        if self._caps_summary_lbl is not None:
            self._caps_summary_lbl.configure(
                text=f"Пакет ORT: {caps.get('ort_package')}  |  Рекомендовано: {rec.get('ai_text', '')}",
                fg="#75e08b" if rec.get("ai_backend") in ("cuda", "directml", "openvino") else "#00e5ff",
            )

        def _add_group(title, items):
            tk.Label(self._caps_container, text=title, fg="#c8d0e4", bg=bg, font=("Segoe UI", 9, "bold"), anchor=tk.W).pack(fill=tk.X, pady=(6, 2))
            for key, info in items.items():
                row = tk.Frame(self._caps_container, bg="#222432", padx=8, pady=3)
                row.pack(fill=tk.X, pady=1)
                sup = bool(info.get("supported"))
                st = str(info.get("status", ""))
                st_col = "#75e08b" if sup else ("#ffcc00" if "Потребує" in st else "#888c9e")
                mark = "✔" if sup else ("⚠" if "Потребує" in st else "✖")
                tk.Label(row, text=f"{mark} {info.get('label', key)}", fg="#ffffff", bg="#222432", font=("Segoe UI", 8, "bold"), width=42, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(row, text=st, fg=st_col, bg="#222432", font=("Segoe UI", 8, "bold"), width=24, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(row, text=info.get("reason", ""), fg="#9ba3b8", bg="#222432", font=("Segoe UI", 8), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

        _add_group("1. Нейромережі (ONNX Runtime):", caps.get("ai_backends") or {})
        _add_group("2. Відеокодування FFmpeg (для запису архіву):", caps.get("video_encoders") or {})

        extra = tk.Frame(self._caps_container, bg="#222432", padx=8, pady=5)
        extra.pack(fill=tk.X, pady=(6, 0))
        hw_dec = ", ".join(caps.get("hw_decoders") or []) or "немає"
        ocl = caps.get("opencv_opencl") or {}
        ocl_s = f"Так ({ocl.get('device', '')})" if ocl.get("supported") else "Ні"
        cuda_cv = caps.get("opencv_cuda") or {}
        cuda_cv_s = f"Так ({cuda_cv.get('count', 0)})" if cuda_cv.get("supported") else "Ні"
        tk.Label(
            extra,
            text=f"FFmpeg HW Decode: {hw_dec}   |   OpenCV OpenCL: {ocl_s}   |   OpenCV CUDA: {cuda_cv_s}   |   Рендер: {caps.get('ui_renderer', '')}",
            fg="#b8c0d4", bg="#222432", font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=740
        ).pack(fill=tk.X)

    def _apply_recommended_hw_profile(self):
        caps = self._last_caps or self.hw_monitor.probe_acceleration_capabilities(force=False)
        rec = caps.get("recommendations") or {}
        best_ai = rec.get("ai_backend", "cpu")
        ai_backends = caps.get("ai_backends") or {}
        if ai_backends.get("cuda", {}).get("supported"):
            chosen_ai = "cuda"
        elif ai_backends.get("directml", {}).get("supported"):
            chosen_ai = "directml"
        elif ai_backends.get("openvino", {}).get("supported"):
            chosen_ai = "openvino"
        else:
            chosen_ai = "cpu"

        hw = self.draft.setdefault("hardware", {})
        hw["ai_backend"] = chosen_ai
        hw["ai_threads"] = 2 if chosen_ai != "cpu" else min(4, max(2, (os.cpu_count() or 4) // 2))
        hw["video_decode_mode"] = "d3d11va" if chosen_ai in ("directml", "cuda") and "d3d11va" in (caps.get("hw_decoders") or []) else "cpu"
        hw["use_opencv_opencl"] = bool((caps.get("opencv_opencl") or {}).get("supported", False))

        if self._hw_widget_vars:
            if "ai_backend" in self._hw_widget_vars:
                self._hw_widget_vars["ai_backend"].set(self._choice_label_for(AI_BACKEND_CHOICES, hw["ai_backend"]))
            if "ai_threads" in self._hw_widget_vars:
                self._hw_widget_vars["ai_threads"].set(str(hw["ai_threads"]))
            if "video_decode_mode" in self._hw_widget_vars:
                self._hw_widget_vars["video_decode_mode"].set(self._choice_label_for(HW_DECODE_CHOICES, hw["video_decode_mode"]))

        note = f"Встановлено '{chosen_ai.upper()}' (поточний активний оптимум). "
        if chosen_ai == "cpu" and best_ai != "cpu":
            note += f"Для GPU-прискорення встановіть '{'onnxruntime-directml' if best_ai == 'directml' else 'onnxruntime-gpu'}' вище або в 1_SETUP.bat."
        self.footer_status_lbl.configure(text=note, fg="#75e08b")
        self._show_page("hardware")

    # ------------------------------------------------------------------
    # Сторінка 4: Модулі (plug-and-play) та стан бібліотек Python
    # ------------------------------------------------------------------
    def _request_app_restart(self):
        if self._is_closing:
            return
        if not messagebox.askyesno(
            "Перезапуск програми",
            "Зберегти поточні налаштування чернетки та перезапустити програму зараз,\nщоб застосувати оновлені нативні бібліотеки або модулі?",
            parent=self.win,
        ):
            return
        self._commit_current_page()
        self.cfg_mgr.save_data(self.draft)
        if self.on_apply_cb:
            try:
                self.on_apply_cb()
            except Exception:
                pass
        if self.on_restart_app_cb:
            self.on_restart_app_cb()

    def _trigger_modules_refresh(self):
        if self._is_closing:
            return
        self.footer_status_lbl.configure(text="Сканування scripts/modules/ та перевірка бібліотек...", fg="#ffcc00")
        self.win.update_idletasks()
        res = None
        if self.on_refresh_modules_cb:
            try:
                res = self.on_refresh_modules_cb(force=True)
            except Exception as e:
                self.footer_status_lbl.configure(text=f"Помилка оновлення модулів: {e}", fg="#ff5555")
                return
        else:
            if self.registry is not None:
                res = self.registry.scan_and_load(force=True)
            self.cfg_mgr.sync_with_registry(self.registry)
            try:
                self.hw_monitor.refresh_all(force=True)
            except Exception:
                pass
            if self.registry is not None:
                try:
                    self.registry.refresh_libraries_capabilities()
                except Exception:
                    pass
            try:
                self.camera_manager.reload_from_registry(force_all=True)
            except Exception:
                pass
            self.on_modules_reloaded()

        if isinstance(res, dict):
            a = len(res.get("added") or [])
            u = len(res.get("updated") or [])
            r = len(res.get("removed") or [])
            err_n = len(res.get("errors") or {})
            summary = f"Модулі та бібліотеки оновлено (+{a}, оновл. {u}, прибр. {r}"
            if err_n:
                summary += f", помилок: {err_n}"
            summary += ")."
            self.footer_status_lbl.configure(text=summary, fg="#ffcc00" if err_n else "#75e08b")
        else:
            self.footer_status_lbl.configure(text="✔ Модулі та бібліотеки оновлено!", fg="#75e08b")

    def _open_modules_folder(self):
        mod_dir = os.path.join(self.cfg_mgr.base_dir, "scripts", "modules")
        os.makedirs(mod_dir, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(mod_dir)
            else:
                subprocess.Popen(["xdg-open", mod_dir])
        except Exception as e:
            messagebox.showerror("Папка модулів", f"Не вдалося відкрити папку:\n{mod_dir}\n{e}", parent=self.win)

    def _build_page_modules(self, parent):
        mod_dir = os.path.join(self.cfg_mgr.base_dir, "scripts", "modules")
        self._section_header(
            parent,
            "🧩 Незалежні модулі та оновлення бібліотек",
            (
                f"Папка модулів: {mod_dir}\n"
                "Будь-який новий або оновлений .py модуль із контрактом MODULE_SPEC автоматично підхоплюється програмою без редагування ядра."
            ),
        )

        # Панель дій
        act_card = self._make_card(parent, "Керування плагінами та середовищем")
        act_bg = str(act_card.cget("bg"))
        btn_bar = tk.Frame(act_card, bg=act_bg)
        btn_bar.pack(fill=tk.X)

        tk.Button(
            btn_bar,
            text="🔄 Оновити модулі та бібліотеки",
            command=self._trigger_modules_refresh,
            bg="#007acc",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_bar,
            text="📂 Відкрити папку модулів",
            command=self._open_modules_folder,
            bg="#2f3346",
            fg="#00e5ff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 8))

        if self.on_restart_app_cb is not None:
            tk.Button(
                btn_bar,
                text="♻ Перезапустити програму",
                command=self._request_app_restart,
                bg="#3a2f22",
                fg="#ffcc00",
                relief=tk.FLAT,
                font=("Segoe UI", 9, "bold"),
                padx=12,
                pady=6,
                cursor="hand2",
            ).pack(side=tk.RIGHT)

        # Картка 1: Завантажені й проблемні модулі
        mods_card = self._make_card(parent, "Виявлені модулі (scripts/modules/)")
        m_bg = str(mods_card.cget("bg"))

        rows = self.registry.describe_modules_status() if self.registry is not None else []
        if not rows:
            tk.Label(
                mods_card,
                text="У папці scripts/modules/ не виявлено жодного модуля.",
                fg="#9ba3b8",
                bg=m_bg,
                font=("Segoe UI", 9),
                anchor=tk.W,
            ).pack(fill=tk.X)
        else:
            extra_pkgs = []
            for r in rows:
                is_err = (r.get("kind") == "error")
                row_bg = "#2a171c" if is_err else "#222432"
                border_col = "#ff5555" if is_err else "#2f3346"
                box = tk.Frame(mods_card, bg=row_bg, padx=10, pady=8, highlightthickness=1, highlightbackground=border_col)
                box.pack(fill=tk.X, pady=3)

                top = tk.Frame(box, bg=row_bg)
                top.pack(fill=tk.X)
                kind_lbl = "ПОМИЛКА" if is_err else ("Відеозапис" if r.get("kind") == "recorder" else "AI-модуль")
                title_col = "#ff7777" if is_err else "#ffffff"
                tk.Label(
                    top,
                    text=f"{r.get('icon', '🧩')} {r.get('name', r['id'])}  (id: {r['id']}  v{r.get('version', '—')})  [{kind_lbl}]",
                    fg=title_col,
                    bg=row_bg,
                    font=("Segoe UI", 9, "bold"),
                    anchor=tk.W,
                ).pack(side=tk.LEFT)
                tk.Label(
                    top,
                    text=os.path.basename(r.get("file", "")),
                    fg="#8d95ab",
                    bg=row_bg,
                    font=("Consolas", 8),
                ).pack(side=tk.RIGHT)

                st_col = "#ff9999" if is_err else ("#75e08b" if "Готовий" in str(r.get("status", "")) else "#ffcc00")
                tk.Label(
                    box,
                    text=f"Стан: {r.get('status', '')}",
                    fg=st_col,
                    bg=row_bg,
                    font=("Segoe UI", 8, "bold"),
                    anchor=tk.W,
                    wraplength=710,
                    justify=tk.LEFT,
                ).pack(fill=tk.X, pady=(2, 0))

                if not is_err and self.registry is not None:
                    spec = self.registry.get_spec(r["id"]) or {}
                    desc = str(spec.get("description") or "").strip()
                    deps = spec.get("library_deps") or []
                    extra_pkgs.extend(deps)
                    if desc:
                        tk.Label(
                            box,
                            text=desc,
                            fg="#b8c0d4",
                            bg=row_bg,
                            font=("Segoe UI", 8),
                            anchor=tk.W,
                            wraplength=710,
                            justify=tk.LEFT,
                        ).pack(fill=tk.X, pady=(2, 0))
                    if deps:
                        tk.Label(
                            box,
                            text=f"Залежності: {', '.join(deps)}",
                            fg="#8892aa",
                            bg=row_bg,
                            font=("Consolas", 8),
                            anchor=tk.W,
                        ).pack(fill=tk.X, pady=(2, 0))

                if is_err and r.get("error"):
                    tk.Label(
                        box,
                        text=str(r["error"]).strip(),
                        fg="#ffb3b3",
                        bg="#1f1014",
                        font=("Consolas", 8),
                        anchor=tk.W,
                        justify=tk.LEFT,
                        wraplength=700,
                        padx=6,
                        pady=4,
                    ).pack(fill=tk.X, pady=(4, 0))

            # Картка 2: Версії Python-бібліотек і виявлення потреби у перезапуску
            libs_card = self._make_card(
                parent,
                "Бібліотеки Python та динамічне оновлення можливостей",
                "Порівняння версій пакетів на диску та вже завантажених у RAM C-розширень (.pyd/.dll).",
            )
            l_bg = str(libs_card.cget("bg"))
            lib_rows = get_library_versions(extra_pkgs=extra_pkgs)

            if any(lr.get("needs_restart") for lr in lib_rows):
                alert = tk.Frame(libs_card, bg="#3b2814", padx=10, pady=8, highlightthickness=1, highlightbackground="#ffcc00")
                alert.pack(fill=tk.X, pady=(0, 8))
                tk.Label(
                    alert,
                    text=(
                        "⚠ Виявлено оновлені нативні C-бібліотеки на диску, стара версія яких вже завантажена в пам'ять процесу.\n"
                        "Для повної активації нових драйверів/бібліотек (ONNX Runtime, OpenCV тощо) виконайте перезапуск програми."
                    ),
                    fg="#ffe082",
                    bg="#3b2814",
                    font=("Segoe UI", 8, "bold"),
                    justify=tk.LEFT,
                    anchor=tk.W,
                    wraplength=560,
                ).pack(side=tk.LEFT, fill=tk.X, expand=True)
                if self.on_restart_app_cb is not None:
                    tk.Button(
                        alert,
                        text="♻ Перезапустити зараз",
                        command=self._request_app_restart,
                        bg="#d97706",
                        fg="#ffffff",
                        relief=tk.FLAT,
                        font=("Segoe UI", 8, "bold"),
                        padx=10,
                        pady=5,
                        cursor="hand2",
                    ).pack(side=tk.RIGHT, padx=(8, 0))

            hdr = tk.Frame(libs_card, bg="#252836", padx=8, pady=4)
            hdr.pack(fill=tk.X, pady=(0, 2))
            tk.Label(hdr, text="Бібліотека", fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), width=24, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(hdr, text="На диску", fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), width=12, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(hdr, text="У пам'яті", fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), width=22, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(hdr, text="Статус", fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

            for lr in lib_rows:
                lrow = tk.Frame(libs_card, bg="#222432", padx=8, pady=3)
                lrow.pack(fill=tk.X, pady=1)
                if lr.get("needs_restart"):
                    st_c = "#ffcc00"
                elif "Активна" in lr["status"]:
                    st_c = "#75e08b"
                elif "На диску" in lr["status"]:
                    st_c = "#9ee8ff"
                else:
                    st_c = "#888c9e"

                tk.Label(lrow, text=lr["name"], fg="#ffffff", bg="#222432", font=("Consolas", 8, "bold"), width=24, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(lrow, text=lr["disk_version"], fg="#d5dae8", bg="#222432", font=("Consolas", 8), width=12, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(lrow, text=lr["loaded_version"], fg="#b8c0d4", bg="#222432", font=("Segoe UI", 8), width=22, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(lrow, text=lr["status"], fg=st_c, bg="#222432", font=("Segoe UI", 8, "bold"), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

            if not rows:
                tk.Label(libs_card, text="Немає даних про бібліотеки.", fg="#888c9e", bg=l_bg).pack()

    # ------------------------------------------------------------------
    # Сторінка 5: Спільна база облич (динамічно через capability 'face_db')
    # ------------------------------------------------------------------
    def _get_face_db_accessor(self):
        try:
            return self.camera_manager.get_capability_accessor("face_db")
        except Exception:
            return None

    def _build_page_faces_db(self, parent):
        self._section_header(
            parent,
            "👥 Спільна база відомих облич (faces.txt)",
            f"Файл бази: {self.cfg_mgr.get_faces_db_path()} (спільна для всіх камер системи).",
        )
        ctrl_bar = tk.Frame(parent, bg="#1c1e2a", padx=12, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
        ctrl_bar.pack(fill=tk.X, pady=(0, 10))

        search_var = tk.StringVar()
        tk.Label(ctrl_bar, text="🔍 Пошук:", fg="#d5dae8", bg="#1c1e2a", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Entry(
            ctrl_bar, textvariable=search_var, width=28,
            bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, ipady=3, padx=(0, 8))

        count_lbl = tk.Label(ctrl_bar, text="", fg="#00e5ff", bg="#1c1e2a", font=("Segoe UI", 9, "bold"))
        count_lbl.pack(side=tk.LEFT, padx=8)

        list_container = tk.Frame(parent, bg="#161720")
        list_container.pack(fill=tk.BOTH, expand=True)

        def _render_list(*_):
            for ch in list(list_container.winfo_children()):
                ch.destroy()
            self._face_thumbs_cache.clear()
            db = self._get_face_db_accessor()
            if db is None:
                count_lbl.configure(text="Недоступно")
                tk.Label(
                    list_container,
                    text="Модуль розпізнавання облич із можливістю 'face_db' зараз не підключений у scripts/modules/.",
                    fg="#ffcc00", bg="#161720", font=("Segoe UI", 10), pady=24
                ).pack()
                return

            records = db.list_records()
            q = search_var.get().strip().upper()
            if q:
                records = [r for r in records if (q in r["id"].upper()) or (q in r["name"].upper())]
            count_lbl.configure(text=f"Записів: {len(records)}")

            if not records:
                tk.Label(
                    list_container,
                    text="База облич порожня (або нічого не знайдено за фільтром).\nОбличчя додаватимуться автоматично при роботі камер.",
                    fg="#888c9e", bg="#161720", font=("Segoe UI", 10), pady=24
                ).pack()
                return

            for rec in records:
                row = tk.Frame(list_container, bg="#1c1e2a", padx=10, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
                row.pack(fill=tk.X, pady=3)

                thumb_lbl = tk.Label(row, bg="#12131a", width=7, height=3, text="👤", fg="#70758a", font=("Segoe UI", 12))
                tpath = rec.get("thumb_path") or ""
                if Image is not None and ImageTk is not None and tpath and os.path.exists(tpath):
                    try:
                        with Image.open(tpath) as pim:
                            pim.load()
                            pim = pim.resize((52, 52), Image.Resampling.LANCZOS)
                            tkimg = ImageTk.PhotoImage(pim)
                            self._face_thumbs_cache.append(tkimg)
                            thumb_lbl.configure(image=tkimg, text="", width=52, height=52)
                    except Exception:
                        pass
                thumb_lbl.pack(side=tk.LEFT, padx=(0, 12))

                info_box = tk.Frame(row, bg="#1c1e2a")
                info_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
                ign = bool(rec.get("ignore", False))
                name_col = "#888c9e" if ign else "#ffffff"
                ign_badge = "  [ІГНОР]" if ign else ""
                tk.Label(
                    info_box,
                    text=f"{rec['name']}  ({rec['id']}){ign_badge}",
                    fg=name_col, bg="#1c1e2a", font=("Segoe UI", 10, "bold"), anchor=tk.W
                ).pack(fill=tk.X)
                tk.Label(
                    info_box,
                    text=f"Ракурсів у галереї: {rec.get('gallery_count', 1)}/4",
                    fg="#8c92a4", bg="#1c1e2a", font=("Segoe UI", 8), anchor=tk.W
                ).pack(fill=tk.X)

                btns = tk.Frame(row, bg="#1c1e2a")
                btns.pack(side=tk.RIGHT)

                def _rename(uid=rec["id"], cur_name=rec["name"]):
                    new_n = _ask_text_dialog(self.win, "Перейменування особи", f"Введіть нове ім'я для {uid}:", cur_name)
                    if new_n:
                        cams_info = [
                            {
                                "uuid": c["uuid"],
                                "name": c.get("name", ""),
                                "archive_dir": self.cfg_mgr.get_camera_archive_dir(c["uuid"]),
                            }
                            for c in (self.cfg_mgr.get_cameras() or [])
                            if c.get("uuid")
                        ]
                        try:
                            ok_ren = db.rename_record(uid, new_n, cameras_info=cams_info)
                        except TypeError:
                            ok_ren = db.rename_record(uid, new_n)
                        if ok_ren:
                            _render_list()

                def _toggle_ign(uid=rec["id"], cur_ign=ign):
                    if db.set_ignore(uid, not cur_ign):
                        _render_list()

                def _delete(uid=rec["id"], cur_name=rec["name"]):
                    if messagebox.askyesno("Видалення обличчя", f"Видалити '{cur_name}' ({uid}) з бази облич?", parent=self.win):
                        if db.delete_record(uid):
                            _render_list()

                tk.Button(btns, text="✎ Ім'я", command=_rename, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=3)
                tk.Button(
                    btns,
                    text="✅ Зняти ігнор" if ign else "🚫 Ігнорувати",
                    command=_toggle_ign,
                    bg="#3a2f22" if not ign else "#1f442c",
                    fg="#ffcc00" if not ign else "#75e08b",
                    relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2"
                ).pack(side=tk.LEFT, padx=3)
                tk.Button(btns, text="🗑", command=_delete, bg="#44252a", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=3)

        tk.Button(ctrl_bar, text="↻ Оновити", command=_render_list, bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2").pack(side=tk.RIGHT)
        search_var.trace_add("write", _render_list)
        _render_list()

    # ------------------------------------------------------------------
    # Сторінка 6: Список усіх камер
    # ------------------------------------------------------------------
    def _build_template_modules(self) -> dict:
        base_mods = make_default_camera_modules(registry=self.registry)
        saved_tpl = self.draft.get("default_camera_modules") or {}
        merged = copy.deepcopy(base_mods)
        for mid, sec in saved_tpl.items():
            if isinstance(sec, dict):
                if isinstance(merged.get(mid), dict):
                    merged[mid].update(copy.deepcopy(sec))
                else:
                    merged[mid] = copy.deepcopy(sec)
        return merged

    def _build_page_cameras_root(self, parent):
        self._section_header(
            parent,
            "📹 Керування камерами",
            "Додавайте необмежену кількість камер. Кожна отримує унікальний UUID, окрему папку архіву та власний набір модулів.",
        )

        bar = tk.Frame(parent, bg="#1c1e2a", padx=12, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
        bar.pack(fill=tk.X, pady=(0, 12))

        def _add_cam():
            cams = self.draft.setdefault("cameras", [])
            new_idx = len(cams) + 1
            tpl = self._build_template_modules()
            new_cam = make_camera_entry(
                name=f"Камера {new_idx}",
                main_source="",
                sub_source="",
                enabled=True,
                modules=tpl,
                registry=self.registry,
            )
            cams.append(new_cam)
            self._rebuild_nav_tree(select_id=f"cam:{new_cam['uuid']}")

        tk.Button(
            bar, text="➕ Додати нову камеру (авто UUID)", command=_add_cam,
            bg="#28a745", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=6, cursor="hand2"
        ).pack(side=tk.LEFT)

        reg_entries = self.registry.list_modules() if self.registry is not None else []
        rec_entries = self.registry.list_modules(kind="recorder") if self.registry is not None else []

        cams = self.draft.get("cameras") or []
        for idx, cam in enumerate(cams):
            uid = cam["uuid"]
            mods = cam.get("modules") or {}

            badges = []
            for ent in reg_entries:
                mid = ent["id"]
                if (mods.get(mid) or {}).get("on", False):
                    badges.append(ent["spec"].get("name", mid))
            mod_txt = ", ".join(badges) if badges else "Немає активних модулів"

            rec_parts = []
            for ent in rec_entries:
                mid = ent["id"]
                msec = mods.get(mid) or {}
                if msec.get("codec"):
                    rec_parts.append(f"{ent['spec'].get('name', mid)}: {msec.get('codec')}")
                elif msec.get("on", False):
                    rec_parts.append(f"{ent['spec'].get('name', mid)}: увімк.")
            rec_txt = "  |  " + ", ".join(rec_parts) if rec_parts else ""

            card = tk.Frame(parent, bg="#1c1e2a", padx=12, pady=10, highlightthickness=1, highlightbackground="#2b2e3d")
            card.pack(fill=tk.X, pady=4)

            top = tk.Frame(card, bg="#1c1e2a")
            top.pack(fill=tk.X)
            en_col = "#75e08b" if cam.get("enabled", True) else "#888c9e"
            en_txt = "АКТИВНА" if cam.get("enabled", True) else "ВИМКНЕНА"
            tk.Label(top, text=f"#{idx + 1}  {cam.get('name', 'Камера')}  [{en_txt}]", fg=en_col, bg="#1c1e2a", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
            tk.Label(top, text=f"UUID: {uid}", fg="#888c9e", bg="#1c1e2a", font=("Consolas", 8)).pack(side=tk.RIGHT)

            tk.Label(
                card,
                text=f"Main: {cam.get('main_source') or '—'}   |   Sub (сітка): {cam.get('sub_source') or '(масштабований Main)'}",
                fg="#c8d0e4", bg="#1c1e2a", font=("Segoe UI", 8), anchor=tk.W
            ).pack(fill=tk.X, pady=(4, 2))
            tk.Label(
                card,
                text=f"Модулі: {mod_txt}{rec_txt}",
                fg="#00e5ff", bg="#1c1e2a", font=("Segoe UI", 8), anchor=tk.W
            ).pack(fill=tk.X, pady=(0, 6))

            actions = tk.Frame(card, bg="#1c1e2a")
            actions.pack(fill=tk.X)

            tk.Button(
                actions, text="⚙ Налаштувати",
                command=lambda u=uid: self._select_nav_node(f"cam:{u}"),
                bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
            ).pack(side=tk.LEFT, padx=(0, 6))

            def _dup(c_src=cam):
                dup_mods = copy.deepcopy(c_src.get("modules") or make_default_camera_modules(registry=self.registry))
                dup_cam = make_camera_entry(
                    name=f"{c_src.get('name', 'Камера')} (копія)",
                    main_source=c_src.get("main_source", ""),
                    sub_source=c_src.get("sub_source", ""),
                    enabled=c_src.get("enabled", True),
                    modules=dup_mods,
                    registry=self.registry,
                )
                self.draft.setdefault("cameras", []).append(dup_cam)
                self._rebuild_nav_tree(select_id="cameras_root")

            def _move(pos, delta):
                clist = self.draft.get("cameras") or []
                new_pos = pos + delta
                if 0 <= new_pos < len(clist):
                    clist[pos], clist[new_pos] = clist[new_pos], clist[pos]
                    self._rebuild_nav_tree(select_id="cameras_root")

            def _delete(cname, u_del):
                clist = self.draft.get("cameras") or []
                if len(clist) <= 1:
                    messagebox.showwarning("Увага", "У системі повинна залишатися хоча б 1 камера.", parent=self.win)
                    return
                if messagebox.askyesno("Видалення камери", f"Видалити камеру '{cname}' зі списку?\n(Файли архіву на диску залишаться недоторканими).", parent=self.win):
                    self.draft["cameras"] = [c for c in clist if c.get("uuid") != u_del]
                    self._rebuild_nav_tree(select_id="cameras_root")

            tk.Button(actions, text="📋 Дублювати", command=_dup, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
            tk.Button(actions, text="⬆", command=lambda i=idx: _move(i, -1), state=tk.NORMAL if idx > 0 else tk.DISABLED, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=3, cursor="hand2").pack(side=tk.LEFT, padx=(0, 2))
            tk.Button(actions, text="⬇", command=lambda i=idx: _move(i, 1), state=tk.NORMAL if idx < len(cams) - 1 else tk.DISABLED, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=3, cursor="hand2").pack(side=tk.LEFT, padx=(0, 6))
            tk.Button(actions, text="🗑 Видалити", command=lambda cn=cam.get("name", ""), u=uid: _delete(cn, u), bg="#44252a", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.RIGHT)

    def _select_nav_node(self, node_id: str):
        if self.nav_tree.exists(node_id):
            self.nav_tree.selection_set(node_id)
            self.nav_tree.focus(node_id)
            self.nav_tree.see(node_id)

    # ------------------------------------------------------------------
    # Сторінка 7: Налаштування конкретної камери + динамічні модулі
    # ------------------------------------------------------------------
    def _build_page_single_camera(self, parent, cam_uuid: str):
        cam = None
        for c in self.draft.get("cameras") or []:
            if c.get("uuid") == cam_uuid:
                cam = c
                break
        if cam is None:
            tk.Label(parent, text="Камеру не знайдено.", fg="#ff5555", bg="#161720").pack()
            return

        self._section_header(
            parent,
            f"📹 Камера: {cam.get('name', 'Камера')}",
            f"Папка архіву цієї камери: {os.path.join(self.draft.get('general', {}).get('archive_root') or self.cfg_mgr.get_archive_root(), 'cameras', cam_uuid)}",
        )

        # Картка 1: Потоки та ідентифікація камери
        src_card = self._make_card(parent, "Підключення та потоки")
        src_bg = str(src_card.cget("bg"))

        uid_var = tk.StringVar(value=str(cam["uuid"]))
        name_var = tk.StringVar(value=str(cam.get("name", "")))
        en_var = tk.BooleanVar(value=bool(cam.get("enabled", True)))
        main_var = tk.StringVar(value=str(cam.get("main_source", "")))
        sub_var = tk.StringVar(value=str(cam.get("sub_source", "")))

        mod_entries = self.registry.list_modules() if self.registry is not None else []
        mods = cam.setdefault("modules", make_default_camera_modules(registry=self.registry))
        mod_vars = {}
        for ent in mod_entries:
            mid = ent["id"]
            spec = ent["spec"]
            def_cfg = self.registry.get_default_module_config(mid)
            msec = mods.setdefault(mid, {})
            for k, v in def_cfg.items():
                msec.setdefault(k, copy.deepcopy(v))
            mod_vars[mid] = tk.BooleanVar(value=bool(msec.get("on", spec.get("default_enabled", True))))

        test_res_lbl = tk.Label(src_card, text="", fg="#9ba3b8", bg=src_bg, font=("Segoe UI", 8), anchor=tk.W)

        def _commit_cam_base():
            cam["name"] = name_var.get().strip() or "Камера"
            cam["enabled"] = bool(en_var.get())
            cam["main_source"] = main_var.get().strip()
            cam["sub_source"] = sub_var.get().strip()
            for m_key, m_var in mod_vars.items():
                mods.setdefault(m_key, {})["on"] = bool(m_var.get())
            if self.nav_tree.exists(f"cam:{cam_uuid}"):
                p_icon = "●" if cam["enabled"] else "⏸"
                suf = "" if cam["enabled"] else " (вимк.)"
                self.nav_tree.item(f"cam:{cam_uuid}", text=f"  {p_icon} {cam['name']}{suf}")

        self._page_committers.append(_commit_cam_base)
        base_committer_count = len(self._page_committers)

        name_var.trace_add("write", lambda *_: _commit_cam_base())
        main_var.trace_add("write", lambda *_: _commit_cam_base())
        sub_var.trace_add("write", lambda *_: _commit_cam_base())

        def _w_uuid(box, bg):
            e = tk.Entry(box, textvariable=uid_var, state="readonly", readonlybackground="#20222e", fg="#00e5ff", relief=tk.FLAT, font=("Consolas", 9))
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 6))

            def _copy_uid():
                try:
                    self.win.clipboard_clear()
                    self.win.clipboard_append(uid_var.get())
                    self.footer_status_lbl.configure(text=f"UUID {uid_var.get()} скопійовано.", fg="#00e5ff")
                except Exception:
                    pass

            tk.Button(box, text="📋 Копіювати", command=_copy_uid, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT)
            return e

        self._add_row(src_card, "UUID камери:", "camera.uuid", _w_uuid)

        def _w_name(box, bg):
            e = tk.Entry(box, textvariable=name_var, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9))
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))
            tk.Checkbutton(box, text="Камера увімкнена", variable=en_var, command=_commit_cam_base, bg=bg, fg="#75e08b", selectcolor="#252836", activebackground=bg, activeforeground="#75e08b", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            return e

        self._add_row(src_card, "Назва / опис:", "camera.name", _w_name)

        def _run_stream_test(kind_label, raw_src):
            src_val = raw_src.strip()
            test_res_lbl.configure(text=f"Тестування '{kind_label}' ({src_val})...", fg="#ffcc00")
            dec_mode = str(self.draft.get("hardware", {}).get("video_decode_mode", "cpu"))

            def _t():
                ok, msg = probe_stream_info(src_val, decode_mode=dec_mode)
                self._safe_set_label(test_res_lbl, f"[{kind_label}] {msg}", "#75e08b" if ok else "#ff5555")

            threading.Thread(target=_t, daemon=True).start()

        def _w_source(var, kind_lbl):
            def _fn(box, bg):
                e = tk.Entry(box, textvariable=var, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9))
                e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 4))

                def _browse_file():
                    fp = filedialog.askopenfilename(
                        parent=self.win,
                        title=f"Оберіть відеофайл ({kind_lbl})",
                        filetypes=[("Відеофайли", "*.mp4 *.avi *.mkv *.mov *.ts"), ("Усі файли", "*.*")],
                    )
                    if fp:
                        var.set(os.path.normpath(fp))
                        _commit_cam_base()

                tk.Button(box, text="📁", command=_browse_file, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=3, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
                tk.Button(box, text="▶ Тест", command=lambda: _run_stream_test(kind_lbl, var.get()), bg="#233b56", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT)
                return e
            return _fn

        self._add_row(src_card, "Основний потік (Main):", "camera.main_source", _w_source(main_var, "Main"))
        self._add_row(src_card, "Стиснутий потік (Sub сітки):", "camera.sub_source", _w_source(sub_var, "Sub"))
        test_res_lbl.pack(fill=tk.X, pady=(4, 0))

        # Картка 2: ОДНА ПАНЕЛЬ З ЧЕКБОКСАМИ МОДУЛІВ В РЯДОК (динамічно з ModuleRegistry)
        mod_bar = self._make_card(parent, "Підключення модулів аналітики та запису для цієї камери")
        m_bg = str(mod_bar.cget("bg"))
        chk_container = tk.Frame(mod_bar, bg=m_bg)
        chk_container.pack(fill=tk.X, pady=(0, 4))

        inline_mods_holder = tk.Frame(parent, bg="#161720")
        inline_mods_holder.pack(fill=tk.X)

        def _rebuild_inline_module_blocks():
            for fn in list(self._page_committers[base_committer_count:]):
                try:
                    fn()
                except Exception:
                    pass
            self._page_committers = self._page_committers[:base_committer_count]
            _commit_cam_base()

            for ch in list(inline_mods_holder.winfo_children()):
                ch.destroy()

            for ent in mod_entries:
                mid = ent["id"]
                if mod_vars[mid].get():
                    self._build_dynamic_module_card(inline_mods_holder, ent["spec"], mods.setdefault(mid, {}))

        if not mod_entries:
            tk.Label(
                chk_container,
                text="У scripts/modules/ не знайдено жодного модуля.",
                fg="#888c9e",
                bg=m_bg,
                font=("Segoe UI", 9),
            ).pack(anchor=tk.W)
        else:
            per_row = 4
            for r_idx in range(0, len(mod_entries), per_row):
                row_fr = tk.Frame(chk_container, bg=m_bg)
                row_fr.pack(fill=tk.X, pady=2)
                for ent in mod_entries[r_idx:r_idx + per_row]:
                    mid = ent["id"]
                    spec = ent["spec"]
                    m_label = f"{spec.get('icon', '🧩')} {spec.get('name', mid)}"
                    grp = tk.Frame(row_fr, bg="#252836", padx=8, pady=5)
                    grp.pack(side=tk.LEFT, padx=(0, 10))
                    tk.Checkbutton(
                        grp,
                        text=m_label,
                        variable=mod_vars[mid],
                        command=_rebuild_inline_module_blocks,
                        bg="#252836",
                        fg="#ffffff",
                        selectcolor="#151720",
                        activebackground="#252836",
                        activeforeground="#00e5ff",
                        font=("Segoe UI", 9, "bold"),
                        cursor="hand2",
                    ).pack(side=tk.LEFT)
                    HelpButton(
                        grp,
                        help_text=ConfigManager.get_help_text(f"{mid}.on", registry=self.registry),
                        title=m_label,
                        bg="#252836",
                    ).pack(side=tk.LEFT, padx=(4, 0))

        tools_row = tk.Frame(mod_bar, bg=m_bg)
        tools_row.pack(fill=tk.X, pady=(6, 0))

        def _copy_modules_to_others():
            self._commit_current_page()
            others = [c for c in (self.draft.get("cameras") or []) if c.get("uuid") != cam_uuid]
            if not others:
                messagebox.showinfo("Копіювання", "Немає інших камер для копіювання.", parent=self.win)
                return
            _open_copy_modules_modal(
                self.win, cam, others,
                on_done=lambda n: self.footer_status_lbl.configure(
                    text=f"Налаштування модулів скопійовано на {n} камер(у/и) (натисніть 'Застосувати').",
                    fg="#00e5ff",
                ),
            )

        def _save_as_template():
            self._commit_current_page()
            self.draft["default_camera_modules"] = copy.deepcopy(mods)
            self.footer_status_lbl.configure(text="Конфігурацію модулів збережено як шаблон для нових камер.", fg="#75e08b")

        tk.Button(
            tools_row, text="📋 Копіювати налаштування модулів на інші камери...",
            command=_copy_modules_to_others, bg="#2f3346", fg="#00e5ff",
            relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            tools_row, text="💾 Зберегти як шаблон нових камер",
            command=_save_as_template, bg="#2f3346", fg="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.LEFT)

        _rebuild_inline_module_blocks()

    # ------------------------------------------------------------------
    # Декларативний рендерер картки модуля з MODULE_SPEC['settings_schema']
    # ------------------------------------------------------------------
    def _build_dynamic_module_card(self, parent, spec: dict, sec: dict):
        mid = str(spec.get("id", "module"))
        title = f"{spec.get('icon', '🧩')} Налаштування: {spec.get('name', mid)}"
        subtitle = str(spec.get("description") or "").strip()
        card = self._make_card(parent, title, subtitle=subtitle)

        schema = spec.get("settings_schema") or []
        if not schema:
            tk.Label(
                card,
                text="Цей модуль не має додаткових параметрів.",
                fg="#8d95ab",
                bg=str(card.cget("bg")),
                font=("Segoe UI", 8),
            ).pack(anchor=tk.W)
            return

        raw_vars = {}
        choices_cache = {}
        conditional_bindings = []  # list of (widgets_list, disable_when_dict)
        rows_by_key = {}
        notice_items = []

        building_card = True

        def _current_values_dict() -> dict:
            vals = {}
            for fld in schema:
                k = fld.get("key")
                if not k or k not in raw_vars:
                    continue
                ftype = str(fld.get("type", "string")).lower()
                if ftype == "bool":
                    vals[k] = bool(raw_vars[k].get())
                elif ftype == "int":
                    vals[k] = _clamp_int(raw_vars[k].get(), fld.get("default", 0), fld.get("min", -10**9), fld.get("max", 10**9))
                elif ftype == "float":
                    vals[k] = _clamp_float(raw_vars[k].get(), fld.get("default", 0.0), fld.get("min", -1e9), fld.get("max", 1e9))
                elif ftype == "choice":
                    choices = choices_cache.get(k) or []
                    dflt = fld.get("default", choices[0][0] if choices else "")
                    vals[k] = self._choice_key_from(choices, raw_vars[k].get(), dflt)
                elif ftype == "date_range":
                    vals[k] = raw_vars[k].get().strip()
                    k_to = fld.get("key_to")
                    if k_to and k_to in raw_vars:
                        vals[k_to] = raw_vars[k_to].get().strip()
                else:
                    vals[k] = raw_vars[k].get().strip()
            return vals

        def _commit():
            if building_card:
                return
            vals = _current_values_dict()
            for k, v in vals.items():
                sec[k] = v

        def _sync_dynamic_states(*_):
            if building_card:
                return
            vals = _current_values_dict()
            for widgets, cond in conditional_bindings:
                is_disabled = _matches_condition(cond, vals)
                for w in widgets:
                    try:
                        if isinstance(w, ttk.Combobox):
                            w.configure(state=tk.DISABLED if is_disabled else "readonly")
                        else:
                            w.configure(state=tk.DISABLED if is_disabled else tk.NORMAL)
                    except Exception:
                        pass

            for n_info in notice_items:
                n_lbl = n_info["label"]
                after_row = rows_by_key.get(n_info["after_key"])
                active = _matches_condition(n_info["when"], vals)
                try:
                    if active:
                        if str(n_lbl.winfo_manager()) != "pack":
                            if after_row is not None:
                                n_lbl.pack(fill=tk.X, pady=(4, 6), after=after_row)
                            else:
                                n_lbl.pack(fill=tk.X, pady=(4, 6))
                    else:
                        n_lbl.pack_forget()
                except Exception:
                    pass
            _commit()

        self._page_committers.append(_commit)

        for fld in schema:
            key = fld.get("key")
            if not key:
                continue
            ftype = str(fld.get("type", "string")).lower()
            label = str(fld.get("label") or f"{key}:")
            help_text = str(fld.get("help") or "")
            disable_cond = fld.get("disable_when")

            created_widgets = []

            if ftype == "bool":
                var = tk.BooleanVar(value=bool(sec.get(key, fld.get("default", False))))
                raw_vars[key] = var
                bool_txt = str(fld.get("bool_text") or "")

                def _fn_bool(box, bg_c, v=var, bt=bool_txt):
                    cb = tk.Checkbutton(
                        box, text=bt, variable=v, command=_sync_dynamic_states,
                        bg=bg_c, fg="#ffffff", selectcolor="#252836",
                        activebackground=bg_c, activeforeground="#ffffff", font=("Segoe UI", 9),
                    )
                    cb.pack(side=tk.LEFT)
                    created_widgets.append(cb)
                    return cb

                row, _ = self._add_row(card, label, f"{mid}.{key}", _fn_bool, help_text=help_text)

            elif ftype in ("int", "float"):
                dflt = fld.get("default", 0 if ftype == "int" else 0.0)
                min_v = fld.get("min", 0)
                max_v = fld.get("max", 10000)
                step_v = fld.get("step", 1)
                var = tk.StringVar(value=str(sec.get(key, dflt)))
                raw_vars[key] = var
                var.trace_add("write", lambda *_: _commit())

                def _fn_num(box, bg_c, v=var, fr=min_v, to_=max_v, inc=step_v):
                    sp = tk.Spinbox(
                        box, from_=fr, to=to_, increment=inc, textvariable=v, width=12,
                        bg="#252836", fg="#ffffff", buttonbackground="#2f3346", relief=tk.FLAT,
                        font=("Segoe UI", 9), command=_sync_dynamic_states,
                    )
                    sp.pack(side=tk.LEFT)
                    created_widgets.append(sp)
                    return sp

                row, _ = self._add_row(card, label, f"{mid}.{key}", _fn_num, help_text=help_text)

            elif ftype == "choice":
                prov_fn = fld.get("choices_provider")
                if callable(prov_fn):
                    try:
                        choices = list(prov_fn() or [])
                    except Exception:
                        choices = []
                else:
                    choices = list(fld.get("choices") or [])
                if not choices:
                    cur_d = str(sec.get(key, fld.get("default", "")))
                    choices = [(cur_d, cur_d)]

                choices_cache[key] = choices
                cur_val = str(sec.get(key, fld.get("default", choices[0][0])))
                # Якщо збережений ключ наразі відсутній у динамічному каталозі, не губимо його
                if cur_val and all(str(ck).lower() != cur_val.lower() for ck, _ in choices):
                    choices.append((cur_val, cur_val))

                var = tk.StringVar(value=self._choice_label_for(choices, cur_val))
                raw_vars[key] = var

                def _fn_choice(box, bg_c, v=var, ch=choices):
                    cb = ttk.Combobox(
                        box, textvariable=v, values=[lbl for _, lbl in ch],
                        state="readonly", width=52,
                    )
                    cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    cb.bind("<<ComboboxSelected>>", _sync_dynamic_states)
                    created_widgets.append(cb)
                    return cb

                row, _ = self._add_row(card, label, f"{mid}.{key}", _fn_choice, help_text=help_text)

            elif ftype == "date_range":
                key_to = str(fld.get("key_to") or "date_to")
                v1 = tk.StringVar(value=str(sec.get(key, fld.get("default", ""))))
                v2 = tk.StringVar(value=str(sec.get(key_to, fld.get("default_to", ""))))
                raw_vars[key] = v1
                raw_vars[key_to] = v2
                v1.trace_add("write", lambda *_: _commit())
                v2.trace_add("write", lambda *_: _commit())

                def _fn_dates(box, bg_c, var1=v1, var2=v2):
                    tk.Label(box, text="З:", fg="#9ba3b8", bg=bg_c, font=("Segoe UI", 8)).pack(side=tk.LEFT)
                    e1 = tk.Entry(box, textvariable=var1, width=12, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Consolas", 9))
                    e1.pack(side=tk.LEFT, ipady=3, padx=(4, 10))
                    tk.Label(box, text="По:", fg="#9ba3b8", bg=bg_c, font=("Segoe UI", 8)).pack(side=tk.LEFT)
                    e2 = tk.Entry(box, textvariable=var2, width=12, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Consolas", 9))
                    e2.pack(side=tk.LEFT, ipady=3, padx=(4, 8))
                    tk.Label(box, text="(YYYY-MM-DD або порожньо)", fg="#767c90", bg=bg_c, font=("Segoe UI", 8)).pack(side=tk.LEFT)
                    created_widgets.extend([e1, e2])
                    return box

                row, _ = self._add_row(card, label, f"{mid}.{key}", _fn_dates, help_text=help_text)

            else:
                var = tk.StringVar(value=str(sec.get(key, fld.get("default", ""))))
                raw_vars[key] = var
                var.trace_add("write", lambda *_: _commit())

                def _fn_str(box, bg_c, v=var):
                    e = tk.Entry(box, textvariable=v, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9))
                    e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3)
                    created_widgets.append(e)
                    return e

                row, _ = self._add_row(card, label, f"{mid}.{key}", _fn_str, help_text=help_text)

            rows_by_key[key] = row
            if isinstance(disable_cond, dict) and disable_cond and created_widgets:
                conditional_bindings.append((created_widgets, disable_cond))

        for n_def in (spec.get("notices") or []):
            if not isinstance(n_def, dict) or not n_def.get("text"):
                continue
            n_lbl = tk.Label(
                card,
                text=str(n_def["text"]),
                fg=str(n_def.get("fg") or "#00e5ff"),
                bg=str(n_def.get("bg") or "#1b2c3b"),
                font=("Segoe UI", 8, "bold"),
                padx=10,
                pady=6,
                justify=tk.LEFT,
                wraplength=720,
                anchor=tk.W,
            )
            notice_items.append({
                "label": n_lbl,
                "after_key": str(n_def.get("after_key") or ""),
                "when": n_def.get("when") or {},
            })

        building_card = False
        _sync_dynamic_states()

    @staticmethod
    def _choice_label_for(choices, key: str) -> str:
        k = str(key or "").strip().lower()
        for ck, lbl in choices:
            if str(ck).strip().lower() == k:
                return lbl
        return choices[0][1] if choices else str(key)

    @staticmethod
    def _choice_key_from(choices, label: str, default: str) -> str:
        for ck, lbl in choices:
            if lbl == label or str(ck) == str(label):
                return ck
        return default

    # ------------------------------------------------------------------
    # Збереження і застосування
    # ------------------------------------------------------------------
    def _apply_changes(self):
        self._commit_current_page()
        self.cfg_mgr.save_data(self.draft)
        self.draft = self.cfg_mgr.get_snapshot()
        if self.on_apply_cb:
            try:
                self.on_apply_cb()
            except Exception as e:
                messagebox.showerror("Помилка застосування", f"Не вдалося застосувати налаштування:\n{e}", parent=self.win)
                return
        self.footer_status_lbl.configure(text="✔ Налаштування збережено та застосовано!", fg="#75e08b")
        self._rebuild_nav_tree(select_id=self._current_node_id)

    def _save_and_close(self):
        self._apply_changes()
        self.close()

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        close_active_help_popover()
        self._cancel_telemetry_timer()
        if getattr(self, "_ui_poll_after_id", None) is not None:
            try:
                self.win.after_cancel(self._ui_poll_after_id)
            except Exception:
                pass
            self._ui_poll_after_id = None
        try:
            self.geom_tracker.close()
        except Exception:
            pass
        try:
            self.win.destroy()
        except Exception:
            pass
        if self.on_close_cb:
            try:
                self.on_close_cb()
            except Exception:
                pass


def _ask_text_dialog(parent, title: str, prompt: str, initial: str = "") -> str:
    res = {"val": None}
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.configure(bg="#1c1e2a")
    dlg.transient(parent)
    dlg.grab_set()
    dlg.resizable(False, False)

    tk.Label(dlg, text=prompt, fg="#ffffff", bg="#1c1e2a", font=("Segoe UI", 9, "bold"), anchor=tk.W).pack(fill=tk.X, padx=16, pady=(14, 6))
    var = tk.StringVar(value=str(initial or ""))
    ent = tk.Entry(dlg, textvariable=var, width=36, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 10))
    ent.pack(fill=tk.X, padx=16, pady=(0, 12), ipady=4)
    ent.select_range(0, tk.END)
    ent.focus_set()

    btns = tk.Frame(dlg, bg="#1c1e2a")
    btns.pack(fill=tk.X, padx=16, pady=(0, 14))

    def _ok(*_):
        v = var.get().strip()
        if v:
            res["val"] = v
        dlg.destroy()

    ent.bind("<Return>", _ok)
    dlg.bind("<Escape>", lambda e: dlg.destroy())
    tk.Button(btns, text="Скасувати", command=dlg.destroy, bg="#353849", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4).pack(side=tk.RIGHT, padx=(6, 0))
    tk.Button(btns, text="Зберегти", command=_ok, bg="#28a745", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4).pack(side=tk.RIGHT)

    parent.update_idletasks()
    px = parent.winfo_rootx() + max(20, (parent.winfo_width() - 360) // 2)
    py = parent.winfo_rooty() + max(20, (parent.winfo_height() - 140) // 2)
    dlg.geometry(f"+{px}+{py}")
    dlg.wait_window()
    return res["val"]


def _open_copy_modules_modal(parent, src_cam: dict, target_cams: list, on_done=None):
    dlg = tk.Toplevel(parent)
    dlg.title("Копіювання модулів")
    dlg.configure(bg="#1c1e2a")
    dlg.transient(parent)
    dlg.grab_set()

    tk.Label(
        dlg,
        text=f"Скопіювати модулі та їхні налаштування з '{src_cam.get('name', 'Камера')}' на:",
        fg="#00e5ff", bg="#1c1e2a", font=("Segoe UI", 10, "bold"), padx=14, pady=12, anchor=tk.W
    ).pack(fill=tk.X)

    vars_map = []
    for c in target_cams:
        v = tk.BooleanVar(value=True)
        vars_map.append((c, v))
        tk.Checkbutton(
            dlg,
            text=f"{c.get('name', 'Камера')}  ({c.get('main_source') or c['uuid'][:8]})",
            variable=v, bg="#1c1e2a", fg="#ffffff", selectcolor="#252836",
            activebackground="#1c1e2a", activeforeground="#ffffff", font=("Segoe UI", 9), anchor=tk.W
        ).pack(fill=tk.X, padx=16, pady=2)

    btns = tk.Frame(dlg, bg="#1c1e2a", padx=14, pady=12)
    btns.pack(fill=tk.X)

    def _apply():
        cnt = 0
        src_mods = src_cam.get("modules") or {}
        for tc, v in vars_map:
            if v.get():
                tc["modules"] = copy.deepcopy(src_mods)
                cnt += 1
        dlg.destroy()
        if on_done and cnt > 0:
            on_done(cnt)

    tk.Button(btns, text="Скасувати", command=dlg.destroy, bg="#353849", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4).pack(side=tk.RIGHT, padx=(6, 0))
    tk.Button(btns, text="Копіювати", command=_apply, bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4).pack(side=tk.RIGHT)

    parent.update_idletasks()
    px = parent.winfo_rootx() + max(20, (parent.winfo_width() - 420) // 2)
    py = parent.winfo_rooty() + max(20, (parent.winfo_height() - 260) // 2)
    dlg.geometry(f"+{px}+{py}")
    dlg.wait_window()
