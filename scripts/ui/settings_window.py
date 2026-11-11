import os
import re
import sys
import copy
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser

from scripts.core.i18n import (
    DEFAULT_LANGUAGE,
    get_language,
    get_language_label,
    list_languages,
    normalize_language_code,
    parse_language_code,
    set_language,
    tr,
)
from scripts.core.config_manager import (
    make_camera_entry,
    make_default_camera_modules,
    ConfigManager,
)
from scripts.core.hardware_info import (
    install_ort_package,
    get_library_versions,
    get_installed_ort_package,
    check_cuda_libraries_status,
    download_cuda_libraries,
    prepare_cuda_dll_environment,
    get_cuda_dir,
    assess_ai_modes,
    ensure_ai_backend_dependencies,
    build_full_autosetup_plan,
    run_full_autosetup,
)
from scripts.engine.camera_manager import probe_stream_info
from scripts.ui.helpers import (
    WindowGeometryTracker,
    ScrollableFrame,
    HelpButton,
    close_active_help_popover,
    bind_single_line_ellipsis,
    set_single_line_ellipsis_text,
    bind_hover_tooltip,
    ensure_clam_theme,
    configure_dark_combobox_style,
    style_dark_combobox_popdown,
)

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

EVT_TREE_SELECT = "<<TreeviewSelect>>"


def _get_ai_backend_choices() -> list:
    return [
        ("cpu", tr("ui.settings.choice.ai.cpu", "CPU (CPUExecutionProvider — default, maximum compatibility)")),
        ("auto", tr("ui.settings.choice.ai.auto", "AUTO (auto-select best available accelerator)")),
        ("directml", tr("ui.settings.choice.ai.directml", "DirectML GPU (DX12 — NVIDIA / AMD / Intel)")),
        ("cuda", tr("ui.settings.choice.ai.cuda", "NVIDIA CUDA GPU (fastest speed on RTX / GeForce)")),
        ("openvino", tr("ui.settings.choice.ai.openvino", "Intel OpenVINO (Intel CPU / iGPU)")),
    ]


def _get_hw_decode_choices() -> list:
    return [
        ("cpu", tr("ui.settings.choice.dec.cpu", "cpu — Standard CPU decoder (most reliable)")),
        ("auto", tr("ui.settings.choice.dec.auto", "auto — Auto-select GPU hardware decoder (CPU fallback)")),
        ("d3d11va", tr("ui.settings.choice.dec.d3d11va", "d3d11va — Windows Direct3D 11 GPU decoder")),
        ("cuda", tr("ui.settings.choice.dec.cuda", "cuda — NVIDIA NVDEC GPU decoder")),
        ("qsv", tr("ui.settings.choice.dec.qsv", "qsv — Intel QuickSync decoder")),
    ]


def _get_resize_choices() -> list:
    return [
        ("linear", tr("ui.settings.choice.resize.linear", "linear / area — Optimal balance of speed and clarity")),
        ("nearest", tr("ui.settings.choice.resize.nearest", "nearest — Lowest CPU load")),
        ("lanczos", tr("ui.settings.choice.resize.lanczos", "lanczos — Maximum sharpness (higher CPU load)")),
    ]


def _get_grid_mode_choices() -> list:
    return [
        ("auto", tr("ui.settings.choice.grid.auto", "auto")),
        ("manual", tr("ui.settings.choice.grid.manual", "manual")),
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
        self._hw_restart_bar = None
        self._hw_auto_box = None
        self._hw_auto_note = None
        self._inspected_ai_mode = None
        self._mode_tiles_frame = None
        self._mode_tile_widgets = {}
        self._mode_detail_frame = None
        self._mode_setup_pb = None
        self._mode_setup_lbl = None
        self._mode_setup_status = None
        self._hw_details_frame = None
        self._hw_details_expanded = False
        self._dir_dialog_open = False
        self._syncing_ai_backend_var = False
        self._ort_installing = False
        self._cuda_downloading = False
        self._autosetup_running = False
        self._autosetup_dlg = None
        self._autosetup_dlg_state = None
        self._last_caps = None
        self._last_cuda_status = None
        self._tree_syncing = False
        self._lang_var = None
        self._lang_combo = None
        self._footer_is_default = True
        self._btn_cancel = None
        self._btn_save_close = None
        self._btn_apply = None
        self._lbl_nav_header = None
        self._ui_queue = queue.SimpleQueue()
        self._ui_poll_after_id = None

        self.win = tk.Toplevel(master)
        self.win.title(tr("ui.settings.window_title", "System Settings — ANPR Multi-Cam NVR"))
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
        style = ensure_clam_theme(self.win)
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
        configure_dark_combobox_style(
            self.win,
            style_name="SettingsDark.TCombobox",
            field_bg="#232635",
            hover_field_bg="#2a2e42",
            btn_bg="#2c3044",
            btn_hover_bg="#363b52",
            fg="#ffffff",
            arrow_color="#00e5ff",
            border_color="#2e3347",
        )
        style.configure("SettingsDark.Horizontal.TProgressbar", troughcolor="#252836", background="#00e5ff")

        footer = tk.Frame(self.win, bg="#1b1d27", padx=14, pady=10, highlightthickness=1, highlightbackground="#2b2e3d")
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self._btn_cancel = tk.Button(
            footer,
            text=tr("ui.common.cancel", "Cancel"),
            command=self.close,
            bg="#353849",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self._btn_cancel.pack(side=tk.RIGHT, padx=(6, 0))

        self._btn_save_close = tk.Button(
            footer,
            text=tr("ui.settings.btn_save_close", "✅ Save & Close"),
            command=self._save_and_close,
            bg="#28a745",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self._btn_save_close.pack(side=tk.RIGHT, padx=6)

        self._btn_apply = tk.Button(
            footer,
            text=tr("ui.settings.btn_apply", "💾 Apply"),
            command=self._apply_changes,
            bg="#007acc",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
        )
        self._btn_apply.pack(side=tk.RIGHT, padx=6)

        self.footer_status_lbl = tk.Label(
            footer,
            text=tr("ui.settings.footer_default", "Editing configuration (changes take effect after clicking 'Apply')"),
            fg="#9ba3b8",
            bg="#1b1d27",
            font=("Segoe UI", 9),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=340,
        )
        self.footer_status_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self._bind_auto_wrap(self.footer_status_lbl, initial=340, min_wrap=200)

        body = tk.Frame(self.win, bg="#14141b")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav_frame = tk.Frame(body, bg="#1a1c26", width=260, highlightthickness=1, highlightbackground="#282b3a")
        nav_frame.pack(side=tk.LEFT, fill=tk.Y)
        nav_frame.pack_propagate(False)

        self._lbl_nav_header = tk.Label(
            nav_frame,
            text=tr("ui.settings.nav_header", "SETTINGS SECTIONS"),
            fg="#00e5ff",
            bg="#202331",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            padx=12,
            pady=10,
        )
        self._lbl_nav_header.pack(fill=tk.X)

        self.nav_tree = ttk.Treeview(nav_frame, style="SettingsNav.Treeview", show="tree", selectmode="browse")
        self.nav_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=6)
        self.nav_tree.bind(EVT_TREE_SELECT, self._on_tree_select)

        self.right_container = tk.Frame(body, bg="#161720")
        self.right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_panel = ScrollableFrame(self.right_container, bg="#161720")
        self.scroll_panel.pack(fill=tk.BOTH, expand=True)
        self.content_frame = self.scroll_panel.inner

    def _set_footer_status(self, text: str, fg: str = "#9ba3b8", is_default: bool = False):
        self._footer_is_default = bool(is_default)
        try:
            if getattr(self, "footer_status_lbl", None) is not None and self.footer_status_lbl.winfo_exists():
                self.footer_status_lbl.configure(text=text, fg=fg)
        except Exception:
            pass

    def apply_language(self):
        if self._is_closing:
            return
        try:
            if not self.win.winfo_exists():
                return
        except Exception:
            return

        self._commit_current_page()
        try:
            caps = self.hw_monitor.relocalize_cached_caps()
            if isinstance(caps, dict):
                self._last_caps = caps
                if caps.get("cuda_libraries"):
                    self._last_cuda_status = caps.get("cuda_libraries")
        except Exception:
            pass

        try:
            self.win.title(tr("ui.settings.window_title", "System Settings — ANPR Multi-Cam NVR"))
            if self._lbl_nav_header is not None and self._lbl_nav_header.winfo_exists():
                self._lbl_nav_header.configure(text=tr("ui.settings.nav_header", "SETTINGS SECTIONS"))
            if self._btn_cancel is not None and self._btn_cancel.winfo_exists():
                self._btn_cancel.configure(text=tr("ui.common.cancel", "Cancel"))
            if self._btn_save_close is not None and self._btn_save_close.winfo_exists():
                self._btn_save_close.configure(text=tr("ui.settings.btn_save_close", "✅ Save & Close"))
            if self._btn_apply is not None and self._btn_apply.winfo_exists():
                self._btn_apply.configure(text=tr("ui.settings.btn_apply", "💾 Apply"))
            if self._footer_is_default:
                self._set_footer_status(
                    tr("ui.settings.footer_default", "Editing configuration (changes take effect after clicking 'Apply')"),
                    fg="#9ba3b8",
                    is_default=True,
                )
        except Exception:
            pass

        self._rebuild_nav_tree(select_id=self._current_node_id, preserve_scroll=True)

    def on_external_language_changed(self, lang_code: str):
        if self._is_closing:
            return
        self._commit_current_page()
        norm = normalize_language_code(lang_code) or get_language()
        self.draft.setdefault("general", {})["language"] = norm
        if self._lang_var is not None:
            try:
                self._lang_var.set(get_language_label(norm))
            except Exception:
                pass
        self.apply_language()

    def _rebuild_nav_tree(self, select_id=None, preserve_scroll: bool = False):
        self._tree_syncing = True
        try:
            self.nav_tree.delete(*self.nav_tree.get_children())
            self.nav_tree.insert("", "end", iid="general", text=tr("ui.settings.nav.general", "  ⚙ General"))
            self.nav_tree.insert("", "end", iid="grid", text=tr("ui.settings.nav.grid", "  ▦ Camera Grid"))
            self.nav_tree.insert("", "end", iid="hardware", text=tr("ui.settings.nav.hardware", "  ⚡ Hardware & Graphics"))
            self.nav_tree.insert("", "end", iid="modules", text=tr("ui.settings.nav.modules", "  🧩 Modules & Libraries"))
            cams_node = self.nav_tree.insert("", "end", iid="cameras_root", text=tr("ui.settings.nav.cameras", "  📹 Cameras"), open=True)

            disabled_suf = tr("ui.settings.nav.cam_disabled_suf", " (disabled)")
            for idx, cam in enumerate(self.draft.get("cameras") or []):
                uid = cam["uuid"]
                name = str(cam.get("name") or tr("camera.default_numbered", "Camera {num}", num=idx + 1))
                prefix = "●" if cam.get("enabled", True) else "⏸"
                suffix = "" if cam.get("enabled", True) else disabled_suf
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
            self._show_page(target, preserve_scroll=preserve_scroll)

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

    def _clear_content(self, reset_scroll: bool = True):
        close_active_help_popover()
        self._cancel_telemetry_timer()
        self._page_committers.clear()
        self._telemetry_labels.clear()
        self._hw_widget_vars.clear()
        self._lang_var = None
        self._lang_combo = None
        self._caps_container = None
        self._caps_summary_lbl = None
        self._hw_restart_bar = None
        self._hw_auto_box = None
        self._mode_tiles_frame = None
        self._mode_tile_widgets = {}
        self._mode_detail_frame = None
        self._mode_setup_pb = None
        self._mode_setup_lbl = None
        self._hw_details_frame = None
        for child in list(self.content_frame.winfo_children()):
            child.destroy()
        if reset_scroll:
            try:
                self.scroll_panel.canvas.yview_moveto(0.0)
            except Exception:
                pass

    def _show_page(self, node_id: str, preserve_scroll: bool = False):
        self._current_node_id = node_id

        def _build():
            self._clear_content(reset_scroll=not preserve_scroll)
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
            elif node_id == "cameras_root":
                self._build_page_cameras_root(pad)
            elif node_id.startswith("cam:"):
                uid = node_id.split("cam:", 1)[1]
                self._build_page_single_camera(pad, uid)

        if preserve_scroll and self.scroll_panel is not None:
            self.scroll_panel.preserve_scroll_during(_build)
        else:
            _build()

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
            self._rebuild_nav_tree(select_id=target, preserve_scroll=(target == self._current_node_id))
        except Exception as e:
            print(f"[SettingsWindow on_modules_reloaded]: {e}")

    def on_cuda_libraries_updated(self, caps=None, cuda_status=None):
        """Оновлює панель режимів AI та стан CUDA після фонового завантаження DLL."""
        if self._is_closing:
            return
        if caps is not None:
            self._last_caps = caps
            if cuda_status is None:
                cuda_status = caps.get("cuda_libraries")
        if cuda_status is not None:
            self._last_cuda_status = cuda_status
            if isinstance(self._last_caps, dict):
                self._last_caps["cuda_libraries"] = cuda_status
        if self._current_node_id == "hardware" and self._caps_container is not None:
            if self._last_caps is not None:
                try:
                    self._render_caps_matrix(self._last_caps)
                except Exception:
                    pass
            else:
                self._start_async_caps_probe(force=False)

    @classmethod
    def _bind_tile_click(cls, widget, callback, skip_types=()):
        if skip_types and isinstance(widget, skip_types):
            return
        try:
            widget.bind("<Button-1>", lambda e: callback(), add="+")
        except Exception:
            pass
        for ch in widget.winfo_children():
            cls._bind_tile_click(ch, callback, skip_types=skip_types)

    def _is_draft_operator_mode(self) -> bool:
        try:
            gen = (self.draft or {}).get("general") or {}
            return str(gen.get("operation_mode", "server")).strip().lower() == "operator"
        except Exception:
            return False

    def _get_effective_archive_root(self) -> str:
        raw_arch = str(((self.draft or {}).get("general") or {}).get("archive_root") or "").strip()
        if not raw_arch:
            return self.cfg_mgr.get_archive_root()
        if os.path.isabs(raw_arch):
            return os.path.normpath(raw_arch)
        return os.path.normpath(os.path.join(self.cfg_mgr.base_dir, raw_arch))

    @staticmethod
    def _is_path_on_mount(path_str: str, mount: str) -> bool:
        if not path_str or not mount:
            return False
        a_norm = os.path.normcase(os.path.abspath(path_str))
        m_norm = os.path.normcase(os.path.abspath(mount))
        m_prefix = m_norm if m_norm.endswith(os.sep) else (m_norm + os.sep)
        m_exact = m_norm.rstrip(os.sep) if len(m_norm.rstrip(os.sep)) > 2 else m_prefix
        return a_norm == m_exact or a_norm.startswith(m_prefix)

    def _ask_archive_folder(self, initialdir: str = "", ask_fn=None):
        if getattr(self, "_dir_dialog_open", False):
            return None
        self._dir_dialog_open = True
        try:
            init = str(initialdir or "").strip() or self._get_effective_archive_root() or self.cfg_mgr.base_dir
            picker = ask_fn or filedialog.askdirectory
            chosen = picker(
                parent=self.win,
                initialdir=init,
                title=tr("ui.settings.disk.choose_title", "Select Archive Folder"),
            )
            return os.path.normpath(str(chosen).strip()) if chosen else None
        finally:
            self._dir_dialog_open = False

    def _choose_archive_folder_for_drive(self, mount: str = "", ask_fn=None):
        if self._is_draft_operator_mode():
            return None
        cur_arch = self._get_effective_archive_root()
        if mount and self._is_path_on_mount(cur_arch, mount) and os.path.isdir(cur_arch):
            init_dir = cur_arch
        elif mount:
            init_dir = mount
        else:
            init_dir = cur_arch if os.path.isdir(cur_arch) else self.cfg_mgr.base_dir

        chosen = self._ask_archive_folder(initialdir=init_dir, ask_fn=ask_fn)
        if not chosen:
            return None

        self.draft.setdefault("general", {})["archive_root"] = chosen
        self._set_footer_status(
            tr("ui.settings.disk.new_archive_hint", "New archive: {path} (click 'Apply')", path=chosen),
            fg="#00e5ff",
        )
        if self._current_node_id == "hardware" and self._telemetry_labels:
            self._tick_telemetry()
        return chosen

    @staticmethod
    def _layout_button_row(row_frame: tk.Frame, buttons: list):
        """Розкладає кнопки горизонтально в grid із переносом на новий рядок при вузькому вікні."""
        if not buttons:
            return
        state = {"sig": None}

        def _relayout(avail_w: int):
            if not row_frame.winfo_exists():
                return
            w = int(avail_w) if int(avail_w) > 40 else 480
            placements = []
            r_idx = 0
            c_idx = 0
            cur_x = 0
            for btn in buttons:
                if not btn.winfo_exists():
                    continue
                try:
                    bw = max(40, int(btn.winfo_reqwidth()))
                except Exception:
                    bw = 110
                pad_r = 6
                if c_idx > 0 and (cur_x + bw) > (w - 4):
                    r_idx += 1
                    c_idx = 0
                    cur_x = 0
                placements.append((btn, r_idx, c_idx))
                cur_x += bw + pad_r
                c_idx += 1
            sig = tuple((id(b), r, c) for b, r, c in placements)
            if state["sig"] == sig:
                return
            state["sig"] = sig
            for btn, r, c in placements:
                btn.grid(row=r, column=c, sticky="w", padx=(0, 6), pady=(2 if r > 0 else 0, 2))

        try:
            row_frame.update_idletasks()
            init_w = int(row_frame.winfo_width() or 0)
        except Exception:
            init_w = 0
        _relayout(init_w if init_w > 40 else 480)
        row_frame.bind("<Configure>", lambda e: _relayout(e.width), add="+")

    # ------------------------------------------------------------------
    # Допоміжні конструктори карток та полів із довідкою '?'
    # ------------------------------------------------------------------
    @staticmethod
    def _bind_auto_wrap(lbl: tk.Label, initial: int = 440, min_wrap: int = 160, pad: int = 8):
        try:
            lbl.configure(wraplength=initial)
        except Exception:
            pass

        def _on_cfg(event):
            try:
                if not lbl.winfo_exists():
                    return
                w = max(int(min_wrap), int(event.width) - int(pad))
                cur = int(float(lbl.cget("wraplength") or 0))
                if abs(cur - w) >= 4:
                    lbl.configure(wraplength=w)
            except Exception:
                pass

        lbl.bind("<Configure>", _on_cfg, add="+")
        return lbl

    @classmethod
    def _section_header(cls, parent, title: str, subtitle: str = ""):
        tk.Label(
            parent, text=title, fg="#00e5ff", bg="#161720",
            font=("Segoe UI", 14, "bold"), anchor=tk.W
        ).pack(fill=tk.X, pady=(0, 2))
        if subtitle:
            sub_lbl = tk.Label(
                parent, text=subtitle, fg="#9ba3b8", bg="#161720",
                font=("Segoe UI", 9), anchor=tk.W, wraplength=440, justify=tk.LEFT
            )
            sub_lbl.pack(fill=tk.X, pady=(0, 12))
            cls._bind_auto_wrap(sub_lbl, initial=440, min_wrap=220)
        else:
            tk.Frame(parent, bg="#161720", height=8).pack()

    @classmethod
    def _make_card(cls, parent, title: str, subtitle: str = "", bg: str = "#1c1e2a"):
        card = tk.Frame(parent, bg=bg, padx=14, pady=12, highlightthickness=1, highlightbackground="#2b2e3d")
        card.pack(fill=tk.X, pady=(0, 12))
        if title:
            tk.Label(
                card, text=title, fg="#ffffff", bg=bg,
                font=("Segoe UI", 10, "bold"), anchor=tk.W
            ).pack(fill=tk.X, pady=(0, 2 if subtitle else 8))
        if subtitle:
            sub_lbl = tk.Label(
                card, text=subtitle, fg="#8d95ab", bg=bg,
                font=("Segoe UI", 8), anchor=tk.W, wraplength=440, justify=tk.LEFT
            )
            sub_lbl.pack(fill=tk.X, pady=(0, 8))
            cls._bind_auto_wrap(sub_lbl, initial=440, min_wrap=220)
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
            tr("ui.settings.general.title", "⚙ General Settings"),
            tr(
                "ui.settings.general.subtitle",
                "Interface language, operation mode, root archive folder, Windows autostart, kiosk mode, and window layout.",
            ),
        )
        gen = self.draft.setdefault("general", {})
        card = self._make_card(parent, tr("ui.settings.general.card_main", "Basic settings"))

        lang_items = list_languages()
        lang_labels = [m["label"] for m in lang_items]
        cur_lang_code = normalize_language_code(gen.get("language") or get_language()) or DEFAULT_LANGUAGE
        lang_var = tk.StringVar(value=get_language_label(cur_lang_code))
        self._lang_var = lang_var

        cur_mode = str(gen.get("operation_mode", "server")).strip().lower()
        if cur_mode not in ("server", "operator"):
            cur_mode = "server"
        op_mode_var = tk.StringVar(value=cur_mode)
        arch_var = tk.StringVar(value=str(gen.get("archive_root", "")))
        auto_var = tk.BooleanVar(value=bool(gen.get("autostart_os_and_play", False)))
        kiosk_var = tk.BooleanVar(value=bool(gen.get("kiosk_mode", False)))

        arch_controls = []
        op_note_holder = [None]
        op_row_holder = [None]

        def _sync_mode_ui(*_):
            is_op = (str(op_mode_var.get() or "server").strip().lower() == "operator")
            for w in arch_controls:
                try:
                    w.configure(state=tk.DISABLED if is_op else tk.NORMAL)
                except Exception:
                    pass
            nlbl = op_note_holder[0]
            if nlbl is not None:
                try:
                    if is_op:
                        if str(nlbl.winfo_manager()) != "pack":
                            if op_row_holder[0] is not None:
                                nlbl.pack(fill=tk.X, pady=(4, 6), after=op_row_holder[0])
                            else:
                                nlbl.pack(fill=tk.X, pady=(4, 6))
                    else:
                        nlbl.pack_forget()
                except Exception:
                    pass

        def _commit():
            chosen_lang = parse_language_code(lang_var.get(), fallback=cur_lang_code or DEFAULT_LANGUAGE)
            gen["language"] = chosen_lang
            m_val = str(op_mode_var.get() or "server").strip().lower()
            gen["operation_mode"] = "operator" if m_val == "operator" else "server"
            gen["archive_root"] = arch_var.get().strip() or self.cfg_mgr.get_archive_root()
            gen["autostart_os_and_play"] = bool(auto_var.get())
            gen["kiosk_mode"] = bool(kiosk_var.get())

        def _on_mode_change():
            _commit()
            _sync_mode_ui()
            if gen["operation_mode"] == "operator":
                self._set_footer_status(
                    tr("ui.settings.general.mode_op_hint", "Operator mode selected: disk writing will be disabled (click 'Apply')."),
                    fg="#ffcc00",
                )
            else:
                self._set_footer_status(
                    tr("ui.settings.general.mode_srv_hint", "Server mode selected: all features and disk recording enabled (click 'Apply')."),
                    fg="#75e08b",
                )

        self._page_committers.append(_commit)

        def _w_lang(box, bg):
            cb = ttk.Combobox(
                box,
                textvariable=lang_var,
                values=lang_labels,
                state="readonly",
                width=44,
                style="SettingsDark.TCombobox",
            )
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            style_dark_combobox_popdown(cb)

            def _on_lang_selected(e=None):
                _commit()
                self._set_footer_status(
                    tr(
                        "ui.settings.general.lang_selected_hint",
                        "Selected language: {lang} (click 'Apply' to switch and save).",
                        lang=lang_var.get(),
                    ),
                    fg="#00e5ff",
                )

            cb.bind("<<ComboboxSelected>>", _on_lang_selected)
            self._lang_combo = cb
            return cb

        self._add_row(card, tr("ui.settings.general.lang_label", "Application language:"), "general.language", _w_lang)

        def _w_mode(box, bg):
            rb_srv = tk.Radiobutton(
                box,
                text=tr("ui.settings.general.mode_server_opt", "🖥 Server Mode — disk recording, archive, and full features"),
                variable=op_mode_var,
                value="server",
                command=_on_mode_change,
                bg=bg,
                fg="#75e08b",
                selectcolor="#252836",
                activebackground=bg,
                activeforeground="#75e08b",
                font=("Segoe UI", 9, "bold"),
                anchor=tk.W,
                cursor="hand2",
            )
            rb_srv.pack(anchor=tk.W)
            rb_op = tk.Radiobutton(
                box,
                text=tr("ui.settings.general.mode_operator_opt", "👤 Operator Mode — live monitoring and AI without disk writes"),
                variable=op_mode_var,
                value="operator",
                command=_on_mode_change,
                bg=bg,
                fg="#ffcc00",
                selectcolor="#252836",
                activebackground=bg,
                activeforeground="#ffcc00",
                font=("Segoe UI", 9, "bold"),
                anchor=tk.W,
                cursor="hand2",
            )
            rb_op.pack(anchor=tk.W, pady=(2, 0))
            return box

        op_row, _ = self._add_row(card, tr("ui.settings.general.mode_label", "Operation mode:"), "general.operation_mode", _w_mode)
        op_row_holder[0] = op_row

        op_note_lbl = tk.Label(
            card,
            text=tr(
                "ui.settings.general.mode_op_note",
                "⚠ In Operator Mode all disk writes are disabled: video recording, snapshot capture (plates, faces, people), faces.txt updates/editing, auto-cleanup, and window position auto-saving do not run. Existing saved server settings remain preserved when switching back to Server Mode.",
            ),
            fg="#ffcc00",
            bg="#2c2416",
            font=("Segoe UI", 8, "bold"),
            padx=10,
            pady=6,
            justify=tk.LEFT,
            wraplength=440,
            anchor=tk.W,
            highlightthickness=1,
            highlightbackground="#66521e",
        )
        self._bind_auto_wrap(op_note_lbl, initial=440, min_wrap=220, pad=24)
        op_note_holder[0] = op_note_lbl

        def _w_archive(box, bg):
            e = tk.Entry(box, textvariable=arch_var, bg="#252836", fg="#ffffff", disabledbackground="#1a1c26", disabledforeground="#7d8499", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9))
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 6))

            def _browse():
                if self._is_draft_operator_mode():
                    return
                chosen = self._ask_archive_folder(initialdir=arch_var.get() or self.cfg_mgr.base_dir)
                if chosen:
                    arch_var.set(chosen)
                    _commit()

            def _open_dir():
                p = arch_var.get().strip()
                if not p:
                    return
                if self._is_draft_operator_mode() or self.cfg_mgr.is_operator_mode():
                    if not os.path.isdir(p):
                        messagebox.showinfo(
                            tr("ui.common.operator_mode", "Operator Mode"),
                            tr(
                                "ui.settings.general.op_no_dir_msg",
                                "Archive folder does not exist yet ({path}).\nCreating folders on disk is disabled in Operator Mode.",
                                path=p,
                            ),
                            parent=self.win,
                        )
                        return
                else:
                    os.makedirs(p, exist_ok=True)
                try:
                    os.startfile(p)
                except Exception:
                    pass

            btn_browse = tk.Button(box, text=tr("ui.common.browse", "📁 Browse..."), command=_browse, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2")
            btn_browse.pack(side=tk.LEFT, padx=(0, 4))
            btn_open = tk.Button(box, text=tr("ui.common.open", "Open"), command=_open_dir, bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2")
            btn_open.pack(side=tk.LEFT)
            arch_controls.extend([e, btn_browse])
            return e

        self._add_row(card, tr("ui.settings.general.archive_root_label", "Root archive folder:"), "general.archive_root", _w_archive)

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
            tr("ui.settings.general.autostart_label", "Windows startup:"),
            "general.autostart_os_and_play",
            _w_check(auto_var, tr("ui.settings.general.autostart_chk", "Add to Windows startup and start cameras automatically")),
        )
        self._add_row(
            card,
            tr("ui.settings.general.kiosk_label", "Kiosk mode:"),
            "general.kiosk_mode",
            _w_check(kiosk_var, tr("ui.settings.general.kiosk_chk", "Block standard exit (exit via Ctrl+Alt+Shift+Q only)")),
        )
        _sync_mode_ui()

        win_card = self._make_card(parent, tr("ui.settings.general.win_card", "Window positions on monitors"))
        win_desc = tk.Label(
            win_card,
            text=tr(
                "ui.settings.general.win_desc",
                "All windows (camera grid, individual camera windows, ROI crop, and settings) are independent:\njust drag them onto any monitor — coordinates and maximized state save automatically.",
            ),
            fg="#9ba3b8", bg=str(win_card.cget("bg")), font=("Segoe UI", 9), justify=tk.LEFT, anchor=tk.W, wraplength=440
        )
        win_desc.pack(fill=tk.X, pady=(0, 8))
        self._bind_auto_wrap(win_desc, initial=440, min_wrap=220)

        def _reset_window_positions():
            self.draft["windows"] = {
                "grid": {"left": 40, "top": 30, "width": 1280, "height": 760, "maximized": False},
                "settings": {"left": 100, "top": 60, "width": 1080, "height": 740, "maximized": False},
            }
            for idx, cam in enumerate(self.draft.get("cameras") or []):
                cam["window"] = {"left": 80 + idx * 30, "top": 60 + idx * 30, "width": 1160, "height": 720, "maximized": False}
                cam["roi_window"] = {"left": 60, "top": 50, "width": 520, "height": 360, "maximized": False}
            self._set_footer_status(
                tr("ui.settings.general.win_reset_msg", "Window positions reset in draft — click 'Apply'."),
                fg="#00e5ff",
            )

        tk.Button(
            win_card,
            text=tr("ui.settings.general.win_reset_btn", "📍 Reset saved positions of all windows to main monitor"),
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
            tr("ui.settings.grid.title", "▦ Camera Grid Settings"),
            tr("ui.settings.grid.subtitle", "The main grid always uses the compressed sub-stream (sub_source) to save CPU/GPU."),
        )
        grid = self.draft.setdefault("grid", {})
        card = self._make_card(parent, tr("ui.settings.grid.card_layout", "Tile Layout & Appearance"))

        mode_choices = _get_grid_mode_choices()
        mode_var = tk.StringVar(value=self._choice_label_for(mode_choices, str(grid.get("mode", "auto"))))
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
            m = self._choice_key_from(mode_choices, mode_var.get(), "auto")
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
            is_man = (self._choice_key_from(mode_choices, mode_var.get(), "auto") == "manual")
            st = tk.NORMAL if is_man else tk.DISABLED
            for w in rows_spin_holder + cols_spin_holder:
                try:
                    w.configure(state=st)
                except Exception:
                    pass
            _commit()

        def _w_mode(box, bg):
            cb = ttk.Combobox(
                box,
                textvariable=mode_var,
                values=[lbl for _, lbl in mode_choices],
                state="readonly",
                width=18,
                style="SettingsDark.TCombobox",
            )
            cb.pack(side=tk.LEFT)
            style_dark_combobox_popdown(cb)
            cb.bind("<<ComboboxSelected>>", _sync_manual_state)
            tk.Label(
                box,
                text=tr("ui.settings.grid.mode_hint", "  (auto = fit camera count, manual = rows × cols)"),
                fg="#9ba3b8", bg=bg, font=("Segoe UI", 8),
            ).pack(side=tk.LEFT)
            return cb

        self._add_row(card, tr("ui.settings.grid.mode_label", "Grid layout mode:"), "grid.mode", _w_mode)

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

        self._add_row(card, tr("ui.settings.grid.rows_label", "Rows (manual):"), "grid.rows", _w_spin(rows_var, 1, 8, rows_spin_holder))
        self._add_row(card, tr("ui.settings.grid.cols_label", "Columns (manual):"), "grid.cols", _w_spin(cols_var, 1, 8, cols_spin_holder))
        _sync_manual_state()

        def _w_color(var):
            def _fn(box, bg):
                e = tk.Entry(box, textvariable=var, width=12, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Consolas", 9))
                e.pack(side=tk.LEFT, ipady=3, padx=(0, 8))
                swatch = tk.Button(box, text=tr("ui.settings.grid.pick_color_btn", "  Pick color  "), relief=tk.FLAT, cursor="hand2", font=("Segoe UI", 8, "bold"))

                def _refresh_swatch(*_):
                    col = var.get().strip()
                    if re.match(r"^#[0-9a-fA-F]{6}$", col):
                        swatch.configure(bg=col, fg="#000000")

                def _pick():
                    initial = var.get().strip() if re.match(r"^#[0-9a-fA-F]{6}$", var.get().strip()) else "#00e5ff"
                    _, hex_c = colorchooser.askcolor(color=initial, parent=self.win, title=tr("ui.settings.grid.pick_color_title", "Choose border color"))
                    if hex_c:
                        var.set(hex_c)
                        _commit()

                swatch.configure(command=_pick)
                swatch.pack(side=tk.LEFT)
                var.trace_add("write", _refresh_swatch)
                _refresh_swatch()
                return e
            return _fn

        self._add_row(card, tr("ui.settings.grid.border_color_label", "Grid line color:"), "grid.border_color", _w_color(bcol_var))
        self._add_row(card, tr("ui.settings.grid.active_border_label", "Active window color:"), "grid.active_border_color", _w_color(acol_var))
        self._add_row(card, tr("ui.settings.grid.offline_border_label", "Connection lost color:"), "grid.offline_border_color", _w_color(ocol_var))
        self._add_row(card, tr("ui.settings.grid.border_width_label", "Grid line thickness (px):"), "grid.border_width", _w_spin(bw_var, 1, 10, []))
        self._add_row(card, tr("ui.settings.grid.tile_gap_label", "Outer grid margin (px):"), "grid.tile_gap", _w_spin(gap_var, 0, 20, []))

        def _w_chk(var, text):
            def _fn(box, bg):
                cb = tk.Checkbutton(box, text=text, variable=var, command=_commit, bg=bg, fg="#ffffff", selectcolor="#252836", activebackground=bg, activeforeground="#ffffff", font=("Segoe UI", 9))
                cb.pack(side=tk.LEFT)
                return cb
            return _fn

        self._add_row(card, tr("ui.settings.grid.show_labels_label", "Camera label:"), "grid.show_labels", _w_chk(lbl_var, tr("ui.settings.grid.show_labels_chk", "Show camera name on tile")))
        self._add_row(card, tr("ui.settings.grid.show_status_label", "Status indicators:"), "grid.show_status", _w_chk(st_var, tr("ui.settings.grid.show_status_chk", "Show ● REC (codec) and active modules")))
        self._add_row(card, tr("ui.settings.grid.show_fps_label", "Frame rate:"), "grid.show_fps", _w_chk(fps_var, tr("ui.settings.grid.show_fps_chk", "Show current sub-stream FPS on tile")))

    # ------------------------------------------------------------------
    # Сторінка 3: Залізо та графіка
    # ------------------------------------------------------------------
    def _build_page_hardware(self, parent):
        self._section_header(
            parent,
            tr("ui.settings.hw.title", "⚡ Hardware, Resources & GPU Acceleration"),
            tr(
                "ui.settings.hw.subtitle",
                "Live CPU/RAM/GPU/disk telemetry, support checks for CUDA/DirectML/OpenVINO/NVENC/copy, and performance settings.",
            ),
        )
        hw = self.draft.setdefault("hardware", {})

        setup_card = self._make_card(
            parent,
            tr("ui.settings.hw.autosetup_card", "Full System Auto-Setup (1 click)"),
            tr(
                "ui.settings.hw.autosetup_desc",
                "Checks the PC, missing Python packages, optimal ONNX Runtime (CUDA / DirectML / CPU), CUDA/cuDNN libraries in cuda/, all module models/resources, and optimal hardware & camera settings — with a preview confirmation plan, progress bar, and instant apply.",
            ),
            bg="#17242c",
        )
        setup_bg = str(setup_card.cget("bg"))
        setup_bar = tk.Frame(setup_card, bg=setup_bg)
        setup_bar.pack(fill=tk.X)

        tk.Button(
            setup_bar,
            text=tr("ui.settings.hw.autosetup_btn", "🚀 Full Auto-Setup (1 click)"),
            command=self._open_full_autosetup_dialog,
            bg="#1565c0",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 8))

        HelpButton(
            setup_bar,
            title=tr("ui.settings.hw.autosetup_card", "Full System Auto-Setup (1 click)"),
            help_text=tr(
                "ui.settings.hw.autosetup_help",
                "Opens the automatic setup wizard with a plan preview and confirmation:\n1) Checks and installs missing Python dependencies (.venv);\n2) Selects and installs optimal ONNX Runtime (NVIDIA CUDA → DirectML → CPU);\n3) Checks and downloads missing CUDA / cuDNN DLLs into ./cuda/;\n4) Checks and downloads models/resources for all active modules;\n5) Tunes optimal hardware, decoding, and camera codec parameters, saves config.json and applies immediately (or prompts restart if older native .dll/.pyd files are already loaded in memory).",
            ),
            bg=setup_bg,
        ).pack(side=tk.LEFT)

        tel_card = self._make_card(parent, tr("ui.settings.hw.tel_card", "Hardware resources of this PC (live updates)"))
        tel_bg = str(tel_card.cget("bg"))

        hw_tiles_wrap = tk.Frame(tel_card, bg=tel_bg)
        hw_tiles_wrap.pack(fill=tk.X, pady=(0, 8))

        sys_bg = "#181a24"
        sys_frame, sys_inner, sys_title_lbl, sys_meta_lbl = self._create_hw_group_frame(
            hw_tiles_wrap,
            border_col="#2b6b88",
            title_txt=tr("ui.settings.hw.sys_title", "🖥 System"),
            title_col="#80e5ff",
            meta_txt=tr("ui.settings.hw.sys_meta", "Processor and memory"),
            bg=sys_bg,
            init_px=300,
        )

        loading_txt = tr("ui.common.loading", "Loading...")
        cpu_tile, cpu_inner, cpu_tbg, cpu_title_lbl, cpu_sub_lbl = self._create_hw_tile_frame(
            sys_inner, "#00e5ff", tr("ui.settings.hw.cpu_title", "Processor (CPU)"), "#9ee8ff"
        )
        lbl_cpu = tk.Label(
            cpu_inner,
            text=loading_txt,
            fg="#d5dae8",
            bg=cpu_tbg,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=200,
        )
        lbl_cpu.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self._bind_auto_wrap(lbl_cpu, initial=200, min_wrap=105, pad=4)
        pb_cpu = ttk.Progressbar(cpu_inner, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
        pb_cpu.pack(side=tk.BOTTOM, fill=tk.X)

        ram_tile, ram_inner, ram_tbg, ram_title_lbl, ram_sub_lbl = self._create_hw_tile_frame(
            sys_inner, "#75e08b", tr("ui.settings.hw.ram_title", "Memory (RAM)"), "#75e08b"
        )
        lbl_ram = tk.Label(
            ram_inner,
            text=loading_txt,
            fg="#d5dae8",
            bg=ram_tbg,
            font=("Segoe UI", 8),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=200,
        )
        lbl_ram.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        self._bind_auto_wrap(lbl_ram, initial=200, min_wrap=105, pad=4)
        pb_ram = ttk.Progressbar(ram_inner, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
        pb_ram.pack(side=tk.BOTTOM, fill=tk.X)

        sys_group = {
            "key": "system",
            "frame": sys_frame,
            "inner_wrap": sys_inner,
            "inner_uniform": "hw_sys_inner",
            "title_lbl": sys_title_lbl,
            "meta_lbl": sys_meta_lbl,
            "tiles": [cpu_tile, ram_tile],
            "prev_in_cols": 0,
            "prev_in_rows": 0,
        }

        disks_frame = tk.Frame(tel_card, bg=tel_bg)
        disks_frame.pack(fill=tk.X, pady=(2, 0))

        self._telemetry_labels = {
            "hw_tiles_wrap": hw_tiles_wrap,
            "gpus_frame": hw_tiles_wrap,
            "sys_group": sys_group,
            "sys_frame": sys_frame,
            "cpu_tile": cpu_tile,
            "ram_tile": ram_tile,
            "cpu_title_lbl": cpu_title_lbl,
            "cpu_sub_lbl": cpu_sub_lbl,
            "cpu_lbl": lbl_cpu,
            "cpu_pb": pb_cpu,
            "ram_title_lbl": ram_title_lbl,
            "ram_sub_lbl": ram_sub_lbl,
            "ram_lbl": lbl_ram,
            "ram_pb": pb_ram,
            "hw_layout_state": {"sig": None, "prev_cols": 0, "prev_rows": 0},
            "gpu_sig": None,
            "gpu_rows": [],
            "gpu_groups": [],
            "disks_frame": disks_frame,
            "disks_mounts": None,
            "disks_sig": None,
            "disk_rows": {},
            "phys_groups": [],
        }
        hw_tiles_wrap.bind("<Configure>", lambda e: self._relayout_hw_tiles(e.width), add="+")
        self._tick_telemetry()

        mode_card = self._make_card(
            parent,
            tr("ui.settings.hw.modes_card", "Neural Network (AI) & Acceleration Modes"),
            tr(
                "ui.settings.hw.modes_desc",
                "Tiles show speed rating (🏆 #1 Fastest → 🐢 Slowest), support level (✔ Full / ⚙ Need pkg / 🚫 Not supported), and short '+ / −' hints. Click any tile to see full pros, cons, and required steps (CUDA auto-installs on selection).",
            ),
        )
        mode_bg = str(mode_card.cget("bg"))

        self._caps_summary_lbl = tk.Label(
            mode_card,
            text=tr("ui.settings.hw.caps_checking", "Checking available AI modes and GPU..."),
            fg="#ffcc00", bg=mode_bg, font=("Segoe UI", 9, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=440
        )
        self._caps_summary_lbl.pack(fill=tk.X, pady=(0, 6))
        self._bind_auto_wrap(self._caps_summary_lbl, initial=440, min_wrap=220)

        self._hw_restart_bar = tk.Frame(
            mode_card, bg="#3b2814", padx=10, pady=7,
            highlightthickness=1, highlightbackground="#ffcc00"
        )

        self._hw_auto_box = tk.Frame(mode_card, bg=mode_bg)
        self._caps_container = tk.Frame(mode_card, bg=mode_bg)
        self._caps_container.pack(fill=tk.X)
        if self._hw_auto_note is not None:
            self._render_hw_auto_box()
        if self._last_caps is not None:
            self._render_caps_matrix(self._last_caps)

        cfg_card = self._make_card(parent, tr("ui.settings.hw.cfg_card", "Processor/Graphics & Display Parameters"))

        ai_choices = _get_ai_backend_choices()
        dec_choices = _get_hw_decode_choices()
        resz_choices = _get_resize_choices()

        ai_var = tk.StringVar(value=self._choice_label_for(ai_choices, str(hw.get("ai_backend", "cpu"))))
        thr_var = tk.StringVar(value=str(hw.get("ai_threads", 2)))
        aifps_var = tk.StringVar(value=str(hw.get("ai_max_fps_per_cam", 8.0)))
        dec_var = tk.StringVar(value=self._choice_label_for(dec_choices, str(hw.get("video_decode_mode", "cpu"))))
        gfps_var = tk.StringVar(value=str(hw.get("display_grid_fps", 12.0)))
        dfps_var = tk.StringVar(value=str(hw.get("display_detail_fps", 25.0)))
        resz_var = tk.StringVar(value=self._choice_label_for(resz_choices, str(hw.get("display_resize_method", "linear"))))
        ocl_var = tk.BooleanVar(value=bool(hw.get("use_opencv_opencl", False)))

        self._hw_widget_vars = {
            "ai_backend": ai_var,
            "ai_choices": ai_choices,
            "ai_threads": thr_var,
            "video_decode_mode": dec_var,
            "display_resize_method": resz_var,
            "use_opencv_opencl": ocl_var,
        }

        def _commit():
            hw["ai_backend"] = self._choice_key_from(ai_choices, ai_var.get(), "cpu")
            hw["ai_threads"] = _clamp_int(thr_var.get(), 2, 1, 8)
            hw["ai_max_fps_per_cam"] = _clamp_float(aifps_var.get(), 8.0, 1.0, 30.0)
            hw["video_decode_mode"] = self._choice_key_from(dec_choices, dec_var.get(), "cpu")
            hw["display_grid_fps"] = _clamp_float(gfps_var.get(), 12.0, 3.0, 30.0)
            hw["display_detail_fps"] = _clamp_float(dfps_var.get(), 25.0, 5.0, 60.0)
            hw["display_resize_method"] = self._choice_key_from(resz_choices, resz_var.get(), "linear")
            hw["use_opencv_opencl"] = bool(ocl_var.get())

        self._page_committers.append(_commit)

        def _w_ai_combo(box, bg):
            cb = ttk.Combobox(
                box,
                textvariable=ai_var,
                values=[lbl for _, lbl in ai_choices],
                state="readonly",
                width=52,
                style="SettingsDark.TCombobox",
            )
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            style_dark_combobox_popdown(cb)

            def _on_ai_select(e=None):
                _commit()
                if not self._syncing_ai_backend_var:
                    chosen_k = self._choice_key_from(ai_choices, ai_var.get(), "cpu")
                    self._select_ai_mode(chosen_k, auto_install=(chosen_k == "cuda"), prompt_confirm=True)

            cb.bind("<<ComboboxSelected>>", _on_ai_select)
            return cb

        def _w_combo(var, choices):
            def _fn(box, bg):
                cb = ttk.Combobox(
                    box,
                    textvariable=var,
                    values=[lbl for _, lbl in choices],
                    state="readonly",
                    width=52,
                    style="SettingsDark.TCombobox",
                )
                cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                style_dark_combobox_popdown(cb)
                cb.bind("<<ComboboxSelected>>", lambda e: _commit())
                return cb
            return _fn

        def _w_spin(var, fr, to_, inc=1):
            def _fn(box, bg):
                sp = tk.Spinbox(box, from_=fr, to=to_, increment=inc, textvariable=var, width=10, bg="#252836", fg="#ffffff", buttonbackground="#2f3346", relief=tk.FLAT, font=("Segoe UI", 9), command=_commit)
                sp.pack(side=tk.LEFT)
                return sp
            return _fn

        self._add_row(cfg_card, tr("ui.settings.hw.ai_backend_label", "AI backend:"), "hardware.ai_backend", _w_ai_combo)
        self._add_row(cfg_card, tr("ui.settings.hw.ai_threads_label", "ONNX / AI threads:"), "hardware.ai_threads", _w_spin(thr_var, 1, 8))
        self._add_row(cfg_card, tr("ui.settings.hw.ai_fps_label", "Max AI FPS per camera:"), "hardware.ai_max_fps_per_cam", _w_spin(aifps_var, 1, 30))
        self._add_row(cfg_card, tr("ui.settings.hw.decode_label", "Stream decoding:"), "hardware.video_decode_mode", _w_combo(dec_var, dec_choices))
        self._add_row(cfg_card, tr("ui.settings.hw.grid_fps_label", "Grid refresh rate (FPS):"), "hardware.display_grid_fps", _w_spin(gfps_var, 3, 30))
        self._add_row(cfg_card, tr("ui.settings.hw.detail_fps_label", "Detail view refresh (FPS):"), "hardware.display_detail_fps", _w_spin(dfps_var, 5, 60))
        self._add_row(cfg_card, tr("ui.settings.hw.resize_label", "Display resize quality:"), "hardware.display_resize_method", _w_combo(resz_var, resz_choices))

        def _w_ocl(box, bg):
            cb = tk.Checkbutton(
                box,
                text=tr("ui.settings.hw.opencl_chk", "Enable OpenCV OpenCL (GPU UMat) for image processing"),
                variable=ocl_var, command=_commit, bg=bg, fg="#ffffff", selectcolor="#252836",
                activebackground=bg, activeforeground="#ffffff", font=("Segoe UI", 9)
            )
            cb.pack(side=tk.LEFT)
            return cb

        self._add_row(cfg_card, tr("ui.settings.hw.opencl_label", "OpenCV OpenCL:"), "hardware.use_opencv_opencl", _w_ocl)

    def _safe_mode_progress(self, mode_key: str, text: str, color: str = "#ffcc00"):
        m = re.search(r"\((\d+)%\)", str(text or ""))
        pct = int(m.group(1)) if m else None

        def _upd():
            if self._is_closing:
                return
            if isinstance(self._mode_setup_status, dict) and self._mode_setup_status.get("mode") == mode_key:
                self._mode_setup_status["text"] = str(text)
                self._mode_setup_status["color"] = color
                if pct is not None:
                    self._mode_setup_status["pct"] = max(0, min(100, pct))
            if self._inspected_ai_mode != mode_key:
                return
            if self._mode_setup_lbl is not None and self._mode_setup_lbl.winfo_exists():
                self._mode_setup_lbl.configure(text=text, fg=color)
            if pct is not None and self._mode_setup_pb is not None and self._mode_setup_pb.winfo_exists():
                self._mode_setup_pb["value"] = max(0, min(100, pct))

        self._ui_queue.put((_upd, ()))

    def _open_cuda_folder(self):
        cdir = get_cuda_dir(self.cfg_mgr.base_dir)
        os.makedirs(cdir, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(cdir)
            else:
                subprocess.Popen(["xdg-open", cdir])
        except Exception as e:
            messagebox.showerror(
                tr("ui.settings.hw.cuda_dir_title", "cuda/ folder"),
                tr("ui.settings.hw.open_dir_err", "Failed to open folder:\n{path}\n{err}", path=cdir, err=e),
                parent=self.win,
            )

    def _toggle_hw_details(self):
        self._hw_details_expanded = not bool(self._hw_details_expanded)
        if self._current_node_id == "hardware" and self._caps_container is not None and self._last_caps is not None:
            self._render_caps_matrix(self._last_caps)

    def _select_ai_mode(
        self,
        mode_key: str,
        auto_install: bool = True,
        prompt_confirm: bool = True,
        confirm_fn=None,
        setup_runner=None,
    ):
        key = str(mode_key or "cpu").strip().lower()
        if key not in ("auto", "cuda", "directml", "openvino", "cpu"):
            return

        caps = self._last_caps
        if caps is None:
            try:
                caps = self.hw_monitor.probe_acceleration_capabilities(force=False)
                self._last_caps = caps
                if isinstance(caps, dict) and caps.get("cuda_libraries"):
                    self._last_cuda_status = caps.get("cuda_libraries")
            except Exception:
                caps = {}

        assessment = assess_ai_modes(self.cfg_mgr.base_dir, caps=caps)
        m_info = (assessment.get("modes") or {}).get(key) or {}
        hw_ok = bool(m_info.get("hw_supported"))

        self._inspected_ai_mode = key
        if hw_ok:
            self.draft.setdefault("hardware", {})["ai_backend"] = key
            ai_var = self._hw_widget_vars.get("ai_backend")
            ai_choices = self._hw_widget_vars.get("ai_choices") or _get_ai_backend_choices()
            if ai_var is not None:
                self._syncing_ai_backend_var = True
                try:
                    ai_var.set(self._choice_label_for(ai_choices, key))
                finally:
                    self._syncing_ai_backend_var = False
            saved_backend = str(((self.cfg_mgr.get_snapshot() or {}).get("hardware") or {}).get("ai_backend", "cpu")).strip().lower()
            if key != saved_backend:
                self._set_footer_status(
                    tr(
                        "ui.settings.hw.mode_selected_hint",
                        "AI mode selected: {mode} — click '💾 Apply' to save.",
                        mode=m_info.get("title", key.upper()),
                    ),
                    fg="#00e5ff",
                )
            else:
                self._set_footer_status(
                    tr(
                        "ui.settings.hw.mode_already_saved",
                        "AI mode: {mode} (already saved).",
                        mode=m_info.get("title", key.upper()),
                    ),
                    fg="#75e08b" if m_info.get("state") == "ready" else "#ffcc00",
                )
        else:
            self._set_footer_status(
                tr(
                    "ui.settings.hw.mode_preview_only",
                    "Preview: {headline} (active AI selection unchanged).",
                    headline=m_info.get("headline", key),
                ),
                fg="#ffcc00",
            )

        if self._current_node_id == "hardware" and self._caps_container is not None:
            self._render_caps_matrix(caps)

        plan = m_info.get("install_plan") or {}
        if auto_install and key == "cuda" and hw_ok and plan.get("can_install"):
            if self._ort_installing or self._cuda_downloading or self._autosetup_running:
                return
            todo_lines = []
            if plan.get("need_ort_pkg"):
                todo_lines.append(tr("ui.settings.hw.cuda_todo_ort", "• Install ONNX Runtime GPU package ('onnxruntime-gpu')"))
            if plan.get("need_cuda_dlls"):
                todo_lines.append(tr("ui.settings.hw.cuda_todo_dlls", "• Download missing CUDA / cuDNN libraries into ./cuda/ folder"))
            prompt_msg = tr(
                "ui.settings.hw.cuda_confirm_body",
                "NVIDIA CUDA GPU mode selected.\n\nOnly the missing steps will be performed automatically:\n{steps}\n\nExisting modules, models, and libraries will not be deleted or reinstalled.\nContinue now?",
                steps="\n".join(todo_lines),
            )
            asked = (
                True
                if not prompt_confirm
                else (confirm_fn or (lambda title, body: messagebox.askyesno(title, body, parent=self.win)))(
                    tr("ui.settings.hw.cuda_confirm_title", "NVIDIA CUDA GPU Auto-Setup"),
                    prompt_msg,
                )
            )
            if asked:
                self._start_mode_setup_async("cuda", installer_fn=setup_runner)

    def _start_mode_setup_async(self, mode_key: str, installer_fn=None, force_local_cuda: bool = False):
        if self._ort_installing or self._cuda_downloading or self._autosetup_running:
            return

        key = str(mode_key or "cpu").strip().lower()
        caps = self._last_caps
        if caps is None:
            try:
                caps = self.hw_monitor.probe_acceleration_capabilities(force=False)
                self._last_caps = caps
            except Exception:
                caps = {}

        assessment = assess_ai_modes(self.cfg_mgr.base_dir, caps=caps)
        target_mode = assessment["best_mode"] if key == "auto" else key
        minfo = (assessment.get("modes") or {}).get(target_mode) or {}
        if not minfo.get("hw_supported"):
            return

        plan = minfo.get("install_plan") or {}
        if not force_local_cuda and not plan.get("can_install"):
            self._mode_setup_status = {
                "mode": key,
                "ok": True,
                "pct": 100,
                "text": tr(
                    "ui.settings.hw.setup_already_done",
                    "✔ Everything needed for {title} is already installed (no reinstall needed).",
                    title=minfo["title"],
                ),
                "color": "#ffcc00" if plan.get("needs_restart") else "#75e08b",
            }
            if self._current_node_id == "hardware" and self._caps_container is not None:
                self._render_caps_matrix(caps)
            return

        self._ort_installing = True
        self._cuda_downloading = bool(force_local_cuda or plan.get("need_cuda_dlls"))
        self._inspected_ai_mode = key
        self._mode_setup_status = {
            "mode": key,
            "ok": None,
            "pct": 10,
            "text": tr(
                "ui.settings.hw.setup_in_progress",
                "⏳ Configuring {title} (existing modules preserved)...",
                title=minfo["title"],
            ),
            "color": "#ffcc00",
        }
        if self._current_node_id == "hardware" and self._caps_container is not None:
            self._render_caps_matrix(caps)

        def _worker():
            res = None
            err_msg = None
            try:
                if force_local_cuda and installer_fn is None:
                    dl_res = download_cuda_libraries(
                        self.cfg_mgr.base_dir,
                        progress_cb=lambda txt: self._safe_mode_progress(key, txt, "#ffcc00"),
                        include_optional=True,
                        force_into_local=True,
                    )
                    prepare_cuda_dll_environment(self.cfg_mgr.base_dir, preload=True)
                    caps_after = self.hw_monitor.refresh_all(force=True)
                    res = {
                        "ok": bool(dl_res.get("ok")),
                        "mode": key,
                        "target_mode": target_mode,
                        "ort_changed": False,
                        "cuda_downloaded": bool(dl_res.get("downloaded_packages")),
                        "restart_required": bool((caps_after or {}).get("ort_restart_required")),
                        "caps": caps_after,
                        "cuda_status": dl_res.get("status") or (caps_after or {}).get("cuda_libraries"),
                        "message": dl_res.get("message") or tr("ui.settings.hw.cuda_checked", "CUDA libraries checked."),
                    }
                else:
                    fn_setup = installer_fn or ensure_ai_backend_dependencies
                    res = fn_setup(
                        self.cfg_mgr.base_dir,
                        key,
                        hw_monitor=self.hw_monitor,
                        registry=self.registry,
                        progress_cb=lambda txt: self._safe_mode_progress(key, txt, "#ffcc00"),
                    )
            except Exception as exc:
                err_msg = str(exc)

            def _done():
                self._ort_installing = False
                self._cuda_downloading = False
                if self._is_closing:
                    return
                if err_msg is not None or not isinstance(res, dict):
                    msg_fail = tr(
                        "ui.settings.hw.setup_err",
                        "✖ Setup error: {err}",
                        err=err_msg or tr("ui.common.unknown_error", "unknown error"),
                    )
                    self._mode_setup_status = {"mode": key, "ok": False, "pct": 0, "text": msg_fail, "color": "#ff5555"}
                    if self._current_node_id == "hardware" and self._caps_container is not None and self._last_caps is not None:
                        self._render_caps_matrix(self._last_caps)
                    return

                ok_all = bool(res.get("ok"))
                restart_req = bool(res.get("restart_required"))
                if res.get("caps") is not None:
                    self._last_caps = res["caps"]
                if res.get("cuda_status") is not None:
                    self._last_cuda_status = res["cuda_status"]
                    if isinstance(self._last_caps, dict):
                        self._last_caps["cuda_libraries"] = res["cuda_status"]

                if ok_all:
                    if res.get("ort_changed") or res.get("cuda_downloaded"):
                        try:
                            self.camera_manager.inference_pool.invalidate_all()
                        except Exception:
                            pass
                    if key in ("auto", "cuda", "directml", "openvino", "cpu"):
                        self.draft.setdefault("hardware", {})["ai_backend"] = key
                        ai_var = self._hw_widget_vars.get("ai_backend")
                        ai_choices = self._hw_widget_vars.get("ai_choices") or _get_ai_backend_choices()
                        if ai_var is not None:
                            self._syncing_ai_backend_var = True
                            try:
                                ai_var.set(self._choice_label_for(ai_choices, key))
                            finally:
                                self._syncing_ai_backend_var = False

                default_msg = (
                    tr("ui.common.done_dot", "Done.")
                    if ok_all
                    else tr("ui.settings.hw.setup_failed", "Failed to complete setup.")
                )
                msg_txt = str(res.get("message") or default_msg).strip()
                res_col = "#ffcc00" if (ok_all and restart_req) else ("#75e08b" if ok_all else "#ff5555")
                self._mode_setup_status = {
                    "mode": key,
                    "ok": ok_all,
                    "pct": 100 if ok_all else 0,
                    "text": msg_txt,
                    "color": res_col,
                }
                if ok_all and restart_req:
                    self._set_footer_status(
                        tr("ui.settings.hw.setup_footer_restart", "⚠ Components installed — click 'Apply' and restart the application."),
                        fg="#ffcc00",
                    )
                elif ok_all:
                    self._set_footer_status(
                        tr("ui.settings.hw.setup_footer_ready", "✔ {mode} mode ready — click '💾 Apply' to save.", mode=target_mode.upper()),
                        fg="#75e08b",
                    )

                if self._current_node_id == "hardware" and self._caps_container is not None and self._last_caps is not None:
                    self._render_caps_matrix(self._last_caps)

            self._ui_queue.put((_done, ()))

        threading.Thread(target=_worker, daemon=True).start()

    def _safe_set_label(self, lbl, text, color):
        def _upd():
            if not self._is_closing and lbl is not None and lbl.winfo_exists():
                lbl.configure(text=text, fg=color)
        self._ui_queue.put((_upd, ()))

    @staticmethod
    def _create_hw_group_frame(
        parent,
        border_col: str,
        title_txt: str,
        title_col: str,
        meta_txt: str,
        bg: str = "#181a24",
        init_px: int = 240,
    ):
        frame = tk.Frame(
            parent,
            bg=bg,
            padx=5,
            pady=5,
            highlightthickness=1,
            highlightbackground=border_col,
        )
        hdr_box = tk.Frame(frame, bg=bg)
        hdr_box.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        title_lbl = tk.Label(
            hdr_box,
            text="",
            fg=title_col,
            bg=bg,
            font=("Segoe UI", 8, "bold"),
            width=1,
            anchor=tk.W,
        )
        title_lbl.pack(side=tk.TOP, fill=tk.X)
        bind_single_line_ellipsis(title_lbl, full_text=title_txt, pad=4, min_px=60, initial_px=init_px)

        meta_lbl = tk.Label(
            hdr_box,
            text="",
            fg="#8f98b3",
            bg=bg,
            font=("Segoe UI", 8),
            width=1,
            anchor=tk.W,
        )
        meta_lbl.pack(side=tk.TOP, fill=tk.X)
        bind_single_line_ellipsis(meta_lbl, full_text=meta_txt, pad=4, min_px=60, initial_px=init_px)

        for hw_lbl in (hdr_box, title_lbl, meta_lbl):
            bind_hover_tooltip(
                hw_lbl,
                lambda tl=title_lbl, ml=meta_lbl: (
                    f"{getattr(tl, '_ellipsis_full_text', '')}\n{getattr(ml, '_ellipsis_full_text', '')}".strip()
                ),
                only_when_truncated=True,
                truncate_sources=(title_lbl, meta_lbl),
            )

        inner_wrap = tk.Frame(frame, bg=bg)
        inner_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return frame, inner_wrap, title_lbl, meta_lbl

    @staticmethod
    def _create_hw_tile_frame(parent, accent_col: str, title_txt: str, sub_col: str = "#9ee8ff", initial_sub: str = None):
        tile_bg = "#222432"
        init_sub_txt = tr("ui.settings.hw.scanning", "scanning...") if initial_sub is None else str(initial_sub)
        tile = tk.Frame(
            parent,
            bg=tile_bg,
            highlightthickness=1,
            highlightbackground="#2e3244",
        )
        tk.Frame(tile, bg=accent_col, height=3).pack(fill=tk.X)
        inner = tk.Frame(tile, bg=tile_bg, padx=7, pady=6)
        inner.pack(fill=tk.BOTH, expand=True)

        hdr_lbl = tk.Label(
            inner,
            text="",
            fg="#ffffff",
            bg=tile_bg,
            font=("Segoe UI", 9, "bold"),
            width=1,
            anchor=tk.W,
        )
        hdr_lbl.pack(side=tk.TOP, fill=tk.X)
        bind_single_line_ellipsis(hdr_lbl, full_text=title_txt, pad=2, min_px=60, initial_px=200)
        bind_hover_tooltip(
            hdr_lbl,
            lambda l=hdr_lbl: str(getattr(l, "_ellipsis_full_text", "") or ""),
            only_when_truncated=True,
            truncate_sources=(hdr_lbl,),
        )

        sub_lbl = tk.Label(
            inner,
            text="",
            fg=sub_col,
            bg=tile_bg,
            font=("Segoe UI", 8, "bold"),
            width=1,
            anchor=tk.W,
        )
        sub_lbl.pack(side=tk.TOP, fill=tk.X, pady=(1, 4))
        bind_single_line_ellipsis(sub_lbl, full_text=init_sub_txt, pad=2, min_px=60, initial_px=200)
        bind_hover_tooltip(
            sub_lbl,
            lambda l=sub_lbl: str(getattr(l, "_ellipsis_full_text", "") or ""),
            only_when_truncated=True,
            truncate_sources=(sub_lbl,),
        )
        return tile, inner, tile_bg, hdr_lbl, sub_lbl

    @staticmethod
    def _build_gpu_telemetry_specs(gpus) -> list:
        unit_gb = tr("ui.unit.gb", "GB")
        unit_mb = tr("ui.unit.mb", "MB")
        gpu_list = list(gpus or [])
        if not gpu_list:
            return [{
                "key": "__empty__",
                "has_gpu": False,
                "group_title": tr("ui.settings.hw.gpu_group_single", "🎮 Graphics Card"),
                "group_meta": tr("ui.settings.hw.gpu_none_meta", "No hardware GPU detected"),
                "gpu_title": tr("ui.settings.hw.gpu_tile_single", "Graphics Card (GPU)"),
                "gpu_sub": tr("ui.settings.hw.gpu_none_sub", "Not detected"),
                "gpu_text": tr("ui.settings.hw.gpu_none_text", "GPU acceleration unavailable\nRunning baseline CPU mode"),
                "gpu_value": 0.0,
                "has_vram": False,
                "vram_title": "",
                "vram_sub": "",
                "vram_text": "",
                "vram_value": 0.0,
            }]

        def _fmt_mem(used_mb: float, tot_mb: float, free_mb: float):
            if tot_mb >= 1024.0:
                return f"{used_mb / 1024.0:.2f}", f"{tot_mb / 1024.0:.2f}", f"{free_mb / 1024.0:.2f}", unit_gb
            return f"{used_mb:.0f}", f"{tot_mb:.0f}", f"{free_mb:.0f}", unit_mb

        multi = len(gpu_list) > 1
        specs = []
        for idx, g in enumerate(gpu_list):
            num = idx + 1
            group_title = (
                tr("ui.settings.hw.gpu_group_num", "🎮 Graphics Card #{n}", n=num)
                if multi
                else tr("ui.settings.hw.gpu_group_single", "🎮 Graphics Card")
            )
            gpu_title = (
                tr("ui.settings.hw.gpu_tile_num", "Graphics Card (GPU #{n})", n=num)
                if multi
                else tr("ui.settings.hw.gpu_tile_single", "Graphics Card (GPU)")
            )
            vram_title = (
                tr("ui.settings.hw.vram_tile_num", "Video Memory (VRAM #{n})", n=num)
                if multi
                else tr("ui.settings.hw.vram_tile_single", "Video Memory (VRAM)")
            )

            vendor = str(g.get("vendor") or "").strip()
            name = str(g.get("name") or "GPU").strip()
            if vendor and vendor != "Other" and vendor.lower() not in name.lower():
                disp_name = f"[{vendor}] {name}"
            else:
                disp_name = name

            drv = str(g.get("driver") or "").strip()
            live = bool(g.get("live_metrics"))
            v_tot_raw = g.get("vram_total_mb")
            v_used_raw = g.get("vram_used_mb")
            has_vram = bool(live and v_used_raw is not None and v_tot_raw is not None and float(v_tot_raw) > 0)

            meta_base = (
                tr("ui.settings.hw.gpu_meta_with_vram", "Graphics processor and video memory")
                if has_vram
                else tr("ui.settings.hw.gpu_meta_only", "Graphics processor (GPU)")
            )
            if drv:
                drv_s = tr("ui.settings.hw.drv_short", "Drv. {drv}", drv=drv)
                group_meta = f"{meta_base} • {drv_s}"
            else:
                group_meta = meta_base

            sub_parts = [disp_name]
            if not has_vram and drv:
                sub_parts.append(tr("ui.settings.hw.drv_short", "Drv. {drv}", drv=drv))
            if not has_vram and v_tot_raw is not None and float(v_tot_raw) > 0:
                vt = float(v_tot_raw)
                v_str = f"{vt / 1024.0:.2f} {unit_gb}" if vt >= 1024.0 else f"{vt:.0f} {unit_mb}"
                sub_parts.append(f"VRAM {v_str}")
            gpu_sub = " • ".join(sub_parts)

            if live:
                util = g.get("gpu_util_pct")
                gpu_val = min(100.0, max(0.0, float(util))) if util is not None else 0.0
                util_s = f"{float(util):.0f}%" if util is not None else "—"
                line1 = tr("ui.settings.hw.gpu_load", "GPU load: {pct}", pct=util_s)
                codec_parts = []
                if g.get("enc_util_pct") is not None:
                    codec_parts.append(f"ENC: {float(g['enc_util_pct']):.0f}%")
                if g.get("dec_util_pct") is not None:
                    codec_parts.append(f"DEC: {float(g['dec_util_pct']):.0f}%")
                line2 = (
                    "  |  ".join(codec_parts)
                    if codec_parts
                    else tr("ui.settings.hw.gpu_vid_ready", "Hardware video acceleration ready")
                )
                gpu_text = f"{line1}\n{line2}"
            else:
                gpu_val = 0.0
                gpu_text = tr("ui.settings.hw.gpu_no_live_text", "Load: unavailable\nActive graphics adapter")

            vram_sub = ""
            vram_text = ""
            vram_val = 0.0
            if has_vram:
                v_tot = float(v_tot_raw)
                v_used = max(0.0, float(v_used_raw))
                v_free_raw = g.get("vram_free_mb")
                v_free = max(0.0, float(v_free_raw) if v_free_raw is not None else (v_tot - v_used))
                vram_val = min(100.0, max(0.0, (v_used / v_tot) * 100.0))
                u_s, t_s, f_s, unit = _fmt_mem(v_used, v_tot, v_free)
                vram_sub = tr("ui.settings.hw.total_mem_val", "Total: {val} {unit}", val=t_s, unit=unit)
                vram_text = tr(
                    "ui.settings.hw.vram_usage",
                    "Used: {used} {unit} ({pct:.0f}%)\nFree: {free} {unit}",
                    used=u_s,
                    unit=unit,
                    pct=vram_val,
                    free=f_s,
                )

            specs.append({
                "key": f"{idx}:{vendor}:{name}",
                "has_gpu": True,
                "group_title": group_title,
                "group_meta": group_meta,
                "gpu_title": gpu_title,
                "gpu_sub": gpu_sub,
                "gpu_text": gpu_text,
                "gpu_value": gpu_val,
                "has_vram": has_vram,
                "vram_title": vram_title,
                "vram_sub": vram_sub,
                "vram_text": vram_text,
                "vram_value": vram_val,
            })
        return specs

    def _relayout_hw_tiles(self, avail_w=None, force: bool = False):
        tl = self._telemetry_labels
        if not tl:
            return
        wrap = tl.get("hw_tiles_wrap")
        if wrap is None or not wrap.winfo_exists():
            return

        hw_groups = []
        sys_g = tl.get("sys_group")
        if isinstance(sys_g, dict) and sys_g.get("frame") is not None and sys_g["frame"].winfo_exists():
            hw_groups.append(sys_g)
        for gg in (tl.get("gpu_groups") or []):
            gf = gg.get("frame") if isinstance(gg, dict) else None
            if gf is not None and gf.winfo_exists():
                hw_groups.append(gg)
        if not hw_groups:
            return

        if avail_w is None or int(avail_w) <= 40:
            try:
                cur_w = int(wrap.winfo_width() or 0)
            except Exception:
                cur_w = 0
            w = cur_w if cur_w > 40 else 720
        else:
            w = int(avail_w)

        max_slots = max(1, w // 150)
        group_spans = [min(len(ge["tiles"]), max_slots) for ge in hw_groups]

        rows = []
        cur_row = []
        cur_slots = 0
        for ge, g_span in zip(hw_groups, group_spans):
            if cur_row and (cur_slots + g_span) > max_slots:
                rows.append(cur_row)
                cur_row = []
                cur_slots = 0
            cur_row.append((ge, g_span))
            cur_slots += g_span
        if cur_row:
            rows.append(cur_row)

        max_g_cols = max(len(r) for r in rows)
        col_weights = []
        col_compatible = True
        for c_i in range(max_g_cols):
            spans_at_c = {r[c_i][1] for r in rows if c_i < len(r)}
            if len(spans_at_c) != 1:
                col_compatible = False
                break
            col_weights.append(next(iter(spans_at_c)))

        if col_compatible:
            cols = max_g_cols
        else:
            total_span = sum(group_spans)
            cols = max(1, min(total_span, max_slots))
            col_weights = [1] * cols

        state = tl.setdefault("hw_layout_state", {"sig": None, "prev_cols": 0, "prev_rows": 0})
        grp_sig = tuple((ge["key"], len(ge["tiles"]), id(ge["frame"])) for ge in hw_groups)
        new_sig = (col_compatible, tuple(col_weights), max_slots, grp_sig)
        if not force and state.get("sig") == new_sig:
            return
        state["sig"] = new_sig

        prev_cols = int(state.get("prev_cols") or 0)
        state["prev_cols"] = cols
        for c_idx in range(max(cols, prev_cols)):
            wrap.grid_columnconfigure(
                c_idx,
                weight=col_weights[c_idx] if c_idx < cols else 0,
                uniform="hw_slot" if c_idx < cols else "",
            )

        used_rows = len(rows)
        for cur_r, row_items in enumerate(rows):
            cur_slot = 0
            tp = 0 if cur_r == 0 else 6
            for idx_in_row, (ge, g_span) in enumerate(row_items):
                if col_compatible:
                    col_pos = idx_in_row
                    span_cols = 1
                    lp = 0 if idx_in_row == 0 else 4
                    rp = 0 if idx_in_row == (cols - 1) else 4
                else:
                    col_pos = cur_slot
                    span_cols = g_span
                    lp = 0 if cur_slot == 0 else 4
                    rp = 0 if (cur_slot + g_span) >= cols else 4
                    cur_slot += g_span

                ge["frame"].grid(
                    row=cur_r,
                    column=col_pos,
                    columnspan=span_cols,
                    sticky="nsew",
                    padx=(lp, rp),
                    pady=(tp, 0),
                )

                iw = ge["inner_wrap"]
                tlist = ge["tiles"]
                in_cols = max(1, min(len(tlist), g_span))
                max_in_cfg = max(in_cols, int(ge.get("prev_in_cols") or 0))
                ge["prev_in_cols"] = in_cols
                for ic_i in range(max_in_cfg):
                    iw.grid_columnconfigure(
                        ic_i,
                        weight=1 if ic_i < in_cols else 0,
                        uniform=ge["inner_uniform"] if ic_i < in_cols else "",
                    )
                in_rows = max(1, (len(tlist) + in_cols - 1) // in_cols)
                max_in_r = max(in_rows, int(ge.get("prev_in_rows") or 0))
                ge["prev_in_rows"] = in_rows
                for ir_i in range(max_in_r):
                    iw.grid_rowconfigure(ir_i, weight=1 if ir_i < in_rows else 0)
                for tidx, tf in enumerate(tlist):
                    ir, ic = divmod(tidx, in_cols)
                    ilp = 0 if ic == 0 else 3
                    irp = 0 if ic == (in_cols - 1) else 3
                    itp = 0 if ir == 0 else 5
                    tf.grid(row=ir, column=ic, sticky="nsew", padx=(ilp, irp), pady=(itp, 0))

        prev_rows = int(state.get("prev_rows") or 0)
        state["prev_rows"] = used_rows
        for r_i in range(max(used_rows, prev_rows)):
            wrap.grid_rowconfigure(r_i, weight=1 if r_i < used_rows else 0)

    @staticmethod
    def _format_phys_disk_size(size_gb) -> str:
        if size_gb is None:
            return ""
        try:
            v = float(size_gb)
        except Exception:
            return ""
        if v <= 0:
            return ""
        if v >= 1024.0:
            return f"{v / 1024.0:.2f} {tr('ui.unit.tb', 'TB')}"
        return f"{v:.0f} {tr('ui.unit.gb', 'GB')}"

    def _tick_telemetry(self):
        self._cancel_telemetry_timer()
        if self._is_closing or self._current_node_id != "hardware" or not self._telemetry_labels:
            return
        try:
            tl = self._telemetry_labels
            m = self.hw_monitor.get_live_metrics()
            cpu = m["cpu"]
            ram = m["ram"]
            gpus = m["gpus"]
            disks = m["disks"]

            unit_gb = tr("ui.unit.gb", "GB")
            unit_mb = tr("ui.unit.mb", "MB")
            unit_mhz = tr("ui.unit.mhz", "MHz")

            cores_txt = (
                tr("ui.settings.hw.cpu_cores_both", "{phys}c / {logi}t", phys=cpu["physical_cores"], logi=cpu["logical_threads"])
                if cpu["physical_cores"]
                else tr("ui.settings.hw.cpu_cores_threads", "{logi}t", logi=cpu["logical_threads"])
            )
            freq_txt = f" @ {int(cpu['freq_mhz'])} {unit_mhz}" if cpu.get("freq_mhz") else ""
            cpu_sub_txt = str(cpu.get("name") or "CPU")
            if tl.get("cpu_sub_lbl") is not None and tl.get("last_cpu_sub") != cpu_sub_txt:
                set_single_line_ellipsis_text(tl["cpu_sub_lbl"], cpu_sub_txt)
                tl["last_cpu_sub"] = cpu_sub_txt
            cpu_txt = (
                f"{cores_txt}{freq_txt}\n"
                + tr("ui.settings.hw.cpu_load_line", "Total: {tot:.1f}% | App: {proc:.1f}%", tot=cpu["usage_pct"], proc=cpu["process_pct"])
            )
            if str(tl["cpu_lbl"].cget("text")) != cpu_txt:
                tl["cpu_lbl"].configure(text=cpu_txt)
            tl["cpu_pb"]["value"] = min(100, max(0, cpu["usage_pct"]))

            ram_sub_txt = tr("ui.settings.hw.total_mem_val", "Total: {val} {unit}", val=f"{ram['total_gb']:.2f}", unit=unit_gb)
            if tl.get("ram_sub_lbl") is not None and tl.get("last_ram_sub") != ram_sub_txt:
                set_single_line_ellipsis_text(tl["ram_sub_lbl"], ram_sub_txt)
                tl["last_ram_sub"] = ram_sub_txt
            ram_txt = tr(
                "ui.settings.hw.ram_usage",
                "Used: {used:.2f} {unit_gb} ({pct:.0f}%)\nFree: {free:.1f} {unit_gb} | App: {proc:.0f} {unit_mb}",
                used=ram["used_gb"],
                pct=ram["percent"],
                free=ram["free_gb"],
                proc=ram["process_mb"],
                unit_gb=unit_gb,
                unit_mb=unit_mb,
            )
            if str(tl["ram_lbl"].cget("text")) != ram_txt:
                tl["ram_lbl"].configure(text=ram_txt)
            tl["ram_pb"]["value"] = min(100, max(0, ram["percent"]))

            hw_wrap = tl["hw_tiles_wrap"]
            gpu_specs = self._build_gpu_telemetry_specs(gpus)
            new_gpu_sig = tuple((s["key"], bool(s["has_vram"]), bool(s["has_gpu"])) for s in gpu_specs)
            gpu_rows = tl.get("gpu_rows")
            if not isinstance(gpu_rows, list):
                gpu_rows = []
                tl["gpu_rows"] = gpu_rows
            gpu_groups = tl.get("gpu_groups")
            if not isinstance(gpu_groups, list):
                gpu_groups = []
                tl["gpu_groups"] = gpu_groups

            if tl.get("gpu_sig") != new_gpu_sig or len(gpu_rows) != len(gpu_specs):
                had_prev_gpus = tl.get("gpu_sig") is not None

                def _rebuild_gpus():
                    if not hw_wrap.winfo_exists():
                        return
                    for old_g in list(gpu_groups):
                        old_f = old_g.get("frame")
                        if old_f is not None and old_f.winfo_exists():
                            old_f.destroy()
                    gpu_groups.clear()
                    gpu_rows.clear()

                    gpu_grp_bg = "#181a24"
                    for gidx, spec in enumerate(gpu_specs):
                        has_g = bool(spec["has_gpu"])
                        grp_border = "#634299" if has_g else "#383d52"
                        grp_title_col = "#d0b8ff" if has_g else "#9ba3b8"
                        tile_cnt = 2 if spec["has_vram"] else 1
                        g_frame, g_grp_inner, g_grp_title_lbl, g_grp_meta_lbl = self._create_hw_group_frame(
                            hw_wrap,
                            border_col=grp_border,
                            title_txt=spec["group_title"],
                            title_col=grp_title_col,
                            meta_txt=spec["group_meta"],
                            bg=gpu_grp_bg,
                            init_px=max(145, 160 * tile_cnt),
                        )

                        g_accent = "#b388ff" if has_g else "#5e6478"
                        g_sub_col = "#d0b8ff" if has_g else "#8f98b3"
                        g_tile, g_inner, tbg, g_title_lbl, g_sub_lbl = self._create_hw_tile_frame(
                            g_grp_inner,
                            g_accent,
                            spec["gpu_title"],
                            sub_col=g_sub_col,
                            initial_sub=spec["gpu_sub"],
                        )

                        g_lbl = tk.Label(
                            g_inner,
                            text=spec["gpu_text"],
                            fg="#d5dae8",
                            bg=tbg,
                            font=("Segoe UI", 8),
                            anchor=tk.W,
                            justify=tk.LEFT,
                            wraplength=200,
                        )
                        g_lbl.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
                        self._bind_auto_wrap(g_lbl, initial=200, min_wrap=105, pad=4)

                        g_pb = ttk.Progressbar(g_inner, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
                        g_pb.pack(side=tk.BOTTOM, fill=tk.X)
                        g_pb["value"] = spec["gpu_value"]

                        grp_tiles = [g_tile]
                        v_tile = None
                        v_title_lbl = None
                        v_sub_lbl = None
                        v_lbl = None
                        v_pb = None
                        if spec["has_vram"]:
                            v_tile, v_inner, vtbg, v_title_lbl, v_sub_lbl = self._create_hw_tile_frame(
                                g_grp_inner,
                                g_accent,
                                spec["vram_title"],
                                sub_col=g_sub_col,
                                initial_sub=spec["vram_sub"],
                            )

                            v_lbl = tk.Label(
                                v_inner,
                                text=spec["vram_text"],
                                fg="#d5dae8",
                                bg=vtbg,
                                font=("Segoe UI", 8),
                                anchor=tk.W,
                                justify=tk.LEFT,
                                wraplength=200,
                            )
                            v_lbl.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
                            self._bind_auto_wrap(v_lbl, initial=200, min_wrap=105, pad=4)

                            v_pb = ttk.Progressbar(v_inner, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
                            v_pb.pack(side=tk.BOTTOM, fill=tk.X)
                            v_pb["value"] = spec["vram_value"]
                            grp_tiles.append(v_tile)

                        gpu_groups.append({
                            "key": f"gpu:{gidx}:{spec['key']}",
                            "frame": g_frame,
                            "inner_wrap": g_grp_inner,
                            "inner_uniform": f"hw_gpu_inner_{gidx}",
                            "title_lbl": g_grp_title_lbl,
                            "meta_lbl": g_grp_meta_lbl,
                            "tiles": grp_tiles,
                            "prev_in_cols": 0,
                            "prev_in_rows": 0,
                        })

                        gpu_rows.append({
                            "key": spec["key"],
                            "group_frame": g_frame,
                            "group_meta_lbl": g_grp_meta_lbl,
                            "tile": g_tile,
                            "vram_tile": v_tile,
                            "gpu_title_lbl": g_title_lbl,
                            "gpu_sub_lbl": g_sub_lbl,
                            "gpu_lbl": g_lbl,
                            "gpu_pb": g_pb,
                            "vram_title_lbl": v_title_lbl,
                            "vram_sub_lbl": v_sub_lbl,
                            "vram_lbl": v_lbl,
                            "vram_pb": v_pb,
                            "last_group_meta": spec["group_meta"],
                            "last_gpu_sub": spec["gpu_sub"],
                            "last_gpu_text": spec["gpu_text"],
                            "last_vram_sub": spec["vram_sub"],
                            "last_vram_text": spec["vram_text"],
                        })
                    tl["gpu_sig"] = new_gpu_sig
                    self._relayout_hw_tiles(force=True)

                if self.scroll_panel is not None and had_prev_gpus:
                    self.scroll_panel.preserve_scroll_during(_rebuild_gpus)
                else:
                    _rebuild_gpus()
            else:
                for spec, rinfo in zip(gpu_specs, gpu_rows):
                    if rinfo.get("last_group_meta") != spec["group_meta"] and rinfo.get("group_meta_lbl") is not None:
                        set_single_line_ellipsis_text(rinfo["group_meta_lbl"], spec["group_meta"])
                        rinfo["last_group_meta"] = spec["group_meta"]
                    if rinfo.get("last_gpu_sub") != spec["gpu_sub"] and rinfo.get("gpu_sub_lbl") is not None:
                        set_single_line_ellipsis_text(rinfo["gpu_sub_lbl"], spec["gpu_sub"])
                        rinfo["last_gpu_sub"] = spec["gpu_sub"]
                    if rinfo["last_gpu_text"] != spec["gpu_text"]:
                        rinfo["gpu_lbl"].configure(text=spec["gpu_text"])
                        rinfo["last_gpu_text"] = spec["gpu_text"]
                    rinfo["gpu_pb"]["value"] = spec["gpu_value"]
                    if spec["has_vram"] and rinfo.get("vram_lbl") is not None and rinfo.get("vram_pb") is not None:
                        if rinfo.get("last_vram_sub") != spec["vram_sub"] and rinfo.get("vram_sub_lbl") is not None:
                            set_single_line_ellipsis_text(rinfo["vram_sub_lbl"], spec["vram_sub"])
                            rinfo["last_vram_sub"] = spec["vram_sub"]
                        if rinfo["last_vram_text"] != spec["vram_text"]:
                            rinfo["vram_lbl"].configure(text=spec["vram_text"])
                            rinfo["last_vram_text"] = spec["vram_text"]
                        rinfo["vram_pb"]["value"] = spec["vram_value"]

            df = tl["disks_frame"]
            bg = str(df.cget("bg"))
            is_op_mode = self._is_draft_operator_mode()
            cur_arch = self._get_effective_archive_root()

            def _disp_media(m_val: str) -> str:
                s_val = str(m_val or "").strip()
                if not s_val or s_val.lower() in ("disk", "диск"):
                    return tr("hw.disk.media_generic", "Disk")
                return s_val

            if is_op_mode:
                arch_lbl_txt = tr(
                    "ui.settings.hw.disk_arch_op",
                    "Saved archive folder (Server Mode): {path}  —  disk writes are disabled in Operator Mode",
                    path=cur_arch,
                )
            else:
                arch_lbl_txt = tr(
                    "ui.settings.hw.disk_arch_cur",
                    "Current archive folder: {path}",
                    path=cur_arch,
                )
            disks = sorted(disks or [], key=lambda item: str(item.get("mount") or "").upper())
            new_mounts = tuple(d["mount"] for d in disks)
            new_disks_sig = (
                is_op_mode,
                tuple(
                    (
                        d["mount"],
                        str(d.get("phys_id") or ""),
                        str(d.get("phys_model") or ""),
                        str(d.get("phys_size_gb") or ""),
                    )
                    for d in disks
                ),
            )
            disk_rows = tl.get("disk_rows")
            if not isinstance(disk_rows, dict):
                disk_rows = {}
                tl["disk_rows"] = disk_rows
            phys_groups_ref = tl.get("phys_groups")
            if not isinstance(phys_groups_ref, list):
                phys_groups_ref = []
                tl["phys_groups"] = phys_groups_ref

            def _make_set_drive_cb(m_path):
                return lambda: self._choose_archive_folder_for_drive(m_path)

            if tl.get("disks_sig") != new_disks_sig or len(disk_rows) != len(disks):
                def _rebuild_disks():
                    if not df.winfo_exists():
                        return
                    for ch in list(df.winfo_children()):
                        ch.destroy()
                    disk_rows.clear()
                    phys_groups_ref.clear()

                    hdr_disk_txt = (
                        tr(
                            "ui.settings.hw.disk_hdr_op",
                            "Drives and archive space (Operator Mode — disk writes and folder selection locked):",
                        )
                        if is_op_mode
                        else tr(
                            "ui.settings.hw.disk_hdr_srv",
                            "Drives and archive space (partitions grouped by physical drive; click a drive to choose folder):",
                        )
                    )
                    hdr_disk_lbl = tk.Label(
                        df,
                        text=hdr_disk_txt,
                        fg="#ffcc00" if is_op_mode else "#e6eaf5",
                        bg=bg,
                        font=("Segoe UI", 9, "bold"),
                        anchor=tk.W,
                        justify=tk.LEFT,
                        wraplength=440,
                    )
                    hdr_disk_lbl.pack(fill=tk.X, pady=(2, 2))
                    self._bind_auto_wrap(hdr_disk_lbl, initial=440, min_wrap=220)

                    disk_arch_lbl = tk.Label(
                        df,
                        text=arch_lbl_txt,
                        fg="#ffcc00" if is_op_mode else "#9ee8ff",
                        bg=bg,
                        font=("Segoe UI", 8),
                        anchor=tk.W,
                        justify=tk.LEFT,
                        wraplength=440,
                    )
                    disk_arch_lbl.pack(fill=tk.X, pady=(0, 6))
                    self._bind_auto_wrap(disk_arch_lbl, initial=440, min_wrap=220)
                    tl["disk_arch_lbl"] = disk_arch_lbl
                    tl["last_disk_arch_txt"] = arch_lbl_txt

                    if not disks:
                        tk.Label(
                            df,
                            text=tr("ui.settings.hw.disks_none", "No storage drives detected."),
                            fg="#9ba3b8",
                            bg=bg,
                            font=("Segoe UI", 9),
                            anchor=tk.W,
                        ).pack(fill=tk.X, pady=(0, 2))
                        tl["disks_mounts"] = new_mounts
                        tl["disks_sig"] = new_disks_sig
                        return

                    # Групуємо логічні розділи за фізичним диском (зі збереженням порядку)
                    grouped_phys = []
                    group_by_pid = {}
                    for d in disks:
                        mount = d["mount"]
                        pid = str(d.get("phys_id") or "").strip() or f"mount:{mount}"
                        if pid not in group_by_pid:
                            grp = {
                                "phys_id": pid,
                                "phys_model": str(d.get("phys_model") or "").strip(),
                                "phys_size_gb": d.get("phys_size_gb"),
                                "media": str(d.get("media") or "Disk").strip(),
                                "disks": [],
                            }
                            group_by_pid[pid] = grp
                            grouped_phys.append(grp)
                        else:
                            grp = group_by_pid[pid]
                            if not grp["phys_model"] and d.get("phys_model"):
                                grp["phys_model"] = str(d["phys_model"]).strip()
                            if not grp["phys_size_gb"] and d.get("phys_size_gb"):
                                grp["phys_size_gb"] = d["phys_size_gb"]
                            if grp["media"] in ("", "Disk", "Диск") and d.get("media"):
                                grp["media"] = str(d["media"]).strip()
                        group_by_pid[pid]["disks"].append(d)

                    groups_wrap = tk.Frame(df, bg=bg)
                    groups_wrap.pack(fill=tk.X)

                    for gidx, grp in enumerate(grouped_phys):
                        part_disks = grp["disks"]
                        has_arch_in_grp = any(self._is_path_on_mount(cur_arch, pd["mount"]) for pd in part_disks)
                        is_multi_part = len(part_disks) > 1
                        phys_bg = "#181a24"
                        phys_border = "#2d7a4d" if has_arch_in_grp else ("#3a435e" if is_multi_part else "#2c3042")

                        phys_frame = tk.Frame(
                            groups_wrap,
                            bg=phys_bg,
                            padx=5,
                            pady=5,
                            highlightthickness=1,
                            highlightbackground=phys_border,
                        )

                        p_model = grp["phys_model"]
                        p_media = _disp_media(grp["media"])
                        p_size_s = self._format_phys_disk_size(grp["phys_size_gb"])
                        meta_items = [p_media]
                        if p_size_s:
                            meta_items.append(p_size_s)
                        if is_multi_part:
                            meta_items.append(tr("ui.settings.hw.disk_parts", "{n} part.", n=len(part_disks)))

                        if p_model:
                            phys_title_txt = f"💿 {p_model}"
                        else:
                            m_list = ", ".join(pd["mount"].rstrip("\\") for pd in part_disks)
                            phys_title_txt = tr(
                                "ui.settings.hw.phys_disk_fallback",
                                "💿 Physical drive ({mounts})",
                                mounts=m_list,
                            )
                        phys_meta_txt = " • ".join(meta_items)
                        phys_full_tip = f"{phys_title_txt}\n{phys_meta_txt}"

                        phys_hdr_col = "#9ee8ff" if is_multi_part else "#e6eaf5"
                        phys_hdr_box = tk.Frame(phys_frame, bg=phys_bg)
                        phys_hdr_box.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

                        init_hdr_px = max(145, 152 * len(part_disks))
                        phys_title_lbl = tk.Label(
                            phys_hdr_box,
                            text="",
                            fg=phys_hdr_col,
                            bg=phys_bg,
                            font=("Segoe UI", 8, "bold"),
                            width=1,
                            anchor=tk.W,
                        )
                        phys_title_lbl.pack(side=tk.TOP, fill=tk.X)
                        bind_single_line_ellipsis(
                            phys_title_lbl,
                            full_text=phys_title_txt,
                            pad=4,
                            min_px=60,
                            initial_px=init_hdr_px,
                        )

                        phys_meta_lbl = tk.Label(
                            phys_hdr_box,
                            text="",
                            fg="#8f98b3",
                            bg=phys_bg,
                            font=("Segoe UI", 8),
                            width=1,
                            anchor=tk.W,
                        )
                        phys_meta_lbl.pack(side=tk.TOP, fill=tk.X)
                        bind_single_line_ellipsis(
                            phys_meta_lbl,
                            full_text=phys_meta_txt,
                            pad=4,
                            min_px=60,
                            initial_px=init_hdr_px,
                        )

                        for hw_lbl in (phys_hdr_box, phys_title_lbl, phys_meta_lbl):
                            bind_hover_tooltip(
                                hw_lbl,
                                phys_full_tip,
                                only_when_truncated=True,
                                truncate_sources=(phys_title_lbl, phys_meta_lbl),
                            )

                        inner_wrap = tk.Frame(phys_frame, bg=phys_bg)
                        inner_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                        grp_tiles = []
                        for d in part_disks:
                            mount = d["mount"]
                            is_arch_drive = self._is_path_on_mount(cur_arch, mount)
                            drive_cb = _make_set_drive_cb(mount)
                            tile_bg = "#222432"
                            tile_cursor = "arrow" if is_op_mode else "hand2"
                            tile = tk.Frame(
                                inner_wrap,
                                bg=tile_bg,
                                padx=7,
                                pady=6,
                                highlightthickness=1,
                                highlightbackground="#2d7a4d" if is_arch_drive else "#2e3244",
                                cursor=tile_cursor,
                            )
                            grp_tiles.append(tile)

                            raw_media = str(d.get("media") or "Disk")
                            badge_txt = f"[{_disp_media(raw_media)}]"
                            badge_col = "#00e5ff" if "SSD" in raw_media.upper() else "#ffcc00"
                            fs_txt = str(d.get("fs") or "").strip()
                            pct_val = min(100.0, max(0.0, float(d.get("percent") or 0.0)))
                            info_txt = tr(
                                "ui.settings.hw.disk_tile_info",
                                "Free: {free:.1f} {unit}\nTotal: {tot:.1f} {unit} ({pct:.0f}%)",
                                free=d["free_gb"],
                                tot=d["total_gb"],
                                pct=pct_val,
                                unit=unit_gb,
                            )
                            if is_op_mode:
                                btn_txt = tr("ui.settings.hw.disk_btn_locked", "🔒 Writes disabled")
                                btn_bg = "#242634"
                                btn_fg = "#8d95ab"
                            else:
                                btn_txt = (
                                    tr("ui.settings.hw.disk_btn_current", "★ Current archive")
                                    if is_arch_drive
                                    else tr("ui.settings.hw.disk_btn_choose", "📁 Choose folder")
                                )
                                btn_bg = "#1e442c" if is_arch_drive else "#2e3142"
                                btn_fg = "#75e08b" if is_arch_drive else "#00e5ff"

                            hdr_row = tk.Frame(tile, bg=tile_bg, cursor=tile_cursor)
                            hdr_row.pack(side=tk.TOP, fill=tk.X)
                            tk.Label(
                                hdr_row, text=f"{mount}", fg="#ffffff", bg=tile_bg,
                                font=("Consolas", 10, "bold"), anchor=tk.W, cursor=tile_cursor
                            ).pack(side=tk.LEFT)
                            media_lbl = tk.Label(
                                hdr_row, text=badge_txt, fg=badge_col, bg=tile_bg,
                                font=("Segoe UI", 8, "bold"), anchor=tk.W, cursor=tile_cursor
                            )
                            media_lbl.pack(side=tk.LEFT, padx=(4, 2))
                            fs_lbl = tk.Label(
                                hdr_row, text=fs_txt, fg="#9ba3b8", bg=tile_bg,
                                font=("Segoe UI", 8), anchor=tk.E, cursor=tile_cursor
                            )
                            fs_lbl.pack(side=tk.RIGHT)

                            info_lbl = tk.Label(
                                tile, text=info_txt, fg="#d5dae8", bg=tile_bg,
                                font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=150, cursor=tile_cursor
                            )
                            info_lbl.pack(side=tk.TOP, fill=tk.X, pady=(4, 4))
                            self._bind_auto_wrap(info_lbl, initial=150, min_wrap=105, pad=14)

                            btn = tk.Button(
                                tile, text=btn_txt, command=drive_cb,
                                state=tk.DISABLED if is_op_mode else tk.NORMAL,
                                bg=btn_bg, fg=btn_fg,
                                disabledforeground="#8d95ab",
                                relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=5, pady=3, cursor=tile_cursor
                            )
                            btn.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

                            disk_pb = ttk.Progressbar(tile, maximum=100, style="SettingsDark.Horizontal.TProgressbar", cursor=tile_cursor)
                            disk_pb.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
                            disk_pb["value"] = pct_val

                            if not is_op_mode:
                                self._bind_tile_click(tile, drive_cb, skip_types=(tk.Button,))

                            disk_rows[mount] = {
                                "phys_frame": phys_frame,
                                "tile": tile,
                                "media_lbl": media_lbl,
                                "fs_lbl": fs_lbl,
                                "info_lbl": info_lbl,
                                "disk_pb": disk_pb,
                                "btn": btn,
                                "last_media": (badge_txt, badge_col),
                                "last_fs": fs_txt,
                                "last_info": info_txt,
                                "last_pct": pct_val,
                                "last_arch": is_arch_drive,
                            }

                        phys_groups_ref.append({
                            "phys_id": grp["phys_id"],
                            "frame": phys_frame,
                            "inner_wrap": inner_wrap,
                            "inner_uniform": f"disk_inner_{gidx}",
                            "tiles": grp_tiles,
                            "mounts": [pd["mount"] for pd in part_disks],
                            "is_multi": is_multi_part,
                            "prev_in_cols": 0,
                            "prev_in_rows": 0,
                        })

                    layout_state = {"sig": None, "prev_cols": 0}

                    def _relayout_disk_groups(avail_w: int):
                        if not phys_groups_ref or not groups_wrap.winfo_exists():
                            return
                        w = int(avail_w) if int(avail_w) > 40 else 720
                        max_slots = max(1, w // 160)
                        group_spans = [min(len(ge["tiles"]), max_slots) for ge in phys_groups_ref]
                        total_span = sum(group_spans)
                        cols = max(1, min(total_span, max_slots))
                        if layout_state["sig"] == (cols, max_slots):
                            return
                        layout_state["sig"] = (cols, max_slots)

                        max_c_cfg = max(cols, layout_state["prev_cols"])
                        layout_state["prev_cols"] = cols
                        for c_idx in range(max_c_cfg):
                            groups_wrap.grid_columnconfigure(
                                c_idx,
                                weight=1 if c_idx < cols else 0,
                                uniform="disk_slot" if c_idx < cols else "",
                            )

                        cur_r = 0
                        cur_c = 0
                        for ge, g_span in zip(phys_groups_ref, group_spans):
                            if cur_c > 0 and (cur_c + g_span) > cols:
                                cur_r += 1
                                cur_c = 0
                            lp = 0 if cur_c == 0 else 4
                            rp = 0 if (cur_c + g_span) >= cols else 4
                            tp = 0 if cur_r == 0 else 6
                            ge["frame"].grid(
                                row=cur_r,
                                column=cur_c,
                                columnspan=g_span,
                                sticky="nsew",
                                padx=(lp, rp),
                                pady=(tp, 0),
                            )

                            iw = ge["inner_wrap"]
                            tlist = ge["tiles"]
                            in_cols = max(1, min(len(tlist), g_span))
                            max_in_cfg = max(in_cols, ge["prev_in_cols"])
                            ge["prev_in_cols"] = in_cols
                            for ic_i in range(max_in_cfg):
                                iw.grid_columnconfigure(
                                    ic_i,
                                    weight=1 if ic_i < in_cols else 0,
                                    uniform=ge["inner_uniform"] if ic_i < in_cols else "",
                                )
                            in_rows = max(1, (len(tlist) + in_cols - 1) // in_cols)
                            max_in_r = max(in_rows, ge.get("prev_in_rows", 0))
                            ge["prev_in_rows"] = in_rows
                            for ir_i in range(max_in_r):
                                iw.grid_rowconfigure(ir_i, weight=1 if ir_i < in_rows else 0)
                            for tidx, tf in enumerate(tlist):
                                ir, ic = divmod(tidx, in_cols)
                                ilp = 0 if ic == 0 else 3
                                irp = 0 if ic == (in_cols - 1) else 3
                                itp = 0 if ir == 0 else 5
                                tf.grid(row=ir, column=ic, sticky="nsew", padx=(ilp, irp), pady=(itp, 0))

                            cur_c += g_span
                            if cur_c >= cols:
                                cur_r += 1
                                cur_c = 0

                    try:
                        df_w = int(df.winfo_width())
                    except Exception:
                        df_w = 0
                    _relayout_disk_groups(df_w if df_w > 40 else 720)
                    groups_wrap.bind("<Configure>", lambda e: _relayout_disk_groups(e.width), add="+")
                    tl["disks_mounts"] = new_mounts
                    tl["disks_sig"] = new_disks_sig

                if self.scroll_panel is not None and tl.get("disks_sig") is not None:
                    self.scroll_panel.preserve_scroll_during(_rebuild_disks)
                else:
                    _rebuild_disks()
            else:
                if tl.get("disk_arch_lbl") is not None and tl.get("last_disk_arch_txt") != arch_lbl_txt:
                    tl["disk_arch_lbl"].configure(text=arch_lbl_txt)
                    tl["last_disk_arch_txt"] = arch_lbl_txt
                for d in disks:
                    mount = d["mount"]
                    rinfo = disk_rows.get(mount)
                    if not rinfo:
                        continue
                    is_arch_drive = self._is_path_on_mount(cur_arch, mount)
                    raw_media = str(d.get("media") or "Disk")
                    badge_txt = f"[{_disp_media(raw_media)}]"
                    badge_col = "#00e5ff" if "SSD" in raw_media.upper() else "#ffcc00"
                    fs_txt = str(d.get("fs") or "").strip()
                    pct_val = min(100.0, max(0.0, float(d.get("percent") or 0.0)))
                    info_txt = tr(
                        "ui.settings.hw.disk_tile_info",
                        "Free: {free:.1f} {unit}\nTotal: {tot:.1f} {unit} ({pct:.0f}%)",
                        free=d["free_gb"],
                        tot=d["total_gb"],
                        pct=pct_val,
                        unit=unit_gb,
                    )

                    if rinfo["last_media"] != (badge_txt, badge_col):
                        rinfo["media_lbl"].configure(text=badge_txt, fg=badge_col)
                        rinfo["last_media"] = (badge_txt, badge_col)
                    if rinfo.get("last_fs") != fs_txt and rinfo.get("fs_lbl") is not None:
                        rinfo["fs_lbl"].configure(text=fs_txt)
                        rinfo["last_fs"] = fs_txt
                    if rinfo["last_info"] != info_txt:
                        rinfo["info_lbl"].configure(text=info_txt)
                        rinfo["last_info"] = info_txt
                    if rinfo.get("last_pct") != pct_val and rinfo.get("disk_pb") is not None:
                        rinfo["disk_pb"]["value"] = pct_val
                        rinfo["last_pct"] = pct_val
                    if rinfo["last_arch"] != is_arch_drive:
                        if not is_op_mode:
                            rinfo["btn"].configure(
                                text=(
                                    tr("ui.settings.hw.disk_btn_current", "★ Current archive")
                                    if is_arch_drive
                                    else tr("ui.settings.hw.disk_btn_choose", "📁 Choose folder")
                                ),
                                bg="#1e442c" if is_arch_drive else "#2e3142",
                                fg="#75e08b" if is_arch_drive else "#00e5ff",
                            )
                        if rinfo.get("tile") is not None:
                            rinfo["tile"].configure(
                                highlightbackground="#2d7a4d" if is_arch_drive else "#2e3244"
                            )
                        rinfo["last_arch"] = is_arch_drive
                for ge in phys_groups_ref:
                    has_arch = any(self._is_path_on_mount(cur_arch, gm) for gm in ge["mounts"])
                    p_col = "#2d7a4d" if has_arch else ("#3a435e" if ge["is_multi"] else "#2c3042")
                    try:
                        if str(ge["frame"].cget("highlightbackground")) != p_col:
                            ge["frame"].configure(highlightbackground=p_col)
                    except Exception:
                        pass
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
            def _set_probing():
                try:
                    if self._caps_summary_lbl is not None and self._caps_summary_lbl.winfo_exists():
                        self._caps_summary_lbl.configure(
                            text=tr(
                                "ui.settings.hw.caps_scanning",
                                "Full scan of GPU, AI modes (CUDA/DirectML/OpenVINO), DLLs, and FFmpeg...",
                            ),
                            fg="#ffcc00",
                        )
                except Exception:
                    pass

            if self.scroll_panel is not None:
                self.scroll_panel.preserve_scroll_during(_set_probing)
            else:
                _set_probing()

        def _bg():
            need_force = bool(force)
            caps = None
            if not need_force:
                caps = self.hw_monitor.probe_acceleration_capabilities(force=False)
                try:
                    if isinstance(caps, dict):
                        cur_ort = get_installed_ort_package()
                        cur_cuda_ok = bool(check_cuda_libraries_status(self.cfg_mgr.base_dir, allow_network=False).get("ready"))
                        cached_cuda_ok = bool((caps.get("cuda_libraries") or {}).get("ready"))
                        if caps.get("ort_package") != cur_ort or cached_cuda_ok != cur_cuda_ok:
                            need_force = True
                except Exception:
                    pass
            if need_force:
                if hasattr(self.hw_monitor, "refresh_all"):
                    caps = self.hw_monitor.refresh_all(force=True)
                else:
                    caps = self.hw_monitor.probe_acceleration_capabilities(force=True)
            self._last_caps = caps
            if isinstance(caps, dict) and caps.get("cuda_libraries"):
                self._last_cuda_status = caps.get("cuda_libraries")

            def _apply():
                if self._is_closing:
                    return
                if self._current_node_id == "hardware":
                    if need_force and self._telemetry_labels:
                        self._tick_telemetry()
                    if self._caps_container is not None:
                        self._render_caps_matrix(caps)
                    if force:
                        self._set_footer_status(
                            tr(
                                "ui.settings.hw.probe_updated",
                                "✔ Hardware, AI modes, CUDA, and FFmpeg check updated.",
                            ),
                            fg="#75e08b",
                        )

            self._ui_queue.put((_apply, ()))

        threading.Thread(target=_bg, daemon=True).start()

    def _render_hw_restart_bar(self, caps: dict):
        if self._hw_restart_bar is None or not self._hw_restart_bar.winfo_exists():
            return
        for ch in list(self._hw_restart_bar.winfo_children()):
            ch.destroy()

        restart_req = bool(caps.get("ort_restart_required"))
        if not restart_req:
            try:
                self._hw_restart_bar.pack_forget()
            except Exception:
                pass
            return

        ort_pkg = caps.get("ort_package") or "GPU"
        ort_loaded = caps.get("ort_loaded_package") or "onnxruntime"
        msg = tr(
            "ui.settings.hw.restart_bar_msg",
            "⚠ '{pkg}' is installed on disk, but the previous module '{loaded}' is still active in memory.\nRestart the application to enable the new GPU acceleration ({pkg}).",
            pkg=ort_pkg,
            loaded=ort_loaded,
        )
        if self.on_restart_app_cb is not None:
            tk.Button(
                self._hw_restart_bar, text=tr("ui.settings.hw.btn_restart_app", "♻ Restart Application"),
                command=self._request_app_restart,
                bg="#d97706", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                padx=10, pady=5, cursor="hand2"
            ).pack(side=tk.RIGHT, padx=(8, 0))
        lbl = tk.Label(
            self._hw_restart_bar, text=msg, fg="#ffe082", bg="#3b2814",
            font=("Segoe UI", 8, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=340
        )
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._bind_auto_wrap(lbl, initial=340, min_wrap=180)

        try:
            pack_kw = {"fill": tk.X, "pady": (0, 8)}
            if (
                self._hw_auto_box is not None
                and self._hw_auto_box.winfo_exists()
                and self._hw_auto_box.winfo_manager() == "pack"
            ):
                pack_kw["before"] = self._hw_auto_box
            elif self._caps_container is not None and self._caps_container.winfo_exists():
                pack_kw["before"] = self._caps_container
            self._hw_restart_bar.pack(**pack_kw)
        except Exception:
            pass

    def _render_hw_auto_box(self):
        if self._hw_auto_box is None or not self._hw_auto_box.winfo_exists():
            return

        def _rebuild_box():
            if self._hw_auto_box is None or not self._hw_auto_box.winfo_exists():
                return
            for ch in list(self._hw_auto_box.winfo_children()):
                ch.destroy()

            note = self._hw_auto_note
            if not isinstance(note, dict):
                try:
                    self._hw_auto_box.pack_forget()
                except Exception:
                    pass
                return

            pending_save = bool(note.get("pending_save"))
            restart_req = bool(note.get("restart_required"))
            if restart_req:
                box_bg, border_col, title_col = "#332814", "#ffcc00", "#ffe082"
            elif pending_save:
                box_bg, border_col, title_col = "#172a23", "#75e08b", "#75e08b"
            elif note.get("is_gpu_ready"):
                box_bg, border_col, title_col = "#16262f", "#00e5ff", "#9ee8ff"
            else:
                box_bg, border_col, title_col = "#222432", "#ffcc00", "#ffe082"

            panel = tk.Frame(
                self._hw_auto_box, bg=box_bg, padx=10, pady=8,
                highlightthickness=1, highlightbackground=border_col
            )
            panel.pack(fill=tk.X)

            hl = tk.Label(
                panel, text=note.get("headline", ""), fg=title_col, bg=box_bg,
                font=("Segoe UI", 9, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=420
            )
            hl.pack(fill=tk.X, pady=(0, 4))
            self._bind_auto_wrap(hl, initial=420, min_wrap=200)

            lines = note.get("lines") or []
            if lines:
                body_lbl = tk.Label(
                    panel, text="\n".join(lines), fg="#e2e6f3", bg=box_bg,
                    font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=420
                )
                body_lbl.pack(fill=tk.X)
                self._bind_auto_wrap(body_lbl, initial=420, min_wrap=200)

            extra = note.get("extra_note") or ""
            if extra:
                ex_lbl = tk.Label(
                    panel, text=extra, fg="#9ee8ff" if not restart_req else "#ffcc00", bg=box_bg,
                    font=("Segoe UI", 8, "bold" if restart_req else "normal"),
                    anchor=tk.W, justify=tk.LEFT, wraplength=420
                )
                ex_lbl.pack(fill=tk.X, pady=(4, 0))
                self._bind_auto_wrap(ex_lbl, initial=420, min_wrap=200)

            if pending_save or (restart_req and self.on_restart_app_cb is not None):
                act = tk.Frame(panel, bg=box_bg)
                act.pack(fill=tk.X, pady=(6, 0))
                if pending_save:
                    tk.Button(
                        act, text=tr("ui.settings.hw.btn_apply_auto", "💾 Apply selected settings"),
                        command=self._apply_changes,
                        bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                        padx=10, pady=4, cursor="hand2"
                    ).pack(side=tk.LEFT, padx=(0, 8))
                if restart_req and self.on_restart_app_cb is not None:
                    tk.Button(
                        act, text=tr("ui.settings.hw.btn_save_restart", "♻ Save & Restart Now"),
                        command=self._request_app_restart,
                        bg="#d97706", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                        padx=10, pady=4, cursor="hand2"
                    ).pack(side=tk.LEFT, padx=(0, 8))
                if pending_save:
                    hint_lbl = tk.Label(
                        panel, text=tr("ui.settings.hw.auto_hint", "Fields below are already filled — just click 'Apply'."),
                        fg="#b8c0d4", bg=box_bg, font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=420
                    )
                    hint_lbl.pack(fill=tk.X, pady=(4, 0))
                    self._bind_auto_wrap(hint_lbl, initial=420, min_wrap=200)

            pack_kw = {"fill": tk.X, "pady": (0, 8)}
            if self._caps_container is not None and self._caps_container.winfo_exists():
                pack_kw["before"] = self._caps_container
            self._hw_auto_box.pack(**pack_kw)

        if self.scroll_panel is not None:
            self.scroll_panel.preserve_scroll_during(_rebuild_box)
        else:
            _rebuild_box()

    def _render_caps_matrix(self, caps: dict):
        if self._caps_container is None or not isinstance(caps, dict):
            return

        def _rebuild():
            if self._caps_container is None or not self._caps_container.winfo_exists():
                return
            bg = str(self._caps_container.cget("bg"))
            for ch in list(self._caps_container.winfo_children()):
                ch.destroy()
            self._mode_tile_widgets = {}
            self._mode_tiles_frame = None
            self._mode_detail_frame = None
            self._mode_setup_pb = None
            self._mode_setup_lbl = None
            self._hw_details_frame = None

            eff_caps = dict(caps)
            if isinstance(self._last_cuda_status, dict) and not eff_caps.get("cuda_libraries"):
                eff_caps["cuda_libraries"] = self._last_cuda_status

            assessment = assess_ai_modes(self.cfg_mgr.base_dir, caps=eff_caps)
            modes = assessment.get("modes") or {}
            best_mode = str(assessment.get("best_mode") or "cpu")
            eff_auto = str(assessment.get("effective_auto_mode") or "cpu")

            hw_draft = (self.draft or {}).get("hardware") or {}
            active_backend = str(hw_draft.get("ai_backend", "cpu")).strip().lower()
            if active_backend not in modes:
                active_backend = "cpu"
            if self._inspected_ai_mode not in modes:
                self._inspected_ai_mode = active_backend
            inspected = self._inspected_ai_mode

            saved_backend = str((((self.cfg_mgr.get_snapshot() or {}).get("hardware")) or {}).get("ai_backend", "cpu")).strip().lower()

            if self._caps_summary_lbl is not None and self._caps_summary_lbl.winfo_exists():
                is_restart = bool(assessment.get("ort_restart_required"))
                best_m = modes.get(best_mode) or {}
                act_m = modes.get(active_backend) or {}
                sum_fg = (
                    "#ffcc00"
                    if is_restart
                    else ("#75e08b" if act_m.get("state") == "ready" else "#00e5ff")
                )
                best_tag = (
                    f"🏆 {best_m.get('title', best_mode.upper())}"
                    if best_mode != "cpu"
                    else tr("ui.settings.hw.best_tag_cpu", "🖥 {title} (no GPU)", title=best_m.get("title", "CPU"))
                )
                self._caps_summary_lbl.configure(
                    text=tr(
                        "ui.settings.hw.caps_summary",
                        "ORT Package: {ort}  |  Selected: {sel}  |  Best for this PC: {best}",
                        ort=assessment.get("ort_package"),
                        sel=act_m.get("short_title", active_backend.upper()),
                        best=best_tag,
                    ),
                    fg=sum_fg,
                )

            self._render_hw_restart_bar(eff_caps)

            # 1. Плитки всіх режимів роботи AI в 1 горизонтальний ряд (швидкість + підтримка + +/-)
            tiles_frame = tk.Frame(self._caps_container, bg=bg)
            tiles_frame.pack(fill=tk.X, pady=(2, 8))
            self._mode_tiles_frame = tiles_frame

            mode_order = ("auto", "cuda", "directml", "openvino", "cpu")
            for idx, m_key in enumerate(mode_order):
                tiles_frame.grid_columnconfigure(idx, weight=1, uniform="ai_mode_tile")
                m_info = modes.get(m_key) or {}
                hw_ok = bool(m_info.get("hw_supported"))
                is_best = bool(m_info.get("is_best"))
                is_inspected = (m_key == inspected)
                is_active = (m_key == active_backend)
                sup_lvl = str(m_info.get("support_level") or ("none" if not hw_ok else "partial"))
                s_rank = m_info.get("speed_rank")

                top_txt = str(m_info.get("speed_tile") or "")
                badge_txt = str(m_info.get("support_badge") or "")
                hint_txt = str(m_info.get("tile_hint") or "")

                if not hw_ok:
                    t_bg = "#251c24" if is_inspected else "#171820"
                    t_border = "#00e5ff" if is_inspected else "#462930"
                    accent_col = "#7a333c"
                    top_col = "#ff8a8a"
                    title_col = "#8b92a8"
                    badge_col = "#ff7b7b"
                    hint_col = "#7f869c"
                else:
                    if is_inspected:
                        t_bg = "#192a3b"
                        t_border = "#00e5ff"
                    elif is_active:
                        t_bg = "#1b2532"
                        t_border = "#0099b8"
                    elif sup_lvl == "full":
                        t_bg = "#202331"
                        t_border = "#2d7a4d" if (m_key != "cpu" or is_best) else "#2c4c3f"
                    else:
                        t_bg = "#202331"
                        t_border = "#8a6d1f"

                    if is_active:
                        accent_col = "#00e5ff"
                    elif is_best and m_key != "cpu":
                        accent_col = "#ffd54f"
                    elif sup_lvl == "full":
                        accent_col = "#75e08b" if (m_key != "cpu" or best_mode == "cpu") else "#4e8a6b"
                    else:
                        accent_col = "#d49a1a"

                    title_col = "#ffffff"
                    if is_best and m_key != "cpu":
                        top_col = "#ffd54f"
                    elif m_key == "auto":
                        top_col = "#ffcc00" if sup_lvl == "partial" else ("#ffe082" if sup_lvl == "restart" else "#9ee8ff")
                    elif s_rank == 2:
                        top_col = "#9ee8ff"
                    elif s_rank == 3:
                        top_col = "#a8c7fa"
                    else:
                        top_col = "#9fb3c8"

                    if sup_lvl == "full":
                        badge_col = "#75e08b" if (m_key != "cpu" or best_mode == "cpu") else "#6bc481"
                    else:
                        badge_col = "#ffcc00"

                    hint_col = "#aeb8cc" if hw_ok else "#7f869c"

                tile = tk.Frame(
                    tiles_frame,
                    bg=t_bg,
                    highlightthickness=2,
                    highlightbackground=t_border,
                    cursor="hand2",
                )
                tile.grid(
                    row=0,
                    column=idx,
                    sticky="nsew",
                    padx=(0 if idx == 0 else 3, 0 if idx == (len(mode_order) - 1) else 3),
                )
                tk.Frame(tile, bg=accent_col, height=3).pack(fill=tk.X)
                inner = tk.Frame(tile, bg=t_bg, padx=4, pady=5)
                inner.pack(fill=tk.BOTH, expand=True)

                top_lbl = tk.Label(
                    inner, text="", fg=top_col, bg=t_bg,
                    font=("Segoe UI", 7, "bold"), width=1, anchor=tk.CENTER
                )
                top_lbl.pack(fill=tk.X)
                bind_single_line_ellipsis(top_lbl, full_text=top_txt, pad=4, min_px=40, initial_px=125)

                name_text = str(m_info.get("short_title", m_key.upper()))
                if is_active and hw_ok:
                    name_text = f"● {name_text}"
                name_lbl = tk.Label(
                    inner, text="", fg=title_col, bg=t_bg,
                    font=("Segoe UI", 9, "bold"), width=1, anchor=tk.CENTER
                )
                name_lbl.pack(fill=tk.X, pady=(3, 2))
                bind_single_line_ellipsis(name_lbl, full_text=name_text, pad=4, min_px=40, initial_px=125)

                st_lbl = tk.Label(
                    inner, text="", fg=badge_col, bg=t_bg,
                    font=("Segoe UI", 8, "bold"), width=1, anchor=tk.CENTER
                )
                st_lbl.pack(fill=tk.X)
                bind_single_line_ellipsis(st_lbl, full_text=badge_txt, pad=4, min_px=40, initial_px=125)

                hint_lbl = tk.Label(
                    inner, text="", fg=hint_col, bg=t_bg,
                    font=("Segoe UI", 7), width=1, anchor=tk.CENTER
                )
                hint_lbl.pack(fill=tk.X, pady=(3, 0))
                bind_single_line_ellipsis(hint_lbl, full_text=hint_txt, pad=4, min_px=40, initial_px=125)

                tile_full_tip = "\n".join(x for x in (name_text, top_txt, badge_txt, hint_txt) if x)
                for tw in (tile, inner, top_lbl, name_lbl, st_lbl, hint_lbl):
                    bind_hover_tooltip(
                        tw,
                        tile_full_tip,
                        only_when_truncated=True,
                        truncate_sources=(top_lbl, name_lbl, st_lbl, hint_lbl),
                    )

                self._bind_tile_click(
                    tile,
                    lambda mk=m_key: self._select_ai_mode(mk, auto_install=(mk == "cuda"), prompt_confirm=True),
                )
                self._mode_tile_widgets[m_key] = tile

            # 2. Фокусована панель деталей ТІЛЬКИ для вибраного режиму (швидкість, підтримка, плюси/мінуси, кроки)
            sel_info = modes.get(inspected) or modes.get("cpu") or {}
            sel_state = str(sel_info.get("state") or "unavailable")
            sel_hw_ok = bool(sel_info.get("hw_supported"))
            sel_plan = sel_info.get("install_plan") or {}
            det_bg = "#19202b"
            det_border = (
                "#2d7a4d"
                if sel_state == "ready"
                else ("#a67c00" if sel_state in ("needs_setup", "restart_required") else "#7a333c")
            )
            det = tk.Frame(
                self._caps_container,
                bg=det_bg,
                padx=12,
                pady=10,
                highlightthickness=1,
                highlightbackground=det_border,
            )
            det.pack(fill=tk.X, pady=(0, 8))
            self._mode_detail_frame = det

            hdr_det = tk.Frame(det, bg=det_bg)
            hdr_det.pack(fill=tk.X, pady=(0, 4))
            tag_suffix = ""
            if sel_info.get("speed_tile"):
                tag_suffix += f"  •  {sel_info['speed_tile']}"
            if inspected == active_backend and sel_hw_ok:
                tag_suffix += tr("ui.settings.hw.tag_active_draft", "  •  ● Active in draft")
            elif not sel_hw_ok:
                tag_suffix += tr("ui.settings.hw.tag_preview_unavail", "  •  Preview (not supported)")
            hdr_lbl = tk.Label(
                hdr_det,
                text=f"{sel_info.get('short_title', inspected.upper())} — {sel_info.get('title', '')}{tag_suffix}",
                fg="#ffffff", bg=det_bg, font=("Segoe UI", 9, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=440
            )
            hdr_lbl.pack(fill=tk.X)
            self._bind_auto_wrap(hdr_lbl, initial=440, min_wrap=200)

            meta_box = tk.Frame(det, bg="#1d2433", padx=8, pady=5)
            meta_box.pack(fill=tk.X, pady=(0, 6))
            spd_txt = str(sel_info.get("speed_detail") or "")
            sup_txt = str(sel_info.get("support_label") or sel_info.get("headline") or "")
            spd_col = "#ffd54f" if sel_info.get("is_best") and sel_info.get("id") != "cpu" else ("#9ee8ff" if sel_hw_ok else "#ff9898")
            sup_col = "#75e08b" if sel_state == "ready" else ("#ffcc00" if sel_hw_ok else "#ff8a8a")

            spd_lbl = tk.Label(
                meta_box, text=tr("ui.settings.hw.det_speed", "⚡ Speed: {text}", text=spd_txt),
                fg=spd_col, bg="#1d2433", font=("Segoe UI", 8, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=420
            )
            spd_lbl.pack(fill=tk.X)
            self._bind_auto_wrap(spd_lbl, initial=420, min_wrap=180, pad=16)

            sup_lbl = tk.Label(
                meta_box, text=tr("ui.settings.hw.det_support", "🛡 Support: {text}", text=sup_txt),
                fg=sup_col, bg="#1d2433", font=("Segoe UI", 8, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=420
            )
            sup_lbl.pack(fill=tk.X, pady=(2, 0))
            self._bind_auto_wrap(sup_lbl, initial=420, min_wrap=180, pad=16)

            pros_list = [str(p).strip() for p in (sel_info.get("pros") or []) if str(p).strip()]
            cons_list = [str(c).strip() for c in (sel_info.get("cons") or []) if str(c).strip()]
            if pros_list:
                pros_box = tk.Frame(det, bg="#162620", padx=8, pady=5, highlightthickness=1, highlightbackground="#29523f")
                pros_box.pack(fill=tk.X, pady=(0, 4))
                pros_hdr = tk.Label(
                    pros_box, text=tr("ui.settings.hw.det_pros", "➕ Pros:"), fg="#75e08b", bg="#162620",
                    font=("Segoe UI", 8, "bold"), anchor=tk.W
                )
                pros_hdr.pack(fill=tk.X)
                pros_body = tk.Label(
                    pros_box,
                    text="\n".join(f"  + {p}" for p in pros_list),
                    fg="#c8f2d1", bg="#162620", font=("Segoe UI", 8),
                    anchor=tk.W, justify=tk.LEFT, wraplength=420,
                )
                pros_body.pack(fill=tk.X)
                self._bind_auto_wrap(pros_body, initial=420, min_wrap=180, pad=16)

            if cons_list:
                cons_bg = "#281f1b" if sel_hw_ok else "#2a1a20"
                cons_bd = "#5c3f2e" if sel_hw_ok else "#6b2d38"
                cons_hdr_col = "#ffcc80" if sel_hw_ok else "#ff8a8a"
                cons_body_col = "#f3d5b5" if sel_hw_ok else "#ffb8b8"
                cons_box = tk.Frame(det, bg=cons_bg, padx=8, pady=5, highlightthickness=1, highlightbackground=cons_bd)
                cons_box.pack(fill=tk.X, pady=(0, 6))
                cons_hdr = tk.Label(
                    cons_box, text=tr("ui.settings.hw.det_cons", "➖ Cons / limitations:"), fg=cons_hdr_col, bg=cons_bg,
                    font=("Segoe UI", 8, "bold"), anchor=tk.W
                )
                cons_hdr.pack(fill=tk.X)
                cons_body = tk.Label(
                    cons_box,
                    text="\n".join(f"  − {c}" for c in cons_list),
                    fg=cons_body_col, bg=cons_bg, font=("Segoe UI", 8),
                    anchor=tk.W, justify=tk.LEFT, wraplength=420,
                )
                cons_body.pack(fill=tk.X)
                self._bind_auto_wrap(cons_body, initial=420, min_wrap=180, pad=16)

            hl_col = "#75e08b" if sel_state == "ready" else ("#ffcc00" if sel_hw_ok else "#ff9898")
            hl_lbl = tk.Label(
                det,
                text=sel_info.get("headline", ""),
                fg=hl_col, bg=det_bg, font=("Segoe UI", 8, "bold"), anchor=tk.W, justify=tk.LEFT, wraplength=440
            )
            hl_lbl.pack(fill=tk.X, pady=(0, 5))
            self._bind_auto_wrap(hl_lbl, initial=440, min_wrap=200)

            for step in (sel_info.get("steps") or []):
                step_done = bool(step.get("done"))
                step_kind = str(step.get("kind") or "")
                if step_done:
                    s_mark, s_col = "✔", "#75e08b"
                elif step_kind in ("ort", "cuda_dll", "restart", "info"):
                    s_mark, s_col = "⚙" if step_kind in ("ort", "cuda_dll") else "⚠", "#ffcc00"
                else:
                    s_mark, s_col = "✖", "#ff6666"

                s_row = tk.Frame(det, bg="#212836", padx=8, pady=4)
                s_row.pack(fill=tk.X, pady=2)
                step_text = f"{s_mark} {step.get('label', '')}"
                if step.get("detail"):
                    step_text += f"\n   {step['detail']}"
                s_lbl = tk.Label(
                    s_row,
                    text=step_text,
                    fg=s_col if not step_done else "#d8e0ef",
                    bg="#212836",
                    font=("Segoe UI", 8, "bold" if not step_done else "normal"),
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=420,
                )
                s_lbl.pack(fill=tk.X)
                self._bind_auto_wrap(s_lbl, initial=420, min_wrap=180, pad=16)

            busy = bool(self._ort_installing or self._cuda_downloading or self._autosetup_running)
            btn_state = tk.DISABLED if busy else tk.NORMAL
            act_buttons = []
            act_row = tk.Frame(det, bg=det_bg)

            if sel_plan.get("can_install"):
                act_buttons.append(
                    tk.Button(
                        act_row,
                        text=sel_plan.get("action_label") or tr("ui.settings.hw.btn_install", "⬇ Install"),
                        state=btn_state,
                        command=lambda mk=inspected: self._start_mode_setup_async(mk),
                        bg="#1f4d3a", fg="#75e08b", relief=tk.FLAT,
                        font=("Segoe UI", 8, "bold"), padx=10, pady=5, cursor="hand2"
                    )
                )
            if sel_plan.get("needs_restart") and self.on_restart_app_cb is not None:
                act_buttons.append(
                    tk.Button(
                        act_row,
                        text=tr("ui.settings.hw.btn_restart_short", "♻ Restart"),
                        command=self._request_app_restart,
                        bg="#d97706", fg="#ffffff", relief=tk.FLAT,
                        font=("Segoe UI", 8, "bold"), padx=10, pady=5, cursor="hand2"
                    )
                )
            if inspected == "cuda" and sel_hw_ok:
                act_buttons.append(
                    tk.Button(
                        act_row,
                        text="📂 cuda/",
                        command=self._open_cuda_folder,
                        bg="#2f3346", fg="#00e5ff", relief=tk.FLAT,
                        font=("Segoe UI", 8, "bold"), padx=9, pady=5, cursor="hand2"
                    )
                )
            if sel_hw_ok and (active_backend != saved_backend):
                act_buttons.append(
                    tk.Button(
                        act_row,
                        text=tr("ui.settings.hw.btn_apply_short", "💾 Apply"),
                        command=self._apply_changes,
                        bg="#007acc", fg="#ffffff", relief=tk.FLAT,
                        font=("Segoe UI", 8, "bold"), padx=10, pady=5, cursor="hand2"
                    )
                )

            if act_buttons:
                act_row.pack(fill=tk.X, pady=(6, 0))
                self._layout_button_row(act_row, act_buttons)
            else:
                act_row.destroy()

            if assessment.get("frozen") and sel_plan.get("need_ort_pkg"):
                frz_lbl = tk.Label(
                    det,
                    text=tr(
                        "ui.settings.hw.frozen_ort_note",
                        "ℹ In compiled EXE builds, switching ONNX Runtime is done via 1_SETUP.bat.",
                    ),
                    fg="#ffcc00", bg=det_bg, font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=420
                )
                frz_lbl.pack(fill=tk.X, pady=(5, 0))
                self._bind_auto_wrap(frz_lbl, initial=420, min_wrap=180)

            st_box_info = self._mode_setup_status if isinstance(self._mode_setup_status, dict) and self._mode_setup_status.get("mode") == inspected else None
            if st_box_info is not None:
                prog_box = tk.Frame(det, bg=det_bg)
                prog_box.pack(fill=tk.X, pady=(6, 0))
                pb = ttk.Progressbar(prog_box, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
                pb.pack(fill=tk.X, pady=(0, 3))
                pb["value"] = max(0, min(100, int(st_box_info.get("pct") or 0)))
                self._mode_setup_pb = pb
                s_res_lbl = tk.Label(
                    prog_box,
                    text=str(st_box_info.get("text") or ""),
                    fg=str(st_box_info.get("color") or "#9ba3b8"),
                    bg=det_bg,
                    font=("Segoe UI", 8),
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=420,
                )
                s_res_lbl.pack(fill=tk.X)
                self._bind_auto_wrap(s_res_lbl, initial=420, min_wrap=180)
                self._mode_setup_lbl = s_res_lbl

            # 3. Компактний підсумок відеоприскорення FFmpeg/OpenCL + згорнута за замовчуванням діагностика
            rec = eff_caps.get("recommendations") or {}
            vid_encs = eff_caps.get("video_encoders") or {}
            hw_enc_keys = [
                ek for ek, einfo in vid_encs.items()
                if ek not in ("copy", "libx264") and bool((einfo or {}).get("supported"))
            ]
            has_hw_vid_enc = bool(hw_enc_keys)
            hw_gpus = ((eff_caps.get("hardware_flags") or {}).get("gpus")) or []
            hw_dec = ", ".join(eff_caps.get("hw_decoders") or []) or "cpu"
            yes_txt = tr("ui.common.yes", "Yes")
            no_txt = tr("ui.common.no", "No")
            ocl = eff_caps.get("opencv_opencl") or {}
            ocl_s = f"{yes_txt} ({ocl.get('device', '')})" if ocl.get("supported") else no_txt

            if has_hw_vid_enc:
                vid_acc_txt = tr("ui.settings.hw.vid_acc_gpu", "✔ GPU ({encs})", encs=", ".join(hw_enc_keys))
                bar_fg = "#b8c0d4"
            else:
                vid_acc_txt = tr(
                    "ui.settings.hw.vid_acc_cpu",
                    "✖ GPU codec unsupported (CPU {enc} / RTSP copy)",
                    enc=rec.get("video_encoder", "libx264"),
                )
                bar_fg = "#ffcc80"

            summary_bar = tk.Frame(self._caps_container, bg="#222432", padx=8, pady=5)
            summary_bar.pack(fill=tk.X)

            tog_txt = (
                tr("ui.settings.hw.btn_hide_details", "▾ Hide")
                if self._hw_details_expanded
                else tr("ui.settings.hw.btn_more_details", "▸ Details")
            )
            tk.Button(
                summary_bar,
                text=tog_txt,
                command=self._toggle_hw_details,
                bg="#2f3346", fg="#9ee8ff", relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"), padx=8, pady=2, cursor="hand2"
            ).pack(side=tk.RIGHT, padx=(8, 0))

            foot_summary = tk.Label(
                summary_bar,
                text=tr(
                    "ui.settings.hw.foot_summary",
                    "Video acceleration: {vid}  |  Decode: {dec}  |  OpenCV OpenCL: {ocl}",
                    vid=vid_acc_txt,
                    dec=hw_dec,
                    ocl=ocl_s,
                ),
                fg=bar_fg, bg="#222432", font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=320
            )
            foot_summary.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._bind_auto_wrap(foot_summary, initial=320, min_wrap=140, pad=12)

            if self._hw_details_expanded:
                det_box = tk.Frame(self._caps_container, bg="#1d202c", padx=8, pady=6, highlightthickness=1, highlightbackground="#2b2e3d")
                det_box.pack(fill=tk.X, pady=(4, 0))
                self._hw_details_frame = det_box

                if not has_hw_vid_enc:
                    no_vid_msg = (
                        tr(
                            "ui.settings.hw.no_vid_nogpu",
                            "ℹ No GPU detected: hardware GPU video encoding (NVENC / QSV / AMF) is unavailable — recording uses CPU (libx264) or direct stream copy (RTSP copy). See codec reasons below:",
                        )
                        if not hw_gpus
                        else tr(
                            "ui.settings.hw.no_vid_fail",
                            "ℹ No hardware GPU video codec (NVENC / QSV / AMF) passed testing — recording uses CPU (libx264) or direct RTSP (copy). See reasons below:",
                        )
                    )
                    no_vid_box = tk.Frame(det_box, bg="#2c231c", padx=8, pady=5, highlightthickness=1, highlightbackground="#a67c00")
                    no_vid_box.pack(fill=tk.X, pady=(0, 6))
                    no_vid_lbl = tk.Label(
                        no_vid_box, text=no_vid_msg, fg="#ffd08a", bg="#2c231c",
                        font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=420
                    )
                    no_vid_lbl.pack(fill=tk.X)
                    self._bind_auto_wrap(no_vid_lbl, initial=420, min_wrap=180, pad=16)

                tk.Label(
                    det_box,
                    text=tr("ui.settings.hw.ffmpeg_enc_hdr", "FFmpeg Video Encoding (for archive recording):"),
                    fg="#c8d0e4", bg="#1d202c", font=("Segoe UI", 8, "bold"), anchor=tk.W
                ).pack(fill=tk.X, pady=(0, 3))
                for key, info in (eff_caps.get("video_encoders") or {}).items():
                    r = tk.Frame(det_box, bg="#222432", padx=6, pady=2)
                    r.pack(fill=tk.X, pady=1)
                    sup = bool(info.get("supported"))
                    mk = "✔" if sup else "✖"
                    col = "#75e08b" if sup else "#888c9e"
                    tk.Label(r, text=f"{mk} {info.get('label', key)}", fg="#ffffff", bg="#222432", font=("Segoe UI", 8, "bold"), width=30, anchor=tk.W).pack(side=tk.LEFT)
                    tk.Label(r, text=str(info.get("status", "")), fg=col, bg="#222432", font=("Segoe UI", 8, "bold"), width=18, anchor=tk.W).pack(side=tk.LEFT)
                    rsn = tk.Label(r, text=str(info.get("reason", "")), fg="#9ba3b8", bg="#222432", font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=140)
                    rsn.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    self._bind_auto_wrap(rsn, initial=140, min_wrap=80)

                cuda_cv = eff_caps.get("opencv_cuda") or {}
                cv_cuda_s = f"{yes_txt} ({cuda_cv.get('count', 0)})" if cuda_cv.get("supported") else no_txt
                info_l = tk.Label(
                    det_box,
                    text=tr(
                        "ui.settings.hw.opencv_info",
                        "OpenCV CUDA: {cv_cuda}  |  UI renderer: {renderer}",
                        cv_cuda=cv_cuda_s,
                        renderer=eff_caps.get("ui_renderer", ""),
                    ),
                    fg="#9ba3b8", bg="#1d202c", font=("Segoe UI", 8), anchor=tk.W, justify=tk.LEFT, wraplength=420
                )
                info_l.pack(fill=tk.X, pady=(4, 6))
                self._bind_auto_wrap(info_l, initial=420, min_wrap=180)

                cst = eff_caps.get("cuda_libraries") or self._last_cuda_status
                if isinstance(cst, dict):
                    tk.Label(
                        det_box,
                        text=tr(
                            "ui.settings.hw.cuda_libs_hdr",
                            "CUDA {maj}.x / cuDNN 9 status (folder {dir}):",
                            maj=cst.get("cuda_major", "13"),
                            dir=get_cuda_dir(self.cfg_mgr.base_dir),
                        ),
                        fg="#c8d0e4", bg="#1d202c", font=("Segoe UI", 8, "bold"), anchor=tk.W, wraplength=420, justify=tk.LEFT
                    ).pack(fill=tk.X, pady=(0, 3))
                    for comp in (cst.get("components") or []):
                        cr = tk.Frame(det_box, bg="#222432", padx=6, pady=2)
                        cr.pack(fill=tk.X, pady=1)
                        src = comp.get("source", "missing")
                        s_col = "#75e08b" if src in ("local_cuda", "mixed") else ("#9ee8ff" if src in ("site_packages", "system") else ("#ffcc00" if src == "partial" else ("#ff6666" if comp.get("required") else "#888c9e")))
                        if src == "local_cuda":
                            s_txt = "✔ cuda/"
                        elif src == "mixed":
                            s_txt = tr("ui.settings.hw.cuda_src_mixed", "✔ cuda/+system")
                        elif src == "site_packages":
                            s_txt = "✔ site-packages"
                        elif src == "system":
                            s_txt = tr("ui.settings.hw.cuda_src_system", "✔ system")
                        elif src == "missing" and not comp.get("required"):
                            s_txt = tr("ui.settings.hw.cuda_src_optional", "— optional")
                        elif src == "partial":
                            s_txt = tr("ui.settings.hw.cuda_src_partial", "⚠ partial")
                        elif src == "missing":
                            s_txt = tr("ui.settings.hw.cuda_src_missing", "✖ missing")
                        else:
                            s_txt = f"✔ {src}"
                        tk.Label(cr, text=str(comp.get("label", comp.get("key", ""))), fg="#ffffff", bg="#222432", font=("Segoe UI", 8, "bold"), width=28, anchor=tk.W).pack(side=tk.LEFT)
                        tk.Label(cr, text=s_txt, fg=s_col, bg="#222432", font=("Segoe UI", 8, "bold"), width=18, anchor=tk.W).pack(side=tk.LEFT)
                        dl_txt = (
                            tr("ui.settings.hw.cuda_dll_count", "DLL: {n} file(s)", n=len(comp.get("found_paths") or []))
                            if src not in ("missing", "partial")
                            else f"{comp.get('pypi')}"
                        )
                        tk.Label(cr, text=dl_txt, fg="#9ba3b8", bg="#222432", font=("Segoe UI", 8), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

                    if (modes.get("cuda") or {}).get("hw_supported") and not cst.get("all_required_in_local"):
                        port_row = tk.Frame(det_box, bg="#1d202c")
                        port_row.pack(fill=tk.X, pady=(6, 0))
                        tk.Button(
                            port_row,
                            text=tr("ui.settings.hw.btn_collect_cuda", "📦 Collect all CUDA DLLs into cuda/ folder"),
                            state=btn_state,
                            command=lambda: self._start_mode_setup_async("cuda", force_local_cuda=True),
                            bg="#233b56", fg="#00e5ff", relief=tk.FLAT,
                            font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
                        ).pack(side=tk.LEFT)

        if self.scroll_panel is not None:
            self.scroll_panel.preserve_scroll_during(_rebuild)
        else:
            _rebuild()

    @staticmethod
    def _format_autosetup_plan_text(plan: dict, current_cfg: dict) -> str:
        lm = plan.get("live_metrics") or {}
        cpu = lm.get("cpu") or {}
        ram = lm.get("ram") or {}
        gpus = lm.get("gpus") or []

        unit_gb = tr("ui.unit.gb", "GB")
        unit_mb = tr("ui.unit.mb", "MB")

        lines = []
        lines.append(tr("ui.settings.autosetup.plan.sec1", "=== 1. HARDWARE PLATFORM ==="))
        cores_s = tr(
            "ui.settings.hw.cpu_cores_both",
            "{phys}c / {logi}t",
            phys=cpu.get("physical_cores", "?"),
            logi=cpu.get("logical_threads", "?"),
        )
        lines.append(f"  • CPU: {cpu.get('name', '—')} ({cores_s})")
        if ram.get("total_gb"):
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.ram_line",
                    "  • RAM: {tot:.1f} {unit} (free {free:.1f} {unit})",
                    tot=ram.get("total_gb", 0),
                    free=ram.get("free_gb", 0),
                    unit=unit_gb,
                )
            )
        if gpus:
            for g in gpus:
                vram_s = (
                    f", VRAM {g['vram_total_mb']:.0f} {unit_mb}"
                    if g.get("vram_total_mb")
                    else ""
                )
                lines.append(f"  • GPU: [{g.get('vendor', '?')}] {g.get('name', '?')}{vram_s}")
        else:
            lines.append(tr("ui.settings.autosetup.plan.gpu_none", "  • GPU: discrete GPU not detected (CPU mode)"))

        lines.append("")
        lines.append(tr("ui.settings.autosetup.plan.sec2", "=== 2. PYTHON LIBRARIES & ONNX RUNTIME ==="))
        missing_pkgs = plan.get("missing_packages") or []
        if missing_pkgs:
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.pip_install",
                    "  ⬇ Will install via pip: {pkgs}",
                    pkgs=", ".join(missing_pkgs),
                )
            )
        else:
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.pip_ok",
                    "  ✔ All base libraries and module dependencies are already installed.",
                )
            )

        cur_ort = plan.get("cur_ort") or tr("hw.ort_not_installed", "not installed")
        target_ort = plan.get("target_ort") or "onnxruntime"
        ort_reason = plan.get("ort_reason") or ""
        if plan.get("need_ort_switch"):
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.ort_switch",
                    "  ⬇ Switch ONNX Runtime: '{cur}' → '{target}'",
                    cur=cur_ort,
                    target=target_ort,
                )
            )
            if ort_reason:
                lines.append(
                    tr("ui.settings.autosetup.plan.ort_reason", "    Reason: {reason}", reason=ort_reason)
                )
        else:
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.ort_ok",
                    "  ✔ Optimal ONNX Runtime ('{target}') is already installed.",
                    target=target_ort,
                )
            )

        cuda_st = plan.get("cuda_status")
        lines.append("")
        lines.append(tr("ui.settings.autosetup.plan.sec3", "=== 3. NVIDIA CUDA / cuDNN LIBRARIES (./cuda/) ==="))
        if cuda_st is None:
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.cuda_not_needed",
                    "  ℹ Additional CUDA DLLs are not needed for the selected profile.",
                )
            )
        elif plan.get("need_cuda_dl"):
            miss_req = cuda_st.get("missing_required") or []
            miss_opt = cuda_st.get("missing_optional") or []
            miss_all = miss_req + [x for x in miss_opt if x not in miss_req]
            items_s = ", ".join(miss_all) if miss_all else tr("ui.settings.autosetup.plan.missing_components", "missing components")
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.cuda_dl",
                    "  ⬇ Will verify and download to ./cuda/ (CUDA {maj}.x): {items}",
                    maj=cuda_st.get("cuda_major", "13"),
                    items=items_s,
                )
            )
        else:
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.cuda_ok",
                    "  ✔ All CUDA {maj}.x + cuDNN 9 libraries are ready.",
                    maj=cuda_st.get("cuda_major", "13"),
                )
            )

        lines.append("")
        lines.append(tr("ui.settings.autosetup.plan.sec4", "=== 4. INDEPENDENT MODULE MODELS & RESOURCES ==="))
        mod_checks = plan.get("module_checks") or []
        if not mod_checks:
            lines.append(tr("ui.settings.autosetup.plan.no_modules", "  — No modules found."))
        else:
            for mc in mod_checks:
                ico = mc.get("icon") or "🧩"
                mname = mc.get("name") or mc.get("id") or "module"
                miss_m = mc.get("missing") or []
                if miss_m:
                    lines.append(
                        tr(
                            "ui.settings.autosetup.plan.mod_dl",
                            "  ⬇ {icon} {name}: will be downloaded ({items}) — {msg}",
                            icon=ico,
                            name=mname,
                            items=", ".join(miss_m),
                            msg=mc.get("message", ""),
                        )
                    )
                elif not mc.get("ok", True):
                    lines.append(
                        f"  ⚠ {ico} {mname}: {mc.get('message') or tr('mod.common.check_needed', 'needs check')}"
                    )
                else:
                    lines.append(
                        f"  ✔ {ico} {mname}: {mc.get('message') or tr('mod.common.ready', 'Ready')}"
                    )

        lines.append("")
        lines.append(tr("ui.settings.autosetup.plan.sec5", "=== 5. OPTIMAL SETTINGS (config['hardware']) ==="))
        cur_hw = (current_cfg or {}).get("hardware") or {}
        pred_hw = plan.get("predicted_hw") or {}
        hw_labels = {
            "ai_backend": tr("hw.setup.hw_label.ai_backend", "AI Backend"),
            "ai_threads": tr("hw.setup.hw_label.ai_threads", "ONNX/AI Threads"),
            "ai_max_fps_per_cam": tr("hw.setup.hw_label.ai_fps", "Max AI FPS/camera"),
            "video_decode_mode": tr("hw.setup.hw_label.decode", "Stream Decoding"),
            "display_grid_fps": tr("hw.setup.hw_label.grid_fps", "Camera Grid FPS"),
            "display_detail_fps": tr("hw.setup.hw_label.detail_fps", "Detail View FPS"),
            "display_resize_method": tr("hw.setup.hw_label.resize", "UI Resize Method"),
            "use_opencv_opencl": tr("hw.setup.hw_label.opencl", "OpenCV OpenCL (GPU UMat)"),
        }
        upd_tag = tr("ui.settings.autosetup.plan.will_update", "will be updated")
        opt_tag = tr("ui.settings.autosetup.plan.already_opt", "already optimal")
        for k, title in hw_labels.items():
            ov = cur_hw.get(k)
            nv = pred_hw.get(k)
            if ov != nv:
                lines.append(f"  • {title}: {ov} → {nv}  ({upd_tag})")
            else:
                lines.append(f"  • {title}: {nv}  ({opt_tag})")

        cam_recs = plan.get("camera_recommendations") or []
        lines.append("")
        lines.append(tr("ui.settings.autosetup.plan.sec6", "=== 6. RECORDING CODECS & CAMERA / TEMPLATE SETTINGS ==="))
        if cam_recs:
            for cr in cam_recs:
                lines.append(f"  • {cr}")
        else:
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.cams_ok",
                    "  ✔ Camera and template module settings are already optimal.",
                )
            )

        if plan.get("frozen"):
            lines.append("")
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.frozen_note",
                    "ℹ Running as compiled EXE — pip package installation (if needed) should be done via 1_SETUP.bat.",
                )
            )

        if str((current_cfg or {}).get("general", {}).get("operation_mode", "server")).strip().lower() == "operator":
            lines.append("")
            lines.append(
                tr(
                    "ui.settings.autosetup.plan.operator_note",
                    "ℹ OPERATOR MODE: AI and decoding parameters will be optimized, while operation mode remains 'Operator Mode' (no disk writes).",
                )
            )

        lines.append("")
        lines.append(
            tr(
                "ui.settings.autosetup.plan.footer",
                "Click '🚀 Confirm & Execute' to automatically download missing components, save config.json, and apply settings.",
            )
        )
        return "\n".join(lines)

    def _open_full_autosetup_dialog(self, autosetup_runner=None, plan_builder=None):
        if self._is_closing:
            return None
        if self._autosetup_dlg is not None:
            try:
                if self._autosetup_dlg.winfo_exists():
                    self._autosetup_dlg.deiconify()
                    self._autosetup_dlg.lift()
                    self._autosetup_dlg.focus_force()
                    return self._autosetup_dlg
            except Exception:
                pass
            self._autosetup_dlg = None

        if self._ort_installing or self._cuda_downloading:
            messagebox.showinfo(
                tr("ui.settings.autosetup.msg_title", "Full Auto-Setup"),
                tr(
                    "ui.settings.autosetup.busy_msg",
                    "Please wait for the current ONNX Runtime installation or CUDA DLL download to finish.",
                ),
                parent=self.win,
            )
            return None

        self._commit_current_page()
        draft_snapshot = copy.deepcopy(self.draft)

        dlg = tk.Toplevel(self.win)
        dlg.title(tr("ui.settings.autosetup.dlg.title", "Full System Auto-Setup (1-Click)"))
        dlg.configure(bg="#181a24")
        dlg.minsize(620, 460)
        dlg.transient(self.win)
        try:
            dlg.grab_set()
        except Exception:
            pass
        self._autosetup_dlg = dlg

        state = {
            "dlg": dlg,
            "phase": "scanning",
            "cancelled": False,
            "plan_ready": False,
            "done": False,
            "plan": None,
            "result": None,
            "error": None,
            "confirm_btn": None,
            "cancel_btn": None,
            "done_btn": None,
            "restart_btn": None,
        }
        self._autosetup_dlg_state = state

        def _close_dlg():
            if self._autosetup_running:
                return
            state["cancelled"] = True
            if self._autosetup_dlg is dlg:
                self._autosetup_dlg = None
            try:
                dlg.grab_release()
            except Exception:
                pass
            try:
                dlg.destroy()
            except Exception:
                pass

        dlg.protocol("WM_DELETE_WINDOW", _close_dlg)

        hdr = tk.Frame(dlg, bg="#1e2233", padx=16, pady=12, highlightthickness=1, highlightbackground="#2b2e3d")
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr,
            text=tr("ui.settings.autosetup.dlg.hdr_title", "🚀 1-Click Full Auto-Setup & Optimization Wizard"),
            fg="#00e5ff",
            bg="#1e2233",
            font=("Segoe UI", 12, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Label(
            hdr,
            text=tr(
                "ui.settings.autosetup.dlg.hdr_sub",
                "Checks hardware, auto-downloads libraries/CUDA/models, and applies optimal parameters immediately.",
            ),
            fg="#9ba3b8",
            bg="#1e2233",
            font=("Segoe UI", 9),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(2, 0))

        status_lbl = tk.Label(
            dlg,
            text=tr(
                "ui.settings.autosetup.dlg.scanning",
                "⏳ Scanning PC, Python packages, CUDA, and module resources...",
            ),
            fg="#ffcc00",
            bg="#181a24",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            padx=16,
            pady=8,
            wraplength=580,
            justify=tk.LEFT,
        )
        status_lbl.pack(side=tk.TOP, fill=tk.X)
        self._bind_auto_wrap(status_lbl, initial=580, min_wrap=300, pad=36)

        # Нижня панель кнопок (пакується до body з expand=True, щоб ніколи не витіснялась)
        btn_bar = tk.Frame(dlg, bg="#1b1d27", padx=16, pady=10, highlightthickness=1, highlightbackground="#2b2e3d")
        btn_bar.pack(side=tk.BOTTOM, fill=tk.X)

        cancel_btn = tk.Button(
            btn_bar,
            text=tr("ui.common.cancel", "Cancel"),
            command=_close_dlg,
            bg="#353849",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
        )
        cancel_btn.pack(side=tk.RIGHT, padx=(6, 0))
        state["cancel_btn"] = cancel_btn

        body = tk.Frame(dlg, bg="#181a24", padx=16, pady=4)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Фаза 1: текст плану підтвердження
        plan_wrap = tk.Frame(body, bg="#181a24")
        plan_wrap.pack(fill=tk.BOTH, expand=True)
        plan_sb = tk.Scrollbar(plan_wrap, orient=tk.VERTICAL)
        plan_sb.pack(side=tk.RIGHT, fill=tk.Y)
        plan_text = tk.Text(
            plan_wrap,
            bg="#12141c",
            fg="#e6eaf5",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            yscrollcommand=plan_sb.set,
        )
        plan_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        plan_sb.configure(command=plan_text.yview)
        plan_text.insert(
            tk.END,
            tr("ui.settings.autosetup.dlg.plan_loading", "System diagnostics in progress...\nPlease wait a few seconds."),
        )
        plan_text.configure(state=tk.DISABLED)

        # Фаза 2-3: прогрес-бар, живий лог та фінальний банер
        progress_wrap = tk.Frame(body, bg="#181a24")
        progress_lbl = tk.Label(
            progress_wrap,
            text=tr("ui.settings.autosetup.dlg.prep_lbl", "Preparing auto-setup..."),
            fg="#9ee8ff",
            bg="#181a24",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
        )
        progress_lbl.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        progress_pb = ttk.Progressbar(progress_wrap, maximum=100, style="SettingsDark.Horizontal.TProgressbar")
        progress_pb.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        done_banner = tk.Frame(progress_wrap, bg="#172a23", padx=12, pady=8, highlightthickness=1, highlightbackground="#75e08b")
        done_banner_lbl = tk.Label(
            done_banner,
            text="",
            fg="#75e08b",
            bg="#172a23",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=560,
        )
        done_banner_lbl.pack(fill=tk.X)
        self._bind_auto_wrap(done_banner_lbl, initial=560, min_wrap=300, pad=28)

        log_frame = tk.Frame(progress_wrap, bg="#181a24")
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        log_sb = tk.Scrollbar(log_frame, orient=tk.VERTICAL)
        log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        log_text = tk.Text(
            log_frame,
            bg="#12141c",
            fg="#d5dae8",
            font=("Consolas", 9),
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            yscrollcommand=log_sb.set,
            state=tk.DISABLED,
        )
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_sb.configure(command=log_text.yview)

        def _restart_now():
            self._autosetup_running = False
            if self._autosetup_dlg is dlg:
                self._autosetup_dlg = None
            try:
                dlg.grab_release()
            except Exception:
                pass
            try:
                dlg.destroy()
            except Exception:
                pass
            if self.on_restart_app_cb is not None:
                self.on_restart_app_cb()

        def _append_log(line: str):
            try:
                if not log_text.winfo_exists():
                    return
                log_text.configure(state=tk.NORMAL)
                log_text.insert(tk.END, str(line).rstrip() + "\n")
                log_text.see(tk.END)
                log_text.configure(state=tk.DISABLED)
            except Exception:
                pass

        def _start_execution():
            if self._autosetup_running or state.get("cancelled") or not state.get("plan_ready"):
                return
            self._autosetup_running = True
            state["phase"] = "running"
            dlg.protocol("WM_DELETE_WINDOW", lambda: None)

            try:
                plan_wrap.pack_forget()
                progress_wrap.pack(fill=tk.BOTH, expand=True)
                status_lbl.configure(
                    text=tr(
                        "ui.settings.autosetup.dlg.running_status",
                        "⏳ Running full auto-setup... Do not close this window until finished.",
                    ),
                    fg="#ffcc00",
                )
                confirm_btn.configure(state=tk.DISABLED, text=tr("ui.settings.autosetup.dlg.running_btn", "⏳ Running..."))
                cancel_btn.configure(state=tk.DISABLED)
            except Exception:
                pass

            def _on_prog(info: dict):
                def _ui_step():
                    if self._is_closing or state.get("cancelled"):
                        return
                    try:
                        if not dlg.winfo_exists():
                            return
                        pct = max(0, min(100, int((info or {}).get("pct", 0))))
                        progress_pb["value"] = pct
                        msg = str((info or {}).get("message") or "")
                        if msg:
                            progress_lbl.configure(text=msg)
                        line = str((info or {}).get("line") or msg)
                        if line:
                            _append_log(line)
                    except Exception:
                        pass

                self._ui_queue.put((_ui_step, ()))

            def _exec_worker():
                res_obj = None
                err_obj = None
                try:
                    fn_run = autosetup_runner or run_full_autosetup
                    res_obj = fn_run(
                        self.cfg_mgr.base_dir,
                        self.hw_monitor,
                        draft_snapshot,
                        registry=self.registry,
                        progress_cb=_on_prog,
                    )
                except Exception as exc:
                    err_obj = exc

                def _ui_done():
                    self._autosetup_running = False
                    if self._is_closing:
                        return
                    dlg.protocol("WM_DELETE_WINDOW", _close_dlg)
                    state["phase"] = "done"
                    state["done"] = True
                    state["result"] = res_obj
                    state["error"] = str(err_obj) if err_obj is not None else None

                    if err_obj is not None or not isinstance(res_obj, dict):
                        try:
                            progress_pb["value"] = 0
                            status_lbl.configure(
                                text=tr("ui.settings.autosetup.dlg.err_status", "✖ Auto-setup error: {err}", err=err_obj),
                                fg="#ff5555",
                            )
                            progress_lbl.configure(
                                text=tr("ui.settings.autosetup.dlg.err_aborted", "Auto-setup aborted due to an error."),
                                fg="#ff5555",
                            )
                            _append_log(tr("ui.settings.autosetup.dlg.err_log", "✖ Error: {err}", err=err_obj))
                            confirm_btn.pack_forget()
                            cancel_btn.configure(text=tr("ui.common.close", "Close"), state=tk.NORMAL)
                            state["done_btn"] = cancel_btn
                        except Exception:
                            pass
                        return

                    # Збереження та гаряче застосування конфігурації й модулів
                    new_cfg = res_obj.get("config")
                    if isinstance(new_cfg, dict):
                        self.cfg_mgr.save_data(new_cfg)
                        self.draft = self.cfg_mgr.get_snapshot()
                    if res_obj.get("caps") is not None:
                        self._last_caps = res_obj.get("caps")
                    if res_obj.get("cuda_status") is not None:
                        self._last_cuda_status = res_obj.get("cuda_status")

                    if self.camera_manager is not None:
                        try:
                            self.camera_manager.reload_from_registry(force_all=True)
                        except Exception as re_err:
                            print(f"[AutoSetup reload_from_registry]: {re_err}")

                    if self.on_apply_cb:
                        try:
                            self.on_apply_cb()
                        except Exception as app_err:
                            print(f"[AutoSetup on_apply_cb]: {app_err}")

                    hw_saved = (self.draft or {}).get("hardware") or {}
                    chosen_ai = str(hw_saved.get("ai_backend", "cpu")).strip().lower()
                    restart_req = bool(res_obj.get("restart_required"))
                    ok_all = bool(res_obj.get("ok", True))

                    summary_lines = []
                    upd_saved_tag = tr("ui.settings.autosetup.dlg.updated_saved", "updated and saved")
                    already_opt_tag = tr("ui.settings.autosetup.plan.already_opt", "already optimal")
                    for ch in (res_obj.get("hw_changes") or []):
                        if ch.get("changed"):
                            summary_lines.append(f"  • {ch['label']}: {ch['old']} → {ch['new']}  ({upd_saved_tag})")
                        else:
                            summary_lines.append(f"  • {ch['label']}: {ch['new']}  ({already_opt_tag})")
                    changed_mods = [m for m in (res_obj.get("module_changes") or []) if m.get("changed")]
                    if changed_mods:
                        summary_lines.append(
                            tr(
                                "ui.settings.autosetup.dlg.sum_mods",
                                "  • Camera module/codec parameters: updated {n} item(s).",
                                n=len(changed_mods),
                            )
                        )
                    if res_obj.get("installed_packages"):
                        summary_lines.append(
                            tr(
                                "ui.settings.autosetup.dlg.sum_pip",
                                "  • Python packages installed: {pkgs}",
                                pkgs=", ".join(res_obj["installed_packages"]),
                            )
                        )
                    if res_obj.get("ort_switched"):
                        summary_lines.append(
                            tr(
                                "ui.settings.autosetup.dlg.sum_ort",
                                "  • ONNX Runtime installed: {pkg}",
                                pkg=res_obj.get("ort_package"),
                            )
                        )
                    if res_obj.get("cuda_downloaded"):
                        summary_lines.append(
                            tr(
                                "ui.settings.autosetup.dlg.sum_cuda",
                                "  • CUDA/cuDNN libraries downloaded to ./cuda/.",
                            )
                        )

                    extra_parts = []
                    for w in (res_obj.get("warnings") or []):
                        extra_parts.append(f"⚠ {w}")
                    for e in (res_obj.get("errors") or []):
                        extra_parts.append(f"✖ {e}")
                    if restart_req:
                        extra_parts.append(
                            tr(
                                "ui.settings.autosetup.dlg.restart_note",
                                "⚠ Previous native library was already loaded in process memory. Configuration has been saved — restart the app to activate GPU changes.",
                            )
                        )

                    if restart_req:
                        hl = tr(
                            "ui.settings.autosetup.dlg.hl_restart",
                            "🚀 Auto-setup saved ({mode}) — application restart required:",
                            mode=chosen_ai.upper(),
                        )
                    elif not ok_all:
                        hl = tr(
                            "ui.settings.autosetup.dlg.hl_warn",
                            "⚠ Auto-setup completed with warnings ({mode}):",
                            mode=chosen_ai.upper(),
                        )
                    else:
                        hl = tr(
                            "ui.settings.autosetup.dlg.hl_ok",
                            "✔ Full auto-setup ({mode}) saved and applied!",
                            mode=chosen_ai.upper(),
                        )

                    self._hw_auto_note = {
                        "headline": hl,
                        "lines": summary_lines,
                        "extra_note": "\n".join(extra_parts),
                        "pending_save": False,
                        "restart_required": restart_req,
                        "is_gpu_ready": bool(chosen_ai != "cpu" and not restart_req),
                        "chosen_ai": chosen_ai,
                        "new_vals": {
                            "ai_backend": chosen_ai,
                            "ai_threads": _clamp_int(hw_saved.get("ai_threads", 2), 2, 1, 8),
                            "video_decode_mode": str(hw_saved.get("video_decode_mode", "cpu")).strip().lower(),
                            "use_opencv_opencl": bool(hw_saved.get("use_opencv_opencl", False)),
                        },
                        "cfg_badge": "",
                    }

                    if restart_req:
                        foot_txt = tr(
                            "ui.settings.autosetup.dlg.foot_restart",
                            "⚠ Auto-setup saved — restart the application to activate GPU libraries.",
                        )
                        foot_col = "#ffcc00"
                    elif not ok_all:
                        foot_txt = tr(
                            "ui.settings.autosetup.dlg.foot_warn",
                            "⚠ Auto-setup completed with warnings — see details.",
                        )
                        foot_col = "#ffcc00"
                    else:
                        foot_txt = tr(
                            "ui.settings.autosetup.dlg.foot_ok",
                            "✔ Full auto-setup saved and applied!",
                        )
                        foot_col = "#75e08b"
                    self._set_footer_status(foot_txt, fg=foot_col)

                    # Перебудовуємо сторінку налаштувань зі збереженням скролу
                    self._inspected_ai_mode = chosen_ai
                    self._rebuild_nav_tree(select_id=self._current_node_id, preserve_scroll=True)

                    # Оновлюємо інтерфейс діалогу у фінальний стан (Фаза 3)
                    try:
                        if not dlg.winfo_exists():
                            return
                        progress_pb["value"] = 100
                        if restart_req:
                            status_lbl.configure(
                                text=tr(
                                    "ui.settings.autosetup.dlg.done_status_restart",
                                    "⚠ Configuration saved and applied. Restart the app to activate updated libraries in RAM.",
                                ),
                                fg="#ffcc00",
                            )
                            done_banner.configure(bg="#332814", highlightbackground="#ffcc00")
                            done_banner_lbl.configure(
                                text=hl + ("\n" + "\n".join(extra_parts) if extra_parts else ""),
                                bg="#332814",
                                fg="#ffe082",
                            )
                        elif not ok_all:
                            status_lbl.configure(
                                text=tr(
                                    "ui.settings.autosetup.dlg.done_status_warn",
                                    "⚠ Auto-setup completed with warnings (configuration saved and applied).",
                                ),
                                fg="#ffcc00",
                            )
                            done_banner.configure(bg="#332814", highlightbackground="#ffcc00")
                            done_banner_lbl.configure(
                                text=hl + ("\n" + "\n".join(extra_parts) if extra_parts else ""),
                                bg="#332814",
                                fg="#ffe082",
                            )
                        else:
                            status_lbl.configure(
                                text=tr(
                                    "ui.settings.autosetup.dlg.done_status_ok",
                                    "✔ Full auto-setup completed successfully! Profile: {mode}.",
                                    mode=chosen_ai.upper(),
                                ),
                                fg="#75e08b",
                            )
                            ocl_state = (
                                tr("ui.common.on_short", "On")
                                if hw_saved.get("use_opencv_opencl")
                                else tr("ui.common.off_short", "Off")
                            )
                            done_banner.configure(bg="#172a23", highlightbackground="#75e08b")
                            done_banner_lbl.configure(
                                text=hl
                                + "\n"
                                + tr(
                                    "ui.settings.autosetup.dlg.done_summary_line",
                                    "AI: {mode}, threads={threads}, decode={dec}, OpenCL={ocl}.",
                                    mode=chosen_ai.upper(),
                                    threads=hw_saved.get("ai_threads"),
                                    dec=hw_saved.get("video_decode_mode"),
                                    ocl=ocl_state,
                                ),
                                bg="#172a23",
                                fg="#75e08b",
                            )
                        done_banner.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0), before=log_frame)

                        confirm_btn.pack_forget()
                        cancel_btn.configure(
                            text=tr("ui.settings.autosetup.dlg.done_btn", "Done (Close)"),
                            bg="#28a745",
                            fg="#ffffff",
                            state=tk.NORMAL,
                        )
                        state["done_btn"] = cancel_btn

                        if restart_req and self.on_restart_app_cb is not None and state["restart_btn"] is None:
                            r_btn = tk.Button(
                                btn_bar,
                                text=tr("ui.settings.autosetup.dlg.restart_now", "♻ Restart Application Now"),
                                command=_restart_now,
                                bg="#d97706",
                                fg="#ffffff",
                                relief=tk.FLAT,
                                font=("Segoe UI", 9, "bold"),
                                padx=14,
                                pady=6,
                                cursor="hand2",
                            )
                            r_btn.pack(side=tk.RIGHT, padx=(6, 0))
                            state["restart_btn"] = r_btn
                    except Exception:
                        pass

                self._ui_queue.put((_ui_done, ()))

            threading.Thread(target=_exec_worker, daemon=True).start()

        confirm_btn = tk.Button(
            btn_bar,
            text=tr("ui.settings.autosetup.dlg.confirm_btn", "🚀 Confirm & Execute"),
            command=_start_execution,
            state=tk.DISABLED,
            bg="#1565c0",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=6,
            cursor="hand2",
        )
        confirm_btn.pack(side=tk.RIGHT)
        state["confirm_btn"] = confirm_btn

        try:
            self.win.update_idletasks()
            w_par = max(400, int(self.win.winfo_width() or 860))
            h_par = max(400, int(self.win.winfo_height() or 560))
            px = max(20, int(self.win.winfo_rootx()) + (w_par - 720) // 2)
            py = max(20, int(self.win.winfo_rooty()) + (h_par - 560) // 2)
            dlg.geometry(f"720x560+{px}+{py}")
        except Exception:
            dlg.geometry("720x560")

        # Фонове сканування для побудови плану підтвердження
        def _scan_worker():
            plan_obj = None
            scan_err = None
            try:
                fn_plan = plan_builder or build_full_autosetup_plan
                plan_obj = fn_plan(
                    self.cfg_mgr.base_dir,
                    self.hw_monitor,
                    draft_snapshot,
                    registry=self.registry,
                )
            except Exception as exc:
                scan_err = exc

            def _apply_plan():
                if self._is_closing or state.get("cancelled"):
                    return
                try:
                    if not dlg.winfo_exists():
                        return
                except Exception:
                    return
                if scan_err is not None or not isinstance(plan_obj, dict):
                    state["error"] = str(scan_err)
                    status_lbl.configure(
                        text=tr("ui.settings.autosetup.dlg.scan_err_status", "✖ Failed to prepare auto-setup plan: {err}", err=scan_err),
                        fg="#ff5555",
                    )
                    plan_text.configure(state=tk.NORMAL)
                    plan_text.delete("1.0", tk.END)
                    plan_text.insert(
                        tk.END,
                        tr("ui.settings.autosetup.dlg.scan_err_body", "Diagnostic error:\n{err}", err=scan_err),
                    )
                    plan_text.configure(state=tk.DISABLED)
                    return

                state["plan"] = plan_obj
                state["plan_ready"] = True
                state["phase"] = "confirm"
                formatted = self._format_autosetup_plan_text(plan_obj, draft_snapshot)
                status_lbl.configure(
                    text=tr(
                        "ui.settings.autosetup.dlg.plan_ready",
                        "✔ Diagnostics complete. Review the plan below and confirm running auto-setup:",
                    ),
                    fg="#75e08b",
                )
                plan_text.configure(state=tk.NORMAL)
                plan_text.delete("1.0", tk.END)
                plan_text.insert(tk.END, formatted)
                plan_text.configure(state=tk.DISABLED)
                confirm_btn.configure(state=tk.NORMAL)

            self._ui_queue.put((_apply_plan, ()))

        threading.Thread(target=_scan_worker, daemon=True).start()
        return dlg

    # ------------------------------------------------------------------
    # Сторінка 4: Модулі (plug-and-play) та стан бібліотек Python
    # ------------------------------------------------------------------
    def _request_app_restart(self):
        if self._is_closing:
            return
        if not messagebox.askyesno(
            tr("ui.settings.restart.title", "Restart Application"),
            tr(
                "ui.settings.restart.confirm",
                "Save current draft settings and restart the application now\nto apply updated native libraries or modules?",
            ),
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
        self._set_footer_status(
            tr("ui.settings.mods.scanning", "Scanning scripts/modules/ and checking libraries..."),
            fg="#ffcc00",
        )
        self.win.update_idletasks()
        res = None
        if self.on_refresh_modules_cb:
            try:
                res = self.on_refresh_modules_cb(force=True)
            except Exception as e:
                self._set_footer_status(
                    tr("ui.settings.mods.refresh_err", "Module refresh error: {err}", err=e),
                    fg="#ff5555",
                )
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
            if err_n:
                summary = tr(
                    "ui.settings.mods.refresh_done_err",
                    "Modules & libraries updated (+{added}, upd. {updated}, rem. {removed}, errors: {errors}).",
                    added=a,
                    updated=u,
                    removed=r,
                    errors=err_n,
                )
            else:
                summary = tr(
                    "ui.settings.mods.refresh_done_ok",
                    "Modules & libraries updated (+{added}, upd. {updated}, rem. {removed}).",
                    added=a,
                    updated=u,
                    removed=r,
                )
            self._set_footer_status(summary, fg="#ffcc00" if err_n else "#75e08b")
        else:
            self._set_footer_status(
                tr("ui.settings.mods.refresh_ok", "✔ Modules & libraries updated!"),
                fg="#75e08b",
            )

    def _open_modules_folder(self):
        mod_dir = os.path.join(self.cfg_mgr.base_dir, "scripts", "modules")
        os.makedirs(mod_dir, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(mod_dir)
            else:
                subprocess.Popen(["xdg-open", mod_dir])
        except Exception as e:
            messagebox.showerror(
                tr("ui.settings.mods.dir_title", "Modules Folder"),
                tr("ui.settings.hw.open_dir_err", "Failed to open folder:\n{path}\n{err}", path=mod_dir, err=e),
                parent=self.win,
            )

    def _build_page_modules(self, parent):
        mod_dir = os.path.join(self.cfg_mgr.base_dir, "scripts", "modules")
        self._section_header(
            parent,
            tr("ui.settings.mods.title", "🧩 Independent Modules & Library Updates"),
            tr(
                "ui.settings.mods.subtitle",
                "Modules folder: {path}\nAny new or updated .py module with the MODULE_SPEC contract is automatically loaded by the application without editing the core.",
                path=mod_dir,
            ),
        )

        # Панель дій
        act_card = self._make_card(parent, tr("ui.settings.mods.card_manage", "Plugins & Environment Management"))
        act_bg = str(act_card.cget("bg"))
        btn_bar = tk.Frame(act_card, bg=act_bg)
        btn_bar.pack(fill=tk.X)

        tk.Button(
            btn_bar,
            text=tr("ui.settings.mods.btn_refresh", "🔄 Refresh Modules & Libraries"),
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
            text=tr("ui.settings.mods.btn_open_dir", "📂 Open Modules Folder"),
            command=self._open_modules_folder,
            bg="#2f3346",
            fg="#00e5ff",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            cursor="hand2",
        ).pack(side=tk.LEFT)

        btn_bar2 = tk.Frame(act_card, bg=act_bg)
        btn_bar2.pack(fill=tk.X, pady=(6, 0))

        tk.Button(
            btn_bar2,
            text=tr("ui.settings.hw.autosetup_btn", "🚀 Full Auto-Setup (1 click)"),
            command=self._open_full_autosetup_dialog,
            bg="#1565c0",
            fg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"),
            padx=10,
            pady=5,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=(0, 6))

        HelpButton(
            btn_bar2,
            title=tr("ui.settings.hw.autosetup_card", "Full System Auto-Setup (1 click)"),
            help_text=tr(
                "ui.settings.hw.autosetup_help",
                "Opens the automatic setup wizard with a plan preview and confirmation:\n1) Checks and installs missing Python dependencies (.venv);\n2) Selects and installs optimal ONNX Runtime (NVIDIA CUDA → DirectML → CPU);\n3) Checks and downloads missing CUDA / cuDNN DLLs into ./cuda/;\n4) Checks and downloads models/resources for all active modules;\n5) Tunes optimal hardware, decoding, and camera codec parameters, saves config.json and applies immediately (or prompts restart if older native .dll/.pyd files are already loaded in memory).",
            ),
            bg=act_bg,
        ).pack(side=tk.LEFT)

        if self.on_restart_app_cb is not None:
            tk.Button(
                btn_bar2,
                text=tr("ui.settings.hw.btn_restart_app", "♻ Restart Application"),
                command=self._request_app_restart,
                bg="#3a2f22",
                fg="#ffcc00",
                relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"),
                padx=10,
                pady=5,
                cursor="hand2",
            ).pack(side=tk.RIGHT)

        # Картка 1: Завантажені й проблемні модулі
        mods_card = self._make_card(parent, tr("ui.settings.mods.card_list", "Detected Modules (scripts/modules/)"))
        m_bg = str(mods_card.cget("bg"))

        rows = self.registry.describe_modules_status() if self.registry is not None else []
        if not rows:
            tk.Label(
                mods_card,
                text=tr("ui.settings.mods.none_found", "No modules found in scripts/modules/."),
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
                if is_err:
                    kind_lbl = tr("ui.settings.mods.kind_error", "ERROR")
                elif r.get("kind") == "recorder":
                    kind_lbl = tr("ui.settings.mods.kind_recorder", "Video Recording")
                else:
                    kind_lbl = tr("ui.settings.mods.kind_ai", "AI Module")
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

                st_col = "#ff9999" if is_err else ("#75e08b" if bool(r.get("status_ok", True)) else "#ffcc00")
                st_lbl = tk.Label(
                    box,
                    text=tr("ui.settings.mods.row_status", "Status: {status}", status=r.get("status", "")),
                    fg=st_col,
                    bg=row_bg,
                    font=("Segoe UI", 8, "bold"),
                    anchor=tk.W,
                    wraplength=440,
                    justify=tk.LEFT,
                )
                st_lbl.pack(fill=tk.X, pady=(2, 0))
                self._bind_auto_wrap(st_lbl, initial=440, min_wrap=200)

                if not is_err and self.registry is not None:
                    spec = self.registry.get_spec(r["id"]) or {}
                    raw_desc = str(spec.get("description") or "").strip()
                    desc = tr(f"mod.{r['id']}.description", raw_desc) if raw_desc else ""
                    deps = spec.get("library_deps") or []
                    extra_pkgs.extend(deps)
                    if desc:
                        d_lbl = tk.Label(
                            box,
                            text=desc,
                            fg="#b8c0d4",
                            bg=row_bg,
                            font=("Segoe UI", 8),
                            anchor=tk.W,
                            wraplength=440,
                            justify=tk.LEFT,
                        )
                        d_lbl.pack(fill=tk.X, pady=(2, 0))
                        self._bind_auto_wrap(d_lbl, initial=440, min_wrap=200)
                    if deps:
                        tk.Label(
                            box,
                            text=tr("ui.settings.mods.row_deps", "Dependencies: {deps}", deps=", ".join(deps)),
                            fg="#8892aa",
                            bg=row_bg,
                            font=("Consolas", 8),
                            anchor=tk.W,
                        ).pack(fill=tk.X, pady=(2, 0))

                if is_err and r.get("error"):
                    err_lbl = tk.Label(
                        box,
                        text=str(r["error"]).strip(),
                        fg="#ffb3b3",
                        bg="#1f1014",
                        font=("Consolas", 8),
                        anchor=tk.W,
                        justify=tk.LEFT,
                        wraplength=440,
                        padx=6,
                        pady=4,
                    )
                    err_lbl.pack(fill=tk.X, pady=(4, 0))
                    self._bind_auto_wrap(err_lbl, initial=440, min_wrap=200)

            # Картка 2: Версії Python-бібліотек і виявлення потреби у перезапуску
            libs_card = self._make_card(
                parent,
                tr("ui.settings.mods.card_libs", "Python Libraries & Dynamic Capability Refresh"),
                tr(
                    "ui.settings.mods.card_libs_sub",
                    "Compares package versions on disk against C extensions (.pyd/.dll) already loaded in RAM.",
                ),
            )
            l_bg = str(libs_card.cget("bg"))
            lib_rows = get_library_versions(extra_pkgs=extra_pkgs)

            if any(lr.get("needs_restart") for lr in lib_rows):
                alert = tk.Frame(libs_card, bg="#3b2814", padx=10, pady=8, highlightthickness=1, highlightbackground="#ffcc00")
                alert.pack(fill=tk.X, pady=(0, 8))
                al_lbl = tk.Label(
                    alert,
                    text=tr(
                        "ui.settings.mods.libs_restart_warn",
                        "⚠ Updated native C libraries detected on disk while an older version is already loaded in process memory.\nRestart the application to fully activate the new drivers/libraries (ONNX Runtime, OpenCV, etc.).",
                    ),
                    fg="#ffe082",
                    bg="#3b2814",
                    font=("Segoe UI", 8, "bold"),
                    justify=tk.LEFT,
                    anchor=tk.W,
                    wraplength=280,
                )
                al_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self._bind_auto_wrap(al_lbl, initial=280, min_wrap=160)
                if self.on_restart_app_cb is not None:
                    tk.Button(
                        alert,
                        text=tr("ui.settings.mods.btn_restart_now", "♻ Restart Now"),
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
            tk.Label(hdr, text=tr("ui.settings.mods.col_lib", "Library"), fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), width=24, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(hdr, text=tr("ui.settings.mods.col_disk", "On Disk"), fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), width=12, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(hdr, text=tr("ui.settings.mods.col_mem", "In Memory"), fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), width=22, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(hdr, text=tr("ui.settings.mods.col_status", "Status"), fg="#00e5ff", bg="#252836", font=("Segoe UI", 8, "bold"), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

            for lr in lib_rows:
                lrow = tk.Frame(libs_card, bg="#222432", padx=8, pady=3)
                lrow.pack(fill=tk.X, pady=1)
                skind = str(lr.get("status_kind") or "")
                if lr.get("needs_restart") or skind == "restart":
                    st_c = "#ffcc00"
                elif skind == "active":
                    st_c = "#75e08b"
                elif skind == "disk_only":
                    st_c = "#9ee8ff"
                else:
                    st_c = "#888c9e"

                tk.Label(lrow, text=lr["name"], fg="#ffffff", bg="#222432", font=("Consolas", 8, "bold"), width=24, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(lrow, text=lr["disk_version"], fg="#d5dae8", bg="#222432", font=("Consolas", 8), width=12, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(lrow, text=lr["loaded_version"], fg="#b8c0d4", bg="#222432", font=("Segoe UI", 8), width=22, anchor=tk.W).pack(side=tk.LEFT)
                tk.Label(lrow, text=lr["status"], fg=st_c, bg="#222432", font=("Segoe UI", 8, "bold"), anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True)

            if not rows:
                tk.Label(libs_card, text=tr("ui.settings.mods.no_libs", "No library data available."), fg="#888c9e", bg=l_bg).pack()

    # ------------------------------------------------------------------
    # Сторінка 5: Список усіх камер
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
            tr("ui.settings.cams.title", "📹 Camera Management"),
            tr(
                "ui.settings.cams.subtitle",
                "Add any number of cameras. Each gets a unique UUID, its own archive folder, and an independent module set.",
            ),
        )

        bar = tk.Frame(parent, bg="#1c1e2a", padx=12, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
        bar.pack(fill=tk.X, pady=(0, 12))

        def _add_cam():
            cams = self.draft.setdefault("cameras", [])
            new_idx = len(cams) + 1
            tpl = self._build_template_modules()
            new_cam = make_camera_entry(
                name=tr("camera.default_numbered", "Camera {num}", num=new_idx),
                main_source="",
                sub_source="",
                enabled=True,
                modules=tpl,
                registry=self.registry,
            )
            cams.append(new_cam)
            self._rebuild_nav_tree(select_id=f"cam:{new_cam['uuid']}")

        tk.Button(
            bar, text=tr("ui.settings.cams.btn_add", "➕ Add New Camera (auto UUID)"), command=_add_cam,
            bg="#28a745", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=6, cursor="hand2"
        ).pack(side=tk.LEFT)

        reg_entries = self.registry.list_modules() if self.registry is not None else []
        rec_entries = self.registry.list_modules(kind="recorder") if self.registry is not None else []

        cams = self.draft.get("cameras") or []
        cam_def_single = tr("camera.default_single", "Camera")
        for idx, cam in enumerate(cams):
            uid = cam["uuid"]
            mods = cam.get("modules") or {}

            badges = []
            for ent in reg_entries:
                mid = ent["id"]
                if (mods.get(mid) or {}).get("on", False):
                    badges.append(tr(f"mod.{mid}.name", ent["spec"].get("name", mid)))
            mod_txt = ", ".join(badges) if badges else tr("ui.settings.cams.no_active_mods", "No active modules")

            rec_parts = []
            for ent in rec_entries:
                mid = ent["id"]
                msec = mods.get(mid) or {}
                rec_disp = tr(f"mod.{mid}.name", ent["spec"].get("name", mid))
                if msec.get("codec"):
                    rec_parts.append(f"{rec_disp}: {msec.get('codec')}")
                elif msec.get("on", False):
                    rec_parts.append(tr("ui.settings.cams.rec_on", "{name}: on", name=rec_disp))
            rec_txt = "  |  " + ", ".join(rec_parts) if rec_parts else ""

            card = tk.Frame(parent, bg="#1c1e2a", padx=12, pady=10, highlightthickness=1, highlightbackground="#2b2e3d")
            card.pack(fill=tk.X, pady=4)

            top = tk.Frame(card, bg="#1c1e2a")
            top.pack(fill=tk.X)
            en_col = "#75e08b" if cam.get("enabled", True) else "#888c9e"
            en_txt = (
                tr("ui.settings.cams.status_active", "ACTIVE")
                if cam.get("enabled", True)
                else tr("ui.settings.cams.status_disabled", "DISABLED")
            )
            cam_disp = cam.get("name") or cam_def_single
            tk.Label(top, text=f"#{idx + 1}  {cam_disp}  [{en_txt}]", fg=en_col, bg="#1c1e2a", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
            tk.Label(top, text=f"UUID: {uid}", fg="#888c9e", bg="#1c1e2a", font=("Consolas", 8)).pack(side=tk.RIGHT)

            sub_disp = cam.get("sub_source") or tr("ui.settings.cams.sub_scaled_main", "(scaled Main)")
            tk.Label(
                card,
                text=tr(
                    "ui.settings.cams.sources_line",
                    "Main: {main}   |   Sub (grid): {sub}",
                    main=cam.get("main_source") or "—",
                    sub=sub_disp,
                ),
                fg="#c8d0e4", bg="#1c1e2a", font=("Segoe UI", 8), anchor=tk.W
            ).pack(fill=tk.X, pady=(4, 2))
            tk.Label(
                card,
                text=tr("ui.settings.cams.modules_line", "Modules: {mods}{rec}", mods=mod_txt, rec=rec_txt),
                fg="#00e5ff", bg="#1c1e2a", font=("Segoe UI", 8), anchor=tk.W
            ).pack(fill=tk.X, pady=(0, 6))

            actions = tk.Frame(card, bg="#1c1e2a")
            actions.pack(fill=tk.X)

            tk.Button(
                actions, text=tr("ui.settings.cams.btn_configure", "⚙ Configure"),
                command=lambda u=uid: self._select_nav_node(f"cam:{u}"),
                bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
            ).pack(side=tk.LEFT, padx=(0, 6))

            def _dup(c_src=cam):
                dup_mods = copy.deepcopy(c_src.get("modules") or make_default_camera_modules(registry=self.registry))
                src_nm = c_src.get("name") or tr("camera.default_single", "Camera")
                dup_cam = make_camera_entry(
                    name=tr("ui.settings.cams.copy_suffix", "{name} (copy)", name=src_nm),
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
                    messagebox.showwarning(
                        tr("ui.common.warning", "Warning"),
                        tr("ui.settings.cams.min_one_warn", "At least 1 camera must remain in the system."),
                        parent=self.win,
                    )
                    return
                if messagebox.askyesno(
                    tr("ui.settings.cams.del_title", "Delete Camera"),
                    tr(
                        "ui.settings.cams.del_confirm",
                        "Remove camera '{name}' from the list?\n(Archive files on disk will remain intact).",
                        name=cname,
                    ),
                    parent=self.win,
                ):
                    self.draft["cameras"] = [c for c in clist if c.get("uuid") != u_del]
                    self._rebuild_nav_tree(select_id="cameras_root")

            tk.Button(actions, text=tr("ui.settings.cams.btn_dup", "📋 Duplicate"), command=_dup, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
            tk.Button(actions, text="⬆", command=lambda i=idx: _move(i, -1), state=tk.NORMAL if idx > 0 else tk.DISABLED, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=3, cursor="hand2").pack(side=tk.LEFT, padx=(0, 2))
            tk.Button(actions, text="⬇", command=lambda i=idx: _move(i, 1), state=tk.NORMAL if idx < len(cams) - 1 else tk.DISABLED, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=3, cursor="hand2").pack(side=tk.LEFT, padx=(0, 6))
            tk.Button(actions, text=tr("ui.settings.cams.btn_del", "🗑 Delete"), command=lambda cn=cam.get("name", ""), u=uid: _delete(cn, u), bg="#44252a", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.RIGHT)

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
            tk.Label(parent, text=tr("ui.settings.cam.not_found", "Camera not found."), fg="#ff5555", bg="#161720").pack()
            return

        is_op_mode = self._is_draft_operator_mode()
        cam_arch_dir = os.path.join(self.draft.get('general', {}).get('archive_root') or self.cfg_mgr.get_archive_root(), 'cameras', cam_uuid)
        sub_txt = (
            tr(
                "ui.settings.cam.arch_op",
                "Camera archive folder (disk writes disabled in Operator Mode): {dir}",
                dir=cam_arch_dir,
            )
            if is_op_mode
            else tr("ui.settings.cam.arch_srv", "Camera archive folder: {dir}", dir=cam_arch_dir)
        )
        cam_title_name = cam.get("name") or tr("camera.default_single", "Camera")
        self._section_header(
            parent,
            tr("ui.settings.cam.title", "📹 Camera: {name}", name=cam_title_name),
            sub_txt,
        )

        # Картка 1: Потоки та ідентифікація камери
        src_card = self._make_card(parent, tr("ui.settings.cam.card_src", "Connection & Streams"))
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
            cam["name"] = name_var.get().strip() or tr("camera.default_single", "Camera")
            cam["enabled"] = bool(en_var.get())
            cam["main_source"] = main_var.get().strip()
            cam["sub_source"] = sub_var.get().strip()
            for m_key, m_var in mod_vars.items():
                mods.setdefault(m_key, {})["on"] = bool(m_var.get())
            if self.nav_tree.exists(f"cam:{cam_uuid}"):
                p_icon = "●" if cam["enabled"] else "⏸"
                suf = "" if cam["enabled"] else tr("ui.settings.nav.cam_disabled_suf", " (disabled)")
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
                    self._set_footer_status(
                        tr("ui.settings.cam.uuid_copied", "UUID {uuid} copied.", uuid=uid_var.get()),
                        fg="#00e5ff",
                    )
                except Exception:
                    pass

            tk.Button(box, text=tr("ui.settings.cam.btn_copy", "📋 Copy"), command=_copy_uid, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT)
            return e

        self._add_row(src_card, tr("ui.settings.cam.uuid_label", "Camera UUID:"), "camera.uuid", _w_uuid)

        def _w_name(box, bg):
            e = tk.Entry(box, textvariable=name_var, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9))
            e.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))
            tk.Checkbutton(box, text=tr("ui.settings.cam.enabled_chk", "Camera enabled"), variable=en_var, command=_commit_cam_base, bg=bg, fg="#75e08b", selectcolor="#252836", activebackground=bg, activeforeground="#75e08b", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            return e

        self._add_row(src_card, tr("ui.settings.cam.name_label", "Name / description:"), "camera.name", _w_name)

        def _run_stream_test(kind_label, raw_src):
            src_val = raw_src.strip()
            test_res_lbl.configure(
                text=tr("ui.settings.cam.testing_stream", "Testing '{kind}' ({src})...", kind=kind_label, src=src_val),
                fg="#ffcc00",
            )
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
                        title=tr("ui.settings.cam.pick_video_title", "Select video file ({kind})", kind=kind_lbl),
                        filetypes=[
                            (tr("ui.settings.cam.ft_video", "Video files"), "*.mp4 *.avi *.mkv *.mov *.ts"),
                            (tr("ui.settings.cam.ft_all", "All files"), "*.*"),
                        ],
                    )
                    if fp:
                        var.set(os.path.normpath(fp))
                        _commit_cam_base()

                tk.Button(box, text="📁", command=_browse_file, bg="#2f3346", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=3, pady=4, cursor="hand2").pack(side=tk.LEFT, padx=(0, 4))
                tk.Button(box, text=tr("ui.settings.cam.btn_test", "▶ Test"), command=lambda: _run_stream_test(kind_lbl, var.get()), bg="#233b56", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=8, pady=4, cursor="hand2").pack(side=tk.LEFT)
                return e
            return _fn

        self._add_row(src_card, tr("ui.settings.cam.main_label", "Main stream (Main):"), "camera.main_source", _w_source(main_var, "Main"))
        self._add_row(src_card, tr("ui.settings.cam.sub_label", "Sub-stream (Grid Sub):"), "camera.sub_source", _w_source(sub_var, "Sub"))
        test_res_lbl.pack(fill=tk.X, pady=(4, 0))

        # Картка 2: ОДНА ПАНЕЛЬ З ЧЕКБОКСАМИ МОДУЛІВ В РЯДОК (динамічно з ModuleRegistry)
        mod_bar = self._make_card(parent, tr("ui.settings.cam.card_mods", "Analytics & Recording Modules for This Camera"))
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
                text=tr("ui.settings.cam.no_mods", "No modules found in scripts/modules/."),
                fg="#888c9e",
                bg=m_bg,
                font=("Segoe UI", 9),
            ).pack(anchor=tk.W)
        else:
            per_row = 4
            has_disk_mods = False
            for r_idx in range(0, len(mod_entries), per_row):
                row_fr = tk.Frame(chk_container, bg=m_bg)
                row_fr.pack(fill=tk.X, pady=2)
                for ent in mod_entries[r_idx:r_idx + per_row]:
                    mid = ent["id"]
                    spec = ent["spec"]
                    is_rec_mod = (spec.get("kind") == "recorder" or bool(spec.get("writes_to_disk")))
                    if is_rec_mod:
                        has_disk_mods = True
                    lock_mod_toggle = bool(is_op_mode and is_rec_mod)
                    mod_disp_name = tr(f"mod.{mid}.name", spec.get("name", mid))
                    m_label = f"{spec.get('icon', '🧩')} {mod_disp_name}"
                    grp = tk.Frame(row_fr, bg="#252836", padx=8, pady=5)
                    grp.pack(side=tk.LEFT, padx=(0, 10))
                    tk.Checkbutton(
                        grp,
                        text=f"{m_label} 🔒" if lock_mod_toggle else m_label,
                        variable=mod_vars[mid],
                        command=_rebuild_inline_module_blocks,
                        state=tk.DISABLED if lock_mod_toggle else tk.NORMAL,
                        bg="#252836",
                        fg="#8d95ab" if lock_mod_toggle else "#ffffff",
                        disabledforeground="#7d849c",
                        selectcolor="#151720",
                        activebackground="#252836",
                        activeforeground="#00e5ff",
                        font=("Segoe UI", 9, "bold"),
                        cursor="arrow" if lock_mod_toggle else "hand2",
                    ).pack(side=tk.LEFT)
                    HelpButton(
                        grp,
                        help_text=ConfigManager.get_help_text(f"{mid}.on", registry=self.registry),
                        title=m_label,
                        bg="#252836",
                    ).pack(side=tk.LEFT, padx=(4, 0))
            if is_op_mode and has_disk_mods:
                op_mod_note = tk.Label(
                    mod_bar,
                    text=tr(
                        "ui.settings.cam.op_mod_note",
                        "👤 Operator Mode: disk-writing modules are temporarily disabled at runtime and locked without losing saved settings.",
                    ),
                    fg="#ffcc00",
                    bg="#2c2416",
                    font=("Segoe UI", 8, "bold"),
                    padx=10,
                    pady=5,
                    anchor=tk.W,
                    justify=tk.LEFT,
                    wraplength=700,
                )
                op_mod_note.pack(fill=tk.X, pady=(4, 2))
                self._bind_auto_wrap(op_mod_note, initial=700, min_wrap=220, pad=20)

        tools_row = tk.Frame(mod_bar, bg=m_bg)
        tools_row.pack(fill=tk.X, pady=(6, 0))

        def _copy_modules_to_others():
            self._commit_current_page()
            others = [c for c in (self.draft.get("cameras") or []) if c.get("uuid") != cam_uuid]
            if not others:
                messagebox.showinfo(
                    tr("ui.settings.cam.copy_title", "Copy Modules"),
                    tr("ui.settings.cam.copy_no_others", "No other cameras available to copy to."),
                    parent=self.win,
                )
                return
            _open_copy_modules_modal(
                self.win, cam, others,
                on_done=lambda n: self._set_footer_status(
                    tr(
                        "ui.settings.cam.copy_done",
                        "Module settings copied to {n} camera(s) (click 'Apply').",
                        n=n,
                    ),
                    fg="#00e5ff",
                ),
            )

        def _save_as_template():
            self._commit_current_page()
            self.draft["default_camera_modules"] = copy.deepcopy(mods)
            self._set_footer_status(
                tr("ui.settings.cam.tpl_saved", "Module configuration saved as template for new cameras."),
                fg="#75e08b",
            )

        tk.Button(
            tools_row, text=tr("ui.settings.cam.btn_copy_mods", "📋 Copy module settings to other cameras..."),
            command=_copy_modules_to_others, bg="#2f3346", fg="#00e5ff",
            relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(
            tools_row, text=tr("ui.settings.cam.btn_save_tpl", "💾 Save as template for new cameras"),
            command=_save_as_template, bg="#2f3346", fg="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.LEFT)

        _rebuild_inline_module_blocks()

    # ------------------------------------------------------------------
    # Декларативний рендерер картки модуля з MODULE_SPEC['settings_schema']
    # ------------------------------------------------------------------
    def _build_dynamic_module_card(self, parent, spec: dict, sec: dict):
        mid = str(spec.get("id", "module"))
        mod_name = tr(f"mod.{mid}.name", spec.get("name", mid))
        title = tr(
            "ui.settings.mod_card.title",
            "{icon} Settings: {name}",
            icon=spec.get("icon", "🧩"),
            name=mod_name,
        )
        raw_subtitle = str(spec.get("description") or "").strip()
        subtitle = tr(f"mod.{mid}.description", raw_subtitle) if raw_subtitle else ""
        card = self._make_card(parent, title, subtitle=subtitle)

        schema = spec.get("settings_schema") or []
        if not schema:
            tk.Label(
                card,
                text=tr("ui.settings.mod_card.no_params", "This module has no additional parameters."),
                fg="#8d95ab",
                bg=str(card.cget("bg")),
                font=("Segoe UI", 8),
            ).pack(anchor=tk.W)
            return

        is_op_mode = self._is_draft_operator_mode()
        is_recorder_card = (spec.get("kind") == "recorder" or bool(spec.get("writes_to_disk")))
        has_disk_fields = any(bool(fld.get("disk_write")) for fld in schema)

        if is_op_mode and is_recorder_card:
            rec_lock_lbl = tk.Label(
                card,
                text=tr(
                    "ui.settings.mod_card.op_rec_locked",
                    "🔒 Operator Mode: disk recording is disabled. Parameters are locked, but saved settings remain preserved.",
                ),
                fg="#ffcc00",
                bg="#2c2416",
                font=("Segoe UI", 8, "bold"),
                padx=10,
                pady=6,
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=700,
            )
            rec_lock_lbl.pack(fill=tk.X, pady=(0, 8))
            self._bind_auto_wrap(rec_lock_lbl, initial=700, min_wrap=220, pad=20)
        elif is_op_mode and has_disk_fields:
            ai_lock_lbl = tk.Label(
                card,
                text=tr(
                    "ui.settings.mod_card.op_ai_disk_locked",
                    "👤 Operator Mode: live detection and overlays remain active, while saving files and auto-cleanup on disk are disabled (corresponding fields are locked without losing values).",
                ),
                fg="#ffcc00",
                bg="#2c2416",
                font=("Segoe UI", 8, "bold"),
                padx=10,
                pady=6,
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=700,
            )
            ai_lock_lbl.pack(fill=tk.X, pady=(0, 8))
            self._bind_auto_wrap(ai_lock_lbl, initial=700, min_wrap=220, pad=20)

        raw_vars = {}
        choices_cache = {}
        conditional_bindings = []  # list of (widgets_list, disable_when_dict)
        operator_locked_widgets = []
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

            for w in operator_locked_widgets:
                try:
                    w.configure(state=tk.DISABLED)
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
            raw_label = str(fld.get("label") or f"{key}:")
            label = tr(f"mod.{mid}.field.{key}.label", raw_label)
            raw_help = str(fld.get("help") or "")
            help_text = tr(f"mod.{mid}.field.{key}.help", raw_help) if raw_help else ""
            disable_cond = fld.get("disable_when")

            created_widgets = []

            if ftype == "bool":
                var = tk.BooleanVar(value=bool(sec.get(key, fld.get("default", False))))
                raw_vars[key] = var
                raw_bt = str(fld.get("bool_text") or "")
                bool_txt = tr(f"mod.{mid}.field.{key}.bool_text", raw_bt) if raw_bt else ""

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
                        style="SettingsDark.TCombobox",
                    )
                    cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    style_dark_combobox_popdown(cb)
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
                    tk.Label(box, text=tr("ui.settings.mod_card.date_from", "From:"), fg="#9ba3b8", bg=bg_c, font=("Segoe UI", 8)).pack(side=tk.LEFT)
                    e1 = tk.Entry(box, textvariable=var1, width=12, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Consolas", 9))
                    e1.pack(side=tk.LEFT, ipady=3, padx=(4, 10))
                    tk.Label(box, text=tr("ui.settings.mod_card.date_to", "To:"), fg="#9ba3b8", bg=bg_c, font=("Segoe UI", 8)).pack(side=tk.LEFT)
                    e2 = tk.Entry(box, textvariable=var2, width=12, bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Consolas", 9))
                    e2.pack(side=tk.LEFT, ipady=3, padx=(4, 8))
                    tk.Label(box, text=tr("ui.settings.mod_card.date_hint", "(YYYY-MM-DD or empty)"), fg="#767c90", bg=bg_c, font=("Segoe UI", 8)).pack(side=tk.LEFT)
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
            if is_op_mode and (is_recorder_card or bool(fld.get("disk_write"))) and created_widgets:
                operator_locked_widgets.extend(created_widgets)

        for n_idx, n_def in enumerate(spec.get("notices") or []):
            if not isinstance(n_def, dict) or not n_def.get("text"):
                continue
            n_lbl = tk.Label(
                card,
                text=tr(f"mod.{mid}.notice.{n_idx}", str(n_def["text"])),
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

        if isinstance(self._hw_auto_note, dict):
            nv = self._hw_auto_note.get("new_vals") or {}
            saved_hw = self.draft.get("hardware") or {}
            matches_auto = bool(nv) and (
                str(saved_hw.get("ai_backend", "cpu")).strip().lower() == nv.get("ai_backend")
                and _clamp_int(saved_hw.get("ai_threads", 2), 2, 1, 8) == nv.get("ai_threads")
                and str(saved_hw.get("video_decode_mode", "cpu")).strip().lower() == nv.get("video_decode_mode")
                and bool(saved_hw.get("use_opencv_opencl", False)) == bool(nv.get("use_opencv_opencl"))
            )
            if matches_auto:
                self._hw_auto_note["pending_save"] = False
            else:
                self._hw_auto_note = None

        saved_lang = normalize_language_code((self.draft.get("general") or {}).get("language")) or DEFAULT_LANGUAGE
        prev_lang = get_language()
        lang_changed = (saved_lang != prev_lang)

        if self.on_apply_cb:
            try:
                self.on_apply_cb()
            except Exception as e:
                messagebox.showerror(
                    tr("ui.settings.apply_err_title", "Apply Error"),
                    tr("ui.settings.apply_err_msg", "Failed to apply settings:\n{err}", err=e),
                    parent=self.win,
                )
                return

        did_self_apply_lang = False
        if saved_lang != get_language():
            set_language(saved_lang)
            try:
                self.hw_monitor.relocalize_cached_caps()
            except Exception:
                pass
            did_self_apply_lang = True

        if isinstance(self._hw_auto_note, dict):
            chosen_u = str(self._hw_auto_note.get("chosen_ai") or "cpu").upper()
            if self._hw_auto_note.get("restart_required"):
                self._hw_auto_note["headline"] = tr(
                    "ui.settings.hw.note_saved_restart",
                    "✔ Settings ({mode}) saved — restart the app to activate GPU:",
                    mode=chosen_u,
                )
            else:
                self._hw_auto_note["headline"] = tr(
                    "ui.settings.hw.note_saved_applied",
                    "✔ Selected settings ({mode}) saved and applied!",
                    mode=chosen_u,
                )

        self._set_footer_status(
            tr("ui.settings.applied_ok", "✔ Settings saved and applied!"),
            fg="#75e08b",
        )
        if did_self_apply_lang:
            self.apply_language()
        elif (not lang_changed) or isinstance(self._hw_auto_note, dict):
            self._rebuild_nav_tree(select_id=self._current_node_id, preserve_scroll=True)

    def _save_and_close(self):
        self._apply_changes()
        self.close()

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        close_active_help_popover()
        self._cancel_telemetry_timer()
        if isinstance(self._autosetup_dlg_state, dict):
            self._autosetup_dlg_state["cancelled"] = True
        if self._autosetup_dlg is not None:
            try:
                self._autosetup_dlg.grab_release()
            except Exception:
                pass
            try:
                self._autosetup_dlg.destroy()
            except Exception:
                pass
            self._autosetup_dlg = None
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


def _open_copy_modules_modal(parent, src_cam: dict, target_cams: list, on_done=None):
    dlg = tk.Toplevel(parent)
    dlg.title(tr("ui.settings.copy_modal.title", "Copy Modules"))
    dlg.configure(bg="#1c1e2a")
    dlg.transient(parent)
    dlg.grab_set()

    def_cam_nm = tr("camera.default_single", "Camera")
    src_name = src_cam.get("name") or def_cam_nm
    tk.Label(
        dlg,
        text=tr(
            "ui.settings.copy_modal.prompt",
            "Copy modules and their settings from '{name}' to:",
            name=src_name,
        ),
        fg="#00e5ff", bg="#1c1e2a", font=("Segoe UI", 10, "bold"), padx=14, pady=12, anchor=tk.W
    ).pack(fill=tk.X)

    vars_map = []
    for c in target_cams:
        v = tk.BooleanVar(value=True)
        vars_map.append((c, v))
        c_nm = c.get("name") or def_cam_nm
        tk.Checkbutton(
            dlg,
            text=f"{c_nm}  ({c.get('main_source') or c['uuid'][:8]})",
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

    tk.Button(btns, text=tr("ui.common.cancel", "Cancel"), command=dlg.destroy, bg="#353849", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4).pack(side=tk.RIGHT, padx=(6, 0))
    tk.Button(btns, text=tr("ui.settings.copy_modal.btn_copy", "Copy"), command=_apply, bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4).pack(side=tk.RIGHT)

    parent.update_idletasks()
    px = parent.winfo_rootx() + max(20, (parent.winfo_width() - 420) // 2)
    py = parent.winfo_rooty() + max(20, (parent.winfo_height() - 260) // 2)
    dlg.geometry(f"+{px}+{py}")
    dlg.wait_window()
