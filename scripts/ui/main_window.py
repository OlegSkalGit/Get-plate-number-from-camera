import os
import sys
import time
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

from scripts.core.i18n import (
    init_i18n,
    tr,
    set_language,
    get_language,
    list_languages,
    get_language_label,
    parse_language_code,
)
from scripts.core.module_registry import ModuleRegistry
from scripts.core.config_manager import ConfigManager
from scripts.core.hardware_info import (
    HardwareMonitor,
    prepare_cuda_dll_environment,
    check_cuda_libraries_status,
    download_cuda_libraries,
    is_onnxruntime_gpu_installed,
    has_nvidia_gpu,
)
from scripts.engine.inference_pool import InferencePool
from scripts.engine.camera_manager import CameraManager
from scripts.ui.helpers import WindowGeometryTracker, compute_auto_grid
from scripts.ui.detail_window import CameraDetailWindow
from scripts.ui.settings_window import SettingsWindow
from scripts.ui.people_window import PeopleWindow

# Базова папка додатку (підтримка PyInstaller EXE та структури scripts/ui/main_window.py)
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import winreg
except ImportError:
    winreg = None

EVT_MOTION = "<Motion>"
EVT_CONFIGURE = "<Configure>"
EVT_ADMIN_EXIT_UP = "<Control-Alt-Shift-KeyPress-Q>"
EVT_ADMIN_EXIT_LOW = "<Control-Alt-Shift-KeyPress-q>"

LOCKDOWN_KEYS = [
    "<Alt-F4>",
    "<Alt-KeyPress-F4>",
    "<Control-w>",
    "<Control-W>",
    "<Control-q>",
    "<Control-Q>",
    "<F11>",
]


class MainGridWindow:
    """
    Головне вікно системи: адаптивна або ручна сітка перегляду стиснутих потоків (sub_source)
    усіх камер. Лівий клік по плитці відкриває незалежне вікно повного перегляду (main_source + архів)
    конкретної камери. Фоновий запис та AI-аналіз усіх камер працюють безперервно незалежно від вікон,
    а незалежні модулі автоматично підхоплюються через ModuleRegistry.
    """
    def __init__(self, root: tk.Tk, base_dir: str = None):
        self.root = root
        self._manage_os_startup = (base_dir is None)
        self.base_dir = os.path.abspath(base_dir or BASE_DIR)
        if self._manage_os_startup:
            try:
                os.chdir(self.base_dir)
            except Exception:
                pass

        self.root.configure(bg="#0b0c10")

        self._is_closing = False
        self._startup_ts = time.time()

        # 1. Автоматичне виявлення незалежних модулів (scripts/modules/), конфіг і локалізація
        try:
            prepare_cuda_dll_environment(self.base_dir, preload=False)
        except Exception:
            pass
        self.registry = ModuleRegistry(self.base_dir)
        self.cfg_mgr = ConfigManager(self.base_dir, registry=self.registry)
        self.cfg_mgr.sync_with_registry(self.registry)
        init_i18n(self.base_dir, lang_code=self.cfg_mgr.get_language())
        self.root.title(tr("ui.main.window_title", "ANPR Multi-Cam NVR — Camera Grid"))

        snap = self.cfg_mgr.get_snapshot()
        gen_cfg = snap.get("general") or {}
        op_mode = str(gen_cfg.get("operation_mode", "server")).strip().lower()
        if op_mode not in ("server", "operator"):
            op_mode = "server"
        allow_write = (op_mode == "server")
        hw_cfg = snap.get("hardware") or {}

        # 2. Телеметрія заліза, спільний пул ресурсів та менеджер камер
        self.hw_monitor = HardwareMonitor(self.base_dir)
        self.inference_pool = InferencePool(
            self.base_dir,
            self.registry,
            app_context={
                "operation_mode": op_mode,
                "allow_disk_write": allow_write,
                "faces_db_path": self.cfg_mgr.get_faces_db_path(create=allow_write),
            },
            ai_backend=hw_cfg.get("ai_backend", "cpu"),
            ai_threads=int(hw_cfg.get("ai_threads", 2)),
        )
        self.camera_manager = CameraManager(
            self.cfg_mgr,
            self.inference_pool,
            self.registry,
            on_saved_event=self._on_camera_saved_cb,
        )

        # Відкриті дочірні вікна
        self.settings_win = None
        self.people_win = None
        self.detail_windows = {}  # cam_uuid -> CameraDetailWindow

        # Стан сітки та пагінації
        self._current_page = 0
        self._total_pages = 1
        self._tiles = {}          # uuid -> dict widget refs & cache
        self._last_layout_key = None
        self._relayout_after_id = None
        self._render_after_id = None
        self._hw_after_id = None
        self._hide_bar_after_id = None
        self._module_poll_after_id = None
        self._start_after_id = None
        self._ui_poll_after_id = None
        self._cuda_note_clear_after_id = None
        self._panel_visible = True
        self._lang_menu_active = False
        self._last_total_cams = 0

        # Потокобезпечна черга UI-викликів із фонових потоків
        self._ui_queue = queue.SimpleQueue()
        self._cuda_ensure_lock = threading.Lock()
        self._cuda_downloading = False
        self._cuda_status_note = ""
        self._cuda_note_color = "#ffcc00"

        # Кеш фонового збору телеметрії (щоб не блокувати UI-потік)
        self._cached_hw_metrics = None
        self._hw_poll_running = True

        # Синхронізація автозапуску та режиму кіоску
        self._sync_windows_startup(bool(gen_cfg.get("autostart_os_and_play", False)))
        self._apply_kiosk_mode(bool(gen_cfg.get("kiosk_mode", False)))
        self.root.bind_all(EVT_ADMIN_EXIT_UP, self._admin_exit)
        self.root.bind_all(EVT_ADMIN_EXIT_LOW, self._admin_exit)

        self._build_ui()

        # Відновлення й автозбереження позиції вікна на будь-якому моніторі
        self.geom_tracker = WindowGeometryTracker(
            self.root,
            get_geom_cb=lambda: self.cfg_mgr.get_window_geometry("grid"),
            set_geom_cb=lambda g: self.cfg_mgr.set_window_geometry("grid", g),
            min_w=640,
            min_h=400,
            default_geom={"left": 40, "top": 30, "width": 1280, "height": 760, "maximized": False},
        )

        self.root.bind(EVT_MOTION, self._on_mouse_motion)
        self.root.bind(EVT_CONFIGURE, self._on_root_configure)

        self._drain_ui_queue()
        self._start_hw_poll_thread()
        self._schedule_hide_bar(5000)

        self._start_after_id = self.root.after(120, self._start_system)

    # ------------------------------------------------------------------
    # Ініціалізація UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_area = tk.Frame(self.root, bg="#0b0c10")
        self.grid_area.pack(fill=tk.BOTH, expand=True)

        # Автоприховувана верхня панель (поверх сітки без зміщення її висоти)
        self.ctrl_frame = tk.Frame(
            self.root, bg="#171923", padx=12, pady=7,
            highlightthickness=1, highlightbackground="#292d3e"
        )

        left_box = tk.Frame(self.ctrl_frame, bg="#171923")
        left_box.pack(side=tk.LEFT)

        tk.Label(
            left_box, text="📹 ANPR Multi-Cam NVR", fg="#ffffff", bg="#171923",
            font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_cam_count = tk.Label(
            left_box, text=tr("ui.main.cam_count", "Cameras: {n}", n=0), fg="#00e5ff", bg="#222534",
            font=("Segoe UI", 8, "bold"), padx=8, pady=3
        )
        self.lbl_cam_count.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_mode_badge = tk.Label(
            left_box, text="", fg="#75e08b", bg="#1b2e24",
            font=("Segoe UI", 8, "bold"), padx=8, pady=3
        )
        self.lbl_mode_badge.pack(side=tk.LEFT, padx=(0, 10))
        self._refresh_mode_badge()

        self.btn_settings = tk.Button(
            left_box, text=tr("ui.main.btn_settings", "⚙ Settings"), command=self.open_settings,
            bg="#007acc", fg="#ffffff", activebackground="#0066ad", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_settings.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_people = tk.Button(
            left_box, text=tr("ui.main.btn_people", "👥 People"), command=self.open_people,
            bg="#254c5a", fg="#ffffff", activebackground="#2f6375", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_people.pack(side=tk.LEFT, padx=(0, 8))

        # Блок пагінації (видимий лише коли в ручній сітці > 1 сторінки)
        self.page_box = tk.Frame(self.ctrl_frame, bg="#171923")
        self.btn_prev_page = tk.Button(
            self.page_box, text="◀", command=lambda: self._change_page(-1),
            bg="#2b2f40", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, pady=2, cursor="hand2"
        )
        self.btn_prev_page.pack(side=tk.LEFT, padx=(0, 4))

        self.lbl_page = tk.Label(
            self.page_box, text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=1, total=1), fg="#d5dae8", bg="#171923",
            font=("Segoe UI", 9, "bold"), padx=6
        )
        self.lbl_page.pack(side=tk.LEFT)

        self.btn_next_page = tk.Button(
            self.page_box, text="▶", command=lambda: self._change_page(1),
            bg="#2b2f40", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, pady=2, cursor="hand2"
        )
        self.btn_next_page.pack(side=tk.LEFT, padx=(4, 0))

        # Правий блок: швидкий перемикач мови + телеметрія заліза + вихід
        right_box = tk.Frame(self.ctrl_frame, bg="#171923")
        right_box.pack(side=tk.RIGHT)

        lang_box = tk.Frame(right_box, bg="#171923")
        lang_box.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(
            lang_box, text="🌐", fg="#00e5ff", bg="#171923",
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=(0, 4))

        try:
            style = ttk.Style(self.root)
            if style.theme_use() in ("vista", "xpnative", "winnative", "default"):
                style.theme_use("clam")
            style.configure(
                "MainLang.TCombobox",
                fieldbackground="#222534",
                background="#2b2f40",
                foreground="#ffffff",
                arrowcolor="#00e5ff",
                bordercolor="#2e3347",
                lightcolor="#2e3347",
                darkcolor="#222534",
                selectbackground="#007acc",
                selectforeground="#ffffff",
            )
            style.map(
                "MainLang.TCombobox",
                fieldbackground=[("readonly", "#222534")],
                foreground=[("readonly", "#ffffff")],
                selectbackground=[("readonly", "#222534")],
                selectforeground=[("readonly", "#ffffff")],
            )
        except Exception:
            pass

        lang_labels = [m["label"] for m in list_languages()]
        self.top_lang_var = tk.StringVar(value=get_language_label())
        self.cmb_top_lang = ttk.Combobox(
            lang_box,
            textvariable=self.top_lang_var,
            values=lang_labels,
            state="readonly",
            width=21,
            font=("Segoe UI", 8, "bold"),
            style="MainLang.TCombobox",
        )
        self.cmb_top_lang.pack(side=tk.LEFT)
        self.cmb_top_lang.bind("<<ComboboxSelected>>", self._on_top_language_selected)
        self.cmb_top_lang.bind("<Button-1>", self._on_top_lang_interact)
        self.cmb_top_lang.bind("<FocusIn>", self._on_top_lang_interact)
        self.cmb_top_lang.bind("<FocusOut>", self._on_top_lang_leave)

        self.lbl_hw_badge = tk.Label(
            right_box, text=tr("ui.main.hw_badge_init", "CPU: --% | RAM: -- | AI: CPU"),
            fg="#9ee8ff", bg="#222534", font=("Segoe UI", 8, "bold"),
            padx=10, pady=4
        )
        self.lbl_hw_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_exit = tk.Button(
            right_box, text=tr("ui.main.btn_exit", "🚪 Exit"), command=self.on_close,
            bg="#64262b", fg="#ffffff", activebackground="#7e2f36", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_exit.pack(side=tk.LEFT)
        self._apply_exit_btn_text()

        self._show_top_bar()

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
                if self.root.winfo_exists():
                    self._ui_poll_after_id = self.root.after(40, self._drain_ui_queue)
            except Exception:
                pass

    def _start_system(self):
        self._start_after_id = None
        if self._is_closing:
            return
        self.camera_manager.start_all()
        self._relayout_grid(force=True)
        self._render_grid_tick()
        self._refresh_hw_badge_tick()
        self._module_poll_after_id = self.root.after(3000, self._poll_modules_tick)
        self._maybe_ensure_cuda_async()

    def _set_cuda_note(self, text: str, color: str = "#ffcc00", clear_after_ms: int = 0):
        if self._is_closing:
            return
        if self._cuda_note_clear_after_id is not None:
            try:
                self.root.after_cancel(self._cuda_note_clear_after_id)
            except Exception:
                pass
            self._cuda_note_clear_after_id = None
        self._cuda_status_note = str(text or "")
        self._cuda_note_color = str(color or "#ffcc00")
        if self._cuda_status_note:
            try:
                self.lbl_hw_badge.configure(text=self._cuda_status_note, fg=self._cuda_note_color)
            except Exception:
                pass
            if clear_after_ms > 0:
                self._cuda_note_clear_after_id = self.root.after(int(clear_after_ms), self._clear_cuda_note)
        else:
            try:
                self.lbl_hw_badge.configure(fg="#9ee8ff")
            except Exception:
                pass

    def _clear_cuda_note(self):
        self._cuda_note_clear_after_id = None
        self._cuda_status_note = ""
        self._cuda_note_color = "#9ee8ff"
        if not self._is_closing:
            try:
                self.lbl_hw_badge.configure(fg="#9ee8ff")
            except Exception:
                pass

    def _maybe_ensure_cuda_async(self):
        """
        Автоматично перевіряє наявність CUDA / cuDNN бібліотек при вибраному GPU-режимі ('cuda' або 'auto' з onnxruntime-gpu).
        Якщо потрібних DLL немає у системі та в <BASE_DIR>/cuda/, фоново завантажує їх у локальну папку cuda/.
        """
        if self._is_closing:
            return
        hw_cfg = self.cfg_mgr.get_snapshot().get("hardware") or {}
        backend = str(hw_cfg.get("ai_backend", "cpu")).strip().lower()
        gpu_ort = is_onnxruntime_gpu_installed()
        if not (backend == "cuda" or (backend == "auto" and gpu_ort)):
            return

        with self._cuda_ensure_lock:
            if self._cuda_downloading:
                return
            self._cuda_downloading = True

        def _worker():
            try:
                if self._is_closing or not has_nvidia_gpu():
                    return
                st = check_cuda_libraries_status(self.base_dir, allow_network=False)
                if st.get("ready"):
                    prepare_cuda_dll_environment(self.base_dir, preload=True)
                    return

                self._ui_queue.put((self._set_cuda_note, (tr("ui.main.cuda_dl_start", "⬇ Checking and downloading CUDA to cuda/..."), "#ffcc00", 0)))
                last_msg_ts = [0.0]

                def _prog(msg):
                    if self._is_closing:
                        return
                    now_t = time.time()
                    if now_t - last_msg_ts[0] < 0.35:
                        return
                    last_msg_ts[0] = now_t
                    short_m = f"⬇ CUDA: {msg[:60]}"
                    self._ui_queue.put((self._set_cuda_note, (short_m, "#ffcc00", 0)))

                res = download_cuda_libraries(
                    self.base_dir,
                    progress_cb=_prog,
                    include_optional=True,
                    force_into_local=False,
                )
                if self._is_closing:
                    return

                prepare_cuda_dll_environment(self.base_dir, preload=True)
                caps = self.hw_monitor.probe_acceleration_capabilities(force=True)
                cuda_ok = bool(((caps.get("ai_backends") or {}).get("cuda") or {}).get("supported"))
                gpu_pkg_now = is_onnxruntime_gpu_installed()

                def _on_done():
                    if self._is_closing:
                        return
                    if res.get("ok"):
                        try:
                            ai_ids = [ent["id"] for ent in self.registry.list_modules(kind="ai")]
                            self.inference_pool.invalidate_all()
                            self.camera_manager.sync_from_config()
                            for u in list(self.camera_manager.units.values()):
                                u.force_reload_modules(ai_ids)
                        except Exception as e:
                            print(f"[CUDA hot-reload error]: {e}")

                        if self.settings_win is not None:
                            try:
                                if self.settings_win.win.winfo_exists():
                                    self.settings_win.on_cuda_libraries_updated(caps=caps)
                            except Exception:
                                pass

                        if cuda_ok:
                            self._set_cuda_note(tr("ui.main.cuda_ok_active", "✔ CUDA libraries downloaded to cuda/ — GPU activated!"), "#75e08b", 8000)
                        elif not gpu_pkg_now:
                            self._set_cuda_note(tr("ui.main.cuda_ok_need_pkg", "✔ CUDA DLLs downloaded to cuda/ (onnxruntime-gpu required)"), "#ffcc00", 12000)
                        else:
                            self._set_cuda_note(tr("ui.main.cuda_ok_restart", "✔ CUDA DLLs in cuda/ — restart to activate GPU"), "#ffcc00", 12000)
                    else:
                        if self.settings_win is not None:
                            try:
                                if self.settings_win.win.winfo_exists():
                                    self.settings_win.on_cuda_libraries_updated(caps=caps)
                            except Exception:
                                pass
                        self._set_cuda_note(tr("ui.main.cuda_dl_fail", "⚠ Failed to download CUDA DLLs (CPU fallback active)"), "#ff8888", 10000)

                self._ui_queue.put((_on_done, ()))
            except Exception as e:
                print(f"[CUDA auto-ensure error]: {e}")
            finally:
                with self._cuda_ensure_lock:
                    self._cuda_downloading = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Автоматичне виявлення нових/оновлених модулів і бібліотек
    # ------------------------------------------------------------------
    def refresh_modules_and_libraries(self, force: bool = True) -> dict:
        if self._is_closing:
            return {"added": [], "updated": [], "removed": [], "errors": {}, "changed": False}
        res = self.registry.scan_and_load(force=force)
        if force:
            try:
                self.hw_monitor.refresh_all(force=True)
            except Exception:
                pass
        return self._apply_registry_update(res, force_reload_all=force)

    def _apply_registry_update(self, res: dict, force_reload_all: bool = False) -> dict:
        if self._is_closing:
            return res
        changed = bool(res and res.get("changed"))
        if changed or force_reload_all:
            try:
                self.registry.refresh_libraries_capabilities()
            except Exception:
                pass
            self.cfg_mgr.sync_with_registry(self.registry)
            self._refresh_mode_badge()

            added = list(res.get("added") or [])
            updated = list(res.get("updated") or [])
            removed = list(res.get("removed") or [])
            try:
                self.camera_manager.reload_from_registry(
                    changed_mod_ids=added + updated,
                    removed_mod_ids=removed,
                    force_all=force_reload_all,
                )
            except Exception as e:
                print(f"[MainGridWindow reload_from_registry]: {e}")

            if self.settings_win is not None:
                try:
                    if self.settings_win.win.winfo_exists():
                        self.settings_win.on_modules_reloaded()
                except Exception:
                    pass

            if self.people_win is not None:
                try:
                    if self.people_win.win.winfo_exists():
                        self.people_win.on_modules_reloaded()
                except Exception:
                    pass

            for dwin in list(self.detail_windows.values()):
                try:
                    if dwin.win.winfo_exists():
                        dwin.on_modules_reloaded()
                except Exception:
                    pass

            self._last_layout_key = None
            self._relayout_grid(force=True)
        return res

    def _poll_modules_tick(self):
        self._module_poll_after_id = None
        if self._is_closing:
            return
        try:
            res = self.registry.check_for_updates()
            if res and res.get("changed"):
                self._apply_registry_update(res, force_reload_all=False)
        except Exception as e:
            print(f"[ModulePoll error]: {e}")
        if not self._is_closing:
            self._module_poll_after_id = self.root.after(3000, self._poll_modules_tick)

    # ------------------------------------------------------------------
    # Реєстр автозапуску Windows, перезапуск і режим кіоску
    # ------------------------------------------------------------------
    def _sync_windows_startup(self, enable: bool):
        if not self._manage_os_startup or winreg is None or sys.platform != "win32":
            return
        app_name = "ANPR_Video_Monitor"
        run_key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        if getattr(sys, "frozen", False):
            launch_cmd = f'"{os.path.abspath(sys.executable)}"'
        else:
            py_exe = sys.executable
            pyw = os.path.join(os.path.dirname(py_exe), "pythonw.exe")
            if os.path.exists(pyw):
                py_exe = pyw
            entry_py = os.path.abspath(os.path.join(self.base_dir, "app.py"))
            launch_cmd = f'"{py_exe}" "{entry_py}"'

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key_path, 0, winreg.KEY_SET_VALUE) as key:
                if enable:
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, launch_cmd)
                else:
                    try:
                        winreg.DeleteValue(key, app_name)
                    except FileNotFoundError:
                        pass
        except Exception:
            pass

    def restart_application(self):
        if self._is_closing:
            return
        if getattr(sys, "frozen", False):
            cmd = [os.path.abspath(sys.executable)]
        else:
            py_exe = sys.executable
            pyw = os.path.join(os.path.dirname(py_exe), "pythonw.exe")
            if os.path.basename(py_exe).lower() == "pythonw.exe" and os.path.exists(pyw):
                py_exe = pyw
            cmd = [py_exe, os.path.abspath(os.path.join(self.base_dir, "app.py"))]

        self.on_close(exit_app=False)
        try:
            subprocess.Popen(cmd, cwd=self.base_dir, close_fds=True)
        except Exception as e:
            print(f"[Restart error]: {e}")
        sys.exit(0)

    def _apply_exit_btn_text(self):
        if hasattr(self, "btn_exit"):
            if getattr(self, "_kiosk_mode", False):
                self.btn_exit.configure(
                    text=tr("ui.main.btn_kiosk", "🔒 Kiosk (Ctrl+Alt+Shift+Q)"),
                    bg="#2e3142",
                    fg="#9ba3b8",
                    command=self._kiosk_hint,
                )
            else:
                self.btn_exit.configure(
                    text=tr("ui.main.btn_exit", "🚪 Exit"),
                    bg="#64262b",
                    fg="#ffffff",
                    command=self.on_close,
                )

    def _apply_kiosk_mode(self, kiosk_on: bool):
        self._kiosk_mode = bool(kiosk_on)
        if self._kiosk_mode:
            self.root.protocol("WM_DELETE_WINDOW", self._prevent_close)
            for key in LOCKDOWN_KEYS:
                self.root.bind(key, lambda e: "break")
        else:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            for key in LOCKDOWN_KEYS:
                try:
                    self.root.unbind(key)
                except Exception:
                    pass

        self._apply_exit_btn_text()

    def _prevent_close(self):
        pass

    def _kiosk_hint(self):
        messagebox.showinfo(
            tr("ui.main.kiosk_hint_title", "Kiosk Mode"),
            tr(
                "ui.main.kiosk_hint_msg",
                "Exit lock is enabled.\nTo exit the application, press Ctrl+Alt+Shift+Q or disable Kiosk Mode in Settings.",
            ),
            parent=self.root,
        )

    def _admin_exit(self, event=None):
        self.on_close()

    # ------------------------------------------------------------------
    # Автоприховувана верхня панель і швидкий перемикач мови
    # ------------------------------------------------------------------
    def _on_top_lang_interact(self, event=None):
        self._lang_menu_active = True
        self._show_top_bar()
        self._schedule_hide_bar(6000)

    def _on_top_lang_leave(self, event=None):
        self._lang_menu_active = False
        if self._panel_visible:
            self._schedule_hide_bar(4500)

    def _on_top_language_selected(self, event=None):
        self._lang_menu_active = False
        try:
            selected = self.top_lang_var.get()
        except Exception:
            return
        code = parse_language_code(selected, fallback=get_language())
        if code:
            self.change_language(code, source="topbar")
        if self._panel_visible:
            self._schedule_hide_bar(4500)

    def _sync_top_lang_combo(self):
        if not hasattr(self, "cmb_top_lang") or not hasattr(self, "top_lang_var"):
            return
        try:
            self.cmb_top_lang.configure(values=[m["label"] for m in list_languages()])
            self.top_lang_var.set(get_language_label(get_language()))
        except Exception:
            pass

    def apply_language(self):
        if self._is_closing:
            return
        try:
            self.root.title(tr("ui.main.window_title", "ANPR Multi-Cam NVR — Camera Grid"))
            self.lbl_cam_count.configure(text=tr("ui.main.cam_count", "Cameras: {n}", n=getattr(self, "_last_total_cams", 0)))
            self._refresh_mode_badge()
            self.btn_settings.configure(text=tr("ui.main.btn_settings", "⚙ Settings"))
            self.btn_people.configure(text=tr("ui.main.btn_people", "👥 People"))
            self._apply_exit_btn_text()
            self.lbl_page.configure(
                text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=self._current_page + 1, total=self._total_pages)
            )
            self._sync_top_lang_combo()
            if hasattr(self, "_empty_lbl") and self._empty_lbl.winfo_exists():
                self._empty_lbl.configure(
                    text=tr(
                        "ui.main.empty_cams",
                        "No enabled cameras.\nOpen ⚙ Settings above to add or enable cameras.",
                    )
                )
            self._render_hw_badge_once()
            for t in self._tiles.values():
                t["last_lbl_text"] = None
                t["last_st_key"] = None
                t["last_warn_text"] = None
            self._last_layout_key = None
            self._relayout_grid(force=True)
            self._render_tiles_once()
        except Exception:
            pass

    def change_language(self, lang_code: str, source: str = "topbar") -> str:
        saved_code = self.cfg_mgr.set_language(lang_code)
        eff_code = set_language(saved_code)
        try:
            self.hw_monitor.relocalize_cached_caps()
        except Exception:
            pass

        self.apply_language()

        if self.settings_win is not None:
            try:
                if self.settings_win.win.winfo_exists() and not getattr(self.settings_win, "_is_closing", False):
                    if source == "topbar":
                        self.settings_win.on_external_language_changed(eff_code)
                    else:
                        self.settings_win.apply_language()
            except Exception:
                pass

        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.apply_language()
            except Exception:
                pass

        for dwin in list(self.detail_windows.values()):
            try:
                if dwin.win.winfo_exists():
                    dwin.apply_language()
            except Exception:
                pass

        return eff_code

    def _show_top_bar(self):
        if self._is_closing:
            return
        if not self._panel_visible or not self.ctrl_frame.winfo_ismapped():
            self.ctrl_frame.place(x=0, y=0, relwidth=1.0)
            self.ctrl_frame.lift()
            self._panel_visible = True

    def _hide_top_bar(self):
        self._hide_bar_after_id = None
        if self._is_closing:
            return
        try:
            focused = self.root.focus_get()
            if self._lang_menu_active or (
                focused is not None
                and (focused is getattr(self, "cmb_top_lang", None) or "popdown" in str(focused).lower())
            ):
                self._schedule_hide_bar(2500)
                return
        except Exception:
            pass
        try:
            px, py = self.root.winfo_pointerxy()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            bh = max(44, self.ctrl_frame.winfo_height())
            if rx <= px <= (rx + self.root.winfo_width()) and ry <= py <= (ry + bh + 4):
                self._schedule_hide_bar(2500)
                return
        except Exception:
            pass
        if self._panel_visible:
            self.ctrl_frame.place_forget()
            self._panel_visible = False

    def _schedule_hide_bar(self, ms=5000):
        if self._hide_bar_after_id is not None:
            try:
                self.root.after_cancel(self._hide_bar_after_id)
            except Exception:
                pass
        self._hide_bar_after_id = self.root.after(ms, self._hide_top_bar)

    def _on_mouse_motion(self, event=None):
        if self._is_closing:
            return
        try:
            ry = self.root.winfo_rooty()
            px_y = event.y_root if event is not None else self.root.winfo_pointery()
            local_y = px_y - ry
            if local_y <= 34:
                self._show_top_bar()
                self._schedule_hide_bar(5000)
            elif self._panel_visible:
                self._schedule_hide_bar(5000)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Керування незалежними вікнами (Налаштування + Детальний перегляд камер)
    # ------------------------------------------------------------------
    def open_settings(self):
        if self._is_closing:
            return
        if self.settings_win is not None:
            try:
                if self.settings_win.win.winfo_exists():
                    self.settings_win.focus()
                    return
            except Exception:
                pass
            self.settings_win = None

        self.settings_win = SettingsWindow(
            self.root,
            cfg_mgr=self.cfg_mgr,
            hw_monitor=self.hw_monitor,
            camera_manager=self.camera_manager,
            registry=self.registry,
            on_apply_cb=self.on_settings_applied,
            on_close_cb=self._on_settings_closed,
            on_refresh_modules_cb=self.refresh_modules_and_libraries,
            on_restart_app_cb=self.restart_application,
        )

    def _on_settings_closed(self):
        self.settings_win = None

    def open_people(self):
        if self._is_closing:
            return
        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.focus()
                    return
            except Exception:
                pass
            self.people_win = None

        self.people_win = PeopleWindow(
            self.root,
            cfg_mgr=self.cfg_mgr,
            camera_manager=self.camera_manager,
            registry=self.registry,
            on_close_cb=self._on_people_closed,
        )

    def _on_people_closed(self):
        self.people_win = None

    def _refresh_mode_badge(self):
        if not hasattr(self, "lbl_mode_badge"):
            return
        try:
            if self.cfg_mgr.is_operator_mode():
                self.lbl_mode_badge.configure(
                    text=tr("ui.mode.operator_badge", "👤 Operator (no recording)"),
                    fg="#ffcc00",
                    bg="#332917",
                )
            else:
                self.lbl_mode_badge.configure(
                    text=tr("ui.mode.server_badge", "🖥 Server"),
                    fg="#75e08b",
                    bg="#1b2e24",
                )
        except Exception:
            pass

    def on_settings_applied(self):
        if self._is_closing:
            return
        snap = self.cfg_mgr.get_snapshot()
        gen_cfg = snap.get("general") or {}
        hw_cfg = snap.get("hardware") or {}

        new_lang = self.cfg_mgr.get_language()
        if new_lang != get_language():
            self.change_language(new_lang, source="settings")

        self._refresh_mode_badge()
        self._sync_windows_startup(bool(gen_cfg.get("autostart_os_and_play", False)))
        self._apply_kiosk_mode(bool(gen_cfg.get("kiosk_mode", False)))
        self.camera_manager.sync_from_config()

        cams_by_uid = {c["uuid"]: c for c in (snap.get("cameras") or [])}
        for uid in list(self.detail_windows.keys()):
            dwin = self.detail_windows[uid]
            cam_cfg = cams_by_uid.get(uid)
            unit = self.camera_manager.get_unit(uid)
            if cam_cfg is None or not cam_cfg.get("enabled", True) or unit is None:
                try:
                    dwin.close()
                except Exception:
                    self.detail_windows.pop(uid, None)
            else:
                try:
                    dwin.update_camera_config(cam_cfg, hw_cfg, unit)
                except Exception:
                    pass

        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.on_modules_reloaded()
            except Exception:
                pass

        self._last_layout_key = None
        self._relayout_grid(force=True)
        self._maybe_ensure_cuda_async()

    def open_camera_detail(self, cam_uuid: str):
        if self._is_closing or not cam_uuid:
            return
        existing = self.detail_windows.get(cam_uuid)
        if existing is not None:
            try:
                if existing.win.winfo_exists():
                    existing.focus()
                    return
            except Exception:
                pass
            self.detail_windows.pop(cam_uuid, None)

        unit = self.camera_manager.get_unit(cam_uuid)
        if unit is None:
            return

        dwin = CameraDetailWindow(
            self.root,
            cam_uuid=cam_uuid,
            camera_unit=unit,
            cfg_mgr=self.cfg_mgr,
            registry=self.registry,
            on_close_cb=self._on_detail_closed,
        )
        self.detail_windows[cam_uuid] = dwin
        self._update_tile_borders()

    def _on_detail_closed(self, cam_uuid: str):
        self.detail_windows.pop(cam_uuid, None)
        if not self._is_closing:
            self._update_tile_borders()

    def _on_camera_saved_cb(self, cam_uuid, mod_name, day_str, time_str, label, fpath):
        if self._is_closing:
            return
        try:
            self._ui_queue.put((self._dispatch_saved_to_ui, (cam_uuid, mod_name)))
        except Exception:
            pass

    def _dispatch_saved_to_ui(self, cam_uuid: str, mod_name: str):
        if self._is_closing:
            return
        dwin = self.detail_windows.get(cam_uuid)
        if dwin is not None:
            try:
                dwin.notify_saved(mod_name)
            except Exception:
                pass
        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.notify_saved(mod_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Побудова, адаптація і рендер сітки камер (тільки sub_source)
    # ------------------------------------------------------------------
    def _on_root_configure(self, event=None):
        if self._is_closing:
            return
        if event is not None and event.widget not in (self.root, self.grid_area):
            return
        if self._relayout_after_id is not None:
            try:
                self.root.after_cancel(self._relayout_after_id)
            except Exception:
                pass
        self._relayout_after_id = self.root.after(50, lambda: self._relayout_grid(force=False))

    def _change_page(self, delta: int):
        new_p = max(0, min(max(0, self._total_pages - 1), self._current_page + delta))
        if new_p != self._current_page:
            self._current_page = new_p
            self._relayout_grid(force=True)

    def _clear_tiles(self):
        for slot in self._tiles.values():
            try:
                slot["frame"].destroy()
            except Exception:
                pass
        self._tiles.clear()

    def _relayout_grid(self, force: bool = False):
        self._relayout_after_id = None
        if self._is_closing:
            return

        self.root.update_idletasks()
        area_w = self.grid_area.winfo_width()
        area_h = self.grid_area.winfo_height()
        if area_w < 80 or area_h < 80:
            return

        snap = self.cfg_mgr.get_snapshot()
        grid_cfg = snap.get("grid") or {}
        units = self.camera_manager.get_ordered_enabled_units()
        total_cams = len(units)
        self._last_total_cams = total_cams
        self.lbl_cam_count.configure(text=tr("ui.main.cam_count", "Cameras: {n}", n=total_cams))

        mode = str(grid_cfg.get("mode", "auto")).lower()
        if mode == "manual":
            rows = max(1, min(8, int(grid_cfg.get("rows", 2))))
            cols = max(1, min(8, int(grid_cfg.get("cols", 2))))
            per_page = rows * cols
            self._total_pages = max(1, (total_cams + per_page - 1) // per_page) if total_cams > 0 else 1
            if self._current_page >= self._total_pages:
                self._current_page = max(0, self._total_pages - 1)
            start_i = self._current_page * per_page
            visible_units = units[start_i:start_i + per_page]
        else:
            rows, cols = compute_auto_grid(max(1, total_cams), area_w, area_h)
            self._total_pages = 1
            self._current_page = 0
            visible_units = units

        if self._total_pages > 1:
            self.lbl_page.configure(
                text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=self._current_page + 1, total=self._total_pages)
            )
            if not self.page_box.winfo_ismapped():
                self.page_box.pack(side=tk.LEFT, padx=14)
        elif self.page_box.winfo_ismapped():
            self.page_box.pack_forget()

        bw = max(1, min(10, int(grid_cfg.get("border_width", 2))))
        gap = max(0, min(20, int(grid_cfg.get("tile_gap", 4))))

        layout_key = (
            area_w, area_h, rows, cols, bw, gap,
            tuple(u.uuid for u in visible_units),
            bool(grid_cfg.get("show_labels", True)),
            bool(grid_cfg.get("show_status", True)),
            bool(grid_cfg.get("show_fps", True)),
            grid_cfg.get("border_color", "#00e5ff"),
            grid_cfg.get("active_border_color", "#00ff88"),
            grid_cfg.get("offline_border_color", "#ff4444"),
        )
        if not force and layout_key == self._last_layout_key:
            return
        self._last_layout_key = layout_key

        avail_w = max(60, area_w - gap * (cols + 1))
        avail_h = max(60, area_h - gap * (rows + 1))
        tile_w = max(48, avail_w // cols)
        tile_h = max(48, avail_h // rows)
        total_grid_w = cols * tile_w + (cols + 1) * gap
        total_grid_h = rows * tile_h + (rows + 1) * gap
        origin_x = max(0, (area_w - total_grid_w) // 2)
        origin_y = max(0, (area_h - total_grid_h) // 2)

        if not visible_units:
            self._clear_tiles()
            empty_txt = tr(
                "ui.main.empty_cams",
                "No enabled cameras.\nOpen ⚙ Settings above to add or enable cameras.",
            )
            if not hasattr(self, "_empty_lbl") or not self._empty_lbl.winfo_exists():
                self._empty_lbl = tk.Label(
                    self.grid_area,
                    text=empty_txt,
                    fg="#9ba3b8", bg="#0b0c10", font=("Segoe UI", 13)
                )
            else:
                self._empty_lbl.configure(text=empty_txt)
            self._empty_lbl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            return
        elif hasattr(self, "_empty_lbl") and self._empty_lbl.winfo_exists():
            self._empty_lbl.place_forget()

        needed_uids = [u.uuid for u in visible_units]
        existing_uids = list(self._tiles.keys())
        if existing_uids != needed_uids:
            self._clear_tiles()
            for u in visible_units:
                frame = tk.Frame(self.grid_area, bg=grid_cfg.get("border_color", "#00e5ff"), cursor="hand2")
                canvas = tk.Canvas(frame, bg="#08090d", highlightthickness=0, bd=0, cursor="hand2")
                canvas.pack(fill=tk.BOTH, expand=True, padx=bw, pady=bw)

                img_id = canvas.create_image(1, 1, anchor=tk.CENTER, tags="vid_img")
                ph_id = canvas.create_text(
                    1, 1, text=tr("ui.main.tile.connecting", "Connecting..."),
                    fill="#ffcc00", font=("Segoe UI", 10, "bold"), tags="placeholder"
                )

                lbl_bg_id = canvas.create_rectangle(0, 0, 0, 0, fill="#10131e", outline="#282d3f", width=1, tags="ov_lbl")
                lbl_txt_id = canvas.create_text(10, 10, text="", fill="#ffffff", font=("Segoe UI", 9, "bold"), anchor=tk.NW, tags="ov_lbl")

                st_bg_id = canvas.create_rectangle(0, 0, 0, 0, fill="#10131e", outline="#282d3f", width=1, tags="ov_st")
                st_txt_id = canvas.create_text(10, 10, text="", fill="#75e08b", font=("Segoe UI", 8, "bold"), anchor=tk.NE, tags="ov_st")

                warn_bg_id = canvas.create_rectangle(0, 0, 0, 0, fill="#351418", outline="#ff5555", width=1, tags="ov_warn")
                warn_txt_id = canvas.create_text(10, 10, text="", fill="#ff9999", font=("Segoe UI", 8, "bold"), anchor=tk.SW, tags="ov_warn")

                frame.bind("<Button-1>", lambda e, uid=u.uuid: self.open_camera_detail(uid))
                canvas.bind("<Button-1>", lambda e, uid=u.uuid: self.open_camera_detail(uid))

                self._tiles[u.uuid] = {
                    "frame": frame,
                    "canvas": canvas,
                    "img_id": img_id,
                    "ph_id": ph_id,
                    "lbl_bg_id": lbl_bg_id,
                    "lbl_txt_id": lbl_txt_id,
                    "st_bg_id": st_bg_id,
                    "st_txt_id": st_txt_id,
                    "warn_bg_id": warn_bg_id,
                    "warn_txt_id": warn_txt_id,
                    "photo_tk": None,
                    "last_seq": -1,
                    "inner_w": 0,
                    "inner_h": 0,
                    "last_border": "",
                    "last_lbl_text": None,
                    "last_st_key": None,
                    "last_warn_text": None,
                }

        inner_w = max(32, tile_w - 2 * bw)
        inner_h = max(24, tile_h - 2 * bw)

        for slot_i, u in enumerate(visible_units):
            r = slot_i // cols
            c = slot_i % cols
            px = origin_x + gap + c * (tile_w + gap)
            py = origin_y + gap + r * (tile_h + gap)
            t = self._tiles[u.uuid]

            t["frame"].place(x=px, y=py, width=tile_w, height=tile_h)
            t["canvas"].pack_configure(padx=bw, pady=bw)
            t["inner_w"] = inner_w
            t["inner_h"] = inner_h
            t["last_lbl_text"] = None
            t["last_st_key"] = None
            t["last_warn_text"] = None

            cv = t["canvas"]
            cv.coords(t["img_id"], inner_w // 2, inner_h // 2)
            cv.coords(t["ph_id"], inner_w // 2, inner_h // 2)

            u.set_grid_target_size(inner_w, inner_h)

        self._update_tile_borders()
        if self._panel_visible:
            self.ctrl_frame.lift()

    def _update_tile_borders(self):
        grid_cfg = (self.cfg_mgr.get_snapshot().get("grid") or {})
        col_normal = grid_cfg.get("border_color", "#00e5ff")
        col_active = grid_cfg.get("active_border_color", "#00ff88")
        col_offline = grid_cfg.get("offline_border_color", "#ff4444")

        for uid, t in self._tiles.items():
            unit = self.camera_manager.get_unit(uid)
            if uid in self.detail_windows:
                bcol = col_active
            elif unit is not None:
                snap = unit.get_grid_snapshot()
                if snap.get("sub_status") in ("offline", "reconnecting") and snap.get("main_status") != "active":
                    bcol = col_offline
                else:
                    bcol = col_normal
            else:
                bcol = col_offline

            if bcol != t["last_border"]:
                t["last_border"] = bcol
                try:
                    t["frame"].configure(bg=bcol)
                except Exception:
                    pass

    def _render_tiles_once(self):
        if self._is_closing:
            return 83

        if not self._tiles and self.camera_manager.get_ordered_enabled_units():
            self._relayout_grid(force=False)

        snap = self.cfg_mgr.get_snapshot()
        grid_cfg = snap.get("grid") or {}
        hw_cfg = snap.get("hardware") or {}
        grid_fps = max(2.0, min(30.0, float(hw_cfg.get("display_grid_fps", 12.0))))
        interval_ms = max(25, int(round(1000.0 / grid_fps)))

        show_labels = bool(grid_cfg.get("show_labels", True))
        show_status = bool(grid_cfg.get("show_status", True))
        show_fps = bool(grid_cfg.get("show_fps", True))
        col_normal = grid_cfg.get("border_color", "#00e5ff")
        col_active = grid_cfg.get("active_border_color", "#00ff88")
        col_offline = grid_cfg.get("offline_border_color", "#ff4444")

        ai_entries = self.registry.list_modules(kind="ai")

        for uid, t in list(self._tiles.items()):
            unit = self.camera_manager.get_unit(uid)
            if unit is None:
                continue
            gs = unit.get_grid_snapshot()
            cv = t["canvas"]
            iw = max(32, t["inner_w"])
            ih = max(24, t["inner_h"])

            # 1. Колір рамки плитки
            if uid in self.detail_windows:
                bcol = col_active
            elif gs.get("sub_status") in ("offline", "reconnecting") and gs.get("main_status") != "active":
                bcol = col_offline
            else:
                bcol = col_normal
            if bcol != t["last_border"]:
                t["last_border"] = bcol
                t["frame"].configure(bg=bcol)

            # 2. Кадр субпотоку
            rgb = gs.get("rgb")
            seq = int(gs.get("seq", 0))
            if rgb is not None:
                if seq != t["last_seq"] or t["photo_tk"] is None:
                    t["last_seq"] = seq
                    try:
                        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
                        t["photo_tk"] = tk_img
                        cv.itemconfigure(t["img_id"], image=tk_img, state=tk.NORMAL)
                        cv.coords(t["img_id"], iw // 2, ih // 2)
                        cv.itemconfigure(t["ph_id"], state=tk.HIDDEN)
                    except Exception:
                        pass
            else:
                cv.itemconfigure(t["img_id"], state=tk.HIDDEN)
                st_map = {
                    "offline": (tr("ui.main.tile.offline", "No signal (offline)"), "#ff6666"),
                    "reconnecting": (tr("ui.main.tile.reconnecting", "Reconnecting sub-stream..."), "#ffcc00"),
                    "connecting": (tr("ui.main.tile.connecting_sub", "Connecting sub-stream..."), "#ffcc00"),
                    "fallback_main": (tr("ui.main.tile.waiting_main", "Waiting for main stream..."), "#ffcc00"),
                }
                p_txt, p_col = st_map.get(
                    gs.get("sub_status", ""),
                    (tr("ui.main.tile.starting", "Starting..."), "#9ba3b8"),
                )
                cv.itemconfigure(t["ph_id"], text=p_txt, fill=p_col, state=tk.NORMAL)
                cv.coords(t["ph_id"], iw // 2, ih // 2)

            # 3. Оверлей назви камери (верхній лівий кут)
            if show_labels:
                cam_nm = str(gs.get("name") or "").strip() or tr("camera.default_single", "Camera")
                name_txt = f"📹 {cam_nm}"
                if uid in self.detail_windows:
                    name_txt += "  " + tr("ui.main.tile.open_tag", "[OPEN]")
                if t["last_lbl_text"] != name_txt:
                    t["last_lbl_text"] = name_txt
                    cv.itemconfigure(t["lbl_txt_id"], text=name_txt, state=tk.NORMAL)
                    cv.coords(t["lbl_txt_id"], 10, 8)
                    bbox = cv.bbox(t["lbl_txt_id"])
                    if bbox:
                        cv.coords(t["lbl_bg_id"], bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3)
                        cv.itemconfigure(t["lbl_bg_id"], state=tk.NORMAL)
            else:
                if t["last_lbl_text"] != "":
                    t["last_lbl_text"] = ""
                    cv.itemconfigure(t["lbl_txt_id"], state=tk.HIDDEN)
                    cv.itemconfigure(t["lbl_bg_id"], state=tk.HIDDEN)

            # 4. Оверлей статусу запису, динамічних AI-модулів і FPS (верхній правий кут)
            if show_status or show_fps:
                parts = []
                rec_on = bool(gs.get("recording"))
                op_mode = str(gs.get("operation_mode") or "server")
                if show_status:
                    if rec_on:
                        enc = gs.get("encoder") or "rec"
                        parts.append(f"● REC ({enc})")
                    elif op_mode == "operator":
                        parts.append(tr("ui.main.tile.no_rec_badge", "👤 NO RECORDING"))
                    m_on = gs.get("modules_on") or {}
                    m_tags = []
                    for ent in ai_entries:
                        mid = ent["id"]
                        if m_on.get(mid):
                            raw_badge = ent["spec"].get("short_badge")
                            if raw_badge != "":
                                default_badge = raw_badge if raw_badge is not None else mid[:2].upper()
                                badge = tr(f"mod.{mid}.short_badge", str(default_badge))
                                if badge:
                                    m_tags.append(str(badge))
                    if m_tags:
                        parts.append(" ".join(m_tags))
                if show_fps and gs.get("sub_status") == "active":
                    parts.append(f"{float(gs.get('fps', 0.0)):.1f} FPS")

                st_txt = "  |  ".join(parts)
                st_col = "#75e08b" if rec_on else ("#ffcc00" if op_mode == "operator" else "#9ee8ff")
                st_key = (st_txt, st_col, iw)
                if st_txt:
                    if t["last_st_key"] != st_key:
                        t["last_st_key"] = st_key
                        cv.itemconfigure(t["st_txt_id"], text=st_txt, fill=st_col, state=tk.NORMAL)
                        cv.coords(t["st_txt_id"], iw - 10, 8)
                        bbox = cv.bbox(t["st_txt_id"])
                        if bbox:
                            cv.coords(t["st_bg_id"], bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3)
                            cv.itemconfigure(t["st_bg_id"], state=tk.NORMAL)
                else:
                    if t["last_st_key"] != ("", "", iw):
                        t["last_st_key"] = ("", "", iw)
                        cv.itemconfigure(t["st_txt_id"], state=tk.HIDDEN)
                        cv.itemconfigure(t["st_bg_id"], state=tk.HIDDEN)
            else:
                if t["last_st_key"] is not None:
                    t["last_st_key"] = None
                    cv.itemconfigure(t["st_txt_id"], state=tk.HIDDEN)
                    cv.itemconfigure(t["st_bg_id"], state=tk.HIDDEN)

            # 5. Нижній бейдж при втраті зв'язку (коли останній кадр ще висить)
            warn_txt = ""
            if rgb is not None and gs.get("sub_status") in ("offline", "reconnecting"):
                warn_txt = tr("ui.main.tile.stream_lost", "⚠ Stream lost (reconnecting...)")
            if t["last_warn_text"] != (warn_txt, ih):
                t["last_warn_text"] = (warn_txt, ih)
                if warn_txt:
                    cv.itemconfigure(t["warn_txt_id"], text=warn_txt, state=tk.NORMAL)
                    cv.coords(t["warn_txt_id"], 10, ih - 8)
                    bbox = cv.bbox(t["warn_txt_id"])
                    if bbox:
                        cv.coords(t["warn_bg_id"], bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3)
                        cv.itemconfigure(t["warn_bg_id"], state=tk.NORMAL)
                else:
                    cv.itemconfigure(t["warn_txt_id"], state=tk.HIDDEN)
                    cv.itemconfigure(t["warn_bg_id"], state=tk.HIDDEN)

            cv.tag_raise("ov_lbl")
            cv.tag_raise("ov_st")
            cv.tag_raise("ov_warn")

        return interval_ms

    def _render_grid_tick(self):
        if self._is_closing:
            return
        interval_ms = self._render_tiles_once()
        if not self._is_closing:
            self._render_after_id = self.root.after(interval_ms, self._render_grid_tick)

    # ------------------------------------------------------------------
    # Телеметрія апаратних ресурсів у фоні (без блокування UI)
    # ------------------------------------------------------------------
    def _start_hw_poll_thread(self):
        def _loop():
            while self._hw_poll_running and not self._is_closing:
                try:
                    self._cached_hw_metrics = self.hw_monitor.get_live_metrics()
                except Exception:
                    pass
                for _ in range(15):
                    if not self._hw_poll_running or self._is_closing:
                        break
                    time.sleep(0.1)

        threading.Thread(target=_loop, daemon=True).start()

    def _render_hw_badge_once(self):
        if self._is_closing or not hasattr(self, "lbl_hw_badge"):
            return
        if self._cuda_status_note:
            try:
                self.lbl_hw_badge.configure(text=self._cuda_status_note, fg=self._cuda_note_color)
            except Exception:
                pass
            return
        m = self._cached_hw_metrics
        if m is None:
            try:
                self.lbl_hw_badge.configure(
                    text=tr("ui.main.hw_badge_init", "CPU: --% | RAM: -- | AI: CPU"),
                    fg="#9ee8ff",
                )
            except Exception:
                pass
            return
        try:
            cpu_pct = float(m["cpu"]["usage_pct"])
            ram_used = float(m["ram"]["used_gb"])
            ram_tot = float(m["ram"]["total_gb"])
            ai_sum = self.inference_pool.get_active_summary()
            gpu_bit = ""
            for g in (m.get("gpus") or []):
                if g.get("live_metrics") and g.get("gpu_util_pct") is not None:
                    gpu_bit = f" | GPU: {g['gpu_util_pct']:.0f}%"
                    break
            self.lbl_hw_badge.configure(
                text=tr(
                    "ui.main.hw_badge_fmt",
                    "CPU: {cpu:.0f}%{gpu_bit} | RAM: {ram_used:.1f}/{ram_tot:.1f} GB | AI: {ai}",
                    cpu=cpu_pct,
                    gpu_bit=gpu_bit,
                    ram_used=ram_used,
                    ram_tot=ram_tot,
                    ai=ai_sum,
                ),
                fg="#9ee8ff",
            )
        except Exception:
            pass

    def _refresh_hw_badge_tick(self):
        if self._is_closing:
            return
        self._render_hw_badge_once()
        if not self._is_closing:
            self._hw_after_id = self.root.after(1500, self._refresh_hw_badge_tick)

    # ------------------------------------------------------------------
    # Завершення роботи
    # ------------------------------------------------------------------
    def on_close(self, exit_app: bool = True):
        if self._is_closing:
            return
        self._is_closing = True
        self._hw_poll_running = False

        for attr in (
            "_start_after_id",
            "_render_after_id",
            "_hw_after_id",
            "_hide_bar_after_id",
            "_relayout_after_id",
            "_module_poll_after_id",
            "_ui_poll_after_id",
            "_cuda_note_clear_after_id",
        ):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)

        if self.settings_win is not None:
            try:
                self.settings_win.close()
            except Exception:
                pass
            self.settings_win = None

        if self.people_win is not None:
            try:
                self.people_win.close()
            except Exception:
                pass
            self.people_win = None

        for dwin in list(self.detail_windows.values()):
            try:
                dwin.close()
            except Exception:
                pass
        self.detail_windows.clear()

        try:
            self.geom_tracker.close()
        except Exception:
            pass

        try:
            self.camera_manager.stop_all()
        except Exception as e:
            print(f"[Shutdown CameraManager error]: {e}")

        try:
            self.root.destroy()
        except Exception:
            pass

        if exit_app:
            sys.exit(0)


def main():
    if not getattr(sys, "frozen", False):
        try:
            os.chdir(BASE_DIR)
        except Exception:
            pass
    root = tk.Tk()
    MainGridWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
