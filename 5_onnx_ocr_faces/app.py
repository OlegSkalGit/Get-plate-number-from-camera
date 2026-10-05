import os
import sys
import time
import configparser
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from plates import PlateEngine
from faces import FaceEngine

# Базова папка додатку
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE_DIR)

# Системний реєстр Windows
try:
    import winreg
except ImportError:
    winreg = None

# Безпечне визначення назв подій Tkinter
EVT_MOTION = "<" + "Motion" + ">"
EVT_TREE_SELECT = "<<" + "TreeviewSelect" + ">>"
EVT_DOUBLE_CLICK = "<" + "Double-1" + ">"
EVT_BTN_PRESS = "<" + "ButtonPress-1" + ">"
EVT_B1_MOTION = "<" + "B1-Motion" + ">"
EVT_BTN_RELEASE = "<" + "ButtonRelease-1" + ">"
EVT_ADMIN_EXIT_UP = "<" + "Control-Alt-Shift-KeyPress-Q" + ">"
EVT_ADMIN_EXIT_LOW = "<" + "Control-Alt-Shift-KeyPress-q" + ">"

EVT_MOUSEWHEEL = "<" + "MouseWheel" + ">"
EVT_BTN4 = "<" + "Button-4" + ">"
EVT_BTN5 = "<" + "Button-5" + ">"
EVT_ENTER = "<" + "Enter" + ">"
EVT_LEAVE = "<" + "Leave" + ">"

LOCKDOWN_KEYS = [
    "<" + "Alt-F4" + ">",
    "<" + "Alt-KeyPress-F4" + ">",
    "<" + "Control-w" + ">",
    "<" + "Control-W" + ">",
    "<" + "Control-q" + ">",
    "<" + "Control-Q" + ">",
    "<" + "Escape" + ">",
    "<" + "F11" + ">",
]


class ANPRViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANPR & Face Video Monitor")
        self.root.configure(bg="#0d0d11")

        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self._prevent_close)
        self._setup_lockdown_bindings()

        self.config_ini_path = os.path.join(BASE_DIR, "config.ini")
        self.config = self._load_config()

        self.plates_on = bool(self.config.get("plates_on", True))
        self.faces_on = bool(self.config.get("faces_on", True))
        self.is_autostart_active = bool(self.config.get("autostart_os_and_play", False))
        self.view_plate_time = float(self.config.get("view_plate_time", 30))
        self.delete_older_days = float(self.config.get("delete_older_days", 1))

        self.min_percent_to_save = int(self.config.get("min_percent_to_save", 80))
        self.sleep_time_after_save = float(self.config.get("sleep_time_after_save", 5.0))
        self.rect_scale_plate = float(self.config.get("rect_scale_plate", 5.0))

        self.roi_user_top = int(self.config.get("rect_scale_user_top", 20))
        self.roi_user_left = int(self.config.get("rect_scale_user_left", 20))
        self.roi_user_width = int(self.config.get("rect_scale_user_width", 360))
        self.roi_user_height = int(self.config.get("rect_scale_user_height", 240))

        self._sync_windows_startup(self.is_autostart_active)

        # 1. Ініціалізація автономних рушіїв
        self.plate_engine = None
        if self.plates_on:
            try:
                self.plate_engine = PlateEngine(BASE_DIR, self.config, on_saved_callback=self._on_plate_saved)
            except Exception as e:
                messagebox.showerror("Помилка модуля номерів", f"Не вдалося ініціалізувати PlateEngine:\n{e}")

        self.face_engine = None
        if self.faces_on:
            try:
                self.face_engine = FaceEngine(BASE_DIR, self.config, on_saved_callback=self._on_face_saved)
            except Exception as e:
                messagebox.showerror("Помилка модуля облич", f"Не вдалося ініціалізувати FaceEngine:\n{e}")

        # Синхронізація потоків
        self.is_running = False
        self.video_thread = None
        self.ai_thread = None

        self.latest_frame = None
        self.new_frame_available = False
        self.frame_lock = threading.Lock()

        self.ai_frame = None
        self.ai_event = threading.Event()
        self.ai_lock = threading.Lock()
        self.is_ai_busy = False

        # Кеш кадрів для відображення
        self.cached_plate_boxes = []
        self.cached_face_boxes = []
        self.last_plate_img = None
        self.last_plate_dims = (0, 0)
        self.last_plate_text = ""
        self.last_plate_conf = 0.0
        self.last_plate_time = 0.0

        self.sidebar_visible = False
        self.sidebar_mode = "plates" if self.plates_on else "faces"
        self.preview_photo_tk = None
        self.screen_photo_tk = None
        self.roi_photo_tk = None
        self.tree_item_map = {}

        self.raw_records = []
        self.sort_mode = 0

        self.current_screen_img_cv = None
        self.last_roi_crop_bgr = None
        self.canvas_scale_x = 1.0
        self.canvas_scale_y = 1.0
        self.canvas_disp_w = 340
        self.canvas_disp_h = 190
        self.drag_start_x = None
        self.drag_start_y = None

        self.panel_visible = True
        self.hide_timer = None

        self._build_ui()
        self.root.bind_all(EVT_MOTION, self._on_mouse_motion)

        self.root.after(5000, self._check_and_delete_old_files)

        if self.is_autostart_active:
            self.root.after(300, self.toggle_stream)

    def _setup_lockdown_bindings(self):
        for key in LOCKDOWN_KEYS:
            self.root.bind_all(key, lambda e: "break")
        self.root.bind_all(EVT_ADMIN_EXIT_UP, self._admin_exit)
        self.root.bind_all(EVT_ADMIN_EXIT_LOW, self._admin_exit)

    def _prevent_close(self):
        pass

    def _admin_exit(self, event=None):
        self.on_close()

    def _sync_windows_startup(self, enable: bool):
        if winreg is None or sys.platform != "win32":
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
            launch_cmd = f'"{py_exe}" "{os.path.abspath(__file__)}"'

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

    def _load_config(self):
        defaults = {
            "source": "0",
            "autostart_os_and_play": False,
            "plates_on": True,
            "faces_on": True,
            "plates_debug": False,          # <--- ДОДАНО
            "face_min_similarity": 0.7,
            "face_det_thresh": 0.45,
            "view_plate_time": 30,
            "delete_older_days": 1,
            "min_percent_to_save": 80,
            "sleep_time_after_save": 5.0,
            "rect_scale_plate": 5.0,
            "rect_scale_user_top": 20,
            "rect_scale_user_left": 20,
            "rect_scale_user_width": 360,
            "rect_scale_user_height": 240
        }
        parser = configparser.ConfigParser()
        if os.path.exists(self.config_ini_path):
            try:
                parser.read(self.config_ini_path, encoding="utf-8")
                if not parser.has_section("SETTINGS"):
                    parser.add_section("SETTINGS")
                needs_saving = False

                for key, val in defaults.items():
                    if not parser.has_option("SETTINGS", key):
                        parser.set("SETTINGS", key, str(val).lower() if isinstance(val, bool) else str(val))
                        needs_saving = True

                cfg = {
                    "source": parser.get("SETTINGS", "source", fallback="0").strip().strip('\'"'),
                    "autostart_os_and_play": parser.getboolean("SETTINGS", "autostart_os_and_play", fallback=False),
                    "plates_on": parser.getboolean("SETTINGS", "plates_on", fallback=True),
                    "faces_on": parser.getboolean("SETTINGS", "faces_on", fallback=True),
                    "plates_debug": parser.getboolean("SETTINGS", "plates_debug", fallback=False),  # <--- ДОДАНО
                    "face_min_similarity": parser.getfloat("SETTINGS", "face_min_similarity", fallback=0.7),
                    "face_det_thresh": parser.getfloat("SETTINGS", "face_det_thresh", fallback=0.45),
                    "view_plate_time": parser.getint("SETTINGS", "view_plate_time", fallback=30),
                    "delete_older_days": parser.getfloat("SETTINGS", "delete_older_days", fallback=1.0),
                    "min_percent_to_save": parser.getint("SETTINGS", "min_percent_to_save", fallback=80),
                    "sleep_time_after_save": parser.getfloat("SETTINGS", "sleep_time_after_save", fallback=5.0),
                    "rect_scale_plate": parser.getfloat("SETTINGS", "rect_scale_plate", fallback=5.0),
                    "rect_scale_user_top": parser.getint("SETTINGS", "rect_scale_user_top", fallback=20),
                    "rect_scale_user_left": parser.getint("SETTINGS", "rect_scale_user_left", fallback=20),
                    "rect_scale_user_width": parser.getint("SETTINGS", "rect_scale_user_width", fallback=360),
                    "rect_scale_user_height": parser.getint("SETTINGS", "rect_scale_user_height", fallback=240)
                }
                if needs_saving:
                    self._save_config_ini(cfg)
                return cfg
            except Exception:
                self._save_config_ini(defaults)
                return defaults
        self._save_config_ini(defaults)
        return defaults

    def _save_config_ini(self, data):
        parser = configparser.ConfigParser()
        parser["SETTINGS"] = {
            "source": str(data.get("source", "0")),
            "autostart_os_and_play": str(data.get("autostart_os_and_play", False)).lower(),
            "plates_on": str(data.get("plates_on", True)).lower(),
            "faces_on": str(data.get("faces_on", True)).lower(),
            "plates_debug": str(data.get("plates_debug", False)).lower(),   # <--- ДОДАНО
            "face_min_similarity": str(data.get("face_min_similarity", 0.7)),
            "face_det_thresh": str(data.get("face_det_thresh", 0.45)),
            "view_plate_time": str(data.get("view_plate_time", 30)),
            "delete_older_days": str(data.get("delete_older_days", 1)),
            "min_percent_to_save": str(data.get("min_percent_to_save", 80)),
            "sleep_time_after_save": str(data.get("sleep_time_after_save", 5.0)),
            "rect_scale_plate": str(data.get("rect_scale_plate", 5.0)),
            "rect_scale_user_top": str(int(data.get("rect_scale_user_top", 20))),
            "rect_scale_user_left": str(int(data.get("rect_scale_user_left", 20))),
            "rect_scale_user_width": str(int(data.get("rect_scale_user_width", 360))),
            "rect_scale_user_height": str(int(data.get("rect_scale_user_height", 240)))
        }
        try:
            with open(self.config_ini_path, "w", encoding="utf-8") as f:
                f.write("; ANPR & Face Video Monitor Configuration\n")
                f.write("; plates_on: true / false (фіксація номерних знаків)\n")
                f.write("; faces_on: true / false (фіксація облич)\n")
                f.write("; plates_debug: true / false (збереження дебаг-логів)\n")  # <--- ДОДАНО
                f.write("; face_min_similarity: поріг схожості векторів облич (0.7 = 70%)\n")
                f.write("; face_det_thresh: поріг детекції облич\n")
                f.write("; source: 0, rtsp://... або d:\\video.mp4\n")
                f.write("; autostart_os_and_play: true / false\n")
                f.write("; view_plate_time: час збереження номера на екрані (с)\n")
                f.write("; delete_older_days: видаляти файли старше зазначених днів\n")
                f.write("; min_percent_to_save: мінімальна точність розпізнавання для збереження (%)\n")
                f.write("; sleep_time_after_save: таймаут ігнорування дублікатів того самого номера (с)\n")
                f.write("; rect_scale_plate: масштаб номера у правому верхньому кутку (базове 5.0)\n")
                f.write("; rect_scale_user_top, left, width, height: геометрія вікна виділення\n\n")
                parser.write(f)
        except Exception:
            pass

    def _save_roi_geometry_to_config(self):
        self.config["rect_scale_user_top"] = self.roi_user_top
        self.config["rect_scale_user_left"] = self.roi_user_left
        self.config["rect_scale_user_width"] = self.roi_user_width
        self.config["rect_scale_user_height"] = self.roi_user_height
        self._save_config_ini(self.config)

    def _check_and_delete_old_files(self):
        try:
            if self.delete_older_days > 0:
                cutoff = time.time() - (self.delete_older_days * 86400.0)
                dirs_to_clean = []
                if self.plate_engine:
                    dirs_to_clean.append(self.plate_engine.plates_dir)
                if self.face_engine:
                    dirs_to_clean.append(self.face_engine.faces_dir)

                for base_d in dirs_to_clean:
                    if not os.path.exists(base_d):
                        continue
                    for root_dir, dirs, files in os.walk(base_d, topdown=False):
                        for fname in files:
                            if fname == "faces.txt":
                                continue
                            fpath = os.path.join(root_dir, fname)
                            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                                try:
                                    os.remove(fpath)
                                except Exception:
                                    pass
                        if root_dir != base_d and not os.listdir(root_dir):
                            try:
                                os.rmdir(root_dir)
                            except Exception:
                                pass
        except Exception:
            pass
        self.root.after(1800000, self._check_and_delete_old_files)

    def _build_ui(self):
        self.ctrl_frame = tk.Frame(self.root, bg="#1e1e24", padx=12, pady=10)
        self.ctrl_frame.pack(fill=tk.X, side=tk.TOP)

        tk.Label(self.ctrl_frame, text="Відеокамера / Джерело:", fg="#e0e0e0", bg="#1e1e24", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(5, 3))
        tk.Button(self.ctrl_frame, text="?", command=self._show_help, bg="#3a3a46", fg="#00ffff", font=("Segoe UI", 9, "bold"), width=2, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 8))

        self.source_var = tk.StringVar(value=str(self.config.get("source", "0")))
        ttk.Entry(self.ctrl_frame, width=34, textvariable=self.source_var, state="readonly").pack(side=tk.LEFT, padx=3)

        self.autostart_var = tk.BooleanVar(value=self.is_autostart_active)
        tk.Checkbutton(self.ctrl_frame, text="Автозапуск", variable=self.autostart_var, state=tk.DISABLED, disabledforeground="#ffffff", bg="#1e1e24", selectcolor="#2b2b36", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=8)

        # Кнопки перегляду бази: пакуються тільки якщо відповідний модуль активовано
        self.btn_saved_plates = tk.Button(
            self.ctrl_frame,
            text="🚘 Номери",
            command=lambda: self.toggle_side_panel("plates"),
            bg="#3a3a46",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        if self.plates_on:
            self.btn_saved_plates.pack(side=tk.LEFT, padx=4)

        self.btn_saved_faces = tk.Button(
            self.ctrl_frame,
            text="👤 Обличчя",
            command=lambda: self.toggle_side_panel("faces"),
            bg="#3a3a46",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        if self.faces_on:
            self.btn_saved_faces.pack(side=tk.LEFT, padx=4)

        if self.is_autostart_active:
            self.btn_toggle = tk.Button(self.ctrl_frame, text="Працює", command=self.toggle_stream, state=tk.DISABLED, disabledforeground="#75e08b", bg="#1b3822", font=("Segoe UI", 10, "bold"), width=10, relief=tk.FLAT)
        else:
            self.btn_toggle = tk.Button(self.ctrl_frame, text="Старт", command=self.toggle_stream, bg="#28a745", fg="white", font=("Segoe UI", 10, "bold"), width=10, relief=tk.FLAT, cursor="hand2")
        self.btn_toggle.pack(side=tk.LEFT, padx=8)

        self.status_lbl = tk.Label(self.ctrl_frame, text="Статус: Очікування", fg="#ffcc00", bg="#1e1e24", font=("Segoe UI", 10))
        self.status_lbl.pack(side=tk.RIGHT, padx=10)

        self.main_body = tk.Frame(self.root, bg="#0d0d11")
        self.main_body.pack(fill=tk.BOTH, expand=True)

        # Бічна панель
        self.side_panel = tk.Frame(self.main_body, bg="#1e1e24", width=360)
        self.side_panel.pack_propagate(False)

        side_header = tk.Frame(self.side_panel, bg="#282832", padx=10, pady=6)
        side_header.pack(fill=tk.X, side=tk.TOP)
        self.side_header_lbl = tk.Label(side_header, text="Історія номерів", fg="#00e5ff", bg="#282832", font=("Segoe UI", 10, "bold"))
        self.side_header_lbl.pack(side=tk.LEFT)
        tk.Button(side_header, text="↻", command=self.refresh_saved_list, bg="#3a3a48", fg="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), width=3, cursor="hand2").pack(side=tk.RIGHT)

        search_frame = tk.Frame(self.side_panel, bg="#22222c", padx=6, pady=6)
        search_frame.pack(fill=tk.X, side=tk.TOP)

        tk.Label(search_frame, text="🔍", fg="#888899", bg="#22222c", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(2, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._apply_search_and_sort())
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var, bg="#14141a", fg="#ffffff", insertbackground="white", bd=1, relief=tk.FLAT, font=("Segoe UI", 9))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=2)

        btn_clear = tk.Button(search_frame, text="✕", command=self._clear_search, bg="#2d2d38", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=2, cursor="hand2")
        btn_clear.pack(side=tk.LEFT, padx=(0, 4))

        self.btn_sort = tk.Button(search_frame, text="▼ Час", command=self._toggle_sort, bg="#333342", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=5, cursor="hand2")
        self.btn_sort.pack(side=tk.RIGHT)

        tree_frame = tk.Frame(self.side_panel, bg="#16161d", padx=6, pady=4)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("ProDark.Treeview", background="#16161d", foreground="#ffffff", fieldbackground="#16161d", font=("Segoe UI", 10), rowheight=26, borderwidth=0)
        style.map("ProDark.Treeview", background=[("selected", "#007acc")], foreground=[("selected", "#ffffff")])

        self.items_tree = ttk.Treeview(tree_frame, style="ProDark.Treeview", show="tree", selectmode="browse", yscrollcommand=scrollbar.set)
        self.items_tree.column("#0", width=330, stretch=True)
        self.items_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.items_tree.yview)

        self.items_tree.tag_configure("day_tag", font=("Segoe UI", 10, "bold"), foreground="#00e5ff")
        self.items_tree.tag_configure("item_tag", font=("Consolas", 10, "bold"), foreground="#75e08b")
        self.items_tree.bind(EVT_TREE_SELECT, self._on_item_selected)
        self.items_tree.bind(EVT_DOUBLE_CLICK, self._on_item_double_clicked)

        for w in (self.items_tree, scrollbar, tree_frame):
            w.bind(EVT_ENTER, self._bind_tree_mousewheel)
            w.bind(EVT_LEAVE, self._unbind_tree_mousewheel)

        self.preview_container = tk.Frame(self.side_panel, bg="#14141a", pady=6, padx=8)
        self.preview_container.pack(fill=tk.X, side=tk.BOTTOM)

        self.preview_info_lbl = tk.Label(self.preview_container, text="Знімок об'єкта:", fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9))
        self.preview_info_lbl.pack(anchor=tk.W, pady=(0, 2))

        self.preview_label = tk.Label(self.preview_container, bg="#1a1a22", text="[Виберіть запис зі списку]", fg="#707080", font=("Segoe UI", 9), pady=10, relief=tk.SOLID, borderwidth=1)
        self.preview_label.pack(fill=tk.X)

        self.screen_info_lbl = tk.Label(self.preview_container, text="Фото скріну (виділіть область мишкою):", fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9))
        self.screen_info_lbl.pack(anchor=tk.W, pady=(6, 2))

        self.screen_canvas = tk.Canvas(self.preview_container, bg="#1a1a22", width=340, height=190, highlightthickness=1, highlightbackground="#333340", cursor="cross")
        self.screen_canvas.pack(fill=tk.X, pady=(0, 2))
        self.screen_canvas.bind(EVT_BTN_PRESS, self._on_screen_canvas_press)
        self.screen_canvas.bind(EVT_B1_MOTION, self._on_screen_canvas_motion)
        self.screen_canvas.bind(EVT_BTN_RELEASE, self._on_screen_canvas_release)

        for pw in (self.preview_container, self.preview_label, self.screen_canvas):
            pw.bind(EVT_ENTER, self._bind_preview_mousewheel)
            pw.bind(EVT_LEAVE, self._unbind_preview_mousewheel)

        self.video_container = tk.Frame(self.main_body, bg="#0d0d11")
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.main_video_label = tk.Label(self.video_container, bg="#0d0d11")
        self.main_video_label.pack(fill=tk.BOTH, expand=True)

        self.main_video_label.bind(EVT_BTN_PRESS, self._on_video_area_click)
        self.video_container.bind(EVT_BTN_PRESS, self._on_video_area_click)

        # Інтерактивне вікно ROI
        self.roi_panel = tk.Frame(self.video_container, bg="#16161e", bd=2, relief=tk.SOLID, highlightbackground="#00ff00", highlightthickness=1)
        
        self.roi_header = tk.Frame(self.roi_panel, bg="#20202c", padx=6, pady=3, cursor="fleur")
        self.roi_header.pack(fill=tk.X, side=tk.TOP)
        
        self.roi_title_lbl = tk.Label(self.roi_header, text="Виділена область", fg="#00e5ff", bg="#20202c", font=("Segoe UI", 9, "bold"), cursor="fleur")
        self.roi_title_lbl.pack(side=tk.LEFT)
        
        btn_close = tk.Button(self.roi_header, text="✕", command=self._hide_roi_panel, bg="#333342", fg="#ff5555", font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, padx=4, cursor="hand2")
        btn_close.pack(side=tk.RIGHT)

        for w in (self.roi_header, self.roi_title_lbl):
            w.bind(EVT_BTN_PRESS, self._on_roi_drag_start)
            w.bind(EVT_B1_MOTION, self._on_roi_drag_motion)
            w.bind(EVT_BTN_RELEASE, self._on_roi_drag_release)

        self.roi_img_label = tk.Label(self.roi_panel, bg="#0d0d11")
        self.roi_img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.roi_resize_grip = tk.Label(self.roi_panel, text="◢", fg="#00e5ff", bg="#16161e", font=("Segoe UI", 10, "bold"), cursor="sizing")
        self.roi_resize_grip.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-1, y=-1)
        self.roi_resize_grip.bind(EVT_BTN_PRESS, self._on_roi_resize_start)
        self.roi_resize_grip.bind(EVT_B1_MOTION, self._on_roi_resize_motion)
        self.roi_resize_grip.bind(EVT_BTN_RELEASE, self._on_roi_resize_release)

    def _bind_tree_mousewheel(self, event=None):
        self.root.bind_all(EVT_MOUSEWHEEL, self._on_tree_mousewheel)
        self.root.bind_all(EVT_BTN4, self._on_tree_mousewheel)
        self.root.bind_all(EVT_BTN5, self._on_tree_mousewheel)

    def _unbind_tree_mousewheel(self, event=None):
        self.root.unbind_all(EVT_MOUSEWHEEL)
        self.root.unbind_all(EVT_BTN4)
        self.root.unbind_all(EVT_BTN5)

    def _on_tree_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta:
            delta = int(-1 * (event.delta / 120))
        else:
            delta = 0

        if getattr(event, "state", 0) & 0x0001:
            self.items_tree.yview_scroll(delta * 2, "units")
        else:
            self._navigate_items(delta)
        return "break"

    def _bind_preview_mousewheel(self, event=None):
        self.root.bind_all(EVT_MOUSEWHEEL, self._on_preview_mousewheel)
        self.root.bind_all(EVT_BTN4, self._on_preview_mousewheel)
        self.root.bind_all(EVT_BTN5, self._on_preview_mousewheel)

    def _unbind_preview_mousewheel(self, event=None):
        self.root.unbind_all(EVT_MOUSEWHEEL)
        self.root.unbind_all(EVT_BTN4)
        self.root.unbind_all(EVT_BTN5)

    def _on_preview_mousewheel(self, event):
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        elif event.delta:
            delta = int(-1 * (event.delta / 120))
        else:
            delta = 0

        if delta != 0:
            self._navigate_items(delta)
        return "break"

    def _navigate_items(self, direction):
        all_items = []
        for day in self.items_tree.get_children():
            for child in self.items_tree.get_children(day):
                if child in self.tree_item_map:
                    all_items.append(child)
        if not all_items:
            return

        sel = self.items_tree.selection()
        if sel and sel[0] in all_items:
            curr_idx = all_items.index(sel[0])
            new_idx = max(0, min(len(all_items) - 1, curr_idx + direction))
        else:
            new_idx = 0 if direction > 0 else len(all_items) - 1

        target = all_items[new_idx]
        self.items_tree.selection_set(target)
        self.items_tree.focus(target)
        self.items_tree.see(target)
        self._on_item_selected(None)

    def _on_video_area_click(self, event=None):
        if hasattr(self, "roi_panel") and self.roi_panel.winfo_ismapped():
            self._hide_roi_panel()

    def _on_roi_drag_start(self, event):
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        self._orig_left = self.roi_user_left
        self._orig_top = self.roi_user_top

    def _on_roi_drag_motion(self, event):
        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y
        self.roi_user_left = max(0, self._orig_left + dx)
        self.roi_user_top = max(0, self._orig_top + dy)
        self.roi_panel.place(x=self.roi_user_left, y=self.roi_user_top, width=self.roi_user_width, height=self.roi_user_height)

    def _on_roi_drag_release(self, event):
        self._save_roi_geometry_to_config()

    def _on_roi_resize_start(self, event):
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._orig_w = self.roi_user_width
        self._orig_h = self.roi_user_height

    def _on_roi_resize_motion(self, event):
        dx = event.x_root - self._resize_start_x
        dy = event.y_root - self._resize_start_y
        self.roi_user_width = max(180, self._orig_w + dx)
        self.roi_user_height = max(120, self._orig_h + dy)
        self.roi_panel.place(x=self.roi_user_left, y=self.roi_user_top, width=self.roi_user_width, height=self.roi_user_height)
        self._render_current_roi_crop()

    def _on_roi_resize_release(self, event):
        self._save_roi_geometry_to_config()

    def toggle_side_panel(self, mode="plates"):
        if self.sidebar_visible and self.sidebar_mode == mode:
            self.side_panel.pack_forget()
            self.video_container.pack_forget()
            self.video_container.pack(fill=tk.BOTH, expand=True)
            self.sidebar_visible = False
            if self.plates_on:
                self.btn_saved_plates.configure(bg="#3a3a46")
            if self.faces_on:
                self.btn_saved_faces.configure(bg="#3a3a46")
        else:
            self.sidebar_mode = mode
            self.video_container.pack_forget()
            self.side_panel.pack(side=tk.LEFT, fill=tk.Y)
            self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.sidebar_visible = True

            if mode == "plates":
                if self.plates_on:
                    self.btn_saved_plates.configure(bg="#007acc")
                if self.faces_on:
                    self.btn_saved_faces.configure(bg="#3a3a46")
                self.side_header_lbl.configure(text="Історія номерів")
            else:
                if self.faces_on:
                    self.btn_saved_faces.configure(bg="#007acc")
                if self.plates_on:
                    self.btn_saved_plates.configure(bg="#3a3a46")
                self.side_header_lbl.configure(text="Історія облич")

            self.refresh_saved_list()

    def _clear_search(self):
        self.search_var.set("")

    def _toggle_sort(self):
        self.sort_mode = (self.sort_mode + 1) % 3
        if self.sort_mode == 0:
            self.btn_sort.configure(text="▼ Час")
        elif self.sort_mode == 1:
            self.btn_sort.configure(text="▲ Час")
        else:
            self.btn_sort.configure(text="🔤 Назва")
        self._apply_search_and_sort()

    def refresh_saved_list(self):
        self.raw_records.clear()
        target_dir = None
        if self.sidebar_mode == "plates" and self.plate_engine:
            target_dir = self.plate_engine.plates_dir
        elif self.sidebar_mode == "faces" and self.face_engine:
            target_dir = self.face_engine.faces_dir

        if not target_dir or not os.path.exists(target_dir):
            self._apply_search_and_sort()
            return

        for root_dir, dirs, files in os.walk(target_dir):
            for fname in files:
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                if "_full" in fname.lower() or fname.endswith("_debug.txt"):
                    continue

                fpath = os.path.join(root_dir, fname)
                base_name, _ = os.path.splitext(fname)
                rel_dir = os.path.relpath(root_dir, target_dir)

                day_key = rel_dir if rel_dir != "." else datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d")

                if "_plate_" in base_name:
                    parts = base_name.split("_plate_")
                    time_disp = ":".join(parts[0].split("-")[:3])
                    rest = parts[1].rsplit("_", 1)
                    item_name = rest[0] if len(rest) == 2 else parts[1]
                elif "_face_" in base_name:
                    parts = base_name.split("_face_")
                    time_disp = ":".join(parts[0].split("-")[:3])
                    rest = parts[1].rsplit("_", 1)
                    item_name = rest[0] if len(rest) == 2 else parts[1]
                else:
                    time_disp = "--:--:--"
                    item_name = base_name

                self.raw_records.append({
                    "day": day_key,
                    "time": time_disp,
                    "name": item_name,
                    "fpath": fpath,
                    "mtime": os.path.getmtime(fpath)
                })

        self._apply_search_and_sort()

    def _apply_search_and_sort(self):
        self.items_tree.delete(*self.items_tree.get_children())
        self.tree_item_map.clear()

        query = self.search_var.get().strip().upper()

        filtered = []
        for r in self.raw_records:
            if query:
                if (query not in r["name"].upper()) and (query not in r["time"]) and (query not in r["day"]):
                    continue
            filtered.append(r)

        if self.sort_mode == 0:
            filtered.sort(key=lambda x: x["mtime"], reverse=True)
        elif self.sort_mode == 1:
            filtered.sort(key=lambda x: x["mtime"], reverse=False)
        else:
            filtered.sort(key=lambda x: x["name"].upper())

        grouped = {}
        for r in filtered:
            d = r["day"]
            if d not in grouped:
                grouped[d] = []
            grouped[d].append(r)

        icon = "🚘" if self.sidebar_mode == "plates" else "👤"
        for idx, day_str in enumerate(sorted(grouped.keys(), reverse=(self.sort_mode != 1))):
            day_node = self.items_tree.insert("", "end", text=f"📁 {day_str}", tags=("day_tag",), open=(idx == 0))
            for rec in grouped[day_str]:
                disp_str = f"🕒 {rec['time']}   {icon} {rec['name']}"
                item_id = self.items_tree.insert(day_node, "end", text=f"  {disp_str}", tags=("item_tag",))
                self.tree_item_map[item_id] = rec["fpath"]

    def _on_plate_saved(self, day_str, time_str, plate_str, fpath):
        if self.sidebar_visible and self.sidebar_mode == "plates":
            self.root.after(0, self.refresh_saved_list)

    def _on_face_saved(self, day_str, time_str, face_name, fpath):
        if self.sidebar_visible and self.sidebar_mode == "faces":
            self.root.after(0, self.refresh_saved_list)

    def _show_item_in_preview(self, fpath):
        if not fpath or not os.path.exists(fpath):
            return
        try:
            pil_img = Image.open(fpath)
            orig_w, orig_h = pil_img.size
            target_w = 340
            target_h = max(1, int(target_w * (orig_h / float(orig_w))))
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            self.preview_photo_tk = ImageTk.PhotoImage(pil_img)
            base = os.path.basename(fpath).replace(".jpg", "")
            lbl_type = "Фото номера:" if self.sidebar_mode == "plates" else "Фото обличчя:"
            self.preview_info_lbl.configure(text=f"{lbl_type} {base}")
            self.preview_label.configure(image=self.preview_photo_tk, text="", pady=0)
        except Exception:
            pass

    def _show_screenshot_in_canvas(self, screen_path):
        self.screen_canvas.delete("all")
        self._hide_roi_panel()
        if not screen_path or not os.path.exists(screen_path):
            self.screen_canvas.create_text(170, 95, text="[Скріншот відсутній]", fill="#707080", font=("Segoe UI", 9))
            self.current_screen_img_cv = None
            return

        try:
            cv_img = cv2.imread(screen_path)
            if cv_img is None:
                return
            self.current_screen_img_cv = cv_img
            orig_h, orig_w = cv_img.shape[:2]

            canvas_w = 340
            aspect = orig_h / float(orig_w) if orig_w > 0 else (9.0 / 16.0)
            canvas_h = max(1, int(round(canvas_w * aspect)))

            self.screen_canvas.config(width=canvas_w, height=canvas_h)
            resized = cv2.resize(cv_img, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.screen_photo_tk = ImageTk.PhotoImage(Image.fromarray(rgb))

            self.screen_canvas.create_image(0, 0, anchor=tk.NW, image=self.screen_photo_tk)
            self.canvas_scale_x = orig_w / float(canvas_w)
            self.canvas_scale_y = orig_h / float(canvas_h)
            self.canvas_disp_w = canvas_w
            self.canvas_disp_h = canvas_h
        except Exception as e:
            print(f"[Screenshot preview error]: {e}")

    def _on_item_selected(self, event):
        sel = self.items_tree.selection()
        if not sel:
            return
        fpath = self.tree_item_map.get(sel[0])
        if not fpath:
            return

        self._show_item_in_preview(fpath)

        item_dir = os.path.dirname(fpath)
        base_name = os.path.basename(fpath)
        time_prefix = base_name.split("_")[0]
        candidate = os.path.join(item_dir, f"{time_prefix}_full.jpg")
        screen_path = candidate if os.path.exists(candidate) else None

        self._show_screenshot_in_canvas(screen_path)

    def _on_item_double_clicked(self, event):
        sel = self.items_tree.selection()
        if not sel:
            return
        fpath = self.tree_item_map.get(sel[0])
        if fpath and os.path.exists(fpath):
            try:
                os.startfile(fpath)
            except Exception:
                pass

    def _on_screen_canvas_press(self, event):
        if self.current_screen_img_cv is None:
            return
        self.drag_start_x = max(0, min(self.canvas_disp_w, event.x))
        self.drag_start_y = max(0, min(self.canvas_disp_h, event.y))
        self.screen_canvas.delete("roi_rect")

    def _on_screen_canvas_motion(self, event):
        if self.drag_start_x is None or self.drag_start_y is None or self.current_screen_img_cv is None:
            return
        cur_x = max(0, min(self.canvas_disp_w, event.x))
        cur_y = max(0, min(self.canvas_disp_h, event.y))
        self.screen_canvas.delete("roi_rect")
        self.screen_canvas.create_rectangle(
            self.drag_start_x, self.drag_start_y, cur_x, cur_y,
            outline="#00ff00", width=2, tags="roi_rect"
        )

    def _on_screen_canvas_release(self, event):
        if self.drag_start_x is None or self.drag_start_y is None or self.current_screen_img_cv is None:
            return

        end_x = max(0, min(self.canvas_disp_w, event.x))
        end_y = max(0, min(self.canvas_disp_h, event.y))

        x1 = min(self.drag_start_x, end_x)
        x2 = max(self.drag_start_x, end_x)
        y1 = min(self.drag_start_y, end_y)
        y2 = max(self.drag_start_y, end_y)

        self.drag_start_x = None
        self.drag_start_y = None

        sel_w = x2 - x1
        sel_h = y2 - y1
        if sel_w < 4 or sel_h < 4:
            return

        orig_h, orig_w = self.current_screen_img_cv.shape[:2]
        crop_x1 = max(0, min(orig_w - 1, int(round(x1 * self.canvas_scale_x))))
        crop_y1 = max(0, min(orig_h - 1, int(round(y1 * self.canvas_scale_y))))
        crop_x2 = max(crop_x1 + 1, min(orig_w, int(round(x2 * self.canvas_scale_x))))
        crop_y2 = max(crop_y1 + 1, min(orig_h, int(round(y2 * self.canvas_scale_y))))

        crop_bgr = self.current_screen_img_cv[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_bgr.size == 0:
            return

        self.last_roi_crop_bgr = crop_bgr.copy()
        
        self.roi_panel.place(x=self.roi_user_left, y=self.roi_user_top, width=self.roi_user_width, height=self.roi_user_height)
        self.roi_panel.lift()
        self._render_current_roi_crop()

    def _render_current_roi_crop(self):
        if self.last_roi_crop_bgr is None:
            return
        h, w = self.last_roi_crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return

        avail_w = max(40, self.roi_user_width - 16)
        avail_h = max(40, self.roi_user_height - 44)

        scale = min(avail_w / float(w), avail_h / float(h))
        disp_w = max(1, int(round(w * scale)))
        disp_h = max(1, int(round(h * scale)))

        resized = cv2.resize(self.last_roi_crop_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.roi_photo_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.roi_img_label.configure(image=self.roi_photo_tk)
        self.roi_title_lbl.configure(text=f"Виділена область ({w}x{h})")

    def _hide_roi_panel(self):
        self.roi_panel.place_forget()

    def _schedule_hide(self, delay=5000):
        self._cancel_hide_timer()
        self.hide_timer = self.root.after(delay, self._hide_panel)

    def _cancel_hide_timer(self):
        if self.hide_timer is not None:
            self.root.after_cancel(self.hide_timer)
            self.hide_timer = None

    def _show_panel(self):
        if not self.panel_visible:
            self.ctrl_frame.pack(fill=tk.X, side=tk.TOP, before=self.main_body)
            self.panel_visible = True

    def _hide_panel(self):
        self.hide_timer = None
        if self.sidebar_visible:
            return
        if self.panel_visible and self.is_running:
            self.ctrl_frame.pack_forget()
            self.panel_visible = False

    def _on_mouse_motion(self, event):
        if not self.is_running:
            return
        if event.y_root <= 15:
            self._show_panel()
            self._cancel_hide_timer()
            return
        if self.panel_visible:
            panel_h = max(self.ctrl_frame.winfo_height(), 55)
            if event.y_root <= panel_h:
                self._cancel_hide_timer()
            else:
                if self.hide_timer is None and not self.sidebar_visible:
                    self._schedule_hide(5000)

    def _show_help(self):
        help_text = (
            "ANPR & Face Video Monitor (Modular Architecture)\n\n"
            f"• Фіксація номерів (plates_ON): {self.plates_on}\n"
            f"• Фіксація облич (faces_ON): {self.faces_on}\n"
            "• Детекція номерів: best.onnx (640x640) + CCT OCR\n"
            "• Детекція облич: SCRFD / InsightFace + AdaFace/MBF (112x112)\n"
            f"• Поріг збереження номерів: >= {self.min_percent_to_save}%\n"
            f"• Поріг схожості облич: >= {int(self.config.get('face_min_similarity', 0.7) * 100)}%\n"
            f"• Ігнорування повторів авто: {self.sleep_time_after_save} с\n"
            f"• Відображення номера на екрані: {self.view_plate_time} с\n"
            "• Збереження номерів: /plates/YYYY-MM-DD/HH-MM-SS_plate_...jpg\n"
            "• Збереження облич: /faces/YYYY-MM-DD/HH-MM-SS_face__Percent%.jpg\n"
            "• База облич: /faces/faces.txt (UniqueFaceID, FaceName, Vector)\n\n"
            "Гарячий вихід: Ctrl + Alt + Shift + Q"
        )
        messagebox.showinfo("Довідка", help_text)

    def toggle_stream(self):
        if not self.is_running:
            raw_source = self.source_var.get().strip().strip('\'"')
            source = int(raw_source) if raw_source.isdigit() else raw_source

            self.is_running = True
            if not self.is_autostart_active:
                self.btn_toggle.configure(text="Стоп", bg="#dc3545")
            else:
                self.btn_toggle.configure(text="Працює", state=tk.DISABLED, disabledforeground="#75e08b", bg="#1b3822")

            self.status_lbl.configure(text="Статус: Активний (Потоки ШІ запущені)", fg="#28a745")

            self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)
            self.ai_thread.start()

            self.video_thread = threading.Thread(target=self._capture_worker, args=(source,), daemon=True)
            self.video_thread.start()

            self.root.after(16, self._render_loop)
            self._schedule_hide(5000)
        else:
            if self.is_autostart_active:
                return
            self._stop_stream()

    def _stop_stream(self):
        self.is_running = False
        self.ai_event.set()
        self.btn_toggle.configure(text="Старт", bg="#28a745")
        self.status_lbl.configure(text="Статус: Зупинено", fg="#ffcc00")
        self._cancel_hide_timer()
        self._show_panel()

    def _ai_worker(self):
        """Паралельний ШІ-потік опитування обох активних модулів."""
        while self.is_running:
            self.ai_event.wait(timeout=0.2)
            if not self.is_running:
                break

            with self.ai_lock:
                frame_to_process = self.ai_frame
                self.ai_frame = None
                self.ai_event.clear()

            if frame_to_process is None:
                continue

            try:
                plate_boxes = []
                overlay_info = None
                if self.plates_on and self.plate_engine is not None:
                    plate_boxes, overlay_info = self.plate_engine.process_frame(frame_to_process)

                face_boxes = []
                if self.faces_on and self.face_engine is not None:
                    face_boxes = self.face_engine.process_frame(frame_to_process)

                with self.frame_lock:
                    self.cached_plate_boxes = plate_boxes
                    self.cached_face_boxes = face_boxes
                    if overlay_info is not None:
                        self.last_plate_img = overlay_info["img"]
                        self.last_plate_dims = overlay_info["dims"]
                        self.last_plate_text = overlay_info["text"]
                        self.last_plate_conf = overlay_info["conf"]
                        self.last_plate_time = overlay_info["time"]

            except Exception as e:
                print(f"[AI Loop Error]: {e}")
            finally:
                self.is_ai_busy = False

    def _capture_worker(self, source):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.root.after(0, lambda: self.status_lbl.configure(text="Помилка джерела!", fg="#ff4444"))
            self.is_running = False
            return

        is_file = isinstance(source, str) and not source.lower().startswith("rtsp://") and not source.lower().startswith("http://")
        target_fps = cap.get(cv2.CAP_PROP_FPS)
        if target_fps <= 0 or target_fps > 120 or np.isnan(target_fps):
            target_fps = 30.0
        frame_delay = 1.0 / target_fps

        while self.is_running and cap.isOpened():
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    continue
                else:
                    time.sleep(0.02)
                    continue

            frame_h, frame_w = frame.shape[:2]
            now = time.time()

            if not self.is_ai_busy:
                self.is_ai_busy = True
                with self.ai_lock:
                    self.ai_frame = frame.copy()
                self.ai_event.set()

            with self.frame_lock:
                plate_boxes_to_draw = list(self.cached_plate_boxes)
                face_boxes_to_draw = list(self.cached_face_boxes)
                plate_img = self.last_plate_img
                plate_dims = self.last_plate_dims
                plate_time = self.last_plate_time

            # Зелені рамки номерних знаків
            for (bx1, by1, bx2, by2) in plate_boxes_to_draw:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            # Блакитні рамки облич з підписом
            for ((fx1, fy1, fx2, fy2), f_name, f_pct) in face_boxes_to_draw:
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 2)
                
                # Безпечна обробка: f_pct може бути 'NEW', '64%', числовим значенням або ''
                pct_str = str(f_pct).strip()
                if pct_str == "NEW":
                    tag = f"{f_name} [NEW]"
                elif pct_str:
                    tag = f"{f_name} {pct_str}" if pct_str.endswith("%") else f"{f_name} {pct_str}%"
                else:
                    tag = f"{f_name}"
                
                cv2.putText(frame, tag, (fx1, max(18, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

            # Оверлей останнього зафіксованого номера у правому верхньому кутку
            if plate_img is not None and (now - plate_time < self.view_plate_time):
                margin = 20
                cur_w, cur_h = plate_dims
                if cur_w > 0 and cur_h > 0:
                    base_scale = max(0.2, self.rect_scale_plate / 5.0)
                    target_ov_w = int(round(cur_w * base_scale))
                    target_ov_h = int(round(cur_h * base_scale))

                    max_allowed_w = int(frame_w * 0.70)
                    if target_ov_w > max_allowed_w and max_allowed_w > 120:
                        fit_scale = max_allowed_w / float(target_ov_w)
                        target_ov_w = int(target_ov_w * fit_scale)
                        target_ov_h = int(target_ov_h * fit_scale)

                    display_img = cv2.resize(plate_img, (target_ov_w, target_ov_h), interpolation=cv2.INTER_LINEAR)
                    dw, dh = display_img.shape[1], display_img.shape[0]
                    border = 3
                    x2_ov = frame_w - margin
                    x1_ov = x2_ov - dw
                    y1_ov = margin
                    y2_ov = y1_ov + dh

                    if x1_ov > 0 and y2_ov < frame_h:
                        cv2.rectangle(frame, (x1_ov - border, y1_ov - border), (x2_ov + border, y2_ov + border), (0, 255, 0), 2)
                        frame[y1_ov:y2_ov, x1_ov:x2_ov] = display_img

            with self.frame_lock:
                self.latest_frame = frame
                self.new_frame_available = True

            if is_file:
                elapsed = time.time() - t_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()

    def _render_loop(self):
        if not self.is_running:
            return

        frame_to_draw = None
        with self.frame_lock:
            if self.new_frame_available and self.latest_frame is not None:
                frame_to_draw = self.latest_frame.copy()
                self.new_frame_available = False

        if frame_to_draw is not None:
            cont_w = self.video_container.winfo_width()
            cont_h = self.video_container.winfo_height()
            lbl_w = cont_w if cont_w > 100 else max(self.main_video_label.winfo_width(), 640)
            lbl_h = cont_h if cont_h > 100 else max(self.main_video_label.winfo_height(), 360)

            h, w = frame_to_draw.shape[:2]
            scale = min(lbl_w / w, lbl_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(frame_to_draw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.img_main_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.main_video_label.configure(image=self.img_main_tk)

            if hasattr(self, "roi_panel") and self.roi_panel.winfo_ismapped():
                self.roi_panel.lift()

        self.root.after(16, self._render_loop)

    def on_close(self):
        self.is_running = False
        self.ai_event.set()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRViewerApp(root)
    root.mainloop()
