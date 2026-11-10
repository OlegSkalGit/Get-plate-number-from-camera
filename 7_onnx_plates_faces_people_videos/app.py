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
from people import PersonEngine
from video import VideoEngine

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
        self.root.title("ANPR, Face, People & Video Monitor")
        self.root.configure(bg="#0d0d11")

        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self._prevent_close)
        self._setup_lockdown_bindings()

        self.config_ini_path = os.path.join(BASE_DIR, "config.ini")
        self.config = self._load_config()

        self.plates_on = bool(self.config.get("plates_on", True))
        self.plates_show_boxes = bool(self.config.get("plates_show_boxes", True))
        self.faces_on = bool(self.config.get("faces_on", True))
        self.faces_show_boxes = bool(self.config.get("faces_show_boxes", True))
        self.people_on = bool(self.config.get("people_on", True))
        self.people_show_boxes = bool(self.config.get("people_show_boxes", True))
        self.video_on = bool(self.config.get("video_on", True))
        self.is_autostart_active = bool(self.config.get("autostart_os_and_play", False))

        # Окреме автовидалення для кожного модуля у днях (0 = необмежено)
        self.plates_delete_older_days = float(self.config.get("plates_delete_older_days", 0))
        self.faces_delete_older_days = float(self.config.get("faces_delete_older_days", 0))
        self.people_delete_older_days = float(self.config.get("people_delete_older_days", 0))
        self.video_delete_older_days = float(self.config.get("video_delete_older_days", 0))

        # Межі дат роботи кожного модуля (None = необмежено)
        self.module_dates = {
            "plates": (
                self._parse_date_str(self.config.get("plates_date_from", "")),
                self._parse_date_str(self.config.get("plates_date_to", "")),
            ),
            "faces": (
                self._parse_date_str(self.config.get("faces_date_from", "")),
                self._parse_date_str(self.config.get("faces_date_to", "")),
            ),
            "people": (
                self._parse_date_str(self.config.get("people_date_from", "")),
                self._parse_date_str(self.config.get("people_date_to", "")),
            ),
            "video": (
                self._parse_date_str(self.config.get("video_date_from", "")),
                self._parse_date_str(self.config.get("video_date_to", "")),
            ),
        }

        # Модуль номерних знаків (plates_)
        self.view_plate_time = float(self.config.get("plates_view_time", 30))
        self.min_percent_to_save = int(self.config.get("plates_min_percent_to_save", 80))
        self.sleep_time_after_save = float(self.config.get("plates_sleep_time_after_save", 5.0))
        self.rect_scale_plate = float(self.config.get("plates_rect_scale", 5.0))
        self.roi_user_top = int(self.config.get("plates_rect_user_top", 20))
        self.roi_user_left = int(self.config.get("plates_rect_user_left", 20))
        self.roi_user_width = int(self.config.get("plates_rect_user_width", 360))
        self.roi_user_height = int(self.config.get("plates_rect_user_height", 240))

        # Модуль облич (faces_)
        self.face_min_similarity = float(self.config.get("faces_min_similarity", 0.48))
        self.face_det_thresh = float(self.config.get("faces_det_thresh", 0.45))

        # Модуль людей та транспорту (people_)
        self.people_det_thresh = float(self.config.get("people_det_thresh", 0.45))
        self.people_timer_ms = int(self.config.get("people_timer_ms", 1000))
        val_veh = str(self.config.get("people_detect_transport", "true")).strip().lower()
        self.people_detect_transport = val_veh in ("true", "1", "yes", "on")

        # Модуль відеозапису (video_)
        self.video_duration_min = float(self.config.get("video_duration_min", 5.0))
        self.video_fps = float(self.config.get("video_fps", 15.0))
        self.video_codec = str(self.config.get("video_codec", "libx264"))
        self.video_crf = int(self.config.get("video_crf", 28))

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

        self.person_engine = None
        if self.people_on:
            try:
                self.person_engine = PersonEngine(BASE_DIR, self.config, on_saved_callback=self._on_person_saved)
            except Exception as e:
                messagebox.showerror("Помилка модуля людей", f"Не вдалося ініціалізувати PersonEngine:\n{e}")

        self.video_engine = None
        if self.video_on:
            try:
                self.video_engine = VideoEngine(BASE_DIR, self.config, on_saved_callback=self._on_video_saved)
            except Exception as e:
                messagebox.showerror("Помилка модуля відеозапису", f"Не вдалося ініціалізувати VideoEngine:\n{e}")

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
        self.cached_person_boxes = []
        self.last_plate_img = None
        self.last_plate_dims = (0, 0)
        self.last_plate_text = ""
        self.last_plate_conf = 0.0
        self.last_plate_time = 0.0

        self.sidebar_visible = False
        if self.plates_on:
            self.sidebar_mode = "plates"
        elif self.faces_on:
            self.sidebar_mode = "faces"
        elif self.people_on:
            self.sidebar_mode = "people"
        elif self.video_on:
            self.sidebar_mode = "video"
        else:
            self.sidebar_mode = "plates"

        self.preview_photo_tk = None
        self.screen_photo_tk = None
        self.roi_photo_tk = None
        self.tree_item_map = {}

        self.raw_records = []
        self.sort_mode = 0

        self.current_screen_img_cv = None
        self.current_preview_fpath = None
        self.last_roi_crop_bgr = None
        self.current_roi_coords = None
        self.canvas_scale_x = 1.0
        self.canvas_scale_y = 1.0
        self.canvas_disp_w = 340
        self.canvas_disp_h = 190
        self.drag_start_x = None
        self.drag_start_y = None

        # Вбудований плеєр архівного відео
        self.arch_cap = None
        self.arch_fpath = None
        self.arch_playing = False
        self.arch_fps = 15.0
        self.arch_total_frames = 0
        self.arch_curr_frame_idx = 0
        self.arch_speed = 1
        self.arch_after_id = None
        self.arch_seeking = False

        # Повноекранне вікно перегляду (на поточному моніторі)
        self.fs_win = None
        self.fs_canvas = None
        self.fs_photo_tk = None
        self.fs_crop_mode = False
        self.fs_scale_x = 1.0
        self.fs_scale_y = 1.0
        self.fs_img_offset_x = 0
        self.fs_img_offset_y = 0
        self.fs_disp_w = 1
        self.fs_disp_h = 1
        self.fs_drag_start_x = None
        self.fs_drag_start_y = None

        self.panel_visible = True
        self.hide_timer = None

        self._build_ui()
        self.root.bind_all(EVT_MOTION, self._on_mouse_motion)

        # Періодичне очищення старих файлів раз на добу (24 години = 86 400 000 мс, не при старті)
        self.root.after(86400000, self._check_and_delete_old_files)

        if self.is_autostart_active:
            self.root.after(300, self.toggle_stream)

    @staticmethod
    def _parse_date_str(val):
        s = str(val or "").strip()
        if not s or s.lower() in ("none", "unlimited", "null", "*"):
            return None
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    def _is_module_date_active(self, mod_name: str) -> bool:
        d_from, d_to = self.module_dates.get(mod_name, (None, None))
        if d_from is None and d_to is None:
            return True
        today = datetime.now().date()
        if d_from is not None and today < d_from:
            return False
        if d_to is not None and today > d_to:
            return False
        return True

    def _format_date_range_str(self, mod_name: str) -> str:
        d_from, d_to = self.module_dates.get(mod_name, (None, None))
        if d_from is None and d_to is None:
            return "Необмежено"
        s_from = d_from.strftime("%Y-%m-%d") if d_from else "..."
        s_to = d_to.strftime("%Y-%m-%d") if d_to else "..."
        return f"{s_from} → {s_to}"

    def _get_current_monitor_geometry(self, widget=None):
        """Визначає (x, y, width, height) саме того монітора, на якому знаходиться вікно/віджет."""
        target = widget if widget is not None else self.root
        try:
            self.root.update_idletasks()
        except Exception:
            pass

        if sys.platform == "win32":
            try:
                import ctypes
                from ctypes import wintypes

                class RECT(ctypes.Structure):
                    _fields_ = [
                        ("left", wintypes.LONG),
                        ("top", wintypes.LONG),
                        ("right", wintypes.LONG),
                        ("bottom", wintypes.LONG),
                    ]

                class MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", wintypes.DWORD),
                        ("rcMonitor", RECT),
                        ("rcWork", RECT),
                        ("dwFlags", wintypes.DWORD),
                    ]

                user32 = ctypes.windll.user32
                hwnd = target.winfo_id()
                hmon = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
                if hmon:
                    mi = MONITORINFO()
                    mi.cbSize = ctypes.sizeof(MONITORINFO)
                    if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                        rc = mi.rcMonitor
                        mw = rc.right - rc.left
                        mh = rc.bottom - rc.top
                        if mw > 0 and mh > 0:
                            return rc.left, rc.top, mw, mh
            except Exception:
                pass

        return (
            self.root.winfo_rootx(),
            self.root.winfo_rooty(),
            max(640, self.root.winfo_width()),
            max(480, self.root.winfo_height()),
        )

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

    @staticmethod
    def _fmt_num(val):
        try:
            f = float(val)
            return str(int(f)) if f.is_integer() else str(f)
        except Exception:
            return str(val)

    def _load_config(self):
        defaults = {
            "source": "0",
            "autostart_os_and_play": False,
            # plates_
            "plates_on": True,
            "plates_show_boxes": True,
            "plates_debug": False,
            "plates_det_thresh": 0.18,
            "plates_min_percent_to_save": 80,
            "plates_sleep_time_after_save": 5.0,
            "plates_view_time": 30,
            "plates_rect_scale": 5.0,
            "plates_rect_user_top": 20,
            "plates_rect_user_left": 20,
            "plates_rect_user_width": 360,
            "plates_rect_user_height": 240,
            "plates_delete_older_days": 0,
            "plates_date_from": "",
            "plates_date_to": "",
            # faces_
            "faces_on": True,
            "faces_show_boxes": True,
            "faces_min_similarity": 0.48,
            "faces_det_thresh": 0.45,
            "faces_min_size": 20,
            "faces_delete_older_days": 0,
            "faces_date_from": "",
            "faces_date_to": "",
            # people_
            "people_on": True,
            "people_show_boxes": True,
            "people_detect_transport": True,
            "people_det_thresh": 0.45,
            "people_timer_ms": 1000,
            "people_delete_older_days": 0,
            "people_date_from": "",
            "people_date_to": "",
            # video_
            "video_on": True,
            "video_duration_min": 5,
            "video_fps": 15,
            "video_codec": "libx264",
            "video_crf": 28,
            "video_delete_older_days": 0,
            "video_date_from": "",
            "video_date_to": "",
        }
        parser = configparser.ConfigParser()
        if os.path.exists(self.config_ini_path):
            try:
                parser.read(self.config_ini_path, encoding="utf-8")
                if not parser.has_section("SETTINGS"):
                    parser.add_section("SETTINGS")

                def _get_bool(key, def_val):
                    if parser.has_option("SETTINGS", key):
                        return parser.getboolean("SETTINGS", key, fallback=def_val)
                    return def_val

                def _get_float(key, def_val):
                    if parser.has_option("SETTINGS", key):
                        return parser.getfloat("SETTINGS", key, fallback=def_val)
                    return def_val

                def _get_int(key, def_val):
                    if parser.has_option("SETTINGS", key):
                        return parser.getint("SETTINGS", key, fallback=def_val)
                    return def_val

                def _get_str(key, def_val):
                    if parser.has_option("SETTINGS", key):
                        return parser.get("SETTINGS", key, fallback=def_val).strip().strip('\'"')
                    return def_val

                cfg = {
                    "source": _get_str("source", "0"),
                    "autostart_os_and_play": _get_bool("autostart_os_and_play", False),

                    # plates_
                    "plates_on": _get_bool("plates_on", True),
                    "plates_show_boxes": _get_bool("plates_show_boxes", True),
                    "plates_debug": _get_bool("plates_debug", False),
                    "plates_det_thresh": _get_float("plates_det_thresh", 0.18),
                    "plates_min_percent_to_save": _get_int("plates_min_percent_to_save", 80),
                    "plates_sleep_time_after_save": _get_float("plates_sleep_time_after_save", 5.0),
                    "plates_view_time": _get_int("plates_view_time", 30),
                    "plates_rect_scale": _get_float("plates_rect_scale", 5.0),
                    "plates_rect_user_top": _get_int("plates_rect_user_top", 20),
                    "plates_rect_user_left": _get_int("plates_rect_user_left", 20),
                    "plates_rect_user_width": _get_int("plates_rect_user_width", 360),
                    "plates_rect_user_height": _get_int("plates_rect_user_height", 240),
                    "plates_delete_older_days": max(0.0, _get_float("plates_delete_older_days", 0.0)),
                    "plates_date_from": _get_str("plates_date_from", ""),
                    "plates_date_to": _get_str("plates_date_to", ""),

                    # faces_
                    "faces_on": _get_bool("faces_on", True),
                    "faces_show_boxes": _get_bool("faces_show_boxes", True),
                    "faces_min_similarity": _get_float("faces_min_similarity", 0.48),
                    "faces_det_thresh": _get_float("faces_det_thresh", 0.45),
                    "faces_min_size": _get_int("faces_min_size", 20),
                    "faces_delete_older_days": max(0.0, _get_float("faces_delete_older_days", 0.0)),
                    "faces_date_from": _get_str("faces_date_from", ""),
                    "faces_date_to": _get_str("faces_date_to", ""),

                    # people_
                    "people_on": _get_bool("people_on", True),
                    "people_show_boxes": _get_bool("people_show_boxes", True),
                    "people_detect_transport": _get_bool("people_detect_transport", True),
                    "people_det_thresh": _get_float("people_det_thresh", 0.45),
                    "people_timer_ms": _get_int("people_timer_ms", 1000),
                    "people_delete_older_days": max(0.0, _get_float("people_delete_older_days", 0.0)),
                    "people_date_from": _get_str("people_date_from", ""),
                    "people_date_to": _get_str("people_date_to", ""),

                    # video_
                    "video_on": _get_bool("video_on", True),
                    "video_duration_min": max(0.05, _get_float("video_duration_min", 5.0)),
                    "video_fps": max(1.0, min(60.0, _get_float("video_fps", 15.0))),
                    "video_codec": _get_str("video_codec", "libx264") or "libx264",
                    "video_crf": _get_int("video_crf", 28),
                    "video_delete_older_days": max(0.0, _get_float("video_delete_older_days", 0.0)),
                    "video_date_from": _get_str("video_date_from", ""),
                    "video_date_to": _get_str("video_date_to", ""),
                }

                needs_saving = False
                for key in defaults:
                    if not parser.has_option("SETTINGS", key):
                        needs_saving = True
                        break

                if needs_saving:
                    self._save_config_ini(cfg)
                return cfg
            except Exception:
                self._save_config_ini(defaults)
                return defaults
        self._save_config_ini(defaults)
        return defaults

    def _save_config_ini(self, data):
        lines = [
            "; =====================================================================",
            "; ANPR, Face, People & Video Monitor Configuration",
            "; =====================================================================",
            ";",
            "; ЗАГАЛЬНІ НАЛАШТУВАННЯ:",
            ";   source                       : 0 (вебкамера), rtsp://... або шлях до файлу .mp4",
            ";   autostart_os_and_play        : true / false (додати у автозапуск Windows та автостарт)",
            ";",
            "; ПРАВИЛА АВТОВИДАЛЕННЯ ТА МЕЖ ДАТ (окремо для кожного модуля):",
            ";   *_delete_older_days          : через скільки днів автоматично видаляти старі файли",
            ";                                  (таймер перевіряє 1 раз на добу).",
            ";                                  За замовчуванням 0 = НЕОБМЕЖЕНО (нічого не видаляється).",
            ";                                  Можна вказувати дробові числа (наприклад: 0.5, 3, 7, 30).",
            ";",
            ";   *_date_from, *_date_to       : календарний період роботи модуля (формат YYYY-MM-DD).",
            ";                                  Поза вказаним діапазоном модуль ПОВНІСТЮ вимикається:",
            ";                                  не детектує, не записує відео і не зберігає файли на диск.",
            ";                                  Якщо обидва поля порожні — працює НЕОБМЕЖЕНО (постійно).",
            ";                                  Можна задавати лише початок або лише кінець періоду",
            ";                                  (наприклад: video_date_from = 2026-11-10, video_date_to = ).",
            ";",
            "; ---------------------------------------------------------------------",
            "; 1. МОДУЛЬ НОМЕРІВ (plates_):",
            ";   plates_on                    : true / false (увімкнути розпізнавання номерних знаків)",
            ";   plates_show_boxes            : true / false (показувати зелені рамки навколо номерів)",
            ";   plates_debug                 : true / false (збереження текстових логів *_debug.txt)",
            ";   plates_det_thresh            : поріг впевненості детектора номерів (0.18 оптимально)",
            ";   plates_min_percent_to_save   : мінімальна точність OCR для збереження (%)",
            ";   plates_sleep_time_after_save : пауза ігнорування повторів того самого номера (сек)",
            ";   plates_view_time             : скільки секунд показувати збільшений номер у кутку екрану",
            ";   plates_rect_scale            : масштаб збільшеного номера в кутку (базове 5.0)",
            ";   plates_rect_user_top, left, width, height : геометрія вікна збільшеного кропу",
            ";   plates_delete_older_days     : автовидалення фото номерів старше N днів (0 = необмежено)",
            ";   plates_date_from, plates_date_to : межі дат роботи модуля (порожньо = необмежено)",
            ";",
            "; ---------------------------------------------------------------------",
            "; 2. МОДУЛЬ ОБЛИЧ (faces_):",
            ";   faces_on                     : true / false (увімкнути розпізнавання облич)",
            ";   faces_show_boxes             : true / false (показувати блакитні рамки та підписи облич)",
            ";   faces_min_similarity         : поріг схожості векторів облич (0.48 = 48%, оптимально 0.45-0.50)",
            ";   faces_det_thresh             : поріг детектора облич (0.45)",
            ";   faces_min_size               : мінімальний розмір обличчя в px (20 для далеких, 45 стандарт)",
            ";   faces_delete_older_days      : автовидалення фото облич старше N днів (0 = необмежено)",
            ";   faces_date_from, faces_date_to : межі дат роботи модуля (порожньо = необмежено)",
            ";",
            "; ---------------------------------------------------------------------",
            "; 3. МОДУЛЬ ЛЮДЕЙ ТА ТРАНСПОРТУ (people_):",
            ";   people_on                    : true / false (увімкнути детекцію людей/транспорту та скріншоти)",
            ";   people_show_boxes            : true / false (показувати рамки людей та транспорту на відео)",
            ";   people_detect_transport      : true / false (також фіксувати авто, автобуси, вантажівки, мото)",
            ";   people_det_thresh            : поріг впевненості детектора YOLO11n (0.45)",
            ";   people_timer_ms              : інтервал зйомки чистих скріншотів у мс (за замовч. 1000)",
            ";   people_delete_older_days     : автовидалення скріншотів старше N днів (0 = необмежено)",
            ";   people_date_from, people_date_to : межі дат роботи модуля (порожньо = необмежено)",
            ";",
            "; ---------------------------------------------------------------------",
            "; 4. МОДУЛЬ ВІДЕОЗАПИСУ (video_):",
            ";   video_on                     : true / false (увімкнути фоновий запис чистого відео без рамок)",
            ";   video_duration_min           : тривалість одного відеофайлу-фрагмента у хвилинах (за замовч. 5)",
            ";   video_fps                    : частота кадрів запису (за замовч. 15 к/с)",
            ";   video_codec                  : libx264 (imageio-ffmpeg preset ultrafast; резерв mp4v)",
            ";   video_crf                    : якість/стиснення libx264 (18-32, 28 оптимально; менше = краще)",
            ";   video_delete_older_days      : автовидалення відео старше N днів (0 = необмежено)",
            ";   video_date_from, video_date_to : межі дат роботи відеозапису (порожньо = необмежено)",
            "; =====================================================================",
            "",
            "[SETTINGS]",
            "; --- ЗАГАЛЬНІ ---",
            f"source = {str(data.get('source', '0'))}",
            f"autostart_os_and_play = {str(bool(data.get('autostart_os_and_play', False))).lower()}",
            "",
            "; --- МОДУЛЬ НОМЕРІВ (plates_) ---",
            f"plates_on = {str(bool(data.get('plates_on', True))).lower()}",
            f"plates_show_boxes = {str(bool(data.get('plates_show_boxes', True))).lower()}",
            f"plates_debug = {str(bool(data.get('plates_debug', False))).lower()}",
            f"plates_det_thresh = {data.get('plates_det_thresh', 0.18)}",
            f"plates_min_percent_to_save = {int(data.get('plates_min_percent_to_save', 80))}",
            f"plates_sleep_time_after_save = {self._fmt_num(data.get('plates_sleep_time_after_save', 5.0))}",
            f"plates_view_time = {self._fmt_num(data.get('plates_view_time', 30))}",
            f"plates_rect_scale = {self._fmt_num(data.get('plates_rect_scale', 5.0))}",
            f"plates_rect_user_top = {int(data.get('plates_rect_user_top', 20))}",
            f"plates_rect_user_left = {int(data.get('plates_rect_user_left', 20))}",
            f"plates_rect_user_width = {int(data.get('plates_rect_user_width', 360))}",
            f"plates_rect_user_height = {int(data.get('plates_rect_user_height', 240))}",
            f"plates_delete_older_days = {self._fmt_num(data.get('plates_delete_older_days', 0))}",
            f"plates_date_from = {str(data.get('plates_date_from', '') or '').strip()}".rstrip(),
            f"plates_date_to = {str(data.get('plates_date_to', '') or '').strip()}".rstrip(),
            "",
            "; --- МОДУЛЬ ОБЛИЧ (faces_) ---",
            f"faces_on = {str(bool(data.get('faces_on', True))).lower()}",
            f"faces_show_boxes = {str(bool(data.get('faces_show_boxes', True))).lower()}",
            f"faces_min_similarity = {data.get('faces_min_similarity', 0.48)}",
            f"faces_det_thresh = {data.get('faces_det_thresh', 0.45)}",
            f"faces_min_size = {int(data.get('faces_min_size', 20))}",
            f"faces_delete_older_days = {self._fmt_num(data.get('faces_delete_older_days', 0))}",
            f"faces_date_from = {str(data.get('faces_date_from', '') or '').strip()}".rstrip(),
            f"faces_date_to = {str(data.get('faces_date_to', '') or '').strip()}".rstrip(),
            "",
            "; --- МОДУЛЬ ЛЮДЕЙ ТА ТРАНСПОРТУ (people_) ---",
            f"people_on = {str(bool(data.get('people_on', True))).lower()}",
            f"people_show_boxes = {str(bool(data.get('people_show_boxes', True))).lower()}",
            f"people_detect_transport = {str(bool(data.get('people_detect_transport', True))).lower()}",
            f"people_det_thresh = {data.get('people_det_thresh', 0.45)}",
            f"people_timer_ms = {int(data.get('people_timer_ms', 1000))}",
            f"people_delete_older_days = {self._fmt_num(data.get('people_delete_older_days', 0))}",
            f"people_date_from = {str(data.get('people_date_from', '') or '').strip()}".rstrip(),
            f"people_date_to = {str(data.get('people_date_to', '') or '').strip()}".rstrip(),
            "",
            "; --- МОДУЛЬ ВІДЕОЗАПИСУ (video_) ---",
            f"video_on = {str(bool(data.get('video_on', True))).lower()}",
            f"video_duration_min = {self._fmt_num(data.get('video_duration_min', 5))}",
            f"video_fps = {self._fmt_num(data.get('video_fps', 15))}",
            f"video_codec = {str(data.get('video_codec', 'libx264') or 'libx264')}",
            f"video_crf = {int(data.get('video_crf', 28))}",
            f"video_delete_older_days = {self._fmt_num(data.get('video_delete_older_days', 0))}",
            f"video_date_from = {str(data.get('video_date_from', '') or '').strip()}".rstrip(),
            f"video_date_to = {str(data.get('video_date_to', '') or '').strip()}".rstrip(),
            "",
        ]
        try:
            with open(self.config_ini_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

    def _save_roi_geometry_to_config(self):
        self.config["plates_rect_user_top"] = self.roi_user_top
        self.config["plates_rect_user_left"] = self.roi_user_left
        self.config["plates_rect_user_width"] = self.roi_user_width
        self.config["plates_rect_user_height"] = self.roi_user_height
        self._save_config_ini(self.config)

    def _check_and_delete_old_files(self):
        try:
            now_ts = time.time()
            plates_dir = self.plate_engine.plates_dir if self.plate_engine else os.path.join(BASE_DIR, "plates")
            faces_dir = self.face_engine.faces_dir if self.face_engine else os.path.join(BASE_DIR, "faces")
            people_dir = self.person_engine.people_dir if self.person_engine else os.path.join(BASE_DIR, "people")
            video_dir = self.video_engine.video_dir if self.video_engine else os.path.join(BASE_DIR, "video")

            cleanup_targets = [
                (plates_dir, self.plates_delete_older_days),
                (faces_dir, self.faces_delete_older_days),
                (people_dir, self.people_delete_older_days),
                (video_dir, self.video_delete_older_days),
            ]

            for base_d, days in cleanup_targets:
                if days <= 0 or not os.path.exists(base_d):
                    continue
                cutoff = now_ts - (days * 86400.0)
                for root_dir, dirs, files in os.walk(base_d, topdown=False):
                    for fname in files:
                        if fname == "faces.txt":
                            continue
                        fpath = os.path.join(root_dir, fname)
                        if self.video_engine and self.video_engine.is_file_busy(fpath):
                            continue
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
        # Періодичний запуск таймера очищення 1 раз на добу (24 години = 86 400 000 мс)
        self.root.after(86400000, self._check_and_delete_old_files)

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

        btn_people_text = "🚶🚗 Люди/Транспорт" if self.people_detect_transport else "🚶 Люди"
        self.btn_saved_people = tk.Button(
            self.ctrl_frame,
            text=btn_people_text,
            command=lambda: self.toggle_side_panel("people"),
            bg="#3a3a46",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=16 if self.people_detect_transport else 10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        if self.people_on:
            self.btn_saved_people.pack(side=tk.LEFT, padx=4)

        self.btn_saved_video = tk.Button(
            self.ctrl_frame,
            text="🎬 Відео",
            command=lambda: self.toggle_side_panel("video"),
            bg="#3a3a46",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
            width=10,
            relief=tk.FLAT,
            cursor="hand2"
        )
        if self.video_on:
            self.btn_saved_video.pack(side=tk.LEFT, padx=4)

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
        self.preview_label.bind(EVT_DOUBLE_CLICK, lambda e: self._open_fullscreen_preview(crop_mode=False, source_widget=self.preview_label))

        self.screen_header_frame = tk.Frame(self.preview_container, bg="#14141a")
        self.screen_header_frame.pack(fill=tk.X, pady=(6, 2))

        self.screen_info_lbl = tk.Label(self.screen_header_frame, text="Фото скріну (кроп / 2x клік):", fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9))
        self.screen_info_lbl.pack(side=tk.LEFT)

        self.btn_expand_preview = tk.Button(
            self.screen_header_frame,
            text="⛶ Розгорнути",
            command=lambda: self._open_fullscreen_preview(crop_mode=False, source_widget=self.screen_canvas),
            bg="#2d2d3a",
            fg="#00e5ff",
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
            padx=6,
            pady=1,
            cursor="hand2"
        )
        self.btn_expand_preview.pack(side=tk.RIGHT)

        self.screen_canvas = tk.Canvas(self.preview_container, bg="#1a1a22", width=340, height=190, highlightthickness=1, highlightbackground="#333340", cursor="cross")
        self.screen_canvas.pack(fill=tk.X, pady=(0, 2))
        self.screen_canvas_img_id = None
        self.drag_moved = False
        self.screen_canvas.bind(EVT_BTN_PRESS, self._on_screen_canvas_press)
        self.screen_canvas.bind(EVT_B1_MOTION, self._on_screen_canvas_motion)
        self.screen_canvas.bind(EVT_BTN_RELEASE, self._on_screen_canvas_release)
        self.screen_canvas.bind(EVT_DOUBLE_CLICK, lambda e: self._open_fullscreen_preview(crop_mode=False, source_widget=self.screen_canvas))

        # Панель керування вбудованим відеоплеєром (показується лише у вкладці "Відео")
        self.video_controls_frame = tk.Frame(self.preview_container, bg="#1b1b26", padx=4, pady=4)
        self.btn_arch_play = tk.Button(
            self.video_controls_frame,
            text="▶",
            command=self._toggle_arch_play,
            bg="#28a745",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            width=3,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_arch_play.pack(side=tk.LEFT, padx=(0, 4))

        self.arch_slider_var = tk.IntVar(value=0)
        self.arch_updating_slider = False
        self.arch_slider = tk.Scale(
            self.video_controls_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.arch_slider_var,
            showvalue=False,
            bg="#1b1b26",
            fg="#00e5ff",
            troughcolor="#2e2e3d",
            highlightthickness=0,
            bd=0,
            sliderlength=14,
            command=self._on_arch_slider_move
        )
        self.arch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.lbl_arch_time = tk.Label(
            self.video_controls_frame,
            text="00:00/00:00",
            fg="#c8c8d8",
            bg="#1b1b26",
            font=("Consolas", 8)
        )
        self.lbl_arch_time.pack(side=tk.LEFT, padx=(4, 4))

        self.btn_arch_speed = tk.Button(
            self.video_controls_frame,
            text="1x",
            command=self._cycle_arch_speed,
            bg="#333344",
            fg="#00e5ff",
            font=("Segoe UI", 8, "bold"),
            width=3,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.btn_arch_speed.pack(side=tk.RIGHT)

        for pw in (self.preview_container, self.preview_label, self.screen_canvas):
            pw.bind(EVT_ENTER, self._bind_preview_mousewheel)
            pw.bind(EVT_LEAVE, self._unbind_preview_mousewheel)

        self.video_container = tk.Frame(self.main_body, bg="#0d0d11")
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.main_video_label = tk.Label(self.video_container, bg="#0d0d11")
        self.main_video_label.pack(fill=tk.BOTH, expand=True)

        self.main_video_label.bind(EVT_BTN_PRESS, self._on_video_area_click)
        self.video_container.bind(EVT_BTN_PRESS, self._on_video_area_click)

        # Інтерактивне вікно ROI (кроп зі скрінів або живий кроп відео)
        self.roi_panel = tk.Frame(self.video_container, bg="#16161e", bd=2, relief=tk.SOLID, highlightbackground="#00ff00", highlightthickness=1)

        self.roi_header = tk.Frame(self.roi_panel, bg="#20202c", padx=6, pady=3, cursor="fleur")
        self.roi_header.pack(fill=tk.X, side=tk.TOP)

        self.roi_title_lbl = tk.Label(self.roi_header, text="Виділена область", fg="#00e5ff", bg="#20202c", font=("Segoe UI", 9, "bold"), cursor="fleur")
        self.roi_title_lbl.pack(side=tk.LEFT)

        btn_close = tk.Button(self.roi_header, text="✕", command=self._hide_roi_panel, bg="#333342", fg="#ff5555", font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, padx=4, cursor="hand2")
        btn_close.pack(side=tk.RIGHT)

        self.btn_expand_roi = tk.Button(
            self.roi_header,
            text="⛶",
            command=lambda: self._open_fullscreen_preview(crop_mode=True, source_widget=self.roi_panel),
            bg="#333342",
            fg="#00e5ff",
            font=("Segoe UI", 8, "bold"),
            bd=0,
            relief=tk.FLAT,
            padx=5,
            cursor="hand2"
        )
        self.btn_expand_roi.pack(side=tk.RIGHT, padx=(0, 3))

        for w in (self.roi_header, self.roi_title_lbl):
            w.bind(EVT_BTN_PRESS, self._on_roi_drag_start)
            w.bind(EVT_B1_MOTION, self._on_roi_drag_motion)
            w.bind(EVT_BTN_RELEASE, self._on_roi_drag_release)

        self.roi_img_label = tk.Label(self.roi_panel, bg="#0d0d11", cursor="hand2")
        self.roi_img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.roi_img_label.bind(EVT_DOUBLE_CLICK, lambda e: self._open_fullscreen_preview(crop_mode=True, source_widget=self.roi_panel))

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

    def _update_preview_layout(self, mode):
        """Для номерів/облич — 2 вікна (кроп + повний скрін); для людей та відео — 1 спільне вікно."""
        if mode in ("people", "video"):
            self.preview_info_lbl.pack_forget()
            self.preview_label.pack_forget()
        else:
            if not self.preview_info_lbl.winfo_ismapped():
                self.preview_info_lbl.pack(anchor=tk.W, pady=(0, 2), before=self.screen_header_frame)
            if not self.preview_label.winfo_ismapped():
                self.preview_label.pack(fill=tk.X, before=self.screen_header_frame)

        if mode == "video":
            self.screen_info_lbl.configure(text="Відеоперегляд (кроп / 2x клік):")
            if not self.video_controls_frame.winfo_ismapped():
                self.video_controls_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            self.video_controls_frame.pack_forget()
            self.screen_info_lbl.configure(text="Фото скріну (кроп / 2x клік):")

    def _clear_screen_preview(self):
        self.current_screen_img_cv = None
        self.current_preview_fpath = None
        self.screen_canvas_img_id = None
        self.screen_canvas.delete("all")
        self.screen_canvas.config(width=340, height=190)
        self.screen_canvas.create_text(170, 95, text="[Виберіть запис зі списку]", fill="#707080", font=("Segoe UI", 9), tags="placeholder")

    def toggle_side_panel(self, mode="plates"):
        if self.sidebar_visible and self.sidebar_mode == mode:
            self._stop_archive_player()
            self.side_panel.pack_forget()
            self.video_container.pack_forget()
            self.video_container.pack(fill=tk.BOTH, expand=True)
            self.sidebar_visible = False
            for attr in ("btn_saved_plates", "btn_saved_faces", "btn_saved_people", "btn_saved_video"):
                if hasattr(self, attr):
                    getattr(self, attr).configure(bg="#3a3a46")
        else:
            prev_mode = self.sidebar_mode
            self.sidebar_mode = mode
            if mode != "video":
                self._stop_archive_player()
            self.video_container.pack_forget()
            self.side_panel.pack(side=tk.LEFT, fill=tk.Y)
            self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.sidebar_visible = True

            if self.plates_on and hasattr(self, "btn_saved_plates"):
                self.btn_saved_plates.configure(bg="#007acc" if mode == "plates" else "#3a3a46")
            if self.faces_on and hasattr(self, "btn_saved_faces"):
                self.btn_saved_faces.configure(bg="#007acc" if mode == "faces" else "#3a3a46")
            if self.people_on and hasattr(self, "btn_saved_people"):
                self.btn_saved_people.configure(bg="#007acc" if mode == "people" else "#3a3a46")
            if self.video_on and hasattr(self, "btn_saved_video"):
                self.btn_saved_video.configure(bg="#007acc" if mode == "video" else "#3a3a46")

            if mode == "plates":
                self.side_header_lbl.configure(text="Історія номерів")
            elif mode == "faces":
                self.side_header_lbl.configure(text="Історія облич")
            elif mode == "video":
                self.side_header_lbl.configure(text="Архів відеозаписів")
            else:
                side_title = "Історія: Люди та Транспорт (скріншоти)" if self.people_detect_transport else "Історія: Люди (скріншоти)"
                self.side_header_lbl.configure(text=side_title)

            self._update_preview_layout(mode)
            if prev_mode != mode:
                self._hide_roi_panel(clear_coords=True)
                self._clear_screen_preview()
                self.items_tree.delete(*self.items_tree.get_children())
                self.tree_item_map.clear()

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
        if self.sidebar_mode == "plates":
            target_dir = self.plate_engine.plates_dir if self.plate_engine else os.path.join(BASE_DIR, "plates")
        elif self.sidebar_mode == "faces":
            target_dir = self.face_engine.faces_dir if self.face_engine else os.path.join(BASE_DIR, "faces")
        elif self.sidebar_mode == "people":
            target_dir = self.person_engine.people_dir if self.person_engine else os.path.join(BASE_DIR, "people")
        elif self.sidebar_mode == "video":
            target_dir = self.video_engine.video_dir if self.video_engine else os.path.join(BASE_DIR, "video")

        if not target_dir or not os.path.exists(target_dir):
            self._apply_search_and_sort()
            return

        for root_dir, dirs, files in os.walk(target_dir):
            for fname in files:
                fpath = os.path.join(root_dir, fname)
                if self.sidebar_mode == "video":
                    if not fname.lower().endswith((".mp4", ".avi", ".mkv")):
                        continue
                    if self.video_engine and self.video_engine.is_file_busy(fpath):
                        continue
                else:
                    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue
                    if self.sidebar_mode != "people" and ("_full" in fname.lower() or fname.endswith("_debug.txt")):
                        continue

                base_name, _ = os.path.splitext(fname)
                rel_dir = os.path.relpath(root_dir, target_dir)

                try:
                    mtime = os.path.getmtime(fpath)
                except Exception:
                    continue

                day_key = rel_dir if rel_dir != "." else datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

                if self.sidebar_mode == "video":
                    time_parts = base_name.split("_")[0].split("-")
                    time_disp = ":".join(time_parts[:3]) if len(time_parts) >= 3 else "--:--:--"
                    try:
                        mb = os.path.getsize(fpath) / (1024.0 * 1024.0)
                    except Exception:
                        mb = 0.0
                    item_name = f"Відео ({mb:.1f} МБ)"
                elif "_plate_" in base_name:
                    parts = base_name.split("_plate_")
                    time_disp = ":".join(parts[0].split("-")[:3])
                    rest = parts[1].rsplit("_", 1)
                    item_name = rest[0] if len(rest) == 2 else parts[1]
                elif "_face_" in base_name:
                    parts = base_name.split("_face_")
                    time_disp = ":".join(parts[0].split("-")[:3])
                    rest = parts[1].rsplit("_", 1)
                    item_name = rest[0] if len(rest) == 2 else parts[1]
                elif self.sidebar_mode == "people":
                    time_parts = base_name.split("_")[0].split("-")
                    time_disp = ":".join(time_parts[:3])
                    parts = base_name.split("_")
                    if len(parts) >= 2 and not parts[1].isdigit():
                        item_name = parts[1].replace("-", ", ")
                    else:
                        item_name = "Людина в кадрі"
                else:
                    time_disp = "--:--:--"
                    item_name = base_name

                self.raw_records.append({
                    "day": day_key,
                    "time": time_disp,
                    "name": item_name,
                    "fpath": fpath,
                    "mtime": mtime
                })

        self._apply_search_and_sort()

    def _apply_search_and_sort(self):
        # Запам'ятовуємо відкриті папки днів та поточний вибраний запис перед оновленням дерева
        prev_day_nodes = self.items_tree.get_children()
        had_days = bool(prev_day_nodes)
        open_days = set()
        known_days = set()
        for d_id in prev_day_nodes:
            d_txt = self.items_tree.item(d_id, "text")
            if d_txt.startswith("📁 "):
                d_name = d_txt[2:].strip()
                known_days.add(d_name)
                if self.items_tree.item(d_id, "open"):
                    open_days.add(d_name)

        prev_sel = self.items_tree.selection()
        selected_fpath = self.tree_item_map.get(prev_sel[0]) if prev_sel else self.current_preview_fpath

        self.items_tree.delete(*prev_day_nodes)
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

        if self.sidebar_mode == "plates":
            icon = "🚘"
        elif self.sidebar_mode == "faces":
            icon = "👤"
        elif self.sidebar_mode == "video":
            icon = "🎬"
        else:
            icon = "🚶"

        item_to_reselect = None
        for idx, day_str in enumerate(sorted(grouped.keys(), reverse=(self.sort_mode != 1))):
            if not had_days:
                is_open = (idx == 0)
            else:
                is_open = (day_str in open_days) or (idx == 0 and day_str not in known_days)

            day_node = self.items_tree.insert("", "end", text=f"📁 {day_str}", tags=("day_tag",), open=is_open)
            for rec in grouped[day_str]:
                disp_str = f"🕒 {rec['time']}   {icon} {rec['name']}"
                item_id = self.items_tree.insert(day_node, "end", text=f"  {disp_str}", tags=("item_tag",))
                self.tree_item_map[item_id] = rec["fpath"]
                if selected_fpath and rec["fpath"] == selected_fpath:
                    item_to_reselect = item_id
                    self.items_tree.item(day_node, open=True)

        if item_to_reselect is not None:
            self.items_tree.selection_set(item_to_reselect)
            self.items_tree.focus(item_to_reselect)

    def _on_plate_saved(self, day_str, time_str, plate_str, fpath):
        if self.sidebar_visible and self.sidebar_mode == "plates":
            self.root.after(0, self.refresh_saved_list)

    def _on_face_saved(self, day_str, time_str, face_name, fpath):
        if self.sidebar_visible and self.sidebar_mode == "faces":
            self.root.after(0, self.refresh_saved_list)

    def _on_person_saved(self, day_str, time_str, name_str, fpath):
        if self.sidebar_visible and self.sidebar_mode == "people":
            self.root.after(0, self.refresh_saved_list)

    def _on_video_saved(self, day_str, time_str, name_str, fpath):
        if self.sidebar_visible and self.sidebar_mode == "video":
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
            if self.sidebar_mode == "plates":
                lbl_type = "Фото номера:"
            elif self.sidebar_mode == "faces":
                lbl_type = "Фото обличчя:"
            else:
                lbl_type = "Скріншот об'єкта:"
            self.preview_info_lbl.configure(text=f"{lbl_type} {base}")
            self.preview_label.configure(image=self.preview_photo_tk, text="", pady=0)
        except Exception:
            pass

    @staticmethod
    def _extract_roi_bgr(cv_img, coords):
        if cv_img is None or coords is None:
            return None
        orig_h, orig_w = cv_img.shape[:2]
        x1, y1, x2, y2 = coords
        cx1 = max(0, min(orig_w - 1, int(x1)))
        cy1 = max(0, min(orig_h - 1, int(y1)))
        cx2 = max(cx1 + 1, min(orig_w, int(x2)))
        cy2 = max(cy1 + 1, min(orig_h, int(y2)))
        crop = cv_img[cy1:cy2, cx1:cx2]
        return crop.copy() if crop.size > 0 else None

    def _draw_roi_rect_on_screen_canvas(self):
        self.screen_canvas.delete("roi_rect")
        if self.current_roi_coords is None or self.current_screen_img_cv is None:
            return
        if self.canvas_scale_x <= 0 or self.canvas_scale_y <= 0:
            return
        cx1, cy1, cx2, cy2 = self.current_roi_coords
        sx1 = cx1 / self.canvas_scale_x
        sy1 = cy1 / self.canvas_scale_y
        sx2 = cx2 / self.canvas_scale_x
        sy2 = cy2 / self.canvas_scale_y
        self.screen_canvas.create_rectangle(
            sx1, sy1, sx2, sy2,
            outline="#00ff00", width=2, tags="roi_rect"
        )
        self.screen_canvas.tag_raise("roi_rect")

    def _display_cv_in_screen_canvas(self, cv_img, keep_roi=True):
        if cv_img is None or cv_img.size == 0:
            return
        try:
            self.current_screen_img_cv = cv_img
            orig_h, orig_w = cv_img.shape[:2]

            canvas_w = 340
            aspect = orig_h / float(orig_w) if orig_w > 0 else (9.0 / 16.0)
            canvas_h = max(1, int(round(canvas_w * aspect)))

            resized = cv2.resize(cv_img, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.screen_photo_tk = ImageTk.PhotoImage(Image.fromarray(rgb))

            size_changed = (canvas_w != self.canvas_disp_w or canvas_h != self.canvas_disp_h)
            self.canvas_scale_x = orig_w / float(canvas_w)
            self.canvas_scale_y = orig_h / float(canvas_h)
            self.canvas_disp_w = canvas_w
            self.canvas_disp_h = canvas_h

            self.screen_canvas.delete("placeholder")
            if size_changed or self.screen_canvas_img_id is None or not self.screen_canvas.find_withtag("frame_img"):
                self.screen_canvas.delete("frame_img")
                self.screen_canvas.config(width=canvas_w, height=canvas_h)
                self.screen_canvas_img_id = self.screen_canvas.create_image(
                    0, 0, anchor=tk.NW, image=self.screen_photo_tk, tags="frame_img"
                )
            else:
                self.screen_canvas.itemconfig(self.screen_canvas_img_id, image=self.screen_photo_tk)

            if keep_roi and self.current_roi_coords is not None:
                # Не перезаписуємо рамку в момент ручного перетягування мишкою
                if self.drag_start_x is None:
                    self._draw_roi_rect_on_screen_canvas()
                crop_bgr = self._extract_roi_bgr(cv_img, self.current_roi_coords)
                if crop_bgr is not None:
                    self.last_roi_crop_bgr = crop_bgr
                    if self.roi_panel.winfo_ismapped():
                        self._render_current_roi_crop()
            else:
                self.screen_canvas.delete("roi_rect")

            if self.screen_canvas.find_withtag("roi_rect"):
                self.screen_canvas.tag_raise("roi_rect")

            if self.fs_win is not None and self.fs_win.winfo_exists():
                self._render_fullscreen_content()
        except Exception as e:
            print(f"[Preview render error]: {e}")

    def _show_screenshot_in_canvas(self, screen_path):
        if not screen_path or not os.path.exists(screen_path):
            self.screen_canvas.delete("all")
            self.screen_canvas_img_id = None
            self.current_screen_img_cv = None
            self.screen_canvas.config(width=340, height=190)
            self.screen_canvas.create_text(170, 95, text="[Скріншот відсутній]", fill="#707080", font=("Segoe UI", 9), tags="placeholder")
            if self.fs_win is not None and self.fs_win.winfo_exists():
                self._render_fullscreen_content()
            return

        try:
            cv_img = cv2.imread(screen_path)
            if cv_img is None:
                return
            keep = bool(self.current_roi_coords is not None and (self.roi_panel.winfo_ismapped() or (self.fs_win is not None and self.fs_win.winfo_exists())))
            if not keep:
                self._hide_roi_panel(clear_coords=True)
            self._display_cv_in_screen_canvas(cv_img, keep_roi=keep)
        except Exception as e:
            print(f"[Screenshot preview error]: {e}")

    @staticmethod
    def _format_sec(sec: float) -> str:
        s = max(0, int(round(sec)))
        if s >= 3600:
            h, rem = divmod(s, 3600)
            m, s2 = divmod(rem, 60)
            return f"{h:d}:{m:02d}:{s2:02d}"
        m, s2 = divmod(s, 60)
        return f"{m:02d}:{s2:02d}"

    def _update_arch_time_ui(self):
        total = max(1, self.arch_total_frames)
        curr = max(0, min(total - 1, self.arch_curr_frame_idx))
        fps = max(1.0, self.arch_fps)
        time_str = f"{self._format_sec(curr / fps)}/{self._format_sec(total / fps)}"

        self.lbl_arch_time.configure(text=time_str)
        self.arch_updating_slider = True
        try:
            self.arch_slider.configure(to=max(1, total - 1))
            self.arch_slider_var.set(curr)
            if self.fs_win is not None and self.fs_win.winfo_exists() and hasattr(self, "fs_slider"):
                self.fs_slider.configure(to=max(1, total - 1))
                self.fs_slider_var.set(curr)
                self.fs_time_lbl.configure(text=time_str)
        finally:
            self.arch_updating_slider = False

    def _update_arch_play_buttons(self):
        icon = "⏸" if self.arch_playing else "▶"
        bg = "#d97706" if self.arch_playing else "#28a745"
        if hasattr(self, "btn_arch_play"):
            self.btn_arch_play.configure(text=icon, bg=bg)
        if self.fs_win is not None and self.fs_win.winfo_exists() and hasattr(self, "fs_btn_play"):
            self.fs_btn_play.configure(text=icon, bg=bg)

    def _stop_archive_player(self):
        self.arch_playing = False
        if self.arch_after_id is not None:
            try:
                self.root.after_cancel(self.arch_after_id)
            except Exception:
                pass
            self.arch_after_id = None
        if self.arch_cap is not None:
            try:
                self.arch_cap.release()
            except Exception:
                pass
            self.arch_cap = None
        self.arch_fpath = None
        self.arch_total_frames = 0
        self.arch_curr_frame_idx = 0
        self._update_arch_play_buttons()
        if hasattr(self, "lbl_arch_time"):
            self.lbl_arch_time.configure(text="00:00/00:00")
        self.arch_updating_slider = True
        try:
            if hasattr(self, "arch_slider_var"):
                self.arch_slider_var.set(0)
        finally:
            self.arch_updating_slider = False

    def _load_archive_video(self, fpath, auto_play=True):
        self._stop_archive_player()
        if not fpath or not os.path.exists(fpath):
            self.screen_canvas.delete("all")
            self.screen_canvas_img_id = None
            self.current_screen_img_cv = None
            self.screen_canvas.config(width=340, height=190)
            self.screen_canvas.create_text(170, 95, text="[Відеофайл відсутній]", fill="#707080", font=("Segoe UI", 9), tags="placeholder")
            return

        if self.video_engine and self.video_engine.is_file_busy(fpath):
            self.screen_canvas.delete("all")
            self.screen_canvas_img_id = None
            self.current_screen_img_cv = None
            self.screen_canvas.config(width=340, height=190)
            self.screen_canvas.create_text(170, 95, text="[Файл ще записується...]", fill="#ffcc00", font=("Segoe UI", 9), tags="placeholder")
            return

        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            self.screen_canvas.delete("all")
            self.screen_canvas_img_id = None
            self.current_screen_img_cv = None
            self.screen_canvas.config(width=340, height=190)
            self.screen_canvas.create_text(170, 95, text="[Не вдалося відкрити відео]", fill="#ff5555", font=("Segoe UI", 9), tags="placeholder")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        if fps <= 0 or fps > 120 or np.isnan(fps):
            fps = 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            total_frames = 1

        self.arch_cap = cap
        self.arch_fpath = fpath
        self.arch_fps = fps
        self.arch_total_frames = total_frames
        self.arch_curr_frame_idx = 0

        ret, frame = cap.read()
        if ret and frame is not None:
            keep = bool(self.current_roi_coords is not None and (self.roi_panel.winfo_ismapped() or (self.fs_win is not None and self.fs_win.winfo_exists())))
            self._display_cv_in_screen_canvas(frame, keep_roi=keep)
        else:
            cap.release()
            self.arch_cap = None
            return

        self._update_arch_time_ui()
        self._sync_fs_controls_visibility()

        if auto_play:
            self.arch_playing = True
            self._update_arch_play_buttons()
            self._schedule_arch_tick(int(round(1000.0 / max(1.0, self.arch_fps))))
        else:
            self.arch_playing = False
            self._update_arch_play_buttons()

    def _schedule_arch_tick(self, delay_ms=None):
        if self.arch_after_id is not None:
            try:
                self.root.after_cancel(self.arch_after_id)
            except Exception:
                pass
            self.arch_after_id = None
        if not self.arch_playing:
            return
        if delay_ms is None:
            # 1x -> реальний fps (наприклад 67мс для 15fps); 2x -> 2x кадр; 4x -> step 2 кадрів @ 2x інтервал (~30 GUI FPS)
            effective_rate = self.arch_fps * (2 if self.arch_speed >= 2 else 1)
            delay_ms = max(16, int(round(1000.0 / max(1.0, effective_rate))))
        self.arch_after_id = self.root.after(delay_ms, self._arch_player_tick)

    def _arch_player_tick(self):
        self.arch_after_id = None
        if not self.arch_playing or self.arch_cap is None:
            return

        t0 = time.perf_counter()
        step = 2 if self.arch_speed >= 4 else 1
        for _ in range(step - 1):
            if not self.arch_cap.grab():
                break
            self.arch_curr_frame_idx += 1

        ret, frame = self.arch_cap.read()
        if not ret or frame is None:
            self.arch_playing = False
            self._update_arch_play_buttons()
            return

        self.arch_curr_frame_idx += 1
        if self.arch_curr_frame_idx >= self.arch_total_frames:
            self.arch_total_frames = self.arch_curr_frame_idx + 1

        self._display_cv_in_screen_canvas(frame, keep_roi=True)
        self._update_arch_time_ui()

        effective_rate = self.arch_fps * (2 if self.arch_speed >= 2 else 1)
        target_ms = 1000.0 / max(1.0, effective_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        next_delay = max(12, int(round(target_ms - elapsed_ms)))
        self._schedule_arch_tick(next_delay)

    def _toggle_arch_play(self):
        if self.arch_cap is None:
            return
        if self.arch_playing:
            self.arch_playing = False
            if self.arch_after_id is not None:
                try:
                    self.root.after_cancel(self.arch_after_id)
                except Exception:
                    pass
                self.arch_after_id = None
            self._update_arch_play_buttons()
        else:
            if self.arch_curr_frame_idx >= max(1, self.arch_total_frames - 1):
                self.arch_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.arch_curr_frame_idx = 0
            self.arch_playing = True
            self._update_arch_play_buttons()
            self._schedule_arch_tick(16)

    def _seek_archive_video(self, frame_idx: int):
        if self.arch_cap is None:
            return
        target = max(0, min(max(0, self.arch_total_frames - 1), int(frame_idx)))
        self.arch_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.arch_cap.read()
        if ret and frame is not None:
            self.arch_curr_frame_idx = target
            self._display_cv_in_screen_canvas(frame, keep_roi=True)
            self._update_arch_time_ui()

    def _on_arch_slider_move(self, val_str):
        if self.arch_updating_slider or self.arch_cap is None:
            return
        try:
            idx = int(float(val_str))
        except Exception:
            return
        self._seek_archive_video(idx)

    def _cycle_arch_speed(self):
        speeds = [1, 2, 4]
        try:
            idx = speeds.index(self.arch_speed)
            self.arch_speed = speeds[(idx + 1) % len(speeds)]
        except ValueError:
            self.arch_speed = 1
        txt = f"{self.arch_speed}x"
        if hasattr(self, "btn_arch_speed"):
            self.btn_arch_speed.configure(text=txt)
        if self.fs_win is not None and self.fs_win.winfo_exists() and hasattr(self, "fs_btn_speed"):
            self.fs_btn_speed.configure(text=txt)

    def _on_item_selected(self, event):
        sel = self.items_tree.selection()
        if not sel:
            return
        fpath = self.tree_item_map.get(sel[0])
        if not fpath:
            return

        # Якщо вибір відновлено після фонового оновлення списку — не перезапускати поточний перегляд
        if event is not None and fpath == self.current_preview_fpath and self.current_screen_img_cv is not None:
            return

        self.current_preview_fpath = fpath

        if self.sidebar_mode == "video":
            self._load_archive_video(fpath, auto_play=True)
            return

        self._stop_archive_player()

        if self.sidebar_mode in ("plates", "faces"):
            self._show_item_in_preview(fpath)

        if self.sidebar_mode == "people":
            screen_path = fpath
        else:
            item_dir = os.path.dirname(fpath)
            base_name = os.path.basename(fpath)
            time_prefix = base_name.split("_")[0]
            candidate = os.path.join(item_dir, f"{time_prefix}_full.jpg")
            screen_path = candidate if os.path.exists(candidate) else fpath

        self._show_screenshot_in_canvas(screen_path)

    def _on_item_double_clicked(self, event):
        sel = self.items_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        if item_id not in self.tree_item_map:
            return
        fpath = self.tree_item_map[item_id]
        if self.current_preview_fpath != fpath or self.current_screen_img_cv is None:
            self._on_item_selected(None)
        if self.current_screen_img_cv is not None:
            self._open_fullscreen_preview(crop_mode=False, source_widget=self.items_tree)

    def _on_screen_canvas_press(self, event):
        if self.current_screen_img_cv is None:
            return
        self.drag_start_x = max(0, min(self.canvas_disp_w, event.x))
        self.drag_start_y = max(0, min(self.canvas_disp_h, event.y))
        self.drag_moved = False

    def _on_screen_canvas_motion(self, event):
        if self.drag_start_x is None or self.drag_start_y is None or self.current_screen_img_cv is None:
            return
        cur_x = max(0, min(self.canvas_disp_w, event.x))
        cur_y = max(0, min(self.canvas_disp_h, event.y))
        if abs(cur_x - self.drag_start_x) >= 4 or abs(cur_y - self.drag_start_y) >= 4:
            self.drag_moved = True
        if self.drag_moved:
            self.screen_canvas.delete("roi_rect")
            self.screen_canvas.create_rectangle(
                self.drag_start_x, self.drag_start_y, cur_x, cur_y,
                outline="#00ff00", width=2, tags="roi_rect"
            )
            self.screen_canvas.tag_raise("roi_rect")

    def _on_screen_canvas_release(self, event):
        if self.drag_start_x is None or self.drag_start_y is None or self.current_screen_img_cv is None:
            return

        end_x = max(0, min(self.canvas_disp_w, event.x))
        end_y = max(0, min(self.canvas_disp_h, event.y))

        x1 = min(self.drag_start_x, end_x)
        x2 = max(self.drag_start_x, end_x)
        y1 = min(self.drag_start_y, end_y)
        y2 = max(self.drag_start_y, end_y)

        was_dragged = self.drag_moved
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_moved = False

        sel_w = x2 - x1
        sel_h = y2 - y1
        if not was_dragged or sel_w < 4 or sel_h < 4:
            return

        orig_h, orig_w = self.current_screen_img_cv.shape[:2]
        crop_x1 = max(0, min(orig_w - 1, int(round(x1 * self.canvas_scale_x))))
        crop_y1 = max(0, min(orig_h - 1, int(round(y1 * self.canvas_scale_y))))
        crop_x2 = max(crop_x1 + 1, min(orig_w, int(round(x2 * self.canvas_scale_x))))
        crop_y2 = max(crop_y1 + 1, min(orig_h, int(round(y2 * self.canvas_scale_y))))

        self._apply_roi_crop_coords((crop_x1, crop_y1, crop_x2, crop_y2), open_panel=True)

    def _apply_roi_crop_coords(self, coords, open_panel=True):
        if self.current_screen_img_cv is None or coords is None:
            return
        crop_bgr = self._extract_roi_bgr(self.current_screen_img_cv, coords)
        if crop_bgr is None:
            return
        self.current_roi_coords = coords
        self.last_roi_crop_bgr = crop_bgr
        self._draw_roi_rect_on_screen_canvas()
        if open_panel:
            self.roi_panel.place(x=self.roi_user_left, y=self.roi_user_top, width=self.roi_user_width, height=self.roi_user_height)
            self.roi_panel.lift()
            self._render_current_roi_crop()
        if self.fs_win is not None and self.fs_win.winfo_exists():
            self._sync_fs_controls_visibility()
            self._render_fullscreen_content()

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
        prefix = "Живий кроп відео" if self.sidebar_mode == "video" else "Виділена область"
        self.roi_title_lbl.configure(text=f"{prefix} ({w}x{h})")

    def _hide_roi_panel(self, clear_coords=True):
        self.roi_panel.place_forget()
        if clear_coords:
            self.current_roi_coords = None
            self.last_roi_crop_bgr = None
            if hasattr(self, "screen_canvas"):
                self.screen_canvas.delete("roi_rect")

    # -----------------------------------------------------------------
    # ПОВНОЕКРАННИЙ ПЕРЕГЛЯД НА ПОТОЧНОМУ МОНІТОРІ (спільний для всіх модулів)
    # -----------------------------------------------------------------
    def _open_fullscreen_preview(self, crop_mode=False, source_widget=None):
        if self.current_screen_img_cv is None:
            sel = self.items_tree.selection()
            if sel and sel[0] in self.tree_item_map:
                self._on_item_selected(None)
        if self.current_screen_img_cv is None:
            return

        use_crop = bool(crop_mode and (self.current_roi_coords is not None or self.last_roi_crop_bgr is not None))
        self.fs_crop_mode = use_crop

        if self.fs_win is not None and self.fs_win.winfo_exists():
            self._sync_fs_controls_visibility()
            self.fs_win.lift()
            self.fs_win.focus_force()
            self._render_fullscreen_content()
            return

        mx, my, mw, mh = self._get_current_monitor_geometry(source_widget if source_widget is not None else self.root)

        win = tk.Toplevel(self.root)
        win.configure(bg="#0a0a0e")
        win.overrideredirect(True)
        win.geometry(f"{mw}x{mh}{mx:+d}{my:+d}")
        win.transient(self.root)
        win.lift()
        win.focus_force()
        self.fs_win = win
        self.fs_drag_start_x = None
        self.fs_drag_start_y = None
        self.fs_drag_moved = False

        win.bind("<Escape>", lambda e: (self._close_fullscreen_preview(), "break")[1])

        # Верхня панель керування повноекранного перегляду
        fs_top = tk.Frame(win, bg="#1b1b24", padx=12, pady=8)
        fs_top.pack(fill=tk.X, side=tk.TOP)

        self.fs_title_lbl = tk.Label(
            fs_top,
            text="Повноекранний перегляд",
            fg="#00e5ff",
            bg="#1b1b24",
            font=("Segoe UI", 11, "bold")
        )
        self.fs_title_lbl.pack(side=tk.LEFT)

        btn_fs_close = tk.Button(
            fs_top,
            text="✕ Згорнути (Esc)",
            command=self._close_fullscreen_preview,
            bg="#dc3545",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2"
        )
        btn_fs_close.pack(side=tk.RIGHT, padx=(6, 0))

        self.fs_btn_clear_crop = tk.Button(
            fs_top,
            text="✕ Скинути кроп",
            command=self._clear_fs_crop,
            bg="#44333a",
            fg="#ffaaaa",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2"
        )

        self.fs_btn_crop_mode = tk.Button(
            fs_top,
            text="🖼 Повний кадр",
            command=self._toggle_fs_crop_mode,
            bg="#2d3b4e",
            fg="#00e5ff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2"
        )

        btn_next = tk.Button(
            fs_top,
            text="Наступний ▶",
            command=lambda: self._navigate_items(1),
            bg="#2e2e3c",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2"
        )
        btn_next.pack(side=tk.RIGHT, padx=(4, 6))

        btn_prev = tk.Button(
            fs_top,
            text="◀ Попередній",
            command=lambda: self._navigate_items(-1),
            bg="#2e2e3c",
            fg="#ffffff",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=3,
            cursor="hand2"
        )
        btn_prev.pack(side=tk.RIGHT, padx=4)

        # Нижня панель плеєра відео (тільки у режимі video)
        self.fs_controls = tk.Frame(win, bg="#1b1b24", padx=14, pady=8)

        self.fs_btn_play = tk.Button(
            self.fs_controls,
            text="⏸" if self.arch_playing else "▶",
            command=self._toggle_arch_play,
            bg="#d97706" if self.arch_playing else "#28a745",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            width=4,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.fs_btn_play.pack(side=tk.LEFT, padx=(0, 8))

        self.fs_slider_var = tk.IntVar(value=self.arch_curr_frame_idx)
        self.fs_slider = tk.Scale(
            self.fs_controls,
            from_=0,
            to=max(1, self.arch_total_frames - 1),
            orient=tk.HORIZONTAL,
            variable=self.fs_slider_var,
            showvalue=False,
            bg="#1b1b24",
            fg="#00e5ff",
            troughcolor="#2e2e3d",
            highlightthickness=0,
            bd=0,
            sliderlength=18,
            command=self._on_arch_slider_move
        )
        self.fs_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.fs_time_lbl = tk.Label(
            self.fs_controls,
            text=self.lbl_arch_time.cget("text") if hasattr(self, "lbl_arch_time") else "00:00/00:00",
            fg="#e0e0f0",
            bg="#1b1b24",
            font=("Consolas", 10, "bold")
        )
        self.fs_time_lbl.pack(side=tk.LEFT, padx=10)

        self.fs_btn_speed = tk.Button(
            self.fs_controls,
            text=f"{self.arch_speed}x",
            command=self._cycle_arch_speed,
            bg="#333344",
            fg="#00e5ff",
            font=("Segoe UI", 9, "bold"),
            width=4,
            relief=tk.FLAT,
            cursor="hand2"
        )
        self.fs_btn_speed.pack(side=tk.RIGHT)

        # Полотно повноекранного зображення/відео
        self.fs_canvas = tk.Canvas(win, bg="#0a0a0e", highlightthickness=0, cursor="cross")
        self.fs_canvas.pack(fill=tk.BOTH, expand=True)
        self.fs_canvas.bind("<Configure>", lambda e: self._render_fullscreen_content())
        self.fs_canvas.bind(EVT_BTN_PRESS, self._on_fs_canvas_press)
        self.fs_canvas.bind(EVT_B1_MOTION, self._on_fs_canvas_motion)
        self.fs_canvas.bind(EVT_BTN_RELEASE, self._on_fs_canvas_release)
        self.fs_canvas.bind(EVT_DOUBLE_CLICK, lambda e: self._close_fullscreen_preview())
        self.fs_canvas.bind(EVT_MOUSEWHEEL, self._on_preview_mousewheel)
        self.fs_canvas.bind(EVT_BTN4, self._on_preview_mousewheel)
        self.fs_canvas.bind(EVT_BTN5, self._on_preview_mousewheel)

        self._sync_fs_controls_visibility()
        win.update_idletasks()
        self._render_fullscreen_content()

    def _sync_fs_controls_visibility(self):
        if self.fs_win is None or not self.fs_win.winfo_exists():
            return

        has_crop = self.current_roi_coords is not None
        if has_crop:
            self.fs_btn_crop_mode.configure(text="🖼 Повний кадр" if self.fs_crop_mode else "🔍 Показати кроп")
            if not self.fs_btn_crop_mode.winfo_ismapped():
                self.fs_btn_crop_mode.pack(side=tk.RIGHT, padx=4)
            if not self.fs_btn_clear_crop.winfo_ismapped():
                self.fs_btn_clear_crop.pack(side=tk.RIGHT, padx=4)
        else:
            self.fs_btn_crop_mode.pack_forget()
            self.fs_btn_clear_crop.pack_forget()

        if self.sidebar_mode == "video" and self.arch_cap is not None:
            if not self.fs_controls.winfo_ismapped():
                self.fs_controls.pack(fill=tk.X, side=tk.BOTTOM, before=self.fs_canvas)
            self._update_arch_play_buttons()
            self._update_arch_time_ui()
        else:
            self.fs_controls.pack_forget()

    def _render_fullscreen_content(self):
        if self.fs_win is None or not self.fs_win.winfo_exists() or self.fs_canvas is None:
            return

        cw = self.fs_canvas.winfo_width()
        ch = self.fs_canvas.winfo_height()
        if cw < 20 or ch < 20:
            return

        if self.current_screen_img_cv is None:
            self.fs_canvas.delete("all")
            self.fs_canvas.create_text(cw // 2, ch // 2, text="[Немає зображення для перегляду]", fill="#707080", font=("Segoe UI", 14))
            return

        if self.fs_crop_mode:
            src_bgr = self._extract_roi_bgr(self.current_screen_img_cv, self.current_roi_coords)
            if src_bgr is None:
                src_bgr = self.last_roi_crop_bgr if self.last_roi_crop_bgr is not None else self.current_screen_img_cv
        else:
            src_bgr = self.current_screen_img_cv

        orig_h, orig_w = src_bgr.shape[:2]
        if orig_w <= 0 or orig_h <= 0:
            return

        scale = min(cw / float(orig_w), ch / float(orig_h))
        disp_w = max(1, int(round(orig_w * scale)))
        disp_h = max(1, int(round(orig_h * scale)))
        off_x = (cw - disp_w) // 2
        off_y = (ch - disp_h) // 2

        self.fs_scale_x = orig_w / float(disp_w)
        self.fs_scale_y = orig_h / float(disp_h)
        self.fs_img_offset_x = off_x
        self.fs_img_offset_y = off_y
        self.fs_disp_w = disp_w
        self.fs_disp_h = disp_h

        interp = cv2.INTER_LANCZOS4 if self.fs_crop_mode else cv2.INTER_LINEAR
        resized = cv2.resize(src_bgr, (disp_w, disp_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.fs_photo_tk = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.fs_canvas.delete("fs_img")
        self.fs_canvas.create_image(off_x, off_y, anchor=tk.NW, image=self.fs_photo_tk, tags="fs_img")

        # Якщо у повному режимі є кроп — показати зелену рамку області кропу
        self.fs_canvas.delete("fs_roi_existing")
        if (not self.fs_crop_mode) and (self.current_roi_coords is not None) and (self.fs_drag_start_x is None):
            cx1, cy1, cx2, cy2 = self.current_roi_coords
            rx1 = off_x + cx1 / self.fs_scale_x
            ry1 = off_y + cy1 / self.fs_scale_y
            rx2 = off_x + cx2 / self.fs_scale_x
            ry2 = off_y + cy2 / self.fs_scale_y
            self.fs_canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="#00ff00", width=2, tags="fs_roi_existing")
            self.fs_canvas.tag_raise("fs_roi_existing")

        if self.fs_canvas.find_withtag("fs_roi_drag"):
            self.fs_canvas.tag_raise("fs_roi_drag")

        fname = os.path.basename(self.current_preview_fpath) if self.current_preview_fpath else ""
        mode_tag = f"Кроп ({orig_w}x{orig_h})" if self.fs_crop_mode else f"Повний екран ({orig_w}x{orig_h}) • виділіть область для кропу"
        self.fs_title_lbl.configure(text=f"{fname}   —   {mode_tag}" if fname else mode_tag)

    def _toggle_fs_crop_mode(self):
        if self.current_roi_coords is None and self.last_roi_crop_bgr is None:
            return
        self.fs_crop_mode = not self.fs_crop_mode
        self._sync_fs_controls_visibility()
        self._render_fullscreen_content()

    def _clear_fs_crop(self):
        self._hide_roi_panel(clear_coords=True)
        self.fs_crop_mode = False
        self._sync_fs_controls_visibility()
        self._render_fullscreen_content()

    def _on_fs_canvas_press(self, event):
        if self.current_screen_img_cv is None or self.fs_crop_mode:
            return
        self.fs_drag_start_x = max(self.fs_img_offset_x, min(self.fs_img_offset_x + self.fs_disp_w, event.x))
        self.fs_drag_start_y = max(self.fs_img_offset_y, min(self.fs_img_offset_y + self.fs_disp_h, event.y))
        self.fs_drag_moved = False

    def _on_fs_canvas_motion(self, event):
        if self.fs_drag_start_x is None or self.fs_drag_start_y is None or self.fs_crop_mode:
            return
        cur_x = max(self.fs_img_offset_x, min(self.fs_img_offset_x + self.fs_disp_w, event.x))
        cur_y = max(self.fs_img_offset_y, min(self.fs_img_offset_y + self.fs_disp_h, event.y))
        if abs(cur_x - self.fs_drag_start_x) >= 6 or abs(cur_y - self.fs_drag_start_y) >= 6:
            self.fs_drag_moved = True
        if self.fs_drag_moved:
            self.fs_canvas.delete("fs_roi_drag")
            self.fs_canvas.create_rectangle(
                self.fs_drag_start_x, self.fs_drag_start_y, cur_x, cur_y,
                outline="#00ff00", width=2, tags="fs_roi_drag"
            )
            self.fs_canvas.tag_raise("fs_roi_drag")

    def _on_fs_canvas_release(self, event):
        if self.fs_drag_start_x is None or self.fs_drag_start_y is None or self.fs_crop_mode:
            return
        end_x = max(self.fs_img_offset_x, min(self.fs_img_offset_x + self.fs_disp_w, event.x))
        end_y = max(self.fs_img_offset_y, min(self.fs_img_offset_y + self.fs_disp_h, event.y))

        x1 = min(self.fs_drag_start_x, end_x) - self.fs_img_offset_x
        x2 = max(self.fs_drag_start_x, end_x) - self.fs_img_offset_x
        y1 = min(self.fs_drag_start_y, end_y) - self.fs_img_offset_y
        y2 = max(self.fs_drag_start_y, end_y) - self.fs_img_offset_y

        was_moved = self.fs_drag_moved
        self.fs_drag_start_x = None
        self.fs_drag_start_y = None
        self.fs_drag_moved = False
        self.fs_canvas.delete("fs_roi_drag")

        if not was_moved or (x2 - x1) < 8 or (y2 - y1) < 8 or self.current_screen_img_cv is None:
            return

        orig_h, orig_w = self.current_screen_img_cv.shape[:2]
        crop_x1 = max(0, min(orig_w - 1, int(round(x1 * self.fs_scale_x))))
        crop_y1 = max(0, min(orig_h - 1, int(round(y1 * self.fs_scale_y))))
        crop_x2 = max(crop_x1 + 1, min(orig_w, int(round(x2 * self.fs_scale_x))))
        crop_y2 = max(crop_y1 + 1, min(orig_h, int(round(y2 * self.fs_scale_y))))

        self.fs_crop_mode = True
        self._apply_roi_crop_coords((crop_x1, crop_y1, crop_x2, crop_y2), open_panel=True)

    def _close_fullscreen_preview(self):
        if self.fs_win is not None:
            try:
                self.fs_win.destroy()
            except Exception:
                pass
            self.fs_win = None
        self.fs_canvas = None
        self.fs_photo_tk = None
        try:
            self.root.focus_force()
        except Exception:
            pass

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
        def _fmt_days(d):
            return "необмежено" if d <= 0 else f"{self._fmt_num(d)} дн."

        help_text = (
            "ANPR, Face, People & Video Monitor (Modular Architecture)\n\n"
            f"• Номери (plates_on): {self.plates_on} | рамки: {self.plates_show_boxes} | дати: {self._format_date_range_str('plates')} | автовид.: {_fmt_days(self.plates_delete_older_days)}\n"
            f"• Обличчя (faces_on): {self.faces_on} | рамки: {self.faces_show_boxes} | дати: {self._format_date_range_str('faces')} | автовид.: {_fmt_days(self.faces_delete_older_days)}\n"
            f"• Люди/транспорт (people_on): {self.people_on} (транспорт: {self.people_detect_transport}) | рамки: {self.people_show_boxes} | дати: {self._format_date_range_str('people')} | автовид.: {_fmt_days(self.people_delete_older_days)}\n"
            f"• Відеозапис (video_on): {self.video_on} | фрагмент: {self._fmt_num(self.video_duration_min)} хв @ {self._fmt_num(self.video_fps)} fps ({self.video_codec} ultrafast) | дати: {self._format_date_range_str('video')} | автовид.: {_fmt_days(self.video_delete_older_days)}\n\n"
            "• Детекція номерів: best.onnx (640x640) + CCT OCR\n"
            "• Детекція облич: SCRFD / InsightFace + AdaFace/MBF (112x112)\n"
            "• Детекція людей та транспорту: yolo11n.onnx (640x640)\n"
            f"• Поріг збереження номерів: >= {self.min_percent_to_save}%\n"
            f"• Поріг схожості облич: >= {int(self.face_min_similarity * 100)}%\n"
            f"• Поріг детекції людей/транспорту: >= {int(self.people_det_thresh * 100)}%\n"
            f"• Інтервал скріншотів людей/транспорту: {self.people_timer_ms} мс\n"
            "• Збереження файлів: /plates, /faces, /people, /video (по днях YYYY-MM-DD)\n"
            "• Повноекранний перегляд: подвійний клік по вікну перегляду / кнопкою '⛶ Розгорнути' (Esc — згорнути)\n\n"
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
        if self.video_engine is not None:
            self.video_engine.stop(wait=False)
        self.btn_toggle.configure(text="Старт", bg="#28a745")
        self.status_lbl.configure(text="Статус: Зупинено", fg="#ffcc00")
        self._cancel_hide_timer()
        self._show_panel()

    def _ai_worker(self):
        """Паралельний ШІ-потік опитування активних модулів з урахуванням меж дат."""
        try:
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
                        if self._is_module_date_active("plates"):
                            plate_boxes, overlay_info = self.plate_engine.process_frame(frame_to_process)
                        elif self.plate_engine.active_tracks:
                            self.plate_engine.flush_tracks()

                    face_boxes = []
                    if self.faces_on and self.face_engine is not None:
                        if self._is_module_date_active("faces"):
                            face_boxes = self.face_engine.process_frame(frame_to_process)
                        elif self.face_engine.active_tracks:
                            self.face_engine.flush_tracks()

                    person_boxes = []
                    if self.people_on and self.person_engine is not None and self._is_module_date_active("people"):
                        person_boxes = self.person_engine.process_frame(frame_to_process)

                    with self.frame_lock:
                        self.cached_plate_boxes = plate_boxes
                        self.cached_face_boxes = face_boxes
                        self.cached_person_boxes = person_boxes
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
        finally:
            # При натисканні "Стоп" або виході примусово зберігаємо всі активні треки
            if self.plate_engine is not None:
                try:
                    self.plate_engine.flush_tracks()
                except Exception:
                    pass
            if self.face_engine is not None:
                try:
                    self.face_engine.flush_tracks()
                except Exception:
                    pass

    def _capture_worker(self, source):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.root.after(0, lambda: self.status_lbl.configure(text="Помилка джерела!", fg="#ff4444"))
            self.is_running = False
            return

        if self.video_on and self.video_engine is not None:
            self.video_engine.start()

        is_file = isinstance(source, str) and not source.lower().startswith("rtsp://") and not source.lower().startswith("http://")
        target_fps = cap.get(cv2.CAP_PROP_FPS)
        if target_fps <= 0 or target_fps > 120 or np.isnan(target_fps):
            target_fps = 30.0
        frame_delay = 1.0 / target_fps

        try:
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

                # Передача чистого кадру (без жодних рамок детекції) у фоновий відеореєстратор
                if self.video_on and self.video_engine is not None:
                    video_allowed = self._is_module_date_active("video")
                    if video_allowed:
                        self.video_engine.push_frame(frame.copy(), allowed=True)
                    else:
                        self.video_engine.set_allowed(False)

                if not self.is_ai_busy:
                    self.is_ai_busy = True
                    with self.ai_lock:
                        self.ai_frame = frame.copy()
                    self.ai_event.set()

                with self.frame_lock:
                    plate_boxes_to_draw = list(self.cached_plate_boxes)
                    face_boxes_to_draw = list(self.cached_face_boxes)
                    person_boxes_to_draw = list(self.cached_person_boxes)
                    plate_img = self.last_plate_img
                    plate_dims = self.last_plate_dims
                    plate_time = self.last_plate_time

                # Зелені рамки номерних знаків
                if self.plates_show_boxes and self._is_module_date_active("plates"):
                    for (bx1, by1, bx2, by2) in plate_boxes_to_draw:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 1)

                # Рамки виявлених людей та транспорту
                if self.people_show_boxes and self._is_module_date_active("people"):
                    for item in person_boxes_to_draw:
                        if len(item) == 3:
                            (px1, py1, px2, py2), p_pct, p_label = item
                        else:
                            (px1, py1, px2, py2), p_pct = item
                            p_label = "Person"

                        is_person = p_label.lower() == "person"
                        # Помаранчевий для людей, золотистий/жовтий для авто й транспорту
                        box_color = (0, 165, 255) if is_person else (0, 215, 255)
                        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 1)
                        p_tag = f"{p_label} {p_pct}%"
                        cv2.putText(frame, p_tag, (px1, max(18, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # Блакитні рамки облич з підписом
                if self.faces_show_boxes and self._is_module_date_active("faces"):
                    for ((fx1, fy1, fx2, fy2), f_name, f_pct) in face_boxes_to_draw:
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)

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
                if plate_img is not None and self._is_module_date_active("plates") and (now - plate_time < self.view_plate_time):
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
                        border = 2
                        x2_ov = frame_w - margin
                        x1_ov = x2_ov - dw
                        y1_ov = margin
                        y2_ov = y1_ov + dh

                        if x1_ov > 0 and y2_ov < frame_h:
                            cv2.rectangle(frame, (x1_ov - border, y1_ov - border), (x2_ov + border, y2_ov + border), (0, 255, 0), 1)
                            frame[y1_ov:y2_ov, x1_ov:x2_ov] = display_img

                with self.frame_lock:
                    self.latest_frame = frame
                    self.new_frame_available = True

                if is_file:
                    elapsed = time.time() - t_start
                    sleep_time = frame_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        finally:
            if self.video_engine is not None:
                self.video_engine.stop(wait=True)
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
        self._close_fullscreen_preview()
        self._stop_archive_player()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        if self.video_engine is not None:
            self.video_engine.stop(wait=True)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = ANPRViewerApp(root)
    root.mainloop()
