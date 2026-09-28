import os
import sys
import re
import time
import configparser
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Локальний OCR-модуль CCT
from fast_alpr.default_ocr import DefaultOCR

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

YOLO_IMGSZ = 640             # Розмір вхідного тензора моделі best.onnx

# Пропорції пластин номерних знаків
PLATE_ASPECT_RATIO_STANDARD = 520.0 / 112.0
PLATE_ASPECT_RATIO_SQUARE = 300.0 / 150.0

# Безпечне визначення назв подій Tkinter (виключає втрату кутових дужок)
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


def async_write_image(path, img):
    """Асинхронний запис зображення на диск без блокування відеопотоку."""
    def _worker():
        try:
            cv2.imwrite(path, img)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


class ONNXDetector:
    """Високопродуктивний детектор ONNX із C++ прискоренням (best.onnx)."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.use_ort = False

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            cpu_cnt = os.cpu_count() or 4
            opts.intra_op_num_threads = max(1, min(2, cpu_cnt - 1))
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.use_ort = True
        except Exception:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_ort = False

    def predict(self, frame, conf_thresh=0.22, imgsz=480):
        h_orig, w_orig = frame.shape[:2]

        r = min(imgsz / h_orig, imgsz / w_orig)
        nw, nh = int(round(w_orig * r)), int(round(h_orig * r))
        dw, dh = (imgsz - nw) / 2.0, (imgsz - nh) / 2.0

        if (w_orig, h_orig) != (nw, nh):
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas[top:top + nh, left:left + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)

        if self.use_ort:
            raw = self.session.run(self.output_names, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            raw = self.net.forward()

        preds = np.squeeze(raw)
        if preds.ndim != 2:
            return []

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        channels = preds.shape[1]
        if channels < 5:
            return []

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]

        if channels == 5:
            scores = preds[:, 4]
        elif channels == 6:
            scores = preds[:, 4] * preds[:, 5]
        else:
            scores = np.max(preds[:, 4:], axis=1)

        mask = scores >= conf_thresh
        if not np.any(mask):
            return []

        cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

        x1 = (cx - w / 2.0 - dw) / r
        y1 = (cy - h / 2.0 - dh) / r
        x2 = (cx + w / 2.0 - dw) / r
        y2 = (cy + h / 2.0 - dh) / r

        boxes_for_nms = []
        scores_list = []
        for i in range(len(scores)):
            bx1 = max(0, min(w_orig - 1, int(x1[i])))
            by1 = max(0, min(h_orig - 1, int(y1[i])))
            bx2 = max(0, min(w_orig, int(x2[i])))
            by2 = max(0, min(h_orig, int(y2[i])))
            bw = bx2 - bx1
            bh = by2 - by1
            if bw > 8 and bh > 8:
                boxes_for_nms.append([bx1, by1, bw, bh])
                scores_list.append(float(scores[i]))

        if not boxes_for_nms:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_list, conf_thresh, 0.45)
        final_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes_for_nms[idx]
                final_boxes.append((bx, by, bx + bw, by + bh))

        return final_boxes


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def rectify_and_enhance(plate_crop):
    """Вирівнювання перспективи за 4 точками та оптимізація контрасту."""
    h, w = plate_crop.shape[:2]
    if h < 12 or w < 24:
        return None, 0, 0

    aspect_ratio = w / float(h)
    if aspect_ratio < 3.0:
        target_w = 520
        target_h = int(target_w / PLATE_ASPECT_RATIO_SQUARE)
    else:
        target_w = 520
        target_h = int(target_w / PLATE_ASPECT_RATIO_STANDARD)

    lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    contrast_crop = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(contrast_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 180)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    plate_quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (w * h * 0.25):
            plate_quad = approx.reshape(4, 2)
            break

    dst = np.array([
        [0, 0],
        [target_w - 1, 0],
        [target_w - 1, target_h - 1],
        [0, target_h - 1]
    ], dtype="float32")

    if plate_quad is not None:
        rect = order_points(plate_quad)
        M = cv2.getPerspectiveTransform(rect, dst)
        rectified = cv2.warpPerspective(contrast_crop, M, (target_w, target_h))
    else:
        rectified = cv2.resize(contrast_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    gaussian = cv2.GaussianBlur(rectified, (0, 0), 1.8)
    enhanced = cv2.addWeighted(rectified, 1.4, gaussian, -0.4, 0)
    return enhanced, target_w, target_h


class ANPRViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANPR Video Monitor (best.onnx + CCT OCR)")
        self.root.configure(bg="#0d0d11")

        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self._prevent_close)
        self._setup_lockdown_bindings()

        self.config_ini_path = os.path.join(BASE_DIR, "config.ini")
        self.config = self._load_config()
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

        self.plates_dir = os.path.join(BASE_DIR, "plates")
        os.makedirs(self.plates_dir, exist_ok=True)

        self._sync_windows_startup(self.is_autostart_active)

        # 1. Завантаження детектора best.onnx
        model_path = os.path.join(BASE_DIR, "model", "best.onnx")
        if not os.path.exists(model_path):
            messagebox.showerror("Помилка моделі", f"Файл детектора не знайдено:\n{model_path}")
            sys.exit(1)
        self.detector = ONNXDetector(model_path)

        # 2. Завантаження розпізнавача CCT OCR
        ocr_model_path = os.path.join(BASE_DIR, "model", "cct_xs_v2_global.onnx")
        ocr_config_path = os.path.join(BASE_DIR, "model", "cct_xs_v2_global_plate_config.yaml")

        if not os.path.exists(ocr_model_path) or not os.path.exists(ocr_config_path):
            messagebox.showerror("Помилка OCR", "Файли cct_xs_v2_global.onnx або config.yaml відсутні в model/")
            sys.exit(1)

        try:
            self.ocr = DefaultOCR(
                hub_ocr_model=None,
                device="cpu",
                model_path=ocr_model_path,
                config_path=ocr_config_path
            )
            print("[OCR] Модель CCT успішно завантажена.")
        except Exception as e:
            messagebox.showerror("Помилка OCR", f"Не вдалося ініціалізувати OCR:\n{e}")
            sys.exit(1)

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

        self.cached_boxes = []
        self.last_plate_img = None
        self.last_plate_dims = (0, 0)
        self.last_plate_text = ""
        self.last_plate_conf = 0.0
        self.last_plate_time = 0.0

        # Змінна останнього збереженого номера для виключення дублікатів підряд
        self.last_saved_plate_text = ""
        self.next_inference_time = 0.0

        self.sidebar_visible = False
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

                migration_map = {
                    "autostart": "autostart_os_and_play",
                    "plate_timeout": "view_plate_time",
                    "min_percent": "min_percent_to_save",
                    "percent": "min_percent_to_save",
                    "save_timeout": "sleep_time_after_save",
                    "save_delay": "sleep_time_after_save",
                    "post_save_delay": "sleep_time_after_save",
                    "rect_scale": "rect_scale_plate",
                    "delete_older": "delete_older_days"
                }

                for old_key, new_key in migration_map.items():
                    if parser.has_option("SETTINGS", old_key):
                        if not parser.has_option("SETTINGS", new_key):
                            parser.set("SETTINGS", new_key, parser.get("SETTINGS", old_key))
                        parser.remove_option("SETTINGS", old_key)
                        needs_saving = True

                for cleanup_key in ["plates", "screenshots"]:
                    if parser.has_option("SETTINGS", cleanup_key):
                        parser.remove_option("SETTINGS", cleanup_key)
                        needs_saving = True

                for key, val in defaults.items():
                    if not parser.has_option("SETTINGS", key):
                        parser.set("SETTINGS", key, str(val).lower() if isinstance(val, bool) else str(val))
                        needs_saving = True

                cfg = {
                    "source": parser.get("SETTINGS", "source", fallback="0").strip().strip('\'"'),
                    "autostart_os_and_play": parser.getboolean("SETTINGS", "autostart_os_and_play", fallback=False),
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
                f.write("; ANPR Video Monitor Configuration\n")
                f.write("; source: 0, rtsp://... або d:\\video.mp4\n")
                f.write("; autostart_os_and_play: true / false\n")
                f.write("; view_plate_time: час збереження номера на екрані (с)\n")
                f.write("; delete_older_days: видаляти файли старше зазначених днів\n")
                f.write("; min_percent_to_save: мінімальна точність розпізнавання для збереження (%)\n")
                f.write("; sleep_time_after_save: таймаут паузи після вдалого визначення номера (с)\n")
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
            if self.delete_older_days > 0 and os.path.exists(self.plates_dir):
                cutoff = time.time() - (self.delete_older_days * 86400.0)
                for root_dir, dirs, files in os.walk(self.plates_dir, topdown=False):
                    for fname in files:
                        fpath = os.path.join(root_dir, fname)
                        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                            try:
                                os.remove(fpath)
                            except Exception:
                                pass
                    if root_dir != self.plates_dir and not os.listdir(root_dir):
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
        ttk.Entry(self.ctrl_frame, width=38, textvariable=self.source_var, state="readonly").pack(side=tk.LEFT, padx=3)

        self.autostart_var = tk.BooleanVar(value=self.is_autostart_active)
        tk.Checkbutton(self.ctrl_frame, text="Автозапуск", variable=self.autostart_var, state=tk.DISABLED, disabledforeground="#ffffff", bg="#1e1e24", selectcolor="#2b2b36", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=12)

        self.btn_saved = tk.Button(self.ctrl_frame, text="Збережені", command=self.toggle_side_panel, bg="#3a3a46", fg="#ffffff", font=("Segoe UI", 10, "bold"), width=11, relief=tk.FLAT, cursor="hand2")
        self.btn_saved.pack(side=tk.LEFT, padx=8)

        if self.is_autostart_active:
            self.btn_toggle = tk.Button(self.ctrl_frame, text="Працює", command=self.toggle_stream, state=tk.DISABLED, disabledforeground="#75e08b", bg="#1b3822", font=("Segoe UI", 10, "bold"), width=10, relief=tk.FLAT)
        else:
            self.btn_toggle = tk.Button(self.ctrl_frame, text="Старт", command=self.toggle_stream, bg="#28a745", fg="white", font=("Segoe UI", 10, "bold"), width=10, relief=tk.FLAT, cursor="hand2")
        self.btn_toggle.pack(side=tk.LEFT, padx=8)

        self.status_lbl = tk.Label(self.ctrl_frame, text="Статус: Очікування", fg="#ffcc00", bg="#1e1e24", font=("Segoe UI", 10))
        self.status_lbl.pack(side=tk.RIGHT, padx=10)

        self.main_body = tk.Frame(self.root, bg="#0d0d11")
        self.main_body.pack(fill=tk.BOTH, expand=True)

        # Бокова панель
        self.side_panel = tk.Frame(self.main_body, bg="#1e1e24", width=360)
        self.side_panel.pack_propagate(False)

        side_header = tk.Frame(self.side_panel, bg="#282832", padx=10, pady=6)
        side_header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(side_header, text="Історія номерів", fg="#00e5ff", bg="#282832", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
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
        style.configure("PlatesDark.Treeview", background="#16161d", foreground="#ffffff", fieldbackground="#16161d", font=("Segoe UI", 10), rowheight=26, borderwidth=0)
        style.map("PlatesDark.Treeview", background=[("selected", "#007acc")], foreground=[("selected", "#ffffff")])

        self.plates_tree = ttk.Treeview(tree_frame, style="PlatesDark.Treeview", show="tree", selectmode="browse", yscrollcommand=scrollbar.set)
        self.plates_tree.column("#0", width=330, stretch=True)
        self.plates_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.plates_tree.yview)

        self.plates_tree.tag_configure("day_tag", font=("Segoe UI", 10, "bold"), foreground="#00e5ff")
        self.plates_tree.tag_configure("plate_tag", font=("Consolas", 10, "bold"), foreground="#75e08b")
        self.plates_tree.bind(EVT_TREE_SELECT, self._on_plate_selected)
        self.plates_tree.bind(EVT_DOUBLE_CLICK, self._on_plate_double_clicked)

        # Прив'язка скролу коліщатком миші над списком номерів
        for w in (self.plates_tree, scrollbar, tree_frame):
            w.bind(EVT_ENTER, self._bind_tree_mousewheel)
            w.bind(EVT_LEAVE, self._unbind_tree_mousewheel)

        self.preview_container = tk.Frame(self.side_panel, bg="#14141a", pady=6, padx=8)
        self.preview_container.pack(fill=tk.X, side=tk.BOTTOM)

        self.preview_info_lbl = tk.Label(self.preview_container, text="Фото номера:", fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9))
        self.preview_info_lbl.pack(anchor=tk.W, pady=(0, 2))

        self.preview_label = tk.Label(self.preview_container, bg="#1a1a22", text="[Виберіть номер зі списку]", fg="#707080", font=("Segoe UI", 9), pady=10, relief=tk.SOLID, borderwidth=1)
        self.preview_label.pack(fill=tk.X)

        self.screen_info_lbl = tk.Label(self.preview_container, text="Фото скріну (виділіть область мишкою):", fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9))
        self.screen_info_lbl.pack(anchor=tk.W, pady=(6, 2))

        self.screen_canvas = tk.Canvas(self.preview_container, bg="#1a1a22", width=340, height=190, highlightthickness=1, highlightbackground="#333340", cursor="cross")
        self.screen_canvas.pack(fill=tk.X, pady=(0, 2))
        self.screen_canvas.bind(EVT_BTN_PRESS, self._on_screen_canvas_press)
        self.screen_canvas.bind(EVT_B1_MOTION, self._on_screen_canvas_motion)
        self.screen_canvas.bind(EVT_BTN_RELEASE, self._on_screen_canvas_release)

        # Прив'язка коліщатка над областю попереднього перегляду для швидкого перемикання номерів
        for pw in (self.preview_container, self.preview_label, self.screen_canvas):
            pw.bind(EVT_ENTER, self._bind_preview_mousewheel)
            pw.bind(EVT_LEAVE, self._unbind_preview_mousewheel)

        self.video_container = tk.Frame(self.main_body, bg="#0d0d11")
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.main_video_label = tk.Label(self.video_container, bg="#0d0d11")
        self.main_video_label.pack(fill=tk.BOTH, expand=True)

        self.main_video_label.bind(EVT_BTN_PRESS, self._on_video_area_click)
        self.video_container.bind(EVT_BTN_PRESS, self._on_video_area_click)

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

    # ------------------ Обробники скролу коліщатком миші ------------------

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

        # Якщо затиснуто Shift — гортаємо список, інакше перемикаємо вибір конкретних номерів
        if getattr(event, "state", 0) & 0x0001:
            self.plates_tree.yview_scroll(delta * 2, "units")
        else:
            self._navigate_plates(delta)
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
            self._navigate_plates(delta)
        return "break"

    def _navigate_plates(self, direction):
        """Перемикання на наступний / попередній номер у дереві та оновлення перегляду."""
        all_items = []
        for day in self.plates_tree.get_children():
            for child in self.plates_tree.get_children(day):
                if child in self.tree_item_map:
                    all_items.append(child)
        if not all_items:
            return

        sel = self.plates_tree.selection()
        if sel and sel[0] in all_items:
            curr_idx = all_items.index(sel[0])
            new_idx = max(0, min(len(all_items) - 1, curr_idx + direction))
        else:
            new_idx = 0 if direction > 0 else len(all_items) - 1

        target = all_items[new_idx]
        self.plates_tree.selection_set(target)
        self.plates_tree.focus(target)
        self.plates_tree.see(target)
        self._on_plate_selected(None)

    # ----------------------------------------------------------------------

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

    def toggle_side_panel(self):
        if self.sidebar_visible:
            self.side_panel.pack_forget()
            self.video_container.pack_forget()
            self.video_container.pack(fill=tk.BOTH, expand=True)
            self.sidebar_visible = False
            self.btn_saved.configure(bg="#3a3a46")
        else:
            self.video_container.pack_forget()
            self.side_panel.pack(side=tk.LEFT, fill=tk.Y)
            self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.sidebar_visible = True
            self.btn_saved.configure(bg="#007acc")
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
            self.btn_sort.configure(text="🔤 Номер")
        self._apply_search_and_sort()

    def refresh_saved_list(self):
        self.raw_records.clear()
        if not os.path.exists(self.plates_dir):
            self._apply_search_and_sort()
            return

        for root_dir, dirs, files in os.walk(self.plates_dir):
            for fname in files:
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue
                if "_full" in fname.lower():
                    continue

                fpath = os.path.join(root_dir, fname)
                base_name, _ = os.path.splitext(fname)
                rel_dir = os.path.relpath(root_dir, self.plates_dir)

                if rel_dir != ".":
                    day_key = rel_dir
                else:
                    day_key = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d")

                if "_plate_" in base_name:
                    parts = base_name.split("_plate_")
                    time_part = parts[0]
                    rest = parts[1]
                    time_disp = ":".join(time_part.split("-")[:3])
                    rest_parts = rest.rsplit("_", 1)
                    plate_num = rest_parts[0] if len(rest_parts) == 2 else rest
                elif base_name.startswith("plate_"):
                    clean_name = base_name.replace("plate_", "", 1)
                    parts = clean_name.split("_")
                    if len(parts) >= 4:
                        time_disp = ":".join(parts[0].split("-")[:3])
                        plate_num = parts[2]
                    else:
                        time_disp = "--:--:--"
                        plate_num = clean_name
                else:
                    time_disp = "--:--:--"
                    plate_num = base_name

                self.raw_records.append({
                    "day": day_key,
                    "time": time_disp,
                    "plate": plate_num,
                    "fpath": fpath,
                    "mtime": os.path.getmtime(fpath)
                })

        self._apply_search_and_sort()

    def _apply_search_and_sort(self):
        self.plates_tree.delete(*self.plates_tree.get_children())
        self.tree_item_map.clear()

        query = self.search_var.get().strip().upper()

        filtered = []
        for r in self.raw_records:
            if query:
                if (query not in r["plate"].upper()) and (query not in r["time"]) and (query not in r["day"]):
                    continue
            filtered.append(r)

        if self.sort_mode == 0:
            filtered.sort(key=lambda x: x["mtime"], reverse=True)
        elif self.sort_mode == 1:
            filtered.sort(key=lambda x: x["mtime"], reverse=False)
        else:
            filtered.sort(key=lambda x: x["plate"].upper())

        grouped = {}
        for r in filtered:
            d = r["day"]
            if d not in grouped:
                grouped[d] = []
            grouped[d].append(r)

        for idx, day_str in enumerate(sorted(grouped.keys(), reverse=(self.sort_mode != 1))):
            day_node = self.plates_tree.insert("", "end", text=f"📁 {day_str}", tags=("day_tag",), open=(idx == 0))
            for rec in grouped[day_str]:
                disp_str = f"🕒 {rec['time']}   🚘 {rec['plate']}"
                item_id = self.plates_tree.insert(day_node, "end", text=f"  {disp_str}", tags=("plate_tag",))
                self.tree_item_map[item_id] = rec["fpath"]

    def _insert_single_plate_to_tree(self, day_str, time_str, plate_str, fpath):
        self.raw_records.append({
            "day": day_str,
            "time": time_str,
            "plate": plate_str,
            "fpath": fpath,
            "mtime": time.time()
        })
        self._apply_search_and_sort()

    def _show_plate_in_preview(self, fpath):
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
            self.preview_info_lbl.configure(text=f"Фото номера: {base}")
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

    def _on_plate_selected(self, event):
        sel = self.plates_tree.selection()
        if not sel:
            return
        fpath = self.tree_item_map.get(sel[0])
        if not fpath:
            return

        self._show_plate_in_preview(fpath)

        plate_dir = os.path.dirname(fpath)
        base_name = os.path.basename(fpath)

        screen_path = None
        if "_plate_" in base_name:
            time_prefix = base_name.split("_plate_")[0]
            candidate = os.path.join(plate_dir, f"{time_prefix}_full.jpg")
            if os.path.exists(candidate):
                screen_path = candidate
        else:
            time_prefix = base_name.split("_")[0]
            candidate = os.path.join(plate_dir, f"{time_prefix}_full.jpg")
            if os.path.exists(candidate):
                screen_path = candidate

        self._show_screenshot_in_canvas(screen_path)

    def _on_plate_double_clicked(self, event):
        sel = self.plates_tree.selection()
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
            "ANPR Video Monitor (best.onnx + CCT OCR)\n\n"
            "• Детекція: best.onnx (640x640)\n"
            "• Вирівнювання: Warp Perspective (520x110)\n"
            "• Розпізнавання: CCT Global OCR\n"
            f"• Фільтрація збереження: точність >= {self.min_percent_to_save}%\n"
            f"• Пауза після збереження: {self.sleep_time_after_save} с\n"
            f"• Відображення номера на екрані: {self.view_plate_time} с\n"
            "• Збереження пари (номер + повний скрін):\n"
            "  /plates/yyyy-MM-dd/hh-mm-ss_plate_OCRNAME_percent.jpg\n"
            "  /plates/yyyy-MM-dd/hh-mm-ss_full.jpg\n"
            f"• rect_scale_plate: {self.rect_scale_plate}\n"
            f"• Автоочищення: старше {self.delete_older_days} днів\n\n"
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

            self.status_lbl.configure(text="Статус: Активний (best.onnx + CCT OCR)", fg="#28a745")

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
        """ШІ-потік: аналіз у реальному часі та умовне збереження номера зі скріншотом."""
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
                frame_h, frame_w = frame_to_process.shape[:2]
                total_area = frame_h * frame_w
                new_boxes = []
                new_plate_img = None
                new_plate_dims = (0, 0)
                recognized_text = "UNKNOWN"
                recognized_conf = 0.0

                detected = self.detector.predict(frame_to_process, conf_thresh=0.25, imgsz=YOLO_IMGSZ)

                for (x1, y1, x2, y2) in detected:
                    bw = x2 - x1
                    bh = y2 - y1

                    if (bw * bh > total_area * 0.35) or (bh > frame_h * 0.45):
                        continue

                    new_boxes.append((x1, y1, x2, y2))
                    pad_x = int(bw * 0.06)
                    pad_y = int(bh * 0.06)
                    crop = frame_to_process[max(0, y1 - pad_y):min(frame_h, y2 + pad_y),
                                            max(0, x1 - pad_x):min(frame_w, x2 + pad_x)]

                    enhanced, ov_w, ov_h = rectify_and_enhance(crop)

                    if enhanced is not None:
                        plate_str = ""
                        plate_conf = 0.0
                        try:
                            res = self.ocr.predict(enhanced)
                            if res is not None:
                                raw_txt = getattr(res, "text", "")
                                raw_conf = getattr(res, "confidence", 0.0)

                                if isinstance(raw_conf, (list, tuple, np.ndarray)):
                                    plate_conf = float(np.mean(raw_conf)) if len(raw_conf) > 0 else 0.0
                                else:
                                    plate_conf = float(raw_conf)

                                clean_txt = re.sub(r'[^A-Z0-9]', '', str(raw_txt).strip().upper())
                                if len(clean_txt) >= 3:
                                    plate_str = clean_txt
                                    print(f"[OCR] Знайдено: {plate_str} (точність: {plate_conf*100:.1f}%)")
                        except Exception as ocr_err:
                            print(f"[OCR Помилка]: {ocr_err}")

                        new_plate_img = enhanced
                        new_plate_dims = (ov_w, ov_h)
                        recognized_text = plate_str if plate_str else "UNKNOWN"
                        recognized_conf = plate_conf
                        break

                now = time.time()
                conf_pct = int(round(recognized_conf * 100))

                # ПРЯМА УМОВА: номер розпізнано, точність >= min_percent_to_save,
                # і він СУВОРО відрізняється від попереднього збереженого.
                if (new_plate_img is not None and 
                    recognized_text != "UNKNOWN" and 
                    conf_pct >= self.min_percent_to_save and 
                    recognized_text != self.last_saved_plate_text):

                    self.last_saved_plate_text = recognized_text

                    # Пауза для детектора на зазначений sleep_time_after_save
                    self.next_inference_time = now + self.sleep_time_after_save

                    now_dt = datetime.now()
                    day_folder = now_dt.strftime("%Y-%m-%d")
                    time_prefix = now_dt.strftime("%H-%M-%S")

                    day_dir = os.path.join(self.plates_dir, day_folder)
                    os.makedirs(day_dir, exist_ok=True)

                    plate_filename = f"{time_prefix}_plate_{recognized_text}_{conf_pct}%.jpg"
                    full_filename = f"{time_prefix}_full.jpg"

                    plate_path = os.path.join(day_dir, plate_filename)
                    full_path = os.path.join(day_dir, full_filename)

                    async_write_image(plate_path, new_plate_img)
                    async_write_image(full_path, frame_to_process)

                    new_boxes = []

                    if self.sidebar_visible:
                        time_display = now_dt.strftime("%H:%M:%S")
                        self.root.after(0, lambda d=day_folder, t=time_display, p=recognized_text, f=plate_path: self._insert_single_plate_to_tree(d, t, p, f))

                with self.frame_lock:
                    self.cached_boxes = new_boxes
                    if new_plate_img is not None:
                        self.last_plate_img = new_plate_img
                        self.last_plate_dims = new_plate_dims
                        self.last_plate_text = recognized_text
                        self.last_plate_conf = recognized_conf
                        self.last_plate_time = now

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

            # Обробка: передача кадру лише тоді, коли минув таймаут паузи після збереження
            if (now >= self.next_inference_time) and not self.is_ai_busy:
                self.is_ai_busy = True
                with self.ai_lock:
                    self.ai_frame = frame.copy()
                self.ai_event.set()

            with self.frame_lock:
                boxes_to_draw = list(self.cached_boxes)
                plate_img = self.last_plate_img
                plate_dims = self.last_plate_dims
                plate_time = self.last_plate_time

            for (bx1, by1, bx2, by2) in boxes_to_draw:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            # Оверлей номера у правому верхньому кутку (масштаб за rect_scale_plate)
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
