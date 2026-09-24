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

# Інтервали та параметри
INFERENCE_INTERVAL = 1.0     # Детекція 1 раз на секунду
SCREENSHOT_INTERVAL = 10.0   # Знімок повного кадру кожні 10 с
PLATE_SAVE_INTERVAL = 5.0    # Збереження номера не частіше 1 разу на 5 с
YOLO_IMGSZ = 640             # Розмір вхідного тензора моделі

# Пропорції пластин номерних знаків
PLATE_ASPECT_RATIO_STANDARD = 520.0 / 112.0
PLATE_ASPECT_RATIO_SQUARE = 300.0 / 150.0


def async_write_image(path, img):
    """Асинхронний запис зображення на диск без блокування відеопотоку."""
    def _worker():
        try:
            cv2.imwrite(path, img)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


class ONNXDetector:
    """Високопродуктивний детектор ONNX із C++ прискоренням (blobFromImage)."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.use_ort = False

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            # Обмежуємо потік ONNX до 2 ядер, щоб відеопотік та GUI не голодували
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
            # Резервний рушій OpenCV DNN
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_ort = False

    def predict(self, frame, conf_thresh=0.22, imgsz=480):
        h_orig, w_orig = frame.shape[:2]

        # Швидкий Letterbox через OpenCV
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

        # C++ апаратна конвертація (у 5-10 разів швидше за чистий NumPy)
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
    h, w = plate_crop.shape[:2]
    if h < 12 or w < 24:
        return None, 0, 0

    aspect_ratio = w / float(h)
    if aspect_ratio < 3.0:
        target_w = 520
        target_h = int(target_w / PLATE_ASPECT_RATIO_SQUARE)
    else:
        target_w = 840
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
        self.root.title("ANPR Video Monitor (ONNX Optimized)")
        self.root.configure(bg="#0d0d11")

        self.root.attributes("-fullscreen", True)
        self.root.protocol("WM_DELETE_WINDOW", self._prevent_close)
        self._setup_lockdown_bindings()

        self.config_ini_path = os.path.join(BASE_DIR, "config.ini")
        self.config = self._load_config()
        self.is_autostart_active = bool(self.config.get("autostart", False))
        self.save_screenshots = bool(self.config.get("screenshots", False))
        self.plate_timeout = float(self.config.get("plate_timeout", 30))
        self.delete_older_hours = float(self.config.get("delete_older", 24))

        self.screenshots_dir = os.path.join(BASE_DIR, "screenshots")
        self.plates_dir = os.path.join(BASE_DIR, "plates")
        os.makedirs(self.plates_dir, exist_ok=True)
        if self.save_screenshots:
            os.makedirs(self.screenshots_dir, exist_ok=True)

        self._sync_windows_startup(self.is_autostart_active)

        model_path = os.path.join(BASE_DIR, "model", "best.onnx")
        if not os.path.exists(model_path):
            messagebox.showerror("Помилка моделі", f"Файл моделі не знайдено:\n{model_path}")
            sys.exit(1)

        self.detector = ONNXDetector(model_path)

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
        self.last_plate_time = 0.0

        self.sidebar_visible = False
        self.preview_photo_tk = None
        self.tree_item_map = {}

        self.panel_visible = True
        self.hide_timer = None

        self._build_ui()
        self.root.bind_all("<Motion>", self._on_mouse_motion)

        self.root.after(5000, self._check_and_delete_old_files)

        if self.is_autostart_active:
            self.root.after(300, self.toggle_stream)

    def _setup_lockdown_bindings(self):
        for key in ["<Alt-F4>", "<Alt-KeyPress-F4>", "<Control-w>", "<Control-W>", "<Control-q>", "<Control-Q>", "<Escape>", "<F11>"]:
            self.root.bind_all(key, lambda e: "break")
        self.root.bind_all("<Control-Alt-Shift-KeyPress-Q>", self._admin_exit)
        self.root.bind_all("<Control-Alt-Shift-KeyPress-q>", self._admin_exit)

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
        defaults = {"source": "0", "autostart": False, "screenshots": False, "plate_timeout": 30, "delete_older": 24}
        parser = configparser.ConfigParser()
        if os.path.exists(self.config_ini_path):
            try:
                parser.read(self.config_ini_path, encoding="utf-8")
                if not parser.has_section("SETTINGS"):
                    parser.add_section("SETTINGS")
                needs_saving = False
                if parser.has_option("SETTINGS", "plates"):
                    parser.remove_option("SETTINGS", "plates")
                    needs_saving = True
                if not parser.has_option("SETTINGS", "source"):
                    parser.set("SETTINGS", "source", str(defaults["source"]))
                    needs_saving = True
                for b_key in ["autostart", "screenshots"]:
                    if not parser.has_option("SETTINGS", b_key):
                        parser.set("SETTINGS", b_key, str(defaults[b_key]).lower())
                        needs_saving = True
                for i_key in ["plate_timeout", "delete_older"]:
                    if not parser.has_option("SETTINGS", i_key):
                        parser.set("SETTINGS", i_key, str(defaults[i_key]))
                        needs_saving = True

                cfg = {
                    "source": parser.get("SETTINGS", "source", fallback="0").strip().strip('\'"'),
                    "autostart": parser.getboolean("SETTINGS", "autostart", fallback=False),
                    "screenshots": parser.getboolean("SETTINGS", "screenshots", fallback=False),
                    "plate_timeout": parser.getint("SETTINGS", "plate_timeout", fallback=30),
                    "delete_older": parser.getint("SETTINGS", "delete_older", fallback=24)
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
            "autostart": str(data.get("autostart", False)).lower(),
            "screenshots": str(data.get("screenshots", False)).lower(),
            "plate_timeout": str(data.get("plate_timeout", 30)),
            "delete_older": str(data.get("delete_older", 24))
        }
        try:
            with open(self.config_ini_path, "w", encoding="utf-8") as f:
                f.write("; ANPR Video Monitor Configuration (Optimized Engine)\n")
                f.write("; source: 0, rtsp://... або d:\\video.mp4\n")
                f.write("; autostart: true / false\n")
                f.write("; screenshots: true / false\n")
                f.write("; plate_timeout: час збереження номера на екрані (с)\n")
                f.write("; delete_older: видаляти файли старше зазначених годин\n\n")
                parser.write(f)
        except Exception:
            pass

    def _check_and_delete_old_files(self):
        try:
            if self.delete_older_hours > 0:
                cutoff = time.time() - (self.delete_older_hours * 3600.0)
                for folder in [self.screenshots_dir, self.plates_dir]:
                    if os.path.exists(folder):
                        for root_dir, dirs, files in os.walk(folder, topdown=False):
                            for fname in files:
                                fpath = os.path.join(root_dir, fname)
                                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                                    try:
                                        os.remove(fpath)
                                    except Exception:
                                        pass
                            if root_dir != folder and not os.listdir(root_dir):
                                try:
                                    os.rmdir(root_dir)
                                except Exception:
                                    pass
        except Exception:
            pass
        self.root.after(3600000, self._check_and_delete_old_files)

    def _build_ui(self):
        self.ctrl_frame = tk.Frame(self.root, bg="#1e1e24", padx=12, pady=10)
        self.ctrl_frame.pack(fill=tk.X, side=tk.TOP)

        tk.Label(self.ctrl_frame, text="Відеокамера / Джерело:", fg="#e0e0e0", bg="#1e1e24", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(5, 3))
        tk.Button(self.ctrl_frame, text="?", command=self._show_help, bg="#3a3a46", fg="#00ffff", font=("Segoe UI", 9, "bold"), width=2, relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 8))

        self.source_var = tk.StringVar(value=str(self.config.get("source", "0")))
        ttk.Entry(self.ctrl_frame, width=38, textvariable=self.source_var, state="readonly").pack(side=tk.LEFT, padx=3)

        self.autostart_var = tk.BooleanVar(value=self.is_autostart_active)
        tk.Checkbutton(self.ctrl_frame, text="Автозавантаження", variable=self.autostart_var, state=tk.DISABLED, disabledforeground="#ffffff", bg="#1e1e24", selectcolor="#2b2b36", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=12)

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

        # Ліва бічна панель (за замовчуванням прихована)
        self.side_panel = tk.Frame(self.main_body, bg="#1e1e24", width=320)
        self.side_panel.pack_propagate(False)

        side_header = tk.Frame(self.side_panel, bg="#282832", padx=10, pady=8)
        side_header.pack(fill=tk.X, side=tk.TOP)
        tk.Label(side_header, text="Історія номерів (День / Час)", fg="#00e5ff", bg="#282832", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        tk.Button(side_header, text="↻", command=self.refresh_saved_list, bg="#3a3a48", fg="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"), width=3, cursor="hand2").pack(side=tk.RIGHT)

        tree_frame = tk.Frame(self.side_panel, bg="#16161d", padx=6, pady=6)
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
        self.plates_tree.column("#0", width=290, stretch=True)
        self.plates_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.plates_tree.yview)

        self.plates_tree.tag_configure("day_tag", font=("Segoe UI", 10, "bold"), foreground="#00e5ff")
        self.plates_tree.tag_configure("plate_tag", font=("Consolas", 10), foreground="#ffffff")
        self.plates_tree.bind("<<TreeviewSelect>>", self._on_plate_selected)
        self.plates_tree.bind("<Double-1>", self._on_plate_double_clicked)

        self.preview_container = tk.Frame(self.side_panel, bg="#14141a", pady=10, padx=8)
        self.preview_container.pack(fill=tk.X, side=tk.BOTTOM)
        self.preview_info_lbl = tk.Label(self.preview_container, text="Перегляд вибраного номера:", fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9))
        self.preview_info_lbl.pack(anchor=tk.W, pady=(0, 5))

        self.preview_label = tk.Label(self.preview_container, bg="#1a1a22", text="[Виберіть номер зі списку]\n(подвійний клік — оригінал)", fg="#707080", font=("Segoe UI", 9), pady=20, relief=tk.SOLID, borderwidth=1)
        self.preview_label.pack(fill=tk.X)

        self.video_container = tk.Frame(self.main_body, bg="#0d0d11")
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.main_video_label = tk.Label(self.video_container, bg="#0d0d11")
        self.main_video_label.pack(fill=tk.BOTH, expand=True)

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

    def refresh_saved_list(self):
        """Повне оновлення списку збережених номерів."""
        self.plates_tree.delete(*self.plates_tree.get_children())
        self.tree_item_map.clear()
        if not os.path.exists(self.plates_dir):
            return

        grouped_data = {}
        for root_dir, dirs, files in os.walk(self.plates_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    fpath = os.path.join(root_dir, fname)
                    base_name, _ = os.path.splitext(fname)
                    rel_dir = os.path.relpath(root_dir, self.plates_dir)
                    if rel_dir != ".":
                        day_key = rel_dir
                        time_disp = base_name.replace("plate_", "", 1)
                    else:
                        parts = base_name.split("_")
                        day_key = parts[1] if len(parts) >= 3 else datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d")
                        time_disp = "_".join(parts[2:]) if len(parts) >= 3 else datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%H-%M-%S")

                    if day_key not in grouped_data:
                        grouped_data[day_key] = []
                    grouped_data[day_key].append((time_disp, fpath))

        for idx, day_str in enumerate(sorted(grouped_data.keys(), reverse=True)):
            day_node = self.plates_tree.insert("", "end", text=f"📁 {day_str}", tags=("day_tag",), open=(idx == 0))
            records = grouped_data[day_str]
            records.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
            for time_disp, fpath in records:
                item_id = self.plates_tree.insert(day_node, "end", text=f"  🕒 {time_disp}", tags=("plate_tag",))
                self.tree_item_map[item_id] = fpath

    def _insert_single_plate_to_tree(self, day_str, time_disp, fpath):
        """Швидка точкова вставка одного номера без блокування інтерфейсу."""
        day_node = None
        for item in self.plates_tree.get_children():
            if day_str in self.plates_tree.item(item, "text"):
                day_node = item
                break

        if not day_node:
            day_node = self.plates_tree.insert("", 0, text=f"📁 {day_str}", tags=("day_tag",), open=True)

        item_id = self.plates_tree.insert(day_node, 0, text=f"  🕒 {time_disp}", tags=("plate_tag",))
        self.tree_item_map[item_id] = fpath

    def _show_plate_in_preview(self, fpath):
        if not fpath or not os.path.exists(fpath):
            return
        try:
            pil_img = Image.open(fpath)
            orig_w, orig_h = pil_img.size
            target_w = 280
            target_h = max(1, int(target_w * (orig_h / float(orig_w))))
            pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            self.preview_photo_tk = ImageTk.PhotoImage(pil_img)
            base = os.path.basename(fpath).replace("plate_", "").replace(".jpg", "")
            self.preview_info_lbl.configure(text=f"Номер: {base}")
            self.preview_label.configure(image=self.preview_photo_tk, text="", pady=0)
        except Exception:
            pass

    def _on_plate_selected(self, event):
        sel = self.plates_tree.selection()
        if not sel:
            return
        fpath = self.tree_item_map.get(sel[0])
        if fpath:
            self._show_plate_in_preview(fpath)

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
            "ANPR Video Monitor (ONNX Ultra-Smooth)\n\n"
            "• Детекція повністю відокремлена від відеотрансляції.\n"
            "• Відео не підгальмовує під час інференсу ШІ.\n"
            "• Номери записуються у /plates/РРРР-ММ-ДД/\n"
            "• Кнопка 'Збережені' керує бічною панеллю.\n\n"
            "Сервісний вихід: Ctrl + Alt + Shift + Q"
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

            self.status_lbl.configure(text="Статус: Працює (ONNX)", fg="#28a745")

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
        last_plate_save_time = 0.0

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

                detected = self.detector.predict(frame_to_process, conf_thresh=0.30, imgsz=YOLO_IMGSZ)

                for (x1, y1, x2, y2) in detected:
                    bw = x2 - x1
                    bh = y2 - y1

                    if (bw * bh > total_area * 0.25) or (bh > frame_h * 0.40):
                        continue

                    new_boxes.append((x1, y1, x2, y2))
                    pad_x = int(bw * 0.05)
                    pad_y = int(bh * 0.05)
                    crop = frame_to_process[max(0, y1 - pad_y):min(frame_h, y2 + pad_y),
                                            max(0, x1 - pad_x):min(frame_w, x2 + pad_x)]
                    enhanced, ov_w, ov_h = rectify_and_enhance(crop)

                    if enhanced is not None:
                        new_plate_img = enhanced
                        new_plate_dims = (ov_w, ov_h)
                        break

                now = time.time()
                if new_plate_img is not None and (now - last_plate_save_time >= PLATE_SAVE_INTERVAL):
                    last_plate_save_time = now
                    now_dt = datetime.now()
                    day_folder = now_dt.strftime("%Y-%m-%d")
                    time_filename = now_dt.strftime("%H-%M-%S_%f")[:-3]

                    day_dir = os.path.join(self.plates_dir, day_folder)
                    os.makedirs(day_dir, exist_ok=True)
                    plate_path = os.path.join(day_dir, f"plate_{time_filename}.jpg")
                    async_write_image(plate_path, new_plate_img)

                    # Неблокуюча швидка вставка у бічну панель
                    if self.sidebar_visible:
                        self.root.after(0, lambda d=day_folder, t=time_filename, p=plate_path: self._insert_single_plate_to_tree(d, t, p))

                with self.frame_lock:
                    self.cached_boxes = new_boxes
                    if new_plate_img is not None:
                        self.last_plate_img = new_plate_img
                        self.last_plate_dims = new_plate_dims
                        self.last_plate_time = now

            except Exception:
                pass
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

        last_inference_dispatch = 0.0
        last_screenshot_time = 0.0

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

            # 1. Асинхронний знімок повного кадру без блокування потоку
            if self.save_screenshots and (now - last_screenshot_time >= SCREENSHOT_INTERVAL):
                last_screenshot_time = now
                shot_path = os.path.join(self.screenshots_dir, f"screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                async_write_image(shot_path, frame.copy())

            # 2. Передача кадру в ШІ-потік
            if (now - last_inference_dispatch >= INFERENCE_INTERVAL) and not self.is_ai_busy:
                last_inference_dispatch = now
                self.is_ai_busy = True
                with self.ai_lock:
                    self.ai_frame = frame.copy()
                self.ai_event.set()

            # 3. Накладання оверлеїв поверх кадру
            with self.frame_lock:
                boxes_to_draw = list(self.cached_boxes)
                plate_img = self.last_plate_img
                plate_dims = self.last_plate_dims
                plate_time = self.last_plate_time

            for (bx1, by1, bx2, by2) in boxes_to_draw:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

            if plate_img is not None and (now - plate_time < self.plate_timeout):
                margin = 20
                cur_w, cur_h = plate_dims
                if cur_w > 0 and cur_h > 0:
                    max_allowed_w = int(frame_w * 0.65)
                    if cur_w > max_allowed_w and max_allowed_w > 120:
                        scale = max_allowed_w / float(cur_w)
                        display_img = cv2.resize(plate_img, (int(cur_w * scale), int(cur_h * scale)), interpolation=cv2.INTER_LINEAR)
                    else:
                        display_img = plate_img

                    dw, dh = display_img.shape[1], display_img.shape[0]
                    header_h = 32
                    x2_ov = frame_w - margin
                    x1_ov = x2_ov - dw
                    y1_ov = margin + header_h
                    y2_ov = y1_ov + dh

                    if x1_ov > 0 and y2_ov < frame_h:
                        cv2.rectangle(frame, (x1_ov - 6, y1_ov - header_h), (x2_ov + 6, y2_ov + 6), (18, 18, 22), -1)
                        cv2.rectangle(frame, (x1_ov - 6, y1_ov - header_h), (x2_ov + 6, y2_ov + 6), (0, 255, 0), 4)
                        cv2.putText(frame, "TARGET PLATE (ZOOM 2X)", (x1_ov + 4, y1_ov - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
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
            # Оновлюємо картинку лише тоді, коли прийшов дійсно новий кадр
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
