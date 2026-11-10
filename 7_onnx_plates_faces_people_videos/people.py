import os
import time
import threading
from datetime import datetime
import cv2
import numpy as np


def async_write_image(path, img):
    """Асинхронний запис зображення на диск без блокування відеопотоку."""
    def _worker():
        try:
            cv2.imwrite(path, img)
        except Exception as e:
            print(f"[Помилка запису скріншота]: {e}")
    threading.Thread(target=_worker, daemon=True).start()


COCO_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    4: "Airplane",
    5: "Bus",
    6: "Train",
    7: "Truck",
    8: "Boat",
}

PERSON_CLASSES = [0]
VEHICLE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]


class PersonDetector:
    """Високошвидкісний детектор людей та транспорту на базі моделі YOLO11n (ONNX)."""
    def __init__(self, model_path: str, detect_vehicles: bool = False):
        self.model_path = model_path
        self.detect_vehicles = detect_vehicles
        self.use_ort = False
        self._update_target_classes()

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
        except Exception as e:
            print(f"[PersonDetector ONNX Runtime fallback to cv2.dnn]: {e}")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_ort = False

    def _update_target_classes(self):
        """Оновлення списку індексів цільових класів моделі."""
        if self.detect_vehicles:
            self.target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.target_classes = [0]
        # Рядки у вихідному тензорі YOLO11 (перші 4 рядки - координати cx, cy, w, h)
        self.target_rows = [4 + c for c in self.target_classes]

    def predict(self, frame: np.ndarray, conf_thresh: float = 0.45, imgsz: int = 640):
        """
        Детекція людей (клас 0) та, за наявності відповідного прапорця, транспорту (класи 1-8).
        """
        if frame is None or frame.size == 0:
            return []

        h_orig, w_orig = frame.shape[:2]

        # 1. Letterbox зі збереженням пропорцій
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

        # 2. Нормалізація та формування blob
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (imgsz, imgsz), swapRB=True, crop=False)

        # 3. Інференс моделі
        if self.use_ort:
            raw = self.session.run(self.output_names, {self.input_name: blob})[0]
        else:
            self.net.setInput(blob)
            raw = self.net.forward()

        # Вихід YOLO11: shape (1, 84, 8400) -> (84, 8400)
        preds = np.squeeze(raw)
        if preds.ndim != 2:
            return []

        # Якщо вихід (8400, 84), транспонуємо до (84, 8400)
        if preds.shape[0] > preds.shape[1]:
            preds = preds.T

        # Векторизована фільтрація за цільовими класами
        class_scores = preds[self.target_rows, :]  # Shape: (len(target_classes), 8400)
        max_scores = np.max(class_scores, axis=0)  # Shape: (8400,)
        best_class_indices = np.argmax(class_scores, axis=0)  # Shape: (8400,)

        mask = max_scores >= conf_thresh
        if not np.any(mask):
            return []

        valid_scores = max_scores[mask]
        valid_class_indices = best_class_indices[mask]
        valid_boxes = preds[:4, mask]

        # Перетворення cx, cy, w, h на координати для NMS
        cx = valid_boxes[0]
        cy = valid_boxes[1]
        bw = valid_boxes[2]
        bh = valid_boxes[3]

        boxes_for_nms = []
        scores_list = []
        original_boxes = []
        detected_classes = []
        detected_labels = []

        for i in range(len(valid_scores)):
            s = float(valid_scores[i])
            c_x, c_y, w, h = float(cx[i]), float(cy[i]), float(bw[i]), float(bh[i])
            c_id = self.target_classes[valid_class_indices[i]]
            lbl = COCO_CLASSES.get(c_id, f"Class_{c_id}")

            # Координати відносно оригінального кадру
            ox1 = (c_x - w / 2.0 - dw) / r
            oy1 = (c_y - h / 2.0 - dh) / r
            ox2 = (c_x + w / 2.0 - dw) / r
            oy2 = (c_y + h / 2.0 - dh) / r

            ox1 = max(0, min(w_orig - 1, int(round(ox1))))
            oy1 = max(0, min(h_orig - 1, int(round(oy1))))
            ox2 = max(0, min(w_orig, int(round(ox2))))
            oy2 = max(0, min(h_orig, int(round(oy2))))

            box_w = ox2 - ox1
            box_h = oy2 - oy1
            if box_w <= 10 or box_h <= 10:
                continue

            # Зміщення координат для роздільного NMS по класах (Batched NMS)
            boxes_for_nms.append([ox1 + c_id * 4096, oy1 + c_id * 4096, box_w, box_h])
            scores_list.append(s)
            original_boxes.append((ox1, oy1, ox2, oy2))
            detected_classes.append(c_id)
            detected_labels.append(lbl)

        if not boxes_for_nms:
            return []

        # Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_list, score_threshold=conf_thresh, nms_threshold=0.45)
        if len(indices) == 0:
            return []

        detections = []
        for idx in indices.flatten():
            detections.append({
                "box": original_boxes[idx],
                "conf": scores_list[idx],
                "class_id": detected_classes[idx],
                "label": detected_labels[idx]
            })

        return detections


class PersonEngine:
    """
    Автономний модуль детекції людей та транспорту, зі щосекундним збереженням чистих скріншотів.
    Зберігає кадри, коли у полі зору камери є відповідні об'єкти (людина або транспорт).
    """
    def __init__(self, base_dir: str, config: dict, on_saved_callback=None):
        self.base_dir = base_dir
        self.config = config
        self.on_saved_callback = on_saved_callback

        self.people_dir = os.path.join(base_dir, "people")
        os.makedirs(self.people_dir, exist_ok=True)

        self.conf_thresh = float(self.config.get("people_det_thresh", 0.45))
        self.timer_ms = int(self.config.get("people_timer_ms", 1000))
        self.interval_sec = max(0.1, float(self.timer_ms) / 1000.0)

        val_veh = str(self.config.get("people_detect_transport", "true")).strip().lower()
        self.detect_vehicles = val_veh in ("true", "1", "yes", "on")

        model_path = os.path.join(base_dir, "model", "yolo11n.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл моделі не знайдено: {model_path}")

        self.detector = PersonDetector(model_path, detect_vehicles=self.detect_vehicles)
        self.last_shot_time = 0.0

    def update_settings(self, conf_thresh: float = None, timer_ms: int = None, detect_transport: bool = None):
        """Динамічне оновлення налаштувань детектора та таймера."""
        if conf_thresh is not None:
            self.conf_thresh = float(conf_thresh)
        if timer_ms is not None:
            self.timer_ms = int(timer_ms)
            self.interval_sec = max(0.1, float(self.timer_ms) / 1000.0)
        if detect_transport is not None:
            self.detect_vehicles = bool(detect_transport)
            self.detector.detect_vehicles = self.detect_vehicles
            self.detector._update_target_classes()

    def process_frame(self, frame: np.ndarray):
        """
        Обробка чергового кадру: детекція людей/транспорту, перевірка таймера та зняття скріншота.
        Повертає список [(box, conf_percent, label)] для малювання рамок у відеоплеєрі.
        """
        detections = self.detector.predict(frame, conf_thresh=self.conf_thresh)
        if not detections:
            return []

        now = time.time()
        # Збереження скріншота, якщо в кадрі є хоча б один шуканий об'єкт і інтервал минув
        if (now - self.last_shot_time) >= self.interval_sec:
            self.last_shot_time = now
            self._save_screenshot(frame, detections)

        display_boxes = [(det["box"], int(round(det["conf"] * 100)), det.get("label", "Person")) for det in detections]
        return display_boxes

    def _save_screenshot(self, frame: np.ndarray, detections: list):
        """Збереження чистого повнорозмірного кадру на диск у папку відповідного дня."""
        now_dt = datetime.now()
        day_folder = now_dt.strftime("%Y-%m-%d")
        day_dir = os.path.join(self.people_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        detected_labels = list(dict.fromkeys(det.get("label", "Person") for det in detections))
        summary_label = ", ".join(detected_labels)
        slug_label = "-".join(detected_labels)

        # Якщо інтервал менше 1 секунди, додаємо мілісекунди для унікальності назви
        if self.timer_ms < 1000:
            time_prefix = now_dt.strftime("%H-%M-%S") + f"_{int(now_dt.microsecond / 1000):03d}"
        else:
            time_prefix = now_dt.strftime("%H-%M-%S")

        filename = f"{time_prefix}_{slug_label}.jpg"
        full_path = os.path.join(day_dir, filename)

        # Запис чистого оригінального кадру
        async_write_image(full_path, frame.copy())

        time_display = now_dt.strftime("%H:%M:%S")
        print(f"[ФІКСАЦІЯ ОБ'ЄКТА ({summary_label})] Збережено скріншот: {full_path}")

        if self.on_saved_callback:
            self.on_saved_callback(day_folder, time_display, summary_label, full_path)
