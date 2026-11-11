import os
import re
import sys
import time
import math
import inspect
import importlib
import threading
from collections import Counter
from datetime import datetime
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from fast_alpr.default_ocr import DefaultOCR
except ImportError:
    DefaultOCR = None

YOLO_IMGSZ = 640

# Пропорції пластин номерних знаків
PLATE_ASPECT_RATIO_STANDARD = 520.0 / 112.0
PLATE_ASPECT_RATIO_SQUARE = 300.0 / 150.0


def _ensure_deps():
    global cv2, DefaultOCR
    if cv2 is None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            pass
    if DefaultOCR is None:
        try:
            DefaultOCR = importlib.import_module("fast_alpr.default_ocr").DefaultOCR
        except ImportError:
            pass


def refresh_capabilities():
    global cv2, DefaultOCR
    importlib.invalidate_caches()
    for mod_name in ("cv2", "fast_alpr", "fast_alpr.default_ocr", "fast_plate_ocr", "onnxruntime"):
        try:
            if mod_name in sys.modules:
                importlib.reload(sys.modules[mod_name])
        except Exception:
            pass
    try:
        cv2 = importlib.import_module("cv2")
    except ImportError:
        cv2 = None
    try:
        DefaultOCR = importlib.import_module("fast_alpr.default_ocr").DefaultOCR
    except ImportError:
        DefaultOCR = None


def get_status_summary(base_dir: str = "") -> str:
    missing_libs = []
    if cv2 is None:
        missing_libs.append("opencv-python")
    if DefaultOCR is None:
        missing_libs.append("fast-alpr")
    if missing_libs:
        return f"Потрібні бібліотеки: {', '.join(missing_libs)}"

    if base_dir:
        req_files = (
            os.path.join(base_dir, "models", "best.onnx"),
            os.path.join(base_dir, "models", "cct_xs_v2_global.onnx"),
            os.path.join(base_dir, "models", "cct_xs_v2_global_plate_config.yaml"),
        )
        missing_m = [os.path.basename(p) for p in req_files if not os.path.exists(p)]
        if missing_m:
            return f"Відсутні моделі: {', '.join(missing_m)}"
    return "Готовий (YOLO best.onnx + CCT OCR)"


def async_write_image(path, img):
    """Асинхронний запис зображення на диск без блокування відеопотоку."""
    def _worker():
        try:
            if cv2 is not None:
                cv2.imwrite(path, img)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def async_write_text(path, content):
    """Асинхронний запис дебаг-лога на диск."""
    def _worker():
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def calculate_sharpness(img):
    """Розрахунок дисперсії Лапласіана для оцінки різкості кадру."""
    if img is None or img.size == 0 or cv2 is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_plate_quality(crop_bgr, text, conf, sharpness):
    """Комплексна оцінка якості кадру з пріоритетом чіткості над розміром."""
    text_len = len(text)
    if text_len < 4 or text == "UNKNOWN":
        return 0.0

    if text_len >= 7:
        len_mult = 1.0
    elif text_len == 6:
        len_mult = 0.85
    elif text_len == 5:
        len_mult = 0.70
    else:
        len_mult = 0.50

    sharp_norm = min(1.0, max(0.0, sharpness / 150.0))
    h, w = crop_bgr.shape[:2]
    area_norm = min(1.0, (w * h) / (180.0 * 45.0))

    return (conf * 0.45 + sharp_norm * 0.45 + area_norm * 0.10) * len_mult


def compute_iou(boxA, boxB):
    """Розрахунок перетину рамок (IoU)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def box_centroid_dist(b1, b2):
    """Евклідова відстань між центрами рамок."""
    c1x, c1y = (b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0
    c2x, c2y = (b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0
    return math.hypot(c1x - c2x, c1y - c2y)


def are_plates_similar(p1, p2, max_diff=1):
    """Перевірка схожості номерів для фільтрації дублікатів з похибкою в 1 символ."""
    if p1 == p2:
        return True
    if abs(len(p1) - len(p2)) > max_diff:
        return False

    if len(p1) == len(p2):
        diffs = sum(1 for a, b in zip(p1, p2) if a != b)
        return diffs <= max_diff

    s_short, s_long = (p1, p2) if len(p1) < len(p2) else (p2, p1)
    for i in range(len(s_long)):
        if s_long[:i] + s_long[i + 1:] == s_short:
            return True
    return False


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
    if cv2 is None or plate_crop is None:
        return None, 0, 0
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
    crop_area = float(w * h)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)
        if len(approx) == 4:
            c_area = cv2.contourArea(approx)
            if c_area >= crop_area * 0.65 and cv2.isContourConvex(approx):
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


class PlateTrack:
    """Об'єкт трекінгу номера із захистом від UNKNOWN і збереженням найрізкішого кадру на кожен варіант тексту."""
    def __init__(self, track_id, bbox, plate_img, frame, text, conf, sharpness):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen = time.time()
        self.first_seen = self.last_seen
        self.saved = False

        self.best_plate_img = plate_img
        self.best_frame = frame.copy()
        self.best_sharpness = sharpness
        self.best_conf = conf if text != "UNKNOWN" else 0.0
        self.best_text = text

        self.history_records = []
        self.text_history = []
        self.conf_history = []

        initial_score = evaluate_plate_quality(plate_img, text, conf, sharpness)
        self.best_score = initial_score
        self.frames_tracked = 0
        self.best_candidates_by_text = {}

        self.append_record(bbox, plate_img, frame, text, conf, sharpness, initial_score)

    def append_record(self, bbox, plate_img, frame, text, conf, sharpness, score):
        self.frames_tracked += 1
        t_rel = round(time.time() - self.first_seen, 3)
        self.history_records.append({
            "rel_time": t_rel,
            "text": text,
            "conf": round(conf * 100, 1),
            "sharpness": round(sharpness, 1),
            "score": round(score, 3),
            "bbox": bbox
        })
        if text != "UNKNOWN" and len(text) >= 4:
            self.text_history.append(text)
            self.conf_history.append(conf)
            existing = self.best_candidates_by_text.get(text)
            if existing is None or sharpness > existing["sharpness"]:
                self.best_candidates_by_text[text] = {
                    "plate_img": plate_img,
                    "frame": frame.copy(),
                    "text": text,
                    "conf": conf,
                    "sharpness": sharpness,
                    "score": score
                }

    def update(self, bbox, plate_img, frame, text, conf, sharpness):
        self.bbox = bbox
        self.last_seen = time.time()
        current_score = evaluate_plate_quality(plate_img, text, conf, sharpness)

        self.append_record(bbox, plate_img, frame, text, conf, sharpness, current_score)

        if current_score > self.best_score and text != "UNKNOWN":
            self.best_score = current_score
            self.best_plate_img = plate_img
            self.best_frame = frame.copy()
            self.best_sharpness = sharpness
            self.best_conf = conf
            self.best_text = text

    def get_best_candidate(self, consensus_text=None):
        """Знаходить абсолютно найчіткіший збережений кадр для узгодженого номера."""
        if not self.best_candidates_by_text:
            return self.best_plate_img, self.best_frame, self.best_sharpness, self.best_conf

        target_text = consensus_text or self.best_text
        if target_text in self.best_candidates_by_text:
            best_item = self.best_candidates_by_text[target_text]
            return best_item["plate_img"], best_item["frame"], best_item["sharpness"], best_item["conf"]

        candidates = list(self.best_candidates_by_text.values())
        matching = [item for item in candidates if are_plates_similar(item["text"], target_text, max_diff=1)]
        if not matching:
            matching = candidates

        best_item = max(matching, key=lambda x: x["sharpness"])
        return best_item["plate_img"], best_item["frame"], best_item["sharpness"], best_item["conf"]

    def get_consensus_text_and_conf(self):
        """Визначення найбільш ймовірного номера методом Majority Voting."""
        if not self.text_history:
            return "UNKNOWN", 0.0

        counter = Counter(self.text_history)
        most_common = counter.most_common()

        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            final_text = most_common[0][0]
        else:
            final_text = self.best_text if self.best_text in self.text_history else most_common[0][0]

        matched_confs = [c for t, c in zip(self.text_history, self.conf_history) if t == final_text]
        final_conf = float(np.mean(matched_confs)) if matched_confs else 0.0

        return final_text, final_conf


class ONNXDetector:
    """Високопродуктивний потокобезпечний детектор ONNX (best.onnx)."""
    def __init__(self, model_path, providers=None, intra_op_threads=2):
        _ensure_deps()
        self.model_path = model_path
        self.use_ort = False
        self.active_providers = ["OpenCV-CPU"]
        self._lock = threading.Lock()

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            cpu_cnt = os.cpu_count() or 4
            opts.intra_op_num_threads = max(1, min(int(intra_op_threads), max(1, cpu_cnt - 1)))
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            req_providers = list(providers) if providers else ["CPUExecutionProvider"]
            try:
                self.session = ort.InferenceSession(model_path, opts, providers=req_providers)
            except Exception:
                self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
            self.active_providers = self.session.get_providers()
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.use_ort = True
        except Exception:
            if cv2 is None:
                raise ImportError("Немає onnxruntime або opencv-python для ONNXDetector")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_ort = False

    def predict(self, frame, conf_thresh=0.22, imgsz=480):
        if cv2 is None or frame is None or frame.size == 0:
            return []
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

        with self._lock:
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
            if bw > 25 and bh > 10:
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


class ThreadSafePlateOCR:
    """Обгортка навколо DefaultOCR з підтримкою GPU-провайдерів і потокобезпечним викликом."""
    def __init__(self, model_path: str, config_path: str, providers=None):
        _ensure_deps()
        if DefaultOCR is None:
            raise ImportError("Пакет 'fast-alpr' / DefaultOCR не встановлено")
        self.model_path = model_path
        self.config_path = config_path
        self._lock = threading.Lock()
        self.ocr = self._create_ocr(model_path, config_path, providers=providers)
        self.active_providers = self._detect_active_providers()

    @staticmethod
    def _create_ocr(model_path: str, config_path: str, providers=None):
        req_providers = list(providers) if providers else ["CPUExecutionProvider"]
        use_cuda = any("CUDA" in p for p in req_providers)
        use_cpu_only = req_providers == ["CPUExecutionProvider"]
        device = "cpu" if use_cpu_only else ("cuda" if use_cuda else "auto")

        try:
            sig = inspect.signature(DefaultOCR.__init__)
            kwargs = {
                "hub_ocr_model": None,
                "device": device,
                "model_path": model_path,
                "config_path": config_path,
            }
            if "providers" in sig.parameters and providers:
                kwargs["providers"] = req_providers
            return DefaultOCR(**kwargs)
        except Exception:
            return DefaultOCR(
                hub_ocr_model=None,
                device="cpu",
                model_path=model_path,
                config_path=config_path,
            )

    def _detect_active_providers(self):
        for attr in ("ocr_model", "model", "session"):
            obj = getattr(self.ocr, attr, None)
            if obj is not None:
                sess = getattr(obj, "session", obj)
                if hasattr(sess, "get_providers"):
                    try:
                        return sess.get_providers()
                    except Exception:
                        pass
        return ["CPUExecutionProvider"]

    def predict(self, img):
        with self._lock:
            return self.ocr.predict(img)


class PlateEngine:
    """Рушій фіксації та розпізнавання номерних знаків (підтримує спільні моделі та per-camera ізоляцію)."""
    def __init__(
        self,
        base_dir,
        config,
        on_saved_callback=None,
        detector: ONNXDetector = None,
        ocr=None,
        output_dir: str = None,
        providers=None,
    ):
        self.base_dir = base_dir
        self.config = dict(config or {})
        self.on_saved_callback = on_saved_callback

        self.plates_dir = output_dir or os.path.join(self.base_dir, "plates")
        os.makedirs(self.plates_dir, exist_ok=True)

        if detector is not None:
            self.detector = detector
        else:
            model_path = os.path.join(self.base_dir, "models", "best.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл детектора не знайдено: {model_path}")
            self.detector = ONNXDetector(model_path, providers=providers)

        if ocr is not None:
            self.ocr = ocr
        else:
            ocr_model_path = os.path.join(self.base_dir, "models", "cct_xs_v2_global.onnx")
            ocr_config_path = os.path.join(self.base_dir, "models", "cct_xs_v2_global_plate_config.yaml")
            if not os.path.exists(ocr_model_path) or not os.path.exists(ocr_config_path):
                raise FileNotFoundError(f"Файли CCT OCR відсутні в {os.path.join(self.base_dir, 'models')}")
            self.ocr = ThreadSafePlateOCR(ocr_model_path, ocr_config_path, providers=providers)

        self.active_tracks = {}
        self.next_track_id = 1
        self.track_timeout = 1.2
        self.recently_saved_plates = {}
        self._state_lock = threading.Lock()
        self._reload_params()

    def _reload_params(self):
        self.plates_det_thresh = float(self.config.get("plates_det_thresh", self.config.get("det_thresh", 0.18)))
        self.min_percent_to_save = int(self.config.get("plates_min_percent_to_save", self.config.get("min_percent_to_save", 80)))
        self.sleep_timeout = float(self.config.get("plates_sleep_time_after_save", self.config.get("sleep_time_after_save", 5.0)))
        raw_debug = self.config.get("plates_debug", self.config.get("debug", False))
        if isinstance(raw_debug, str):
            self.save_debug = raw_debug.strip().lower() in ("true", "1", "yes", "on")
        else:
            self.save_debug = bool(raw_debug)

    def update_config(self, config: dict):
        with self._state_lock:
            self.config = dict(config or {})
            self._reload_params()

    def process_frame(self, frame):
        """Обробка кадру з детекцією, OCR, трекінгом та поверненням даних для оверлею."""
        if frame is None or frame.size == 0:
            return [], None

        frame_h, frame_w = frame.shape[:2]
        total_area = frame_h * frame_w
        current_detections = []
        detected_boxes = self.detector.predict(frame, conf_thresh=self.plates_det_thresh, imgsz=YOLO_IMGSZ)

        for (x1, y1, x2, y2) in detected_boxes:
            bw = x2 - x1
            bh = y2 - y1

            if (bw * bh > total_area * 0.40) or (bh > frame_h * 0.50):
                continue
            if bw < 26 or bh < 8:
                continue

            pad_x = int(bw * 0.12)
            pad_y = int(bh * 0.14)
            crop = frame[max(0, y1 - pad_y):min(frame_h, y2 + pad_y),
                         max(0, x1 - pad_x):min(frame_w, x2 + pad_x)]

            enhanced, ov_w, ov_h = rectify_and_enhance(crop)
            if enhanced is None:
                continue

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

                    clean_txt = re.sub(r"[^A-Z0-9]", "", str(raw_txt).strip().upper())
                    if len(clean_txt) >= 3:
                        plate_str = clean_txt
            except Exception as ocr_err:
                print(f"[OCR Error]: {ocr_err}")

            sharpness = calculate_sharpness(enhanced)

            if plate_str:
                text_res = plate_str
                conf_res = plate_conf
            else:
                text_res = "UNKNOWN"
                conf_res = 0.0

            current_detections.append({
                "box": (x1, y1, x2, y2),
                "plate_img": enhanced,
                "dims": (ov_w, ov_h),
                "text": text_res,
                "conf": conf_res,
                "sharpness": sharpness
            })

        now = time.time()
        unmatched_detections = list(range(len(current_detections)))

        with self._state_lock:
            for t_id, track in list(self.active_tracks.items()):
                best_match_idx = -1
                min_distance = float("inf")

                for idx in unmatched_detections:
                    det = current_detections[idx]
                    iou = compute_iou(track.bbox, det["box"])
                    dist = box_centroid_dist(track.bbox, det["box"])

                    if (iou >= 0.15 or dist < 220.0) and dist < min_distance:
                        min_distance = dist
                        best_match_idx = idx

                if best_match_idx != -1:
                    det = current_detections[best_match_idx]
                    track.update(
                        det["box"],
                        det["plate_img"],
                        frame,
                        det["text"],
                        det["conf"],
                        det["sharpness"]
                    )
                    unmatched_detections.remove(best_match_idx)

                    if (not track.saved and
                        track.best_conf >= 0.96 and
                        track.best_sharpness >= 45.0 and
                        hasattr(track, "best_text") and
                        len(track.best_text) >= 8 and
                        track.frames_tracked >= 3):

                        self._save_tracked_plate_locked(track, reason="high_confidence_peak")
                        track.saved = True

            for idx in unmatched_detections:
                det = current_detections[idx]
                t_id = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[t_id] = PlateTrack(
                    track_id=t_id,
                    bbox=det["box"],
                    plate_img=det["plate_img"],
                    frame=frame,
                    text=det["text"],
                    conf=det["conf"],
                    sharpness=det["sharpness"]
                )

            for t_id, track in list(self.active_tracks.items()):
                if (now - track.last_seen) > self.track_timeout:
                    if not track.saved:
                        self._save_tracked_plate_locked(track, reason="track_exit")
                    del self.active_tracks[t_id]

            display_boxes = [d["box"] for d in current_detections]
            best_active = None
            for track in self.active_tracks.values():
                if track.best_plate_img is not None and track.text_history:
                    if best_active is None or track.best_score > best_active.best_score:
                        best_active = track

            overlay_info = None
            if best_active is not None:
                c_text, c_conf = best_active.get_consensus_text_and_conf()
                overlay_info = {
                    "img": best_active.best_plate_img,
                    "dims": (best_active.best_plate_img.shape[1], best_active.best_plate_img.shape[0]),
                    "text": c_text,
                    "conf": c_conf,
                    "time": now
                }

        return display_boxes, overlay_info

    def flush_tracks(self):
        """Примусове збереження всіх активних незавершених треків."""
        with self._state_lock:
            for t_id, track in list(self.active_tracks.items()):
                if not track.saved:
                    self._save_tracked_plate_locked(track, reason="stream_stop")
                del self.active_tracks[t_id]

    def _save_tracked_plate_locked(self, track: PlateTrack, reason="track_exit"):
        text, conf = track.get_consensus_text_and_conf()
        conf_pct = int(round(conf * 100))

        if text == "UNKNOWN" or len(text) < 4:
            return

        best_plate_img, best_frame, best_sharpness, best_conf = track.get_best_candidate(text)
        if best_plate_img is None:
            return

        if best_sharpness < 28.0:
            print(f"[ВІДХИЛЕНО НОМЕР] {text}: низька різкість ({best_sharpness:.1f} < 28.0)")
            return

        if conf_pct < self.min_percent_to_save:
            return

        now = time.time()
        self.recently_saved_plates = {
            p: t for p, t in self.recently_saved_plates.items()
            if (now - t) < self.sleep_timeout
        }

        for saved_text in self.recently_saved_plates:
            if are_plates_similar(text, saved_text, max_diff=1):
                return

        self.recently_saved_plates[text] = now

        now_dt = datetime.now()
        day_folder = now_dt.strftime("%Y-%m-%d")
        time_prefix = now_dt.strftime("%H-%M-%S") + f"-{int(now_dt.microsecond / 1000):03d}"

        day_dir = os.path.join(self.plates_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        base_name = f"{time_prefix}_plate_{text}_{conf_pct}%"
        plate_filename = f"{base_name}.jpg"
        full_filename = f"{time_prefix}_full.jpg"
        debug_filename = f"{base_name}_debug.txt"

        plate_path = os.path.join(day_dir, plate_filename)
        full_path = os.path.join(day_dir, full_filename)
        debug_path = os.path.join(day_dir, debug_filename)

        async_write_image(plate_path, best_plate_img)
        async_write_image(full_path, best_frame)

        if self.save_debug:
            track_dur = round(now - track.first_seen, 2)
            counter = Counter(track.text_history)
            voting_summary = ", ".join([f"{txt}: {cnt}" for txt, cnt in counter.most_common()])

            debug_content = (
                f"=== ANPR Track Debug Log ===\n"
                f"Track ID: {track.track_id}\n"
                f"Saved Reason: {reason}\n"
                f"Duration: {track_dur}s\n"
                f"Frames Tracked: {track.frames_tracked}\n"
                f"Final Text: {text}\n"
                f"Consensus Confidence: {conf_pct}%\n"
                f"Best Sharpness: {track.best_sharpness:.1f}\n"
                f"Best Score: {track.best_score:.3f}\n"
                f"Voting Summary: {voting_summary}\n\n"
                f"--- Frame-by-Frame History ---\n"
            )
            for rec in track.history_records:
                debug_content += (
                    f"+{rec['rel_time']}s | Text: {rec['text']} | Conf: {rec['conf']}% | "
                    f"Sharp: {rec['sharpness']} | Score: {rec['score']} | Box: {rec['bbox']}\n"
                )

            async_write_text(debug_path, debug_content)

        print(f"[BEST-SHOT НОМЕР] {text} ({conf_pct}%), різкість: {track.best_sharpness:.1f}, кадрів: {track.frames_tracked}")

        if self.on_saved_callback:
            time_display = now_dt.strftime("%H:%M:%S")
            self.on_saved_callback(day_folder, time_display, text, plate_path)


# ------------------------------------------------------------------
#  Контракт незалежного модуля для ModuleRegistry
# ------------------------------------------------------------------
def _create_shared_resources(base_dir: str, providers: list, ai_threads: int, app_context: dict = None):
    det_path = os.path.join(base_dir, "models", "best.onnx")
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Файл детектора номерів не знайдено: {det_path}")
    ocr_path = os.path.join(base_dir, "models", "cct_xs_v2_global.onnx")
    cfg_path = os.path.join(base_dir, "models", "cct_xs_v2_global_plate_config.yaml")
    if not os.path.exists(ocr_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError("Файли CCT OCR відсутні в models/")

    det = ONNXDetector(det_path, providers=providers, intra_op_threads=ai_threads)
    ocr = ThreadSafePlateOCR(ocr_path, cfg_path, providers=providers)
    return {
        "detector": det,
        "ocr": ocr,
        "providers": list(getattr(det, "active_providers", providers or ["CPUExecutionProvider"])),
    }


def _create_engine(base_dir, flat_cfg, output_dir, shared_resources=None, on_saved_callback=None, source=None, app_context=None):
    det = shared_resources.get("detector") if isinstance(shared_resources, dict) else None
    ocr = shared_resources.get("ocr") if isinstance(shared_resources, dict) else None
    return PlateEngine(
        base_dir=base_dir,
        config=flat_cfg,
        on_saved_callback=on_saved_callback,
        detector=det,
        ocr=ocr,
        output_dir=output_dir,
    )


def _update_engine_config(engine, flat_cfg, source=None, shared_resources=None, app_context=None):
    if engine is None:
        return
    if isinstance(shared_resources, dict):
        if shared_resources.get("detector") is not None:
            engine.detector = shared_resources["detector"]
        if shared_resources.get("ocr") is not None:
            engine.ocr = shared_resources["ocr"]
    engine.update_config(flat_cfg)


def _process_and_pack(engine, frame, prev_state=None):
    boxes, overlay_info = engine.process_frame(frame)
    state = dict(prev_state or {})
    state["boxes"] = boxes
    if overlay_info is not None:
        state["last_plate_img"] = overlay_info["img"]
        state["last_plate_dims"] = overlay_info["dims"]
        state["last_plate_text"] = overlay_info["text"]
        state["last_plate_conf"] = overlay_info["conf"]
        state["last_plate_time"] = overlay_info["time"]
    return state


def _draw_overlays(frame, state_data: dict, mod_cfg: dict, now: float):
    if cv2 is None or frame is None or not mod_cfg.get("show_boxes", True):
        return
    for (bx1, by1, bx2, by2) in (state_data or {}).get("boxes") or []:
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    plate_img = (state_data or {}).get("last_plate_img")
    plate_dims = (state_data or {}).get("last_plate_dims") or (0, 0)
    plate_time = float((state_data or {}).get("last_plate_time", 0.0))
    view_time = float(mod_cfg.get("view_time", 30))
    rect_scale = float(mod_cfg.get("rect_scale", 5.0))
    frame_h, frame_w = frame.shape[:2]

    if plate_img is not None and (now - plate_time) < view_time:
        margin = 20
        cur_w, cur_h = plate_dims
        if cur_w > 0 and cur_h > 0:
            base_scale = max(0.2, rect_scale / 5.0)
            target_ov_w = int(round(cur_w * base_scale))
            target_ov_h = int(round(cur_h * base_scale))
            max_allowed_w = int(frame_w * 0.70)
            if target_ov_w > max_allowed_w and max_allowed_w > 120:
                fit = max_allowed_w / float(target_ov_w)
                target_ov_w = int(target_ov_w * fit)
                target_ov_h = int(target_ov_h * fit)
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


def _parse_archive_record(fpath: str, base_name: str, root_dir: str, rel_dir: str):
    fname = os.path.basename(fpath)
    lower = fname.lower()
    if "_full" in lower or lower.endswith("_debug.txt"):
        return None
    if "_plate_" in base_name:
        parts = base_name.split("_plate_")
        time_disp = ":".join(parts[0].split("-")[:3])
        rest = parts[1].rsplit("_", 1)
        item_name = rest[0] if len(rest) == 2 else parts[1]
    else:
        time_disp = "--:--:--"
        item_name = base_name
    return {"time": time_disp, "name": item_name}


MODULE_SPEC = {
    "id": "plates",
    "version": "2.0.0",
    "order": 10,
    "kind": "ai",
    "name": "Номери",
    "icon": "🚘",
    "short_badge": "Н",
    "description": "Виявлення, вирівнювання та OCR-розпізнавання автомобільних номерних знаків.",
    "on_help": "Увімкнути виявлення, OCR-розпізнавання й архівування автомобільних номерів для цієї камери.",
    "provides": ["anpr"],
    "library_deps": ["opencv-python", "onnxruntime", "fast-alpr"],
    "settings_schema": [
        {
            "key": "show_boxes",
            "type": "bool",
            "default": True,
            "label": "Показувати рамки:",
            "bool_text": "Малювати зелені рамки та збільшений номер у куті",
            "help": "Малювати зелену рамку навколо виявлених номерів та показувати збільшений номер у куті вікна повного перегляду.",
        },
        {
            "key": "debug",
            "type": "bool",
            "default": False,
            "label": "Debug-логи:",
            "bool_text": "Зберігати текстові *_debug.txt із голосуванням символів",
            "help": "Зберігати текстовий лог по кожному зафіксованому треку (*_debug.txt) зі статистикою голосування символів.",
        },
        {
            "key": "det_thresh",
            "type": "float",
            "default": 0.18,
            "min": 0.05,
            "max": 0.95,
            "step": 0.01,
            "label": "Поріг детектора:",
            "help": "Поріг впевненості детектора номерів (0.12–0.45; 0.18 оптимум). Нижче — бачить дальші/брудні номери, вище — менше хибних рамок.",
        },
        {
            "key": "min_percent_to_save",
            "type": "int",
            "default": 80,
            "min": 10,
            "max": 100,
            "step": 1,
            "label": "Мін. точність OCR (%):",
            "help": "Мінімальна впевненість OCR (%) для збереження номера в архів (рекомендовано 75–90%).",
        },
        {
            "key": "sleep_time_after_save",
            "type": "float",
            "default": 5.0,
            "min": 0.0,
            "max": 3600.0,
            "step": 0.5,
            "label": "Пауза повторів (сек):",
            "help": "Пауза (сек), протягом якої повтор тієї самої таблички ігнорується (з похибкою в 1 символ), щоб припарковане авто не спамило архів.",
        },
        {
            "key": "view_time",
            "type": "int",
            "default": 30,
            "min": 1,
            "max": 600,
            "step": 1,
            "label": "Час показу у куті (сек):",
            "help": "Скільки секунд показувати збільшений знімок останнього розпізнаного номера в кутку екрану.",
        },
        {
            "key": "rect_scale",
            "type": "float",
            "default": 5.0,
            "min": 1.0,
            "max": 15.0,
            "step": 0.5,
            "label": "Масштаб у куті:",
            "help": "Масштаб збільшеної плашки номера в кутку вікна (базово 5.0).",
        },
        {
            "key": "delete_older_days",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 3650.0,
            "step": 1.0,
            "label": "Автовидалення старше (днів):",
            "help": "Автовидалення фото номерів цієї камери, старших за N днів (0 = зберігати безстроково; можна дробові напр. 0.5).",
        },
        {
            "key": "date_from",
            "key_to": "date_to",
            "type": "date_range",
            "default": "",
            "default_to": "",
            "label": "Період роботи (дати):",
            "help": "Початкова та кінцева дата роботи модуля номерів (YYYY-MM-DD, порожньо = без обмеження).",
        },
    ],
    "archive": {
        "has_archive": True,
        "button_label": "🚘 Номери",
        "sidebar_title": "Історія номерів",
        "item_preview_label": "Фото номера:",
        "icon": "🚘",
        "record_type": "pair_image",
        "extensions": (".jpg", ".jpeg", ".png"),
        "parse_record": _parse_archive_record,
    },
    "refresh_capabilities": refresh_capabilities,
    "get_status_summary": get_status_summary,
    "create_shared_resources": _create_shared_resources,
    "create_engine": _create_engine,
    "update_engine_config": _update_engine_config,
    "process_and_pack": _process_and_pack,
    "draw_overlays": _draw_overlays,
    "flush_tracks": lambda eng: eng.flush_tracks() if eng is not None else None,
    "has_active_tracks": lambda eng: bool(eng is not None and getattr(eng, "active_tracks", None)),
}
