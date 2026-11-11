import os
import re
import sys
import json
import time
import math
import inspect
import importlib
import threading
from collections import Counter
from datetime import datetime
import numpy as np

from scripts.core.i18n import tr
from scripts.core.io_pool import submit_io

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

try:
    from fast_alpr.default_ocr import DefaultOCR
    from fast_alpr.base import OcrResult
except ImportError:
    DefaultOCR = None
    OcrResult = None

YOLO_IMGSZ = 640
_PLATE_TL = threading.local()

# Пропорції пластин номерних знаків
PLATE_ASPECT_RATIO_STANDARD = 520.0 / 112.0
PLATE_ASPECT_RATIO_SQUARE = 300.0 / 150.0


def _ensure_deps():
    global cv2, DefaultOCR, OcrResult
    if cv2 is None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            pass
    if DefaultOCR is None:
        try:
            mod = importlib.import_module("fast_alpr.default_ocr")
            DefaultOCR = mod.DefaultOCR
            OcrResult = getattr(importlib.import_module("fast_alpr.base"), "OcrResult", None)
        except ImportError:
            pass


def _get_clahe():
    clahe = getattr(_PLATE_TL, "clahe", None)
    if clahe is None and cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _PLATE_TL.clahe = clahe
    return clahe


def _expand_cuda_provider(req_providers: list) -> list:
    out = []
    for p in req_providers:
        if p == "CUDAExecutionProvider":
            out.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        else:
            out.append(p)
    return out


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
        mod = importlib.import_module("fast_alpr.default_ocr")
        DefaultOCR = mod.DefaultOCR
        OcrResult = getattr(importlib.import_module("fast_alpr.base"), "OcrResult", None)
    except ImportError:
        DefaultOCR = None
        OcrResult = None


def is_ready(base_dir: str = "") -> bool:
    if cv2 is None or DefaultOCR is None:
        return False
    if base_dir:
        req_files = (
            os.path.join(base_dir, "models", "best.onnx"),
            os.path.join(base_dir, "models", "cct_xs_v2_global.onnx"),
            os.path.join(base_dir, "models", "cct_xs_v2_global_plate_config.yaml"),
        )
        if any(not os.path.exists(p) for p in req_files):
            return False
    return True


def get_status_summary(base_dir: str = "") -> str:
    missing_libs = []
    if cv2 is None:
        missing_libs.append("opencv-python")
    if DefaultOCR is None:
        missing_libs.append("fast-alpr")
    if missing_libs:
        return tr("mod.common.libs_needed", "Required libraries: {libs}", libs=", ".join(missing_libs))

    if base_dir:
        req_files = (
            os.path.join(base_dir, "models", "best.onnx"),
            os.path.join(base_dir, "models", "cct_xs_v2_global.onnx"),
            os.path.join(base_dir, "models", "cct_xs_v2_global_plate_config.yaml"),
        )
        missing_m = [os.path.basename(p) for p in req_files if not os.path.exists(p)]
        if missing_m:
            return tr("mod.common.models_missing", "Missing models: {models}", models=", ".join(missing_m))
    return tr("mod.plates.status.ready", "Ready (YOLO best.onnx + CCT OCR)")


def ensure_resources(base_dir: str = "", download: bool = True, progress_cb=None) -> dict:
    """Перевіряє та за потреби довантажує CCT OCR модель номерів у <base_dir>/models/."""
    refresh_capabilities()
    missing = []
    downloaded = []

    if cv2 is None:
        missing.append("opencv-python")
    if DefaultOCR is None:
        missing.append("fast-alpr / fast-plate-ocr")

    if base_dir:
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        det_p = os.path.join(models_dir, "best.onnx")
        ocr_p = os.path.join(models_dir, "cct_xs_v2_global.onnx")
        cfg_p = os.path.join(models_dir, "cct_xs_v2_global_plate_config.yaml")

        if download and (not os.path.exists(ocr_p) or not os.path.exists(cfg_p)):
            try:
                from pathlib import Path
                from fast_plate_ocr.inference.hub import download_model
                if progress_cb:
                    progress_cb(tr("mod.plates.res.downloading", "Downloading CCT OCR model (cct-xs-v2-global-model)..."))
                dl_model, dl_cfg = download_model(
                    model_name="cct-xs-v2-global-model",
                    save_directory=Path(models_dir),
                    force_download=False,
                )
                if os.path.exists(str(dl_model)):
                    downloaded.append(os.path.basename(str(dl_model)))
                if os.path.exists(str(dl_cfg)):
                    downloaded.append(os.path.basename(str(dl_cfg)))
            except Exception as e:
                if progress_cb:
                    progress_cb(tr("mod.plates.res.dl_failed", "Failed to download CCT OCR: {err}", err=e))

        for req_p in (det_p, ocr_p, cfg_p):
            if not os.path.exists(req_p):
                missing.append(os.path.basename(req_p))

    ok = len(missing) == 0
    if ok and downloaded:
        msg = tr("mod.plates.res.downloaded", "Downloaded OCR resources: {items}", items=", ".join(downloaded))
    elif ok:
        msg = tr("mod.plates.res.ready", "License plate models (best.onnx + CCT OCR) are ready.")
    else:
        msg = tr("mod.common.res_missing", "Missing resources: {items}", items=", ".join(missing))
    return {
        "ok": ok,
        "downloaded": downloaded,
        "missing": missing,
        "message": msg,
    }


def _imwrite_bgr(path: str, bgr_img) -> bool:
    if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if cv2 is not None:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        if ok:
            buf.tofile(path)
            return True
    if _PILImage is not None:
        rgb = bgr_img[:, :, ::-1]
        _PILImage.fromarray(rgb).save(path, quality=94)
        return True
    return False


def _imread_bgr(path: str):
    if not path or not os.path.exists(path):
        return None
    if cv2 is not None:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
    if _PILImage is not None:
        try:
            with _PILImage.open(path) as pim:
                rgb = pim.convert("RGB")
                return np.asarray(rgb, dtype=np.uint8)[:, :, ::-1].copy()
        except Exception:
            pass
    return None


def async_write_image(path, img):
    """Асинхронний запис зображення на диск через спільний обмежений пул."""
    def _worker():
        try:
            _imwrite_bgr(path, img)
        except Exception:
            pass
    submit_io(_worker)


def async_write_text(path, content):
    """Асинхронний запис дебаг-лога на диск через спільний обмежений пул."""
    def _worker():
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass
    submit_io(_worker)


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
    clahe = _get_clahe()
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
    def __init__(self, track_id, bbox, plate_img, frame, text, conf, sharpness, frame_ts=None, seen_ts=None):
        self.track_id = track_id
        self.bbox = bbox
        s_ts = float(seen_ts) if (seen_ts is not None and float(seen_ts) > 0) else time.time()
        ev_ts = float(frame_ts) if (frame_ts is not None and float(frame_ts) > 0) else s_ts
        self.last_seen = s_ts
        self.first_seen = s_ts
        self.saved = False

        self.best_plate_img = plate_img
        self.best_frame = frame
        self.best_frame_ts = ev_ts
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

        self.append_record(bbox, plate_img, frame, text, conf, sharpness, initial_score, frame_ts=ev_ts, seen_ts=s_ts)

    def append_record(self, bbox, plate_img, frame, text, conf, sharpness, score, frame_ts=None, seen_ts=None):
        self.frames_tracked += 1
        s_ts = float(seen_ts) if (seen_ts is not None and float(seen_ts) > 0) else time.time()
        ev_ts = float(frame_ts) if (frame_ts is not None and float(frame_ts) > 0) else s_ts
        t_rel = round(s_ts - self.first_seen, 3)
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
                    "frame": frame,
                    "frame_ts": float(ev_ts),
                    "text": text,
                    "conf": conf,
                    "sharpness": sharpness,
                    "score": score
                }

    def update(self, bbox, plate_img, frame, text, conf, sharpness, frame_ts=None, seen_ts=None):
        s_ts = float(seen_ts) if (seen_ts is not None and float(seen_ts) > 0) else time.time()
        ev_ts = float(frame_ts) if (frame_ts is not None and float(frame_ts) > 0) else s_ts
        self.bbox = bbox
        self.last_seen = s_ts
        current_score = evaluate_plate_quality(plate_img, text, conf, sharpness)

        self.append_record(bbox, plate_img, frame, text, conf, sharpness, current_score, frame_ts=ev_ts, seen_ts=s_ts)

        if current_score > self.best_score and text != "UNKNOWN":
            self.best_score = current_score
            self.best_plate_img = plate_img
            self.best_frame = frame
            self.best_frame_ts = ev_ts
            self.best_sharpness = sharpness
            self.best_conf = conf
            self.best_text = text

    def get_best_candidate_info(self, consensus_text=None):
        """Знаходить абсолютно найчіткіший збережений кадр та його timestamp для узгодженого номера."""
        if not self.best_candidates_by_text:
            return {
                "plate_img": self.best_plate_img,
                "frame": self.best_frame,
                "sharpness": self.best_sharpness,
                "conf": self.best_conf,
                "frame_ts": float(getattr(self, "best_frame_ts", 0.0) or 0.0),
            }

        target_text = consensus_text or self.best_text
        if target_text in self.best_candidates_by_text:
            best_item = self.best_candidates_by_text[target_text]
            return {
                "plate_img": best_item["plate_img"],
                "frame": best_item["frame"],
                "sharpness": best_item["sharpness"],
                "conf": best_item["conf"],
                "frame_ts": float(best_item.get("frame_ts") or getattr(self, "best_frame_ts", 0.0) or 0.0),
            }

        candidates = list(self.best_candidates_by_text.values())
        matching = [item for item in candidates if are_plates_similar(item["text"], target_text, max_diff=1)]
        if not matching:
            matching = candidates

        best_item = max(matching, key=lambda x: x["sharpness"])
        return {
            "plate_img": best_item["plate_img"],
            "frame": best_item["frame"],
            "sharpness": best_item["sharpness"],
            "conf": best_item["conf"],
            "frame_ts": float(best_item.get("frame_ts") or getattr(self, "best_frame_ts", 0.0) or 0.0),
        }

    def get_best_candidate(self, consensus_text=None):
        """Знаходить абсолютно найчіткіший збережений кадр для узгодженого номера."""
        info = self.get_best_candidate_info(consensus_text=consensus_text)
        return info["plate_img"], info["frame"], info["sharpness"], info["conf"]

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
        self._supports_batch = False
        self._lock = threading.Lock()

        try:
            import onnxruntime as ort
            try:
                ort.set_default_logger_severity(3)
            except Exception:
                pass
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            cpu_cnt = os.cpu_count() or 4
            opts.intra_op_num_threads = max(1, min(int(intra_op_threads), max(1, cpu_cnt - 1)))
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            req_providers = list(providers) if providers else ["CPUExecutionProvider"]
            opt_providers = _expand_cuda_provider(req_providers)
            try:
                self.session = ort.InferenceSession(model_path, opts, providers=opt_providers)
            except Exception:
                try:
                    self.session = ort.InferenceSession(model_path, opts, providers=req_providers)
                except Exception:
                    self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
            self.active_providers = self.session.get_providers()
            in0 = self.session.get_inputs()[0]
            self.input_name = in0.name
            self.output_names = [o.name for o in self.session.get_outputs()]
            self.use_ort = True
            self._supports_batch = True
            try:
                if in0.shape and isinstance(in0.shape[0], int) and in0.shape[0] > 0:
                    self._supports_batch = False
            except Exception:
                pass
        except Exception:
            if cv2 is None:
                raise ImportError("onnxruntime or opencv-python is required for ONNXDetector")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_ort = False
            self._supports_batch = False

    @staticmethod
    def _preprocess_one(frame, imgsz: int):
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

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32)
        chw *= np.float32(1.0 / 255.0)
        return chw, (w_orig, h_orig, r, dw, dh)

    @staticmethod
    def _postprocess_one(raw_i, meta, conf_thresh: float):
        w_orig, h_orig, r, dw, dh = meta
        preds = np.asarray(raw_i)
        if preds.ndim > 2:
            preds = np.squeeze(preds)
        if preds.ndim != 2:
            return []

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        channels = preds.shape[1]
        if channels < 5:
            return []

        if channels == 5:
            scores = preds[:, 4]
        elif channels == 6:
            scores = preds[:, 4] * preds[:, 5]
        else:
            scores = np.max(preds[:, 4:], axis=1)

        mask = scores >= conf_thresh
        if not np.any(mask):
            return []

        sub = preds[mask]
        sc = scores[mask]
        cx, cy, w, h = sub[:, 0], sub[:, 1], sub[:, 2], sub[:, 3]
        hw = w * 0.5
        hh = h * 0.5

        bx1 = np.clip((cx - hw - dw) / r, 0, w_orig - 1).astype(np.int32)
        by1 = np.clip((cy - hh - dh) / r, 0, h_orig - 1).astype(np.int32)
        bx2 = np.clip((cx + hw - dw) / r, 0, w_orig).astype(np.int32)
        by2 = np.clip((cy + hh - dh) / r, 0, h_orig).astype(np.int32)

        bw = bx2 - bx1
        bh = by2 - by1
        keep = (bw > 25) & (bh > 10)
        if not np.any(keep):
            return []

        boxes_for_nms = np.column_stack((bx1[keep], by1[keep], bw[keep], bh[keep])).tolist()
        scores_list = sc[keep].astype(np.float64).tolist()

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_list, conf_thresh, 0.45)
        final_boxes = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, w_i, h_i = boxes_for_nms[idx]
                final_boxes.append((int(bx), int(by), int(bx + w_i), int(by + h_i)))
        return final_boxes

    def predict(self, frame, conf_thresh=0.22, imgsz=480):
        if cv2 is None or frame is None or frame.size == 0:
            return []
        chw, meta = self._preprocess_one(frame, imgsz)
        blob = chw[None, ...]

        with self._lock:
            if self.use_ort:
                raw = self.session.run(self.output_names, {self.input_name: blob})[0]
            else:
                self.net.setInput(blob)
                raw = self.net.forward()

        return self._postprocess_one(raw[0] if raw.ndim >= 3 else raw, meta, conf_thresh)

    def predict_batch(self, frames: list, conf_threshs=0.22, imgsz: int = YOLO_IMGSZ) -> list:
        n = len(frames)
        if n == 0 or cv2 is None:
            return [[] for _ in range(n)]
        if isinstance(conf_threshs, (int, float)):
            thresh_list = [float(conf_threshs)] * n
        else:
            thresh_list = [float(x) for x in conf_threshs]

        results = [[] for _ in range(n)]
        valid_indices = []
        chws = []
        metas = []
        for i, fr in enumerate(frames):
            if fr is None or getattr(fr, "size", 0) == 0:
                continue
            chw, meta = self._preprocess_one(fr, imgsz)
            valid_indices.append(i)
            chws.append(chw)
            metas.append(meta)

        if not valid_indices:
            return results

        if len(valid_indices) == 1 or not self.use_ort or not self._supports_batch:
            for k, orig_idx in enumerate(valid_indices):
                blob = chws[k][None, ...]
                with self._lock:
                    if self.use_ort:
                        raw = self.session.run(self.output_names, {self.input_name: blob})[0]
                    else:
                        self.net.setInput(blob)
                        raw = self.net.forward()
                results[orig_idx] = self._postprocess_one(raw[0] if raw.ndim >= 3 else raw, metas[k], thresh_list[orig_idx])
            return results

        batch_blob = np.stack(chws, axis=0)
        try:
            with self._lock:
                raw_batch = self.session.run(self.output_names, {self.input_name: batch_blob})[0]
            for k, orig_idx in enumerate(valid_indices):
                results[orig_idx] = self._postprocess_one(raw_batch[k], metas[k], thresh_list[orig_idx])
            return results
        except Exception:
            self._supports_batch = False
            for k, orig_idx in enumerate(valid_indices):
                blob = chws[k][None, ...]
                with self._lock:
                    raw = self.session.run(self.output_names, {self.input_name: blob})[0]
                results[orig_idx] = self._postprocess_one(raw[0] if raw.ndim >= 3 else raw, metas[k], thresh_list[orig_idx])
            return results


class ThreadSafePlateOCR:
    """Обгортка навколо DefaultOCR з підтримкою GPU-провайдерів і потокобезпечним викликом."""
    def __init__(self, model_path: str, config_path: str, providers=None):
        _ensure_deps()
        if DefaultOCR is None:
            raise ImportError("Package 'fast-alpr' / DefaultOCR is not installed")
        self.model_path = model_path
        self.config_path = config_path
        self._lock = threading.Lock()
        self.ocr = self._create_ocr(model_path, config_path, providers=providers)
        self.active_providers = self._detect_active_providers()

    @staticmethod
    def _create_ocr(model_path: str, config_path: str, providers=None):
        req_providers = list(providers) if providers else ["CPUExecutionProvider"]
        opt_providers = _expand_cuda_provider(req_providers)
        use_cuda = any("CUDA" in p for p in req_providers)
        use_cpu_only = req_providers == ["CPUExecutionProvider"]
        device = "cpu" if use_cpu_only else ("cuda" if use_cuda else "auto")

        opts = None
        try:
            import onnxruntime as ort
            try:
                ort.set_default_logger_severity(3)
            except Exception:
                pass
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            opts = None

        sig = inspect.signature(DefaultOCR.__init__)
        for prov_cand, dev_cand in (
            (opt_providers, device),
            (req_providers, device),
            (["CPUExecutionProvider"], "cpu"),
        ):
            try:
                kwargs = {
                    "hub_ocr_model": None,
                    "device": dev_cand,
                    "model_path": model_path,
                    "config_path": config_path,
                }
                if "providers" in sig.parameters:
                    kwargs["providers"] = prov_cand
                if opts is not None and "sess_options" in sig.parameters:
                    kwargs["sess_options"] = opts
                return DefaultOCR(**kwargs)
            except Exception:
                continue
        return DefaultOCR(hub_ocr_model=None, device="cpu", model_path=model_path, config_path=config_path)

    def _detect_active_providers(self):
        for attr in ("ocr_model", "model", "session"):
            obj = getattr(self.ocr, attr, None)
            if obj is None:
                continue
            for inner_attr in ("session", "model"):
                sess = getattr(obj, inner_attr, None)
                if sess is not None and hasattr(sess, "get_providers"):
                    try:
                        return sess.get_providers()
                    except Exception:
                        pass
            if hasattr(obj, "get_providers"):
                try:
                    return obj.get_providers()
                except Exception:
                    pass
        return ["CPUExecutionProvider"]

    def predict(self, img):
        with self._lock:
            return self.ocr.predict(img)

    def predict_batch(self, imgs: list) -> list:
        n = len(imgs)
        if n == 0:
            return []
        if n == 1:
            return [self.predict(imgs[0])]

        ocr_model = getattr(self.ocr, "ocr_model", None)
        if ocr_model is not None and hasattr(ocr_model, "run") and OcrResult is not None and cv2 is not None:
            try:
                color_mode = getattr(getattr(ocr_model, "config", None), "image_color_mode", None)
                results = [None] * n
                valid_idx = []
                converted = []
                for i, im in enumerate(imgs):
                    if im is None:
                        continue
                    if color_mode == "grayscale":
                        conv = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    elif color_mode == "rgb":
                        conv = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    else:
                        conv = im
                    valid_idx.append(i)
                    converted.append(conv)
                if not valid_idx:
                    return results
                with self._lock:
                    preds = ocr_model.run(converted, return_confidence=True)
                if len(preds) == len(valid_idx):
                    for k, orig_i in enumerate(valid_idx):
                        p = preds[k]
                        cp = getattr(p, "char_probs", None)
                        conf = 0.0 if cp is None else [float(x) for x in cp.tolist()]
                        results[orig_i] = OcrResult(
                            text=p.plate,
                            confidence=conf,
                            region=getattr(p, "region", None),
                            region_confidence=getattr(p, "region_prob", None),
                        )
                    return results
            except Exception:
                pass

        with self._lock:
            return [self.ocr.predict(im) for im in imgs]


def _parse_allow_disk_write(config: dict = None, app_context: dict = None) -> bool:
    for src in (app_context, config):
        if not isinstance(src, dict):
            continue
        if "allow_disk_write" in src and src.get("allow_disk_write") is not None:
            val = src.get("allow_disk_write")
            if isinstance(val, str):
                return val.strip().lower() not in ("false", "0", "no", "off")
            return bool(val)
        mode = src.get("operation_mode")
        if mode is not None and str(mode).strip() != "":
            return str(mode).strip().lower() != "operator"
    return True


def _merge_runtime_write_cfg(flat_cfg: dict, app_context: dict = None) -> dict:
    cfg = dict(flat_cfg or {})
    allow_write = _parse_allow_disk_write(cfg, app_context)
    cfg["allow_disk_write"] = allow_write
    cfg["operation_mode"] = "server" if allow_write else "operator"
    return cfg


_CYR_TO_LAT_PLATE = str.maketrans("АВЕІКМНОРСТХУЁЫЭЯИЙ", "ABEIKMHOPCTXYEYERII")
_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_PFX_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})(?:-(\d{1,6}))?")
_PLATE_CONF_TAIL_RE = re.compile(r"_(\d{1,3})%?$")
VALID_PLATE_STATUSES = ("normal", "whitelist", "blacklist", "ignore")


def normalize_plate_text(text: str) -> str:
    raw = str(text or "").strip().upper().translate(_CYR_TO_LAT_PLATE)
    return re.sub(r"[^A-Z0-9]", "", raw)


def sanitize_filename_part(name: str, fallback: str = "PLATE") -> str:
    s = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", str(name or "").strip())
    s = s.strip(" .")
    return s or fallback


def _debug_path_for_crop(crop_path: str) -> str:
    if not crop_path:
        return ""
    stem, _ = os.path.splitext(crop_path)
    return f"{stem}_debug.txt"


def _sidecar_path_for_crop(crop_path: str) -> str:
    if not crop_path:
        return ""
    stem, _ = os.path.splitext(crop_path)
    return f"{stem}.json"


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _parse_ts_from_day_and_filename(day_key: str, fname: str, fallback_mtime: float):
    day_str = str(day_key or "").strip().replace("\\", "/")
    if "/" in day_str:
        for part in reversed(day_str.split("/")):
            if _DAY_DIR_RE.match(part):
                day_str = part
                break
    m_t = _TIME_PFX_RE.match(str(fname or ""))
    if _DAY_DIR_RE.match(day_str) and m_t:
        hh, mm, ss, ms = m_t.group(1), m_t.group(2), m_t.group(3), m_t.group(4)
        time_str = f"{hh}:{mm}:{ss}"
        try:
            dt = datetime.strptime(f"{day_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            extra_ms = (int(ms[:3]) / 1000.0) if ms else 0.0
            return dt.timestamp() + extra_ms, day_str, time_str
        except Exception:
            pass
    dt_f = datetime.fromtimestamp(float(fallback_mtime or time.time()))
    if not _DAY_DIR_RE.match(day_str):
        day_str = dt_f.strftime("%Y-%m-%d")
    time_str = f"{m_t.group(1)}:{m_t.group(2)}:{m_t.group(3)}" if m_t else dt_f.strftime("%H:%M:%S")
    return float(fallback_mtime or dt_f.timestamp()), day_str, time_str


def _parse_plate_filename(fname: str):
    """Розбирає {time_prefix}_plate_{TEXT}_{CONF}%[_fixN].jpg у нормалізовані поля."""
    if not fname:
        return None
    lower = fname.lower()
    if "_full" in lower or lower.endswith(("_debug.txt", ".json")):
        return None
    stem = os.path.splitext(fname)[0]
    if "_plate_" not in stem:
        return None
    time_prefix, tail = stem.split("_plate_", 1)
    # Прибираємо технічний суфікс колізій (_fix2, _fix3, ...)
    tail_clean = re.sub(r"_fix\d+$", "", tail, flags=re.IGNORECASE)
    conf = None
    m_conf = _PLATE_CONF_TAIL_RE.search(tail_clean)
    if m_conf:
        raw_plate = tail_clean[:m_conf.start()]
        try:
            conf_val = int(m_conf.group(1))
            if 0 <= conf_val <= 100:
                conf = conf_val
        except Exception:
            conf = None
    else:
        raw_plate = tail_clean

    norm_plate = normalize_plate_text(raw_plate)
    if not norm_plate:
        norm_plate = raw_plate.strip().upper()
    if not norm_plate:
        return None

    m_t = _TIME_PFX_RE.match(time_prefix)
    time_display = f"{m_t.group(1)}:{m_t.group(2)}:{m_t.group(3)}" if m_t else "--:--:--"
    return {
        "time_prefix": time_prefix,
        "time_display": time_display,
        "plate": norm_plate,
        "raw_plate": raw_plate,
        "conf": conf,
    }


class _BasePlateDB:
    """
    Крос-камерна база метаданих номерних знаків (<archive_root>/plates_db/plates.json),
    колекціонування подій по архівах усіх камер, виправлення OCR / перейменування та безпечне видалення.
    """

    def __init__(self, db_path: str, base_dir: str = "", ocr=None, allow_disk_write: bool = True):
        self.db_path = os.path.abspath(db_path)
        self.base_dir = os.path.abspath(base_dir) if base_dir else ""
        self.allow_disk_write = bool(allow_disk_write)
        self._ocr = ocr
        self._ocr_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._db_sig = None
        self._meta_by_plate = {}
        self._load_database_locked()

    def set_allow_disk_write(self, allow: bool):
        self.allow_disk_write = bool(allow)

    @staticmethod
    def _file_sig(path: str):
        if not path or not os.path.exists(path):
            return None
        try:
            st = os.stat(path)
            return (os.path.normpath(path), int(st.st_mtime_ns), int(st.st_size))
        except Exception:
            return None

    def _get_ocr(self):
        with self._ocr_lock:
            if self._ocr is not None:
                return self._ocr
            if not self.base_dir or DefaultOCR is None:
                return None
            ocr_p = os.path.join(self.base_dir, "models", "cct_xs_v2_global.onnx")
            cfg_p = os.path.join(self.base_dir, "models", "cct_xs_v2_global_plate_config.yaml")
            if not os.path.exists(ocr_p) or not os.path.exists(cfg_p):
                return None
            try:
                self._ocr = ThreadSafePlateOCR(ocr_p, cfg_p, providers=["CPUExecutionProvider"])
                return self._ocr
            except Exception as e:
                print(f"[PlateDB OCR Init Error]: {e}")
                return None

    def _load_database_locked(self):
        self._meta_by_plate = {}
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                plates_dict = raw.get("plates") if isinstance(raw, dict) else {}
                if isinstance(plates_dict, dict):
                    for k, v in plates_dict.items():
                        if not isinstance(v, dict):
                            continue
                        norm_k = normalize_plate_text(v.get("plate") or k)
                        if not norm_k:
                            continue
                        st = str(v.get("status") or "normal").strip().lower()
                        if st not in VALID_PLATE_STATUSES:
                            st = "normal"
                        self._meta_by_plate[norm_k] = {
                            "plate": norm_k,
                            "label": str(v.get("label") or "").strip(),
                            "status": st,
                            "notes": str(v.get("notes") or "").strip(),
                            "manual": bool(v.get("manual", False)),
                            "created_ts": float(v.get("created_ts") or 0.0),
                            "updated_ts": float(v.get("updated_ts") or 0.0),
                        }
            except Exception as e:
                print(f"[PlateDB Load Error]: {e}")
        self._db_sig = self._file_sig(self.db_path)

    def _sync_if_modified_locked(self):
        if not os.path.exists(self.db_path):
            if self._meta_by_plate:
                self._meta_by_plate.clear()
            self._db_sig = None
            return
        cur_sig = self._file_sig(self.db_path)
        if cur_sig != self._db_sig:
            self._load_database_locked()

    def _save_database_locked(self):
        if not getattr(self, "allow_disk_write", True):
            return
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(db_dir, exist_ok=True)
        tmp_path = f"{self.db_path}.tmp"
        bak_path = f"{self.db_path}.bak"
        payload = {
            "version": 1,
            "plates": {k: self._meta_by_plate[k] for k in sorted(self._meta_by_plate.keys())},
        }
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        if os.path.exists(self.db_path):
            try:
                if os.path.exists(bak_path):
                    os.remove(bak_path)
                os.replace(self.db_path, bak_path)
            except Exception:
                pass
        os.replace(tmp_path, self.db_path)
        self._db_sig = self._file_sig(self.db_path)

    def list_records(self) -> list:
        with self._db_lock:
            self._sync_if_modified_locked()
            return [dict(v) for v in self._meta_by_plate.values()]

    def get_plate_status(self, plate_text: str) -> str:
        norm = normalize_plate_text(plate_text)
        if not norm:
            return "normal"
        with self._db_lock:
            self._sync_if_modified_locked()
            rec = self._meta_by_plate.get(norm)
            return rec["status"] if rec else "normal"

    def is_ignored(self, plate_text: str) -> bool:
        return self.get_plate_status(plate_text) == "ignore"

    def set_plate_metadata(
        self,
        plate_text: str,
        label: str = None,
        status: str = None,
        notes: str = None,
        manual: bool = None,
    ) -> dict:
        if not getattr(self, "allow_disk_write", True):
            return {"ok": False, "error": tr("mod.plates.err.operator_write_disabled", "Operator mode: disk writes are disabled.")}
        norm = normalize_plate_text(plate_text)
        if len(norm) < 2:
            return {"ok": False, "error": tr("mod.plates.err.invalid_plate", "Enter a valid license plate (at least 2 alphanumeric characters).")}

        now = time.time()
        with self._db_lock:
            self._sync_if_modified_locked()
            cur = dict(self._meta_by_plate.get(norm) or {
                "plate": norm,
                "label": "",
                "status": "normal",
                "notes": "",
                "manual": False,
                "created_ts": now,
                "updated_ts": now,
            })
            if label is not None:
                cur["label"] = str(label).strip()
            if notes is not None:
                cur["notes"] = str(notes).strip()
            if status is not None:
                st = str(status).strip().lower()
                if st in VALID_PLATE_STATUSES:
                    cur["status"] = st
            if manual is not None:
                cur["manual"] = bool(manual)
            cur["plate"] = norm
            cur["updated_ts"] = now
            if not cur.get("created_ts"):
                cur["created_ts"] = now

            if cur["status"] == "normal" and not cur["label"] and not cur["notes"] and not cur["manual"]:
                self._meta_by_plate.pop(norm, None)
            else:
                self._meta_by_plate[norm] = cur
            self._save_database_locked()
        return {"ok": True, "plate": norm, "record": dict(cur)}

    def add_plate_record(self, plate_text: str, label: str = "", status: str = "whitelist", notes: str = "") -> dict:
        return self.set_plate_metadata(
            plate_text,
            label=label,
            status=status,
            notes=notes,
            manual=True,
        )

    def delete_db_record(self, plate_text: str) -> bool:
        if not getattr(self, "allow_disk_write", True):
            return False
        norm = normalize_plate_text(plate_text)
        if not norm:
            return False
        with self._db_lock:
            self._sync_if_modified_locked()
            if norm in self._meta_by_plate:
                self._meta_by_plate.pop(norm, None)
                self._save_database_locked()
                return True
        return False

    def collect_plate_events(self, cameras_info: list) -> dict:
        """
        Сканує архіви номерів (<cam_archive_dir>/plates/) по всіх камерах
        та об'єднує їх із базою метаданих <archive_root>/plates_db/plates.json.
        """
        with self._db_lock:
            self._sync_if_modified_locked()
            meta_by_plate = {k: dict(v) for k, v in self._meta_by_plate.items()}

        events_by_plate = {}

        for cam in (cameras_info or []):
            cam_uid = str(cam.get("uuid") or "")
            cam_name = str(cam.get("name") or cam_uid or tr("camera.default_single", "Camera"))
            cam_arch = str(cam.get("archive_dir") or "")
            plates_dir = os.path.join(cam_arch, "plates") if cam_arch else ""
            if not plates_dir or not os.path.isdir(plates_dir):
                continue

            for root_dir, _, files in os.walk(plates_dir):
                if "_staging" in root_dir:
                    continue
                files_set = set(files)
                rel_dir = os.path.relpath(root_dir, plates_dir)
                day_key = rel_dir if rel_dir != "." else ""
                for fname in files:
                    lower = fname.lower()
                    if not lower.endswith((".jpg", ".jpeg", ".png")):
                        continue
                    parsed = _parse_plate_filename(fname)
                    if not parsed:
                        continue

                    crop_path = os.path.join(root_dir, fname)
                    try:
                        mtime = os.path.getmtime(crop_path)
                    except Exception:
                        continue

                    time_prefix = parsed["time_prefix"]
                    full_cand_name = f"{time_prefix}_full.jpg"
                    full_path = os.path.join(root_dir, full_cand_name) if full_cand_name in files_set else ""
                    stem = os.path.splitext(fname)[0]
                    dbg_name = f"{stem}_debug.txt"
                    debug_path = os.path.join(root_dir, dbg_name) if dbg_name in files_set else ""

                    ts_val, day_str, time_str = _parse_ts_from_day_and_filename(day_key, fname, mtime)
                    if parsed["time_display"] != "--:--:--":
                        time_str = parsed["time_display"]

                    plate_key = parsed["plate"]
                    ev = {
                        "event_id": crop_path,
                        "plate": plate_key,
                        "conf": parsed["conf"],
                        "sharpness": None,
                        "camera_uuid": cam_uid,
                        "camera_name": cam_name,
                        "day": day_str,
                        "time": time_str,
                        "timestamp": float(ts_val),
                        "crop_path": crop_path,
                        "full_path": full_path,
                        "debug_path": debug_path,
                        "filename": fname,
                    }
                    events_by_plate.setdefault(plate_key, []).append(ev)

        all_keys = sorted(set(meta_by_plate.keys()) | set(events_by_plate.keys()))
        plate_records = []
        summary = {}

        for plate_key in all_keys:
            ev_list = events_by_plate.setdefault(plate_key, [])
            ev_list.sort(key=lambda e: (e["timestamp"], e["filename"]))

            cams_seen = []
            confs = []
            best_conf_ev = None
            for e in ev_list:
                if e["camera_name"] not in cams_seen:
                    cams_seen.append(e["camera_name"])
                if isinstance(e.get("conf"), (int, float)):
                    c_val = int(e["conf"])
                    confs.append(c_val)
                    if best_conf_ev is None or c_val >= int(best_conf_ev.get("conf") or -1):
                        best_conf_ev = e

            meta = meta_by_plate.get(plate_key) or {}
            thumb_path = ""
            if ev_list:
                thumb_path = (best_conf_ev or ev_list[-1]).get("crop_path") or ev_list[-1].get("crop_path") or ""

            best_conf = max(confs) if confs else None
            avg_conf = int(round(sum(confs) / float(len(confs)))) if confs else None

            plate_records.append({
                "plate": plate_key,
                "label": meta.get("label", ""),
                "status": meta.get("status", "normal"),
                "notes": meta.get("notes", ""),
                "manual": bool(meta.get("manual", False)),
                "in_db": plate_key in meta_by_plate,
                "updated_ts": float(meta.get("updated_ts", 0.0)),
                "thumb_path": thumb_path,
            })
            summary[plate_key] = {
                "event_count": len(ev_list),
                "screen_count": sum(1 for e in ev_list if e.get("full_path")),
                "first_ts": ev_list[0]["timestamp"] if ev_list else 0.0,
                "last_ts": ev_list[-1]["timestamp"] if ev_list else float(meta.get("updated_ts", 0.0) or 0.0),
                "best_conf": best_conf,
                "avg_conf": avg_conf,
                "cameras": cams_seen,
            }

        return {
            "plates": plate_records,
            "events_by_plate": events_by_plate,
            "summary": summary,
        }

    def reanalyze_crop_ocr(
        self,
        crop_path: str,
        current_plate: str = "",
        known_plates: list = None,
        max_diff: int = 2,
    ) -> dict:
        """Запускає OCR на збереженому кропі номера та знаходить схожі відомі номери в архіві/БД."""
        if not crop_path or not os.path.exists(crop_path):
            return {"ok": False, "error": tr("mod.plates.err.crop_missing_disk", "Plate snapshot file was not found on disk.")}

        bgr = _imread_bgr(crop_path)
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return {"ok": False, "error": tr("mod.plates.err.crop_read", "Failed to read plate snapshot image.")}

        ocr_text = ""
        ocr_conf_pct = 0
        ocr = self._get_ocr()
        if ocr is not None:
            try:
                res = ocr.predict(bgr)
                txt, conf = PlateEngine._parse_ocr_result(res)
                if txt and txt != "UNKNOWN":
                    ocr_text = txt
                    ocr_conf_pct = int(round(float(conf) * 100.0))
            except Exception as e:
                print(f"[PlateDB Re-OCR Error]: {e}")

        sharpness = calculate_sharpness(bgr)
        dbg_path = _debug_path_for_crop(crop_path)
        debug_text = ""
        if dbg_path and os.path.exists(dbg_path):
            try:
                with open(dbg_path, "r", encoding="utf-8") as df:
                    debug_text = df.read()
            except Exception:
                debug_text = ""

        cur_norm = normalize_plate_text(current_plate)
        pool_keys = set()
        for p in (known_plates or []):
            np_k = normalize_plate_text(p)
            if np_k:
                pool_keys.add(np_k)
        with self._db_lock:
            self._sync_if_modified_locked()
            pool_keys.update(self._meta_by_plate.keys())
            meta_snap = {k: dict(v) for k, v in self._meta_by_plate.items()}

        ref_tokens = [t for t in (cur_norm, ocr_text) if t]
        similar = []
        for cand in sorted(pool_keys):
            if cand == cur_norm:
                continue
            best_dist = min((_levenshtein(cand, rt) for rt in ref_tokens), default=99)
            same_len_sim = any(are_plates_similar(cand, rt, max_diff=1) for rt in ref_tokens)
            if best_dist <= int(max_diff) or same_len_sim:
                meta = meta_snap.get(cand) or {}
                similar.append({
                    "plate": cand,
                    "distance": best_dist,
                    "label": meta.get("label", ""),
                    "status": meta.get("status", "normal"),
                })
        similar.sort(key=lambda item: (item["distance"], item["plate"]))

        return {
            "ok": True,
            "ocr_plate": ocr_text,
            "ocr_conf": ocr_conf_pct,
            "sharpness": round(float(sharpness), 1),
            "ocr_available": ocr is not None,
            "similar": similar[:12],
            "debug_text": debug_text,
        }

    def correct_event_plate(self, event_info: dict, new_plate_text: str, new_conf: int = None) -> dict:
        if not getattr(self, "allow_disk_write", True):
            return {"ok": False, "error": tr("mod.plates.err.operator_write_disabled", "Operator mode: disk writes are disabled.")}
        if not event_info:
            return {"ok": False, "error": tr("mod.plates.err.crop_missing_disk", "Plate snapshot file was not found on disk.")}

        norm_new = normalize_plate_text(new_plate_text)
        if len(norm_new) < 2:
            return {"ok": False, "error": tr("mod.plates.err.invalid_plate", "Enter a valid license plate (at least 2 alphanumeric characters).")}

        crop_path = str(event_info.get("crop_path") or "")
        if not crop_path or not os.path.exists(crop_path):
            return {"ok": False, "error": tr("mod.plates.err.crop_missing_disk", "Plate snapshot file was not found on disk.")}

        day_dir = os.path.dirname(crop_path)
        fname = os.path.basename(crop_path)
        ext = os.path.splitext(fname)[1] or ".jpg"
        parsed = _parse_plate_filename(fname) or {}
        time_prefix = parsed.get("time_prefix") or fname.split("_plate_", 1)[0]
        old_plate = parsed.get("plate") or str(event_info.get("plate") or "")

        if new_conf is not None:
            conf_val = max(0, min(100, int(new_conf)))
        else:
            conf_val = parsed.get("conf")
            if conf_val is None and event_info.get("conf") is not None:
                try:
                    conf_val = max(0, min(100, int(event_info["conf"])))
                except Exception:
                    conf_val = None

        safe_plate = sanitize_filename_part(norm_new, fallback="PLATE")
        base_stem = f"{time_prefix}_plate_{safe_plate}_{conf_val}%" if conf_val is not None else f"{time_prefix}_plate_{safe_plate}"

        target_crop = os.path.join(day_dir, f"{base_stem}{ext}")
        suffix_idx = 2
        while os.path.normcase(os.path.abspath(target_crop)) != os.path.normcase(os.path.abspath(crop_path)) and os.path.exists(target_crop):
            target_crop = os.path.join(day_dir, f"{base_stem}_fix{suffix_idx}{ext}")
            suffix_idx += 1

        old_debug = _debug_path_for_crop(crop_path)
        new_debug = _debug_path_for_crop(target_crop)
        try:
            if os.path.normcase(os.path.abspath(target_crop)) != os.path.normcase(os.path.abspath(crop_path)):
                os.replace(crop_path, target_crop)
                if old_debug and os.path.exists(old_debug):
                    try:
                        os.replace(old_debug, new_debug)
                    except Exception:
                        pass
            return {
                "ok": True,
                "old_plate": old_plate,
                "new_plate": norm_new,
                "old_crop_path": crop_path,
                "new_crop_path": target_crop,
                "conf": conf_val,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def merge_or_rename_plate(self, old_plate: str, new_plate: str, events: list) -> dict:
        if not getattr(self, "allow_disk_write", True):
            return {"ok": False, "error": tr("mod.plates.err.operator_write_disabled", "Operator mode: disk writes are disabled.")}
        norm_old = normalize_plate_text(old_plate)
        norm_new = normalize_plate_text(new_plate)
        if len(norm_new) < 2:
            return {"ok": False, "error": tr("mod.plates.err.invalid_plate", "Enter a valid license plate (at least 2 alphanumeric characters).")}

        moved = 0
        failed = 0
        for ev in (events or []):
            r = self.correct_event_plate(ev, norm_new)
            if r and r.get("ok"):
                moved += 1
            else:
                failed += 1

        if norm_old and norm_old != norm_new:
            with self._db_lock:
                self._sync_if_modified_locked()
                old_rec = self._meta_by_plate.pop(norm_old, None)
                if old_rec is not None:
                    new_rec = self._meta_by_plate.get(norm_new)
                    if new_rec is None:
                        old_rec["plate"] = norm_new
                        old_rec["updated_ts"] = time.time()
                        self._meta_by_plate[norm_new] = old_rec
                    else:
                        if not new_rec.get("label") and old_rec.get("label"):
                            new_rec["label"] = old_rec["label"]
                        if not new_rec.get("notes") and old_rec.get("notes"):
                            new_rec["notes"] = old_rec["notes"]
                        if new_rec.get("status", "normal") == "normal" and old_rec.get("status", "normal") != "normal":
                            new_rec["status"] = old_rec["status"]
                        new_rec["manual"] = bool(new_rec.get("manual") or old_rec.get("manual"))
                        new_rec["updated_ts"] = time.time()
                    self._save_database_locked()

        return {"ok": failed == 0 or moved > 0, "new_plate": norm_new, "moved": moved, "failed": failed}

    def delete_crop_event(self, event_info: dict) -> dict:
        if not getattr(self, "allow_disk_write", True):
            return {"ok": False, "error": tr("mod.plates.err.operator_write_disabled", "Operator mode: disk writes are disabled.")}
        if not event_info:
            return {"ok": False, "error": tr("mod.plates.err.event_missing", "Snapshot event is missing.")}

        crop_path = str(event_info.get("crop_path") or "")
        full_path = str(event_info.get("full_path") or "")
        debug_path = str(event_info.get("debug_path") or _debug_path_for_crop(crop_path))

        removed = []
        for p in (crop_path, debug_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    removed.append(p)
                except Exception as e:
                    return {"ok": False, "error": str(e)}

        # Видаляємо _full.jpg лише якщо жоден інший _plate_ кроп не ділить цей самий time_prefix
        if crop_path and not full_path:
            day_dir = os.path.dirname(crop_path)
            fname = os.path.basename(crop_path)
            if "_plate_" in fname:
                tpfx = fname.split("_plate_", 1)[0]
                cand_full = os.path.join(day_dir, f"{tpfx}_full.jpg")
                if os.path.exists(cand_full):
                    full_path = cand_full

        if full_path and os.path.exists(full_path):
            day_dir = os.path.dirname(full_path)
            time_pfx = os.path.basename(full_path).split("_full", 1)[0]
            has_sibling = False
            try:
                for fn in os.listdir(day_dir):
                    low = fn.lower()
                    if not low.endswith((".jpg", ".jpeg", ".png")) or "_full" in low:
                        continue
                    if fn.startswith(f"{time_pfx}_plate_"):
                        has_sibling = True
                        break
            except Exception:
                has_sibling = True
            if not has_sibling:
                try:
                    os.remove(full_path)
                    removed.append(full_path)
                except Exception:
                    pass

        return {"ok": True, "removed_files": removed}

    def delete_plate_completely(self, plate_text: str, plate_events: list, delete_files: bool = True) -> dict:
        if not getattr(self, "allow_disk_write", True):
            return {"ok": False, "error": tr("mod.plates.err.operator_write_disabled", "Operator mode: disk writes are disabled.")}
        deleted_files = 0
        if delete_files:
            for ev in (plate_events or []):
                r = self.delete_crop_event(ev)
                if r and r.get("ok"):
                    deleted_files += len(r.get("removed_files") or [])
        rec_deleted = self.delete_db_record(plate_text)
        return {"ok": True, "record_deleted": rec_deleted, "deleted_files": deleted_files}


class PlateDBAccessor(_BasePlateDB):
    """Аксесор бази номерів із прив'язаним спільним ThreadSafePlateOCR."""


class FallbackPlateDB(_BasePlateDB):
    """Полегшений доступ до бази номерів для GUI (коли жодна камера не активна)."""


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
        plate_db=None,
    ):
        self.base_dir = base_dir
        self.config = dict(config or {})
        self.on_saved_callback = on_saved_callback
        self.plate_db = plate_db

        self.plates_dir = output_dir or os.path.join(self.base_dir, "plates")
        self.allow_disk_write = True
        self._state_lock = threading.Lock()
        self._reload_params()
        if self.allow_disk_write:
            os.makedirs(self.plates_dir, exist_ok=True)

        if detector is not None:
            self.detector = detector
        else:
            model_path = os.path.join(self.base_dir, "models", "best.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Plate detector file not found: {model_path}")
            self.detector = ONNXDetector(model_path, providers=providers)

        if ocr is not None:
            self.ocr = ocr
        else:
            ocr_model_path = os.path.join(self.base_dir, "models", "cct_xs_v2_global.onnx")
            ocr_config_path = os.path.join(self.base_dir, "models", "cct_xs_v2_global_plate_config.yaml")
            if not os.path.exists(ocr_model_path) or not os.path.exists(ocr_config_path):
                raise FileNotFoundError(f"CCT OCR files are missing in {os.path.join(self.base_dir, 'models')}")
            self.ocr = ThreadSafePlateOCR(ocr_model_path, ocr_config_path, providers=providers)

        self.active_tracks = {}
        self.next_track_id = 1
        self.track_timeout = 1.2
        self.recently_saved_plates = {}

    def _reload_params(self):
        self.allow_disk_write = _parse_allow_disk_write(self.config)
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

    def _extract_rectified_candidates(self, frame, detected_boxes):
        frame_h, frame_w = frame.shape[:2]
        total_area = frame_h * frame_w
        cands = []
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

            sharpness = calculate_sharpness(enhanced)
            cands.append({
                "box": (x1, y1, x2, y2),
                "plate_img": enhanced,
                "dims": (ov_w, ov_h),
                "sharpness": sharpness,
            })
        return cands

    @staticmethod
    def _parse_ocr_result(res):
        plate_str = ""
        plate_conf = 0.0
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
        if plate_str:
            return plate_str, plate_conf
        return "UNKNOWN", 0.0

    def _update_tracks_from_detections(self, frame, current_detections, now: float = None, frame_ts: float = None):
        if now is None:
            now = time.time()
        ev_ts = float(frame_ts) if (frame_ts is not None and float(frame_ts) > 0) else float(now)
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
                        det["sharpness"],
                        frame_ts=ev_ts,
                        seen_ts=now,
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
                    sharpness=det["sharpness"],
                    frame_ts=ev_ts,
                    seen_ts=now,
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

    def process_frame(self, frame, frame_ts: float = None):
        """Обробка кадру з детекцією, OCR, трекінгом та поверненням даних для оверлею."""
        if frame is None or frame.size == 0:
            return [], None

        with self._state_lock:
            det_thresh = self.plates_det_thresh

        detected_boxes = self.detector.predict(frame, conf_thresh=det_thresh, imgsz=YOLO_IMGSZ)
        cands = self._extract_rectified_candidates(frame, detected_boxes)
        current_detections = []
        if cands:
            try:
                if hasattr(self.ocr, "predict_batch"):
                    ocr_results = self.ocr.predict_batch([c["plate_img"] for c in cands])
                else:
                    ocr_results = [self.ocr.predict(c["plate_img"]) for c in cands]
            except Exception as ocr_err:
                print(f"[OCR Error]: {ocr_err}")
                ocr_results = [None] * len(cands)

            for cand, res in zip(cands, ocr_results):
                text_res, conf_res = self._parse_ocr_result(res)
                current_detections.append({
                    "box": cand["box"],
                    "plate_img": cand["plate_img"],
                    "dims": cand["dims"],
                    "text": text_res,
                    "conf": conf_res,
                    "sharpness": cand["sharpness"],
                })

        return self._update_tracks_from_detections(frame, current_detections, frame_ts=frame_ts)

    def flush_tracks(self):
        """Примусове збереження всіх активних незавершених треків."""
        with self._state_lock:
            for t_id, track in list(self.active_tracks.items()):
                if not track.saved:
                    self._save_tracked_plate_locked(track, reason="stream_stop")
                del self.active_tracks[t_id]

    def _save_tracked_plate_locked(self, track: PlateTrack, reason="track_exit"):
        if not self.allow_disk_write:
            return
        text, conf = track.get_consensus_text_and_conf()
        conf_pct = int(round(conf * 100))

        if text == "UNKNOWN" or len(text) < 4:
            return

        cand_info = track.get_best_candidate_info(text)
        best_plate_img = cand_info["plate_img"]
        best_frame = cand_info["frame"]
        best_sharpness = cand_info["sharpness"]
        best_conf = cand_info["conf"]
        if best_plate_img is None:
            return

        if best_sharpness < 28.0:
            print(f"[ВІДХИЛЕНО НОМЕР] {text}: низька різкість ({best_sharpness:.1f} < 28.0)")
            return

        if conf_pct < self.min_percent_to_save:
            return

        if self.plate_db is not None and self.plate_db.is_ignored(text):
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

        event_ts = float(cand_info.get("frame_ts") or getattr(track, "best_frame_ts", 0.0) or 0.0)
        if event_ts <= 0.0:
            event_ts = now
        ev_dt = datetime.fromtimestamp(event_ts)
        day_folder = ev_dt.strftime("%Y-%m-%d")
        time_prefix = ev_dt.strftime("%H-%M-%S") + f"-{int(ev_dt.microsecond / 1000):03d}"
        time_display = ev_dt.strftime("%H:%M:%S")

        os.makedirs(self.plates_dir, exist_ok=True)
        day_dir = os.path.join(self.plates_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        safe_plate = sanitize_filename_part(text, fallback="PLATE")
        base_name = f"{time_prefix}_plate_{safe_plate}_{conf_pct}%"
        plate_filename = f"{base_name}.jpg"
        plate_path = os.path.join(day_dir, plate_filename)
        if os.path.exists(plate_path):
            c_idx = 2
            while os.path.exists(os.path.join(day_dir, f"{base_name}_fix{c_idx}.jpg")):
                c_idx += 1
            base_name = f"{base_name}_fix{c_idx}"
            plate_filename = f"{base_name}.jpg"
            plate_path = os.path.join(day_dir, plate_filename)
        full_filename = f"{time_prefix}_full.jpg"
        debug_filename = f"{base_name}_debug.txt"

        full_path = os.path.join(day_dir, full_filename)
        debug_path = os.path.join(day_dir, debug_filename)

        debug_content = None
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

        plate_copy = best_plate_img.copy() if best_plate_img is not None else None
        full_copy = best_frame
        cb = self.on_saved_callback

        def _persist_plate():
            try:
                _imwrite_bgr(plate_path, plate_copy)
                if full_copy is not None:
                    _imwrite_bgr(full_path, full_copy)
                if debug_content is not None:
                    with open(debug_path, "w", encoding="utf-8") as df:
                        df.write(debug_content)
            except Exception as e:
                print(f"[PlateEngine Save Error]: {e}")
            if cb:
                try:
                    cb(day_folder, time_display, text, plate_path)
                except Exception:
                    pass

        print(f"[BEST-SHOT НОМЕР] {text} ({conf_pct}%), різкість: {track.best_sharpness:.1f}, кадрів: {track.frames_tracked}")
        submit_io(_persist_plate)


# ------------------------------------------------------------------
#  Контракт незалежного модуля для ModuleRegistry
# ------------------------------------------------------------------
def _resolve_db_path(base_dir: str, app_context: dict = None) -> str:
    ctx = app_context or {}
    dbp = ctx.get("plates_db_path")
    if dbp:
        return os.path.abspath(dbp)
    return os.path.join(os.path.abspath(base_dir), "archive", "plates_db", "plates.json")


def _create_shared_resources(base_dir: str, providers: list, ai_threads: int, app_context: dict = None):
    det_path = os.path.join(base_dir, "models", "best.onnx")
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"Plate detector file not found: {det_path}")
    ocr_path = os.path.join(base_dir, "models", "cct_xs_v2_global.onnx")
    cfg_path = os.path.join(base_dir, "models", "cct_xs_v2_global_plate_config.yaml")
    if not os.path.exists(ocr_path) or not os.path.exists(cfg_path):
        raise FileNotFoundError("CCT OCR files are missing in models/")

    db_path = _resolve_db_path(base_dir, app_context)
    allow_write = _parse_allow_disk_write(app_context)
    det = ONNXDetector(det_path, providers=providers, intra_op_threads=ai_threads)
    ocr = ThreadSafePlateOCR(ocr_path, cfg_path, providers=providers)
    db = PlateDBAccessor(db_path, base_dir=base_dir, ocr=ocr, allow_disk_write=allow_write)
    return {
        "detector": det,
        "ocr": ocr,
        "db": db,
        "db_path": db_path,
        "providers": list(getattr(det, "active_providers", providers or ["CPUExecutionProvider"])),
    }


def _on_app_context_changed(shared_resources, app_context, base_dir: str, providers: list, ai_threads: int):
    if not isinstance(shared_resources, dict):
        return shared_resources
    new_db = _resolve_db_path(base_dir, app_context)
    allow_write = _parse_allow_disk_write(app_context)
    cur_db = shared_resources.get("db_path") or ""
    if os.path.abspath(new_db) != os.path.abspath(cur_db):
        shared_resources["db"] = PlateDBAccessor(
            new_db,
            base_dir=base_dir,
            ocr=shared_resources.get("ocr"),
            allow_disk_write=allow_write,
        )
        shared_resources["db_path"] = new_db
    else:
        db = shared_resources.get("db")
        if db is not None and hasattr(db, "set_allow_disk_write"):
            db.set_allow_disk_write(allow_write)
    return shared_resources


_FALLBACK_PLATE_DB_LOCK = threading.Lock()
_FALLBACK_PLATE_DB_CACHE = {}


def _get_db_accessor(app_context: dict = None, shared_resources=None):
    allow_write = _parse_allow_disk_write(app_context)
    if isinstance(shared_resources, dict) and shared_resources.get("db") is not None:
        db = shared_resources["db"]
        if hasattr(db, "set_allow_disk_write"):
            db.set_allow_disk_write(allow_write)
        return db
    base_dir = (app_context or {}).get("base_dir", "")
    db_path = _resolve_db_path(base_dir, app_context)
    cache_key = (os.path.abspath(db_path), os.path.abspath(base_dir) if base_dir else "")
    with _FALLBACK_PLATE_DB_LOCK:
        inst = _FALLBACK_PLATE_DB_CACHE.get(cache_key)
        if inst is None:
            ocr_ref = shared_resources.get("ocr") if isinstance(shared_resources, dict) else None
            inst = FallbackPlateDB(db_path, base_dir=base_dir, ocr=ocr_ref, allow_disk_write=allow_write)
            _FALLBACK_PLATE_DB_CACHE[cache_key] = inst
        elif hasattr(inst, "set_allow_disk_write"):
            inst.set_allow_disk_write(allow_write)
        return inst


def _create_engine(base_dir, flat_cfg, output_dir, shared_resources=None, on_saved_callback=None, source=None, app_context=None):
    det = shared_resources.get("detector") if isinstance(shared_resources, dict) else None
    ocr = shared_resources.get("ocr") if isinstance(shared_resources, dict) else None
    db = shared_resources.get("db") if isinstance(shared_resources, dict) else None
    cfg_copy = _merge_runtime_write_cfg(flat_cfg, app_context)
    if db is not None and hasattr(db, "set_allow_disk_write"):
        db.set_allow_disk_write(_parse_allow_disk_write(cfg_copy))
    return PlateEngine(
        base_dir=base_dir,
        config=cfg_copy,
        on_saved_callback=on_saved_callback,
        detector=det,
        ocr=ocr,
        output_dir=output_dir,
        plate_db=db,
    )


def _update_engine_config(engine, flat_cfg, source=None, shared_resources=None, app_context=None):
    if engine is None:
        return
    if isinstance(shared_resources, dict):
        if shared_resources.get("detector") is not None:
            engine.detector = shared_resources["detector"]
        if shared_resources.get("ocr") is not None:
            engine.ocr = shared_resources["ocr"]
        if shared_resources.get("db") is not None:
            engine.plate_db = shared_resources["db"]
    cfg_copy = _merge_runtime_write_cfg(flat_cfg, app_context)
    if getattr(engine, "plate_db", None) is not None and hasattr(engine.plate_db, "set_allow_disk_write"):
        engine.plate_db.set_allow_disk_write(_parse_allow_disk_write(cfg_copy))
    engine.update_config(cfg_copy)


def _process_and_pack(engine, frame, prev_state=None, meta=None):
    frame_ts = meta.get("frame_ts") if isinstance(meta, dict) else None
    boxes, overlay_info = engine.process_frame(frame, frame_ts=frame_ts)
    state = dict(prev_state or {})
    state["boxes"] = boxes
    if overlay_info is not None:
        state["last_plate_img"] = overlay_info["img"]
        state["last_plate_dims"] = overlay_info["dims"]
        state["last_plate_text"] = overlay_info["text"]
        state["last_plate_conf"] = overlay_info["conf"]
        state["last_plate_time"] = overlay_info["time"]
    return state


def _process_batch_and_pack(items: list, shared_resources=None) -> list:
    """Батч-обробка кадрів з різних камер за один виклик YOLO-детектора та один батч-виклик CCT OCR."""
    n = len(items)
    if n == 0:
        return []
    out_states = [None] * n

    groups = {}
    for idx, it in enumerate(items):
        eng = it["engine"]
        fr = it["frame"]
        if eng is None or fr is None or getattr(fr, "size", 0) == 0:
            state = dict(it.get("prev_state") or {})
            state["boxes"] = []
            out_states[idx] = state
            continue
        key = (id(eng.detector), id(eng.ocr))
        groups.setdefault(key, []).append(idx)

    for idxs in groups.values():
        first_eng = items[idxs[0]]["engine"]
        detector = first_eng.detector
        ocr = first_eng.ocr

        frames_g = []
        threshs_g = []
        for idx in idxs:
            eng = items[idx]["engine"]
            frames_g.append(items[idx]["frame"])
            with eng._state_lock:
                threshs_g.append(eng.plates_det_thresh)

        if hasattr(detector, "predict_batch"):
            boxes_list = detector.predict_batch(frames_g, conf_threshs=threshs_g, imgsz=YOLO_IMGSZ)
        else:
            boxes_list = [detector.predict(fr, conf_thresh=th, imgsz=YOLO_IMGSZ) for fr, th in zip(frames_g, threshs_g)]

        per_item_cands = []
        all_enhanced = []
        crop_owners = []
        for local_i, idx in enumerate(idxs):
            eng = items[idx]["engine"]
            cands = eng._extract_rectified_candidates(frames_g[local_i], boxes_list[local_i])
            per_item_cands.append(cands)
            for c_i, c in enumerate(cands):
                all_enhanced.append(c["plate_img"])
                crop_owners.append((local_i, c_i))

        ocr_results = []
        if all_enhanced:
            try:
                if hasattr(ocr, "predict_batch"):
                    ocr_results = ocr.predict_batch(all_enhanced)
                else:
                    ocr_results = [ocr.predict(im) for im in all_enhanced]
            except Exception as ocr_err:
                print(f"[OCR Batch Error]: {ocr_err}")
                ocr_results = [None] * len(all_enhanced)

        per_item_dets = [[] for _ in range(len(idxs))]
        for (local_i, c_i), res in zip(crop_owners, ocr_results):
            cand = per_item_cands[local_i][c_i]
            text_res, conf_res = PlateEngine._parse_ocr_result(res)
            per_item_dets[local_i].append({
                "box": cand["box"],
                "plate_img": cand["plate_img"],
                "dims": cand["dims"],
                "text": text_res,
                "conf": conf_res,
                "sharpness": cand["sharpness"],
            })

        now = time.time()
        for local_i, idx in enumerate(idxs):
            it = items[idx]
            eng = it["engine"]
            meta = it.get("meta") if isinstance(it.get("meta"), dict) else {}
            frame_ts = meta.get("frame_ts")
            boxes, overlay_info = eng._update_tracks_from_detections(
                frames_g[local_i],
                per_item_dets[local_i],
                now=now,
                frame_ts=frame_ts,
            )
            state = dict(it.get("prev_state") or {})
            state["boxes"] = boxes
            if overlay_info is not None:
                state["last_plate_img"] = overlay_info["img"]
                state["last_plate_dims"] = overlay_info["dims"]
                state["last_plate_text"] = overlay_info["text"]
                state["last_plate_conf"] = overlay_info["conf"]
                state["last_plate_time"] = overlay_info["time"]
            out_states[idx] = state

    return out_states


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
    if "_full" in lower or lower.endswith(("_debug.txt", ".json")):
        return None
    parsed = _parse_plate_filename(fname)
    if parsed:
        return {"time": parsed["time_display"], "name": parsed["plate"]}
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
    "name": "License Plates",
    "icon": "🚘",
    "short_badge": "LPR",
    "description": "Detection, perspective rectification, and OCR recognition of vehicle license plates.",
    "on_help": "Enable detection, OCR recognition, and archiving of vehicle license plates for this camera.",
    "provides": ["anpr", "plate_db"],
    "library_deps": ["opencv-python", "onnxruntime", "fast-alpr", "fast-plate-ocr"],
    "settings_schema": [
        {
            "key": "show_boxes",
            "type": "bool",
            "default": True,
            "label": "Show Boxes:",
            "bool_text": "Draw green boxes and zoomed plate in corner",
            "help": "Draw a green box around detected plates and show an enlarged crop in the corner of the detail view.",
        },
        {
            "key": "debug",
            "type": "bool",
            "default": False,
            "disk_write": True,
            "label": "Debug Logs:",
            "bool_text": "Save text *_debug.txt logs with character voting",
            "help": "Save a text log for each tracked plate (*_debug.txt) with character voting statistics.",
        },
        {
            "key": "det_thresh",
            "type": "float",
            "default": 0.18,
            "min": 0.05,
            "max": 0.95,
            "step": 0.01,
            "label": "Detector Threshold:",
            "help": "Plate detector confidence threshold (0.12–0.45; 0.18 optimal). Lower catches distant/dirty plates; higher avoids false boxes.",
        },
        {
            "key": "min_percent_to_save",
            "type": "int",
            "default": 80,
            "min": 10,
            "max": 100,
            "step": 1,
            "disk_write": True,
            "label": "Min OCR Confidence (%):",
            "help": "Minimum OCR confidence (%) required to archive a plate (75–90% recommended).",
        },
        {
            "key": "sleep_time_after_save",
            "type": "float",
            "default": 5.0,
            "min": 0.0,
            "max": 3600.0,
            "step": 0.5,
            "disk_write": True,
            "label": "Repeat Cooldown (sec):",
            "help": "Cooldown (sec) ignoring repeats of the same plate (with 1-char tolerance) so parked cars do not spam the archive.",
        },
        {
            "key": "view_time",
            "type": "int",
            "default": 30,
            "min": 1,
            "max": 600,
            "step": 1,
            "label": "Corner Display Time (sec):",
            "help": "How many seconds to show the zoomed crop of the last recognized plate in the corner.",
        },
        {
            "key": "rect_scale",
            "type": "float",
            "default": 5.0,
            "min": 1.0,
            "max": 15.0,
            "step": 0.5,
            "label": "Corner Zoom Scale:",
            "help": "Scale of the enlarged plate thumbnail in the window corner (default 5.0).",
        },
        {
            "key": "delete_older_days",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 3650.0,
            "step": 1.0,
            "disk_write": True,
            "label": "Auto-delete older than (days):",
            "help": "Automatically delete plate snapshots from this camera older than N days (0 = keep forever; decimals like 0.5 allowed).",
        },
        {
            "key": "date_from",
            "key_to": "date_to",
            "type": "date_range",
            "default": "",
            "default_to": "",
            "label": "Active Schedule (Dates):",
            "help": "Start and end dates for the plates module (YYYY-MM-DD, empty = unlimited).",
        },
    ],
    "archive": {
        "has_archive": True,
        "button_label": "🚘 Plates",
        "sidebar_title": "Plates History",
        "item_preview_label": "Plate Snapshot:",
        "icon": "🚘",
        "record_type": "pair_image",
        "extensions": (".jpg", ".jpeg", ".png"),
        "parse_record": _parse_archive_record,
    },
    "is_ready": is_ready,
    "refresh_capabilities": refresh_capabilities,
    "get_status_summary": get_status_summary,
    "ensure_resources": ensure_resources,
    "create_shared_resources": _create_shared_resources,
    "on_app_context_changed": _on_app_context_changed,
    "get_db_accessor": _get_db_accessor,
    "create_engine": _create_engine,
    "update_engine_config": _update_engine_config,
    "process_and_pack": _process_and_pack,
    "process_batch_and_pack": _process_batch_and_pack,
    "draw_overlays": _draw_overlays,
    "flush_tracks": lambda eng: eng.flush_tracks() if eng is not None else None,
    "has_active_tracks": lambda eng: bool(eng is not None and getattr(eng, "active_tracks", None)),
}
