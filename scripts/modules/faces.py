import os
import re
import sys
import time
import json
import uuid
import math
import shutil
import importlib
import threading
from datetime import datetime
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None

REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Кінчик носа
    [-30.0, -30.0, -30.0],    # Ліве око
    [30.0, -30.0, -30.0],     # Праве око
    [-20.0, 40.0, -20.0],     # Лівий кут рота
    [20.0, 40.0, -20.0]       # Правий кут рота
], dtype=np.float64)

_ID_PATTERN = re.compile(r"^ID_(\d+)$")
_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|\r\n\t]+')

DETECTOR_CANDIDATES = (
    "scrfd_face_detector.onnx",
    "det_500m.onnx",
    "scrfd_2.5g_kps.onnx",
    "scrfd_500m_kps.onnx",
)

RECOGNIZER_CANDIDATES = (
    "face_recognizer.onnx",
    "w600k_r50.onnx",
    "w600k_mbf.onnx",
    "adaface_mobilefacenet.onnx",
    "adaface_ir50.onnx",
)


def _ensure_deps():
    global cv2, ort
    if cv2 is None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            pass
    if ort is None:
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError:
            pass


def refresh_capabilities():
    """Перечитує cv2 / onnxruntime після встановлення або оновлення бібліотек."""
    global cv2, ort
    importlib.invalidate_caches()
    for mod_name in ("cv2", "onnxruntime"):
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
        ort = importlib.import_module("onnxruntime")
    except ImportError:
        ort = None


def _find_model_file(base_dir: str, candidates: tuple):
    for c in candidates:
        p = os.path.join(base_dir, "models", c)
        if os.path.exists(p):
            return p
    return None


def get_status_summary(base_dir: str = "") -> str:
    missing_libs = []
    if cv2 is None:
        missing_libs.append("opencv-python")
    if ort is None:
        missing_libs.append("onnxruntime")
    if missing_libs:
        return f"Потрібні бібліотеки: {', '.join(missing_libs)}"

    if base_dir:
        det_p = _find_model_file(base_dir, DETECTOR_CANDIDATES)
        rec_p = _find_model_file(base_dir, RECOGNIZER_CANDIDATES)
        missing_m = []
        if not det_p:
            missing_m.append("scrfd_face_detector.onnx")
        if not rec_p:
            missing_m.append("face_recognizer.onnx")
        if missing_m:
            return f"Відсутні моделі: {', '.join(missing_m)}"
        return f"Готовий ({os.path.basename(det_p)} + {os.path.basename(rec_p)})"
    return "Готовий (SCRFD + ArcFace/AdaFace)"


def sanitize_filename_part(name: str, fallback: str = "Unknown", max_len: int = 48) -> str:
    cleaned = _UNSAFE_FILENAME_RE.sub("_", str(name or "").strip()).strip(" .")
    if not cleaned:
        return fallback
    return cleaned[:max_len]


def sanitize_csv_field(name: str, fallback: str = "Unknown", max_len: int = 64) -> str:
    cleaned = str(name or "").replace(",", " ").replace("|", " ").replace("\r", " ").replace("\n", " ").strip()
    if not cleaned:
        return fallback
    return cleaned[:max_len]


def _imread_bgr(path: str):
    _ensure_deps()
    if not path or not os.path.exists(path):
        return None
    if cv2 is not None:
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            if raw.size == 0:
                return None
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            try:
                img = cv2.imread(path)
                if img is not None:
                    return img
            except Exception:
                pass
    if _PILImage is not None:
        try:
            with _PILImage.open(path) as im:
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                return rgb[:, :, ::-1].copy()
        except Exception:
            return None
    return None


def _imwrite_bgr(path: str, img) -> bool:
    _ensure_deps()
    if img is None or getattr(img, "size", 0) == 0:
        return False
    if cv2 is not None:
        try:
            ext = os.path.splitext(path)[1] or ".jpg"
            ok, buf = cv2.imencode(ext, img)
            if ok:
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
                return True
        except Exception:
            pass
        try:
            if cv2.imwrite(path, img):
                return True
        except Exception:
            pass
    if _PILImage is not None:
        try:
            rgb = img[:, :, ::-1] if (img.ndim == 3 and img.shape[2] >= 3) else img
            _PILImage.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path)
            return True
        except Exception:
            return False
    return False


def async_write_image(path, img):
    def _worker():
        try:
            _imwrite_bgr(path, img)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def _sidecar_path_for_crop(face_path: str) -> str:
    return f"{os.path.splitext(face_path)[0]}.json"


_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_PREFIX_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})(?:-(\d{3}))?$")


def _parse_ts_from_day_and_filename(day_key: str, fname: str, fallback_mtime: float):
    base = os.path.splitext(os.path.basename(fname))[0]
    time_part = base.split("_face_", 1)[0] if "_face_" in base else base.split("_", 1)[0]
    m_time = _TIME_PREFIX_RE.match(time_part)
    m_day = _DAY_DIR_RE.match(str(day_key or "").strip())

    day_str = str(day_key).strip() if m_day else datetime.fromtimestamp(fallback_mtime).strftime("%Y-%m-%d")
    if m_time:
        hh, mm, ss, ms = m_time.group(1), m_time.group(2), m_time.group(3), m_time.group(4) or "000"
        time_disp = f"{hh}:{mm}:{ss}"
        try:
            dt = datetime.strptime(f"{day_str} {hh}:{mm}:{ss}.{ms}", "%Y-%m-%d %H:%M:%S.%f")
            return float(dt.timestamp()), day_str, time_disp
        except Exception:
            pass

    dt_fb = datetime.fromtimestamp(fallback_mtime)
    return float(fallback_mtime), dt_fb.strftime("%Y-%m-%d"), dt_fb.strftime("%H:%M:%S")


def calculate_sharpness(img):
    if img is None or getattr(img, "size", 0) == 0:
        return 0.0
    if cv2 is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    arr = img.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = 0.114 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.299 * arr[:, :, 2]
    else:
        gray = arr
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return float(np.var(gray))
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


def align_face(img, landmarks, box=None):
    _ensure_deps()
    if cv2 is None:
        return img
    if landmarks is not None and len(landmarks) == 5:
        src = np.array(landmarks, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, REFERENCE_LANDMARKS)
        if M is not None:
            return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    if box is not None:
        bx1, by1, bx2, by2 = box
        h_orig, w_orig = img.shape[:2]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w_orig, bx2), min(h_orig, by2)
        crop = img[by1:by2, bx1:bx2]
        if crop.size > 0:
            return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

    return cv2.resize(img, (112, 112))


def estimate_head_pose_pnp(landmarks, frame_w=640, frame_h=640):
    if cv2 is None or landmarks is None or len(landmarks) != 5:
        return None, None, None

    pts_2d = np.array([
        landmarks[2],
        landmarks[0],
        landmarks[1],
        landmarks[3],
        landmarks[4]
    ], dtype=np.float64)

    fw = max(64.0, float(frame_w or 640))
    fh = max(64.0, float(frame_h or 640))
    focal_length = fw
    center = (fw / 2.0, fh / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL, pts_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None, None, None

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return float(angles[1]), float(angles[0]), float(angles[2])


def evaluate_frontal_face(landmarks, box, min_size=20, max_yaw=32.0, max_pitch=28.0, frame_w=640, frame_h=640):
    """Перевіряє фронтальність обличчя та повертає (is_valid, abs_yaw, abs_pitch)."""
    if landmarks is None or len(landmarks) != 5:
        return False, 999.0, 999.0

    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    if bw < min_size or bh < min_size:
        return False, 999.0, 999.0

    yaw, pitch, _ = estimate_head_pose_pnp(landmarks, frame_w=frame_w, frame_h=frame_h)
    if yaw is None:
        return False, 999.0, 999.0

    ayaw = abs(yaw)
    apitch = abs(pitch)
    if ayaw > max_yaw or apitch > max_pitch:
        return False, ayaw, apitch

    left_eye, right_eye = landmarks[0], landmarks[1]
    eye_dist = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    min_eye_dist = max(6.0, float(min_size) * 0.35)
    if eye_dist < min_eye_dist:
        return False, ayaw, apitch

    return True, ayaw, apitch


def is_valid_frontal_face(landmarks, box, min_size=20, max_yaw=32.0, max_pitch=28.0, frame_w=640, frame_h=640):
    ok, _, _ = evaluate_frontal_face(
        landmarks, box, min_size=min_size, max_yaw=max_yaw, max_pitch=max_pitch, frame_w=frame_w, frame_h=frame_h
    )
    return ok


def _build_ort_session(model_path: str, providers=None, intra_op_threads: int = 2):
    _ensure_deps()
    if ort is None:
        raise ImportError("Пакет 'onnxruntime' не встановлено для модуля облич")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = max(1, int(intra_op_threads))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    req_providers = list(providers) if providers else ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, opts, providers=req_providers)
        return sess
    except Exception:
        if req_providers != ["CPUExecutionProvider"]:
            return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        raise


class SCRFDDetector:
    def __init__(self, model_path, input_size=(640, 640), providers=None, intra_op_threads=2):
        _ensure_deps()
        if cv2 is None:
            raise ImportError("Пакет 'opencv-python' не встановлено для SCRFDDetector")
        self.model_path = model_path
        self.input_size = input_size
        self._lock = threading.Lock()
        self.session = _build_ort_session(model_path, providers=providers, intra_op_threads=intra_op_threads)
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self._init_params()

    def _init_params(self):
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        num_outputs = len(outputs)

        if num_outputs == 6:
            self.fmc = 2
            self._feat_stride_fpn = [8, 16, 32]
            self.has_kps = False
        elif num_outputs == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self.has_kps = True
        elif num_outputs in (10, 15):
            self.fmc = 3 if num_outputs == 10 else 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.has_kps = True
        else:
            self.fmc = 0
            self._feat_stride_fpn = []
            self.has_kps = num_outputs >= 2

    def detect(self, img, conf_thresh=0.40):
        if cv2 is None or img is None or img.size == 0:
            return []
        h_orig, w_orig = img.shape[:2]
        target_w, target_h = self.input_size

        r = min(target_w / w_orig, target_h / h_orig)
        nw, nh = int(round(w_orig * r)), int(round(h_orig * r))
        dw, dh = (target_w - nw) / 2.0, (target_h - nh) / 2.0

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))

        if (w_orig, h_orig) != (nw, nh):
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            resized = img

        canvas[top:top + nh, left:left + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (target_w, target_h), (127.5, 127.5, 127.5))
        with self._lock:
            outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []

        if self.fmc == 0 and len(outs) in (2, 3):
            raw_boxes = np.squeeze(outs[0])
            raw_kps = np.squeeze(outs[1]) if len(outs) > 1 else None

            if raw_boxes.ndim == 2 and raw_boxes.shape[1] >= 5:
                scores = raw_boxes[:, 4]
                mask = scores >= conf_thresh
                for i in np.where(mask)[0]:
                    bx1 = (raw_boxes[i, 0] - dw) / r
                    by1 = (raw_boxes[i, 1] - dh) / r
                    bx2 = (raw_boxes[i, 2] - dw) / r
                    by2 = (raw_boxes[i, 3] - dh) / r

                    scores_list.append(float(scores[i]))
                    bboxes_list.append([int(bx1), int(by1), int(bx2), int(by2)])

                    if raw_kps is not None:
                        kp = raw_kps[i].copy()
                        kp[:, 0] = (kp[:, 0] - dw) / r
                        kp[:, 1] = (kp[:, 1] - dh) / r
                        kpss_list.append(kp)
                    else:
                        kpss_list.append(None)
        else:
            for idx, stride in enumerate(self._feat_stride_fpn):
                score_map = np.squeeze(outs[idx])
                bbox_map = np.squeeze(outs[idx + self.fmc])
                kps_map = np.squeeze(outs[idx + self.fmc * 2]) if self.has_kps else None

                height, width = target_h // stride, target_w // stride
                if score_map.ndim == 1:
                    score_map = score_map.reshape((-1, 1))

                num_anchors = max(1, score_map.shape[0] // (height * width))
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

                valid_len = min(len(anchor_centers), len(score_map), len(bbox_map))
                scores = score_map[:valid_len].reshape(-1)
                bbox_deltas = bbox_map[:valid_len] * stride
                anchors = anchor_centers[:valid_len]

                pos_inds = np.where(scores >= conf_thresh)[0]
                for p in pos_inds:
                    cx, cy = anchors[p]
                    sc = scores[p]
                    del_x1, del_y1, del_x2, del_y2 = bbox_deltas[p][:4]

                    bx1 = ((cx - del_x1) - dw) / r
                    by1 = ((cy - del_y1) - dh) / r
                    bx2 = ((cx + del_x2) - dw) / r
                    by2 = ((cy + del_y2) - dh) / r

                    scores_list.append(float(sc))
                    bboxes_list.append([int(bx1), int(by1), int(bx2), int(by2)])

                    if kps_map is not None and p < len(kps_map):
                        kps_delta = kps_map[p] * stride
                        kps = kps_delta.reshape((5, 2))
                        kps[:, 0] = ((kps[:, 0] + cx) - dw) / r
                        kps[:, 1] = ((kps[:, 1] + cy) - dh) / r
                        kpss_list.append(kps)
                    else:
                        kpss_list.append(None)

        if not bboxes_list:
            return []

        boxes_for_nms = [
            [max(0, b[0]), max(0, b[1]), max(1, b[2] - b[0]), max(1, b[3] - b[1])]
            for b in bboxes_list
        ]
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_list, conf_thresh, 0.40)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx1, by1, bw, bh = boxes_for_nms[i]
                results.append({
                    "box": [bx1, by1, bx1 + bw, by1 + bh],
                    "conf": scores_list[i],
                    "landmarks": kpss_list[i]
                })
        return results


class _BaseFaceDB:
    """
    Спільна логіка роботи з базою облич (faces.txt, thumbs/) та архівом кропів/скрінів
    по усіх камерах (колекціонування подій, видалення кропів, реаналіз бази кропів).
    """
    def __init__(self, db_file_path: str, thumbs_dir: str = None):
        self.db_path = db_file_path
        db_dir = os.path.dirname(os.path.abspath(db_file_path))
        os.makedirs(db_dir, exist_ok=True)
        self.thumbs_dir = thumbs_dir or os.path.join(db_dir, "thumbs")
        os.makedirs(self.thumbs_dir, exist_ok=True)

        self._db_lock = threading.Lock()
        self.known_templates = []
        self._db_sig = None
        self._last_template_save_ts = {}

    def _get_embedder(self):
        return None

    def has_neural_engine(self) -> bool:
        return self._get_embedder() is not None

    @staticmethod
    def _file_sig(path: str):
        try:
            st = os.stat(path)
            return (int(st.st_mtime_ns), int(st.st_size))
        except Exception:
            return None

    def _load_database_locked(self):
        """
        Завантажує базу faces.txt.
        Формат: UniqueFaceID,FaceName,Ignore,Vector1|Vector2|...
        Підтримує старий формат (ID_XXXX, 3 колонки та одиночний вектор без '|').
        """
        self.known_templates.clear()
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                f.write("# UniqueFaceID,FaceName,Ignore,VectorGallery\n")
            self._db_sig = self._file_sig(self.db_path)
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) >= 4:
                    u_id = parts[0].strip()
                    name = sanitize_csv_field(parts[1].strip(), fallback=u_id)
                    ign_str = parts[2].strip().lower()
                    ignore_flag = ign_str in ("true", "1", "yes")
                    raw_gallery = ",".join(parts[3:]).strip()
                elif len(parts) == 3:
                    u_id = parts[0].strip()
                    name = sanitize_csv_field(parts[1].strip(), fallback=u_id)
                    ignore_flag = False
                    raw_gallery = parts[2].strip()
                else:
                    continue

                if not u_id:
                    continue

                gallery = []
                for vec_part in raw_gallery.split("|"):
                    vec_part = vec_part.strip()
                    if not vec_part:
                        continue
                    vec = np.fromstring(vec_part, sep=" ", dtype=np.float32)
                    if vec.size == 0:
                        continue
                    norm = np.linalg.norm(vec)
                    if norm > 1e-8:
                        gallery.append(vec / norm)

                if not gallery:
                    continue

                mean_v = np.mean(gallery[:4], axis=0)
                centroid = mean_v / (np.linalg.norm(mean_v) + 1e-10)
                self.known_templates.append({
                    "id": u_id,
                    "name": name,
                    "ignore": ignore_flag,
                    "vector": centroid,
                    "gallery": gallery[:4],
                })

        self._db_sig = self._file_sig(self.db_path)

    def _sync_if_modified_locked(self):
        if not os.path.exists(self.db_path):
            if self.known_templates:
                self.known_templates.clear()
            self._db_sig = None
            return
        cur_sig = self._file_sig(self.db_path)
        if cur_sig != self._db_sig:
            self._load_database_locked()

    def _save_all_database_locked(self):
        """Атомарний перезапис faces.txt через тимчасовий файл та резервну копію."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(db_dir, exist_ok=True)
        tmp_path = f"{self.db_path}.tmp"
        bak_path = f"{self.db_path}.bak"

        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# UniqueFaceID,FaceName,Ignore,VectorGallery\n")
            for r in self.known_templates:
                ign_str = "true" if r.get("ignore", False) else "false"
                safe_name = sanitize_csv_field(r.get("name", r["id"]), fallback=r["id"])
                gallery = r.get("gallery") or [r["vector"]]
                gal_strs = [" ".join(f"{x:.6f}" for x in v) for v in gallery[:4]]
                f.write(f"{r['id']},{safe_name},{ign_str},{'|'.join(gal_strs)}\n")
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

    def _next_unique_id_locked(self) -> str:
        """Генерує унікальний UUID v4 для кожного нового обличчя в базі."""
        unique_ids = {r["id"] for r in self.known_templates}
        for _ in range(16):
            cand = str(uuid.uuid4())
            if cand not in unique_ids:
                return cand
        return str(uuid.uuid4())

    def _save_thumb_if_needed(self, u_id: str, crop_img, overwrite: bool = False):
        if crop_img is None or getattr(crop_img, "size", 0) == 0:
            return
        os.makedirs(self.thumbs_dir, exist_ok=True)
        tpath = os.path.join(self.thumbs_dir, f"{sanitize_filename_part(u_id)}.jpg")
        if overwrite or not os.path.exists(tpath):
            _imwrite_bgr(tpath, crop_img)

    def get_thumb_path(self, u_id: str) -> str:
        tpath = os.path.join(self.thumbs_dir, f"{sanitize_filename_part(u_id)}.jpg")
        return tpath if os.path.exists(tpath) else ""

    @staticmethod
    def _find_best_match_locked_in(templates, norm_emb, exclude_uids=None):
        best_idx = -1
        best_sim = -1.0
        excl = exclude_uids or set()
        for idx, record in enumerate(templates):
            if record["id"] in excl:
                continue
            gallery = record.get("gallery", [record["vector"]])
            sim = max(float(np.dot(norm_emb, v)) for v in gallery)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        return best_idx, best_sim

    def _find_best_match_locked(self, norm_emb, exclude_uids=None):
        return self._find_best_match_locked_in(self.known_templates, norm_emb, exclude_uids=exclude_uids)

    # --- Базові CRUD методи для GUI ---
    def list_records(self):
        with self._db_lock:
            self._sync_if_modified_locked()
            out = []
            for r in self.known_templates:
                out.append({
                    "id": r["id"],
                    "name": r["name"],
                    "ignore": bool(r.get("ignore", False)),
                    "gallery_count": len(r.get("gallery") or [r["vector"]]),
                    "thumb_path": self.get_thumb_path(r["id"]),
                })
            return out

    def rename_record(self, u_id: str, new_name: str, cameras_info: list = None) -> bool:
        safe = sanitize_csv_field(new_name, fallback="")
        if not safe:
            return False
        changed = False
        with self._db_lock:
            self._sync_if_modified_locked()
            for r in self.known_templates:
                if r["id"] == u_id:
                    r["name"] = safe
                    self._save_all_database_locked()
                    changed = True
                    break
        if changed and cameras_info:
            self._sync_sidecar_person_name(u_id, safe, cameras_info)
        return changed

    @staticmethod
    def _sync_sidecar_person_name(u_id: str, new_name: str, cameras_info: list):
        for cam in (cameras_info or []):
            faces_dir = os.path.join(str(cam.get("archive_dir") or ""), "faces")
            if not os.path.isdir(faces_dir):
                continue
            for root_dir, _, files in os.walk(faces_dir):
                if "_staging" in root_dir:
                    continue
                for fname in files:
                    if not fname.lower().endswith(".json"):
                        continue
                    mpath = os.path.join(root_dir, fname)
                    try:
                        with open(mpath, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        if isinstance(meta, dict) and str(meta.get("person_uuid") or "") == u_id:
                            meta["person_name"] = new_name
                            with open(mpath, "w", encoding="utf-8") as mf:
                                json.dump(meta, mf, ensure_ascii=False, indent=2)
                    except Exception:
                        pass

    def set_ignore(self, u_id: str, ignore_flag: bool) -> bool:
        with self._db_lock:
            self._sync_if_modified_locked()
            for r in self.known_templates:
                if r["id"] == u_id:
                    r["ignore"] = bool(ignore_flag)
                    self._save_all_database_locked()
                    return True
        return False

    def delete_record(self, u_id: str) -> bool:
        with self._db_lock:
            self._sync_if_modified_locked()
            before = len(self.known_templates)
            self.known_templates = [r for r in self.known_templates if r["id"] != u_id]
            if len(self.known_templates) != before:
                self._save_all_database_locked()
                tpath = os.path.join(self.thumbs_dir, f"{sanitize_filename_part(u_id)}.jpg")
                try:
                    if os.path.exists(tpath):
                        os.remove(tpath)
                except Exception:
                    pass
                return True
        return False

    # --- Збір подій та реаналіз кропів облич ---
    def collect_person_events(self, cameras_info: list) -> dict:
        """
        Сканує архіви облич (<cam_archive_dir>/faces/) по всіх камерах,
        об'єднує кропи, повні кадри (_full.jpg) та sidecar-метадані (.json)
        у хронологічному порядку по кожному UniqueFaceID.
        """
        recs = self.list_records()
        rec_by_id = {r["id"]: r for r in recs}

        sanitized_id_to_uid = {sanitize_filename_part(r["id"]): r["id"] for r in recs}
        sanitized_name_to_uids = {}
        for r in recs:
            sname = sanitize_filename_part(r["name"])
            sanitized_name_to_uids.setdefault(sname, []).append(r["id"])

        events_by_person = {r["id"]: [] for r in recs}
        unassigned_events = []

        for cam in (cameras_info or []):
            cam_uid = str(cam.get("uuid") or "")
            cam_name = str(cam.get("name") or cam_uid or "Камера")
            cam_arch = str(cam.get("archive_dir") or "")
            faces_dir = os.path.join(cam_arch, "faces") if cam_arch else ""
            if not faces_dir or not os.path.isdir(faces_dir):
                continue

            for root_dir, _, files in os.walk(faces_dir):
                if "_staging" in root_dir:
                    continue
                rel_dir = os.path.relpath(root_dir, faces_dir)
                day_key = rel_dir if rel_dir != "." else ""
                for fname in files:
                    lower = fname.lower()
                    if not lower.endswith((".jpg", ".jpeg", ".png")):
                        continue
                    if "_full" in lower or "_face_" not in lower:
                        continue

                    crop_path = os.path.join(root_dir, fname)
                    try:
                        mtime = os.path.getmtime(crop_path)
                    except Exception:
                        continue

                    meta_path = _sidecar_path_for_crop(crop_path)
                    meta = None
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                parsed = json.load(mf)
                            if isinstance(parsed, dict):
                                meta = parsed
                        except Exception:
                            meta = None

                    base_name = os.path.splitext(fname)[0]
                    time_prefix = fname.split("_face_", 1)[0]
                    full_candidate = os.path.join(root_dir, f"{time_prefix}_full.jpg")
                    if meta and meta.get("full_file"):
                        meta_full = os.path.join(root_dir, str(meta["full_file"]))
                        if os.path.exists(meta_full):
                            full_candidate = meta_full
                    full_path = full_candidate if os.path.exists(full_candidate) else ""

                    parsed_ts, day_str, time_str = _parse_ts_from_day_and_filename(day_key, fname, mtime)
                    if meta and isinstance(meta.get("timestamp"), (int, float)) and meta["timestamp"] > 0:
                        ts_val = float(meta["timestamp"])
                        dt_m = datetime.fromtimestamp(ts_val)
                        if not _DAY_DIR_RE.match(day_str):
                            day_str = dt_m.strftime("%Y-%m-%d")
                        if not meta.get("time"):
                            time_str = dt_m.strftime("%H:%M:%S")
                        else:
                            time_str = str(meta.get("time"))
                    else:
                        ts_val = parsed_ts

                    if meta:
                        p_uid = str(meta.get("person_uuid") or "").strip()
                        meta_name = str(meta.get("person_name") or "").strip()
                        tag = str(meta.get("tag") or "")
                        sharpness = meta.get("sharpness")
                    else:
                        after_face = base_name.split("_face_", 1)[1]
                        rest = after_face.rsplit("_", 1)
                        item_name = rest[0] if len(rest) == 2 else after_face
                        tag = rest[1] if len(rest) == 2 else ""
                        sharpness = None
                        meta_name = item_name

                        p_uid = ""
                        if item_name in rec_by_id:
                            p_uid = item_name
                        elif item_name in sanitized_id_to_uid:
                            p_uid = sanitized_id_to_uid[item_name]
                        else:
                            cands = sanitized_name_to_uids.get(item_name) or []
                            if len(cands) == 1:
                                p_uid = cands[0]

                    in_db = p_uid in rec_by_id
                    ev = {
                        "event_id": crop_path,
                        "person_uuid": p_uid,
                        "person_name": (rec_by_id[p_uid]["name"] if in_db else (meta_name or p_uid or "Невідомо")),
                        "in_db": in_db,
                        "camera_uuid": cam_uid,
                        "camera_name": cam_name,
                        "day": day_str,
                        "time": time_str,
                        "timestamp": float(ts_val),
                        "tag": tag,
                        "sharpness": sharpness,
                        "crop_path": crop_path,
                        "full_path": full_path,
                        "meta_path": meta_path if os.path.exists(meta_path) else "",
                        "filename": fname,
                    }
                    if in_db:
                        events_by_person[p_uid].append(ev)
                    else:
                        unassigned_events.append(ev)

        summary = {}
        for uid, ev_list in events_by_person.items():
            ev_list.sort(key=lambda e: (e["timestamp"], e["filename"]))
            cams_seen = []
            for e in ev_list:
                if e["camera_name"] not in cams_seen:
                    cams_seen.append(e["camera_name"])
            summary[uid] = {
                "crop_count": len(ev_list),
                "screen_count": sum(1 for e in ev_list if e["full_path"]),
                "first_ts": ev_list[0]["timestamp"] if ev_list else 0.0,
                "last_ts": ev_list[-1]["timestamp"] if ev_list else 0.0,
                "cameras": cams_seen,
            }

        unassigned_events.sort(key=lambda e: (e["timestamp"], e["filename"]))
        return {
            "records": recs,
            "events_by_person": events_by_person,
            "unassigned_events": unassigned_events,
            "summary": summary,
        }

    def _compute_crop_embedding_and_sharpness(self, crop_path: str):
        embedder = self._get_embedder()
        if embedder is None:
            return None, 0.0, None
        bgr = _imread_bgr(crop_path)
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return None, 0.0, None
        sh = calculate_sharpness(bgr)
        if bgr.shape[0] != 112 or bgr.shape[1] != 112:
            if cv2 is not None:
                bgr_112 = cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_AREA)
            elif _PILImage is not None:
                rgb_pil = _PILImage.fromarray(bgr[:, :, ::-1]).resize((112, 112), _PILImage.Resampling.BILINEAR)
                bgr_112 = np.asarray(rgb_pil, dtype=np.uint8)[:, :, ::-1].copy()
            else:
                return None, 0.0, None
        else:
            bgr_112 = bgr
        emb = embedder.compute_embedding(bgr_112)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb, sh, bgr

    @staticmethod
    def _sample_paths(paths: list, limit: int = 16) -> list:
        clean = [p for p in (paths or []) if p and os.path.exists(p)]
        if len(clean) <= limit:
            return clean
        step = (len(clean) - 1) / float(max(1, limit - 1))
        chosen = []
        seen_idx = set()
        for i in range(limit):
            idx = int(round(i * step))
            if idx not in seen_idx and idx < len(clean):
                seen_idx.add(idx)
                chosen.append(clean[idx])
        return chosen

    def _build_gallery_from_crop_paths(self, crop_paths: list, max_eval: int = 20):
        sampled = self._sample_paths(crop_paths, limit=max_eval)
        samples = []
        for cp in sampled:
            try:
                emb, sh, bgr = self._compute_crop_embedding_and_sharpness(cp)
                if emb is not None:
                    samples.append((emb, sh, cp, bgr))
            except Exception:
                continue
        if not samples:
            return [], None, None

        samples.sort(key=lambda item: item[1], reverse=True)
        chosen = [samples[0]]
        remaining = list(samples[1:])
        while len(chosen) < 4 and remaining:
            # Обираємо найрізноманітніший ракурс серед якісних кропів
            best_pos = 0
            best_max_sim = 999.0
            for idx, cand in enumerate(remaining):
                msim = max(float(np.dot(cand[0], g[0])) for g in chosen)
                if msim < best_max_sim:
                    best_max_sim = msim
                    best_pos = idx
            chosen.append(remaining.pop(best_pos))

        gallery = [c[0] for c in chosen]
        mean_v = np.mean(gallery, axis=0)
        centroid = mean_v / (np.linalg.norm(mean_v) + 1e-10)
        best_bgr = samples[0][3]
        return gallery, centroid, best_bgr

    def recalculate_person_profile_from_crops(self, u_id: str, remaining_crop_paths: list) -> dict:
        """
        Перераховує вектори галереї (gallery), центроїд та мініатюру (thumb) людини u_id
        лише на основі дійсних кропів, що залишилися на диску.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return {"ok": False, "recalculated": False, "reason": "Нейромережа облич (face_recognizer.onnx / cv2 / onnxruntime) недоступна"}

        gallery, centroid, best_bgr = self._build_gallery_from_crop_paths(remaining_crop_paths)
        with self._db_lock:
            self._sync_if_modified_locked()
            target = None
            for r in self.known_templates:
                if r["id"] == u_id:
                    target = r
                    break
            if target is None:
                return {"ok": False, "recalculated": False, "reason": "Особу не знайдено в базі"}
            if not gallery:
                return {"ok": True, "recalculated": False, "used_crops": 0, "reason": "Не залишилося кропів для перерахунку"}

            target["gallery"] = gallery
            target["vector"] = centroid
            self._save_thumb_if_needed(u_id, best_bgr, overwrite=True)
            self._save_all_database_locked()
            return {"ok": True, "recalculated": True, "gallery_count": len(gallery), "used_crops": len(gallery)}

    def analyze_crop_candidates(
        self,
        crop_path: str,
        exclude_uids=None,
        person_crops_map: dict = None,
        min_similarity: float = 0.45,
        max_candidates: int = 8,
    ) -> dict:
        """
        Обчислює ембедінг вказаного кропу й реаналізує базу облич та збережених кропів на диску
        (окрім exclude_uids), повертаючи ранжований список найближчих кандидатів.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return {"ok": False, "error": "Нейромережа облич (face_recognizer.onnx / opencv / onnxruntime) недоступна."}

        if not crop_path or not os.path.exists(crop_path):
            return {"ok": False, "error": "Файл кропу обличчя не знайдено на диску."}

        target_emb, sh, _ = self._compute_crop_embedding_and_sharpness(crop_path)
        if target_emb is None:
            return {"ok": False, "error": "Не вдалося прочитати зображення кропу або обчислити вектор."}

        excl = set(exclude_uids or [])
        with self._db_lock:
            self._sync_if_modified_locked()
            snap_recs = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "ignore": bool(r.get("ignore", False)),
                    "gallery": [v.copy() for v in (r.get("gallery") or [r["vector"]])],
                    "thumb_path": self.get_thumb_path(r["id"]),
                }
                for r in self.known_templates
                if r["id"] not in excl
            ]

        norm_target = os.path.normpath(crop_path)
        ranked = []
        for rec in snap_recs:
            uid = rec["id"]
            gal_sim = max(float(np.dot(target_emb, v)) for v in rec["gallery"])

            crop_sim = -1.0
            matched_crop = ""
            cands_to_probe = []
            if person_crops_map and uid in person_crops_map:
                cands_to_probe.extend(
                    p for p in self._sample_paths(person_crops_map[uid], limit=6)
                    if os.path.normpath(p) != norm_target
                )
            tp = rec["thumb_path"]
            if tp and os.path.exists(tp) and os.path.normpath(tp) != norm_target:
                if os.path.normpath(tp) not in {os.path.normpath(p) for p in cands_to_probe}:
                    cands_to_probe.append(tp)

            for cp in cands_to_probe:
                try:
                    c_emb, _, _ = self._compute_crop_embedding_and_sharpness(cp)
                    if c_emb is not None:
                        s_val = float(np.dot(target_emb, c_emb))
                        if s_val > crop_sim:
                            crop_sim = s_val
                            matched_crop = cp
                except Exception:
                    continue

            eff_sim = max(gal_sim, crop_sim)
            ranked.append({
                "id": uid,
                "name": rec["name"],
                "ignore": rec["ignore"],
                "similarity": eff_sim,
                "gallery_sim": gal_sim,
                "crop_sim": crop_sim if crop_sim >= 0.0 else None,
                "passes_threshold": eff_sim >= float(min_similarity),
                "thumb_path": tp,
                "matched_crop": matched_crop,
            })

        ranked.sort(key=lambda x: x["similarity"], reverse=True)
        best = ranked[0] if (ranked and ranked[0]["passes_threshold"]) else None
        return {
            "ok": True,
            "candidates": ranked[:max(1, int(max_candidates))],
            "all_candidates_count": len(ranked),
            "best_candidate": best,
            "target_crop": crop_path,
            "sharpness": sh,
            "min_similarity": float(min_similarity),
        }

    def reanalyze_wrong_crop(
        self,
        crop_path: str,
        old_u_id: str,
        remaining_old_crops: list,
        target_uid: str = None,
        person_crops_map: dict = None,
        min_similarity: float = 0.45,
        full_path: str = "",
        day_str: str = "",
        time_str: str = "",
        timestamp: float = 0.0,
        sharpness: float = None,
        remove_empty_old: bool = False,
    ) -> dict:
        """
        Позначає кроп як помилковий для old_u_id:
        1) Очищує й перераховує еталони old_u_id за його рештою кропів (щоб помилка не засмічувала базу).
        2) Виконує реаналіз кропу проти інших людей (або призначає вказаного target_uid чи нову людину з UUID).
        3) Перейменовує файл кропу та оновлює .json-sidecar під нового власника.
        """
        embedder = self._get_embedder()
        if embedder is None:
            return {"ok": False, "error": "Нейромережа облич (face_recognizer.onnx / opencv / onnxruntime) недоступна."}

        if not crop_path or not os.path.exists(crop_path):
            return {"ok": False, "error": "Файл кропу обличчя не знайдено."}

        wrong_emb, calc_sh, wrong_bgr = self._compute_crop_embedding_and_sharpness(crop_path)
        if wrong_emb is None:
            return {"ok": False, "error": "Не вдалося обробити зображення кропу."}

        # Попередньо рахуємо нові еталони старого профілю поза блокуванням
        norm_wrong = os.path.normpath(crop_path)
        clean_old_remaining = [p for p in (remaining_old_crops or []) if os.path.normpath(p) != norm_wrong and os.path.exists(p)]
        old_gal, old_cent, old_best_bgr = ([], None, None)
        if old_u_id and clean_old_remaining:
            old_gal, old_cent, old_best_bgr = self._build_gallery_from_crop_paths(clean_old_remaining)

        best_sim = 0.0
        if target_uid is None:
            scan = self.analyze_crop_candidates(
                crop_path,
                exclude_uids=[old_u_id] if old_u_id else None,
                person_crops_map=person_crops_map,
                min_similarity=min_similarity,
                max_candidates=1,
            )
            if scan.get("ok") and scan.get("best_candidate"):
                target_uid = scan["best_candidate"]["id"]
                best_sim = float(scan["best_candidate"]["similarity"])
            else:
                target_uid = "__NEW__"

        created_new = False
        with self._db_lock:
            self._sync_if_modified_locked()

            if old_u_id:
                for r in self.known_templates:
                    if r["id"] == old_u_id:
                        if old_gal:
                            r["gallery"] = old_gal
                            r["vector"] = old_cent
                            self._save_thumb_if_needed(old_u_id, old_best_bgr, overwrite=True)
                        break

            dest_rec = None
            if target_uid and target_uid != "__NEW__":
                for r in self.known_templates:
                    if r["id"] == target_uid and r["id"] != old_u_id:
                        dest_rec = r
                        break

            if dest_rec is None:
                new_id = self._next_unique_id_locked()
                new_name = f"Особа-{new_id[:8]}"
                dest_rec = {
                    "id": new_id,
                    "name": new_name,
                    "ignore": False,
                    "vector": wrong_emb,
                    "gallery": [wrong_emb],
                }
                self.known_templates.append(dest_rec)
                self._save_thumb_if_needed(new_id, wrong_bgr, overwrite=True)
                created_new = True
                best_sim = 1.0
            else:
                if best_sim <= 0.0:
                    best_sim = max(float(np.dot(wrong_emb, v)) for v in (dest_rec.get("gallery") or [dest_rec["vector"]]))
                gal = dest_rec.setdefault("gallery", [dest_rec["vector"]])
                max_g = max(float(np.dot(wrong_emb, v)) for v in gal)
                if len(gal) < 4 and max_g < 0.96:
                    gal.append(wrong_emb)
                    mean_v = np.mean(gal, axis=0)
                    dest_rec["vector"] = mean_v / (np.linalg.norm(mean_v) + 1e-10)
                self._save_thumb_if_needed(dest_rec["id"], wrong_bgr, overwrite=False)

            if remove_empty_old and old_u_id and not clean_old_remaining and dest_rec["id"] != old_u_id:
                self.known_templates = [r for r in self.known_templates if r["id"] != old_u_id]
                old_tp = os.path.join(self.thumbs_dir, f"{sanitize_filename_part(old_u_id)}.jpg")
                try:
                    if os.path.exists(old_tp):
                        os.remove(old_tp)
                except Exception:
                    pass

            self._save_all_database_locked()
            dest_uid = dest_rec["id"]
            dest_name = dest_rec["name"]

        # Оновлюємо файли кропу та sidecar .json на диску
        old_dir = os.path.dirname(crop_path)
        old_fname = os.path.basename(crop_path)
        old_meta = _sidecar_path_for_crop(crop_path)
        time_prefix = old_fname.split("_face_", 1)[0] if "_face_" in old_fname else os.path.splitext(old_fname)[0]
        ext = os.path.splitext(old_fname)[1] or ".jpg"
        safe_dest_name = sanitize_filename_part(dest_name, fallback=dest_uid[:8])
        tag_str = "NEW" if created_new else f"{int(round(best_sim * 100))}%"
        safe_tag = sanitize_filename_part(tag_str, fallback="FIX")

        new_crop_path = os.path.join(old_dir, f"{time_prefix}_face_{safe_dest_name}_{safe_tag}{ext}")
        if os.path.normpath(new_crop_path) != os.path.normpath(crop_path):
            idx = 1
            while os.path.exists(new_crop_path) and os.path.normpath(new_crop_path) != os.path.normpath(crop_path):
                new_crop_path = os.path.join(old_dir, f"{time_prefix}_face_{safe_dest_name}_{safe_tag}_{idx}{ext}")
                idx += 1
            try:
                os.replace(crop_path, new_crop_path)
            except Exception:
                new_crop_path = crop_path

        if os.path.exists(old_meta) and os.path.normpath(old_meta) != os.path.normpath(_sidecar_path_for_crop(new_crop_path)):
            try:
                os.remove(old_meta)
            except Exception:
                pass

        try:
            mtime = os.path.getmtime(new_crop_path)
        except Exception:
            mtime = time.time()
        parsed_ts, p_day, p_time = _parse_ts_from_day_and_filename(day_str, os.path.basename(new_crop_path), mtime)
        meta_obj = {
            "person_uuid": dest_uid,
            "person_name": dest_name,
            "tag": tag_str,
            "timestamp": float(timestamp) if timestamp and timestamp > 0 else float(parsed_ts),
            "day": day_str or p_day,
            "time": time_str or p_time,
            "crop_file": os.path.basename(new_crop_path),
            "full_file": os.path.basename(full_path) if full_path else f"{time_prefix}_full.jpg",
            "sharpness": round(float(sharpness if sharpness is not None else calc_sh), 2),
        }
        try:
            with open(_sidecar_path_for_crop(new_crop_path), "w", encoding="utf-8") as mf:
                json.dump(meta_obj, mf, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {
            "ok": True,
            "old_uid": old_u_id,
            "new_uid": dest_uid,
            "new_name": dest_name,
            "created_new": created_new,
            "similarity": float(best_sim),
            "new_crop_path": new_crop_path,
            "old_remaining": len(clean_old_remaining),
        }

    def reanalyze_all_profiles_from_disk(self, cameras_info: list) -> dict:
        """Масово перераховує еталонні вектори (faces.txt + thumbs) усіх людей за реальними кропами на диску."""
        embedder = self._get_embedder()
        if embedder is None:
            return {"ok": False, "error": "Нейромережа облич (face_recognizer.onnx / opencv / onnxruntime) недоступна."}

        collected = self.collect_person_events(cameras_info)
        ev_map = collected.get("events_by_person") or {}
        updated = 0
        skipped = 0
        for rec in list(collected.get("records") or []):
            uid = rec["id"]
            paths = [e["crop_path"] for e in (ev_map.get(uid) or []) if os.path.exists(e.get("crop_path", ""))]
            if not paths:
                skipped += 1
                continue
            res = self.recalculate_person_profile_from_crops(uid, paths)
            if res.get("recalculated"):
                updated += 1
            else:
                skipped += 1
        return {"ok": True, "updated_profiles": updated, "skipped_empty": skipped}

    def delete_crop_event(self, event_info: dict, remaining_old_crops: list) -> dict:
        """Видаляє вказаний кроп, його sidecar і незалежний _full.jpg, після чого перераховує профіль."""
        if not isinstance(event_info, dict):
            return {"ok": False, "error": "Некоректний запис події"}

        crop_path = str(event_info.get("crop_path") or "")
        meta_path = str(event_info.get("meta_path") or (_sidecar_path_for_crop(crop_path) if crop_path else ""))
        full_path = str(event_info.get("full_path") or "")
        old_uid = str(event_info.get("person_uuid") or "")

        if crop_path and os.path.exists(crop_path):
            try:
                os.remove(crop_path)
            except Exception as e:
                return {"ok": False, "error": f"Не вдалося видалити файл кропу: {e}"}

        if meta_path and os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except Exception:
                pass

        deleted_full = False
        if full_path and os.path.exists(full_path):
            day_dir = os.path.dirname(full_path)
            time_prefix = os.path.basename(crop_path).split("_face_", 1)[0] if crop_path else ""
            shared = False
            if time_prefix and os.path.isdir(day_dir):
                target_pref = f"{time_prefix}_face_".lower()
                for fn in os.listdir(day_dir):
                    if fn.lower().startswith(target_pref) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        shared = True
                        break
            if not shared:
                try:
                    os.remove(full_path)
                    deleted_full = True
                except Exception:
                    pass

        norm_del = os.path.normpath(crop_path) if crop_path else ""
        clean_rem = [p for p in (remaining_old_crops or []) if os.path.normpath(p) != norm_del and os.path.exists(p)]
        if old_uid and clean_rem and self.has_neural_engine():
            self.recalculate_person_profile_from_crops(old_uid, clean_rem)

        return {"ok": True, "deleted_full": deleted_full, "old_remaining": len(clean_rem)}

    def delete_person_completely(self, u_id: str, person_events: list, delete_files: bool = True) -> dict:
        deleted_files = 0
        if delete_files:
            for ev in list(person_events or []):
                cp = str(ev.get("crop_path") or "")
                mp = str(ev.get("meta_path") or (_sidecar_path_for_crop(cp) if cp else ""))
                fp = str(ev.get("full_path") or "")
                if cp and os.path.exists(cp):
                    try:
                        os.remove(cp)
                        deleted_files += 1
                    except Exception:
                        pass
                if mp and os.path.exists(mp):
                    try:
                        os.remove(mp)
                    except Exception:
                        pass
                if fp and os.path.exists(fp):
                    day_dir = os.path.dirname(fp)
                    tpref = os.path.basename(cp).split("_face_", 1)[0] if cp else ""
                    shared = False
                    if tpref and os.path.isdir(day_dir):
                        target_pref = f"{tpref}_face_".lower()
                        for fn in os.listdir(day_dir):
                            if fn.lower().startswith(target_pref) and fn.lower().endswith((".jpg", ".jpeg", ".png")):
                                shared = True
                                break
                    if not shared:
                        try:
                            os.remove(fp)
                        except Exception:
                            pass
        ok = self.delete_record(u_id)
        return {"ok": ok, "deleted_files": deleted_files}


class FaceRecognizer(_BaseFaceDB):
    def __init__(self, model_path, db_file_path, providers=None, intra_op_threads=2, thumbs_dir=None):
        super().__init__(db_file_path=db_file_path, thumbs_dir=thumbs_dir)
        self.model_path = model_path
        self._infer_lock = threading.Lock()
        self.session = _build_ort_session(model_path, providers=providers, intra_op_threads=intra_op_threads)
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

        with self._db_lock:
            self._load_database_locked()

    def _get_embedder(self):
        return self

    def compute_embedding(self, face_112_bgr):
        _ensure_deps()
        if cv2 is not None:
            rgb = cv2.cvtColor(face_112_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = face_112_bgr[:, :, ::-1]
        tensor = (rgb.astype(np.float32) - 127.5) / 128.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        with self._infer_lock:
            outs = self.session.run(None, {self.input_name: tensor})
        emb = outs[0][0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-10)

    def identify_readonly(self, fused_embedding, min_similarity=0.45):
        """Повертає поточний збіг без створення нового ID або запису на диск (для live-підпису)."""
        norm_emb = fused_embedding / (np.linalg.norm(fused_embedding) + 1e-10)
        with self._db_lock:
            self._sync_if_modified_locked()
            best_idx, best_sim = self._find_best_match_locked(norm_emb)
            if best_sim >= min_similarity and best_idx != -1:
                rec = self.known_templates[best_idx]
                return rec["id"], rec["name"], f"{int(round(best_sim * 100))}%", rec.get("ignore", False)
        return None, "Особа", "", False

    def identify_or_register(
        self,
        fused_embedding,
        current_sharpness=0.0,
        min_similarity=0.45,
        allow_register=True,
        allow_template_update=True,
        crop_img=None,
    ):
        """
        Порівнює з мульти-шаблонною галереєю:
        - Якщо особа впізнана: адаптивно оновлює або додає стабільний ракурс лише при високому запасі схожості.
        - Якщо особа нова та allow_register=True: створює унікальний UUID з Ignore=False.
        Повертає: (UniqueFaceID | None, FaceName, status_tag, is_ignored: bool)
        """
        norm_emb = fused_embedding / (np.linalg.norm(fused_embedding) + 1e-10)
        now = time.time()

        with self._db_lock:
            self._sync_if_modified_locked()
            best_idx, best_sim = self._find_best_match_locked(norm_emb)

            if best_sim >= min_similarity and best_idx != -1:
                rec = self.known_templates[best_idx]
                sim_pct_val = int(round(best_sim * 100))
                sim_tag = f"{sim_pct_val}%"

                if "gallery" not in rec or not rec["gallery"]:
                    rec["gallery"] = [rec["vector"]]

                self._save_thumb_if_needed(rec["id"], crop_img, overwrite=False)

                adapt_thresh = max(float(min_similarity) + 0.08, 0.58)
                if allow_template_update and current_sharpness >= 28.0 and best_sim >= adapt_thresh:
                    db_dirty = False
                    if best_sim >= 0.82:
                        last_ts = self._last_template_save_ts.get(rec["id"], 0.0)
                        if (now - last_ts) >= 20.0:
                            best_v_idx = int(np.argmax([float(np.dot(norm_emb, v)) for v in rec["gallery"]]))
                            rec["gallery"][best_v_idx] = 0.85 * rec["gallery"][best_v_idx] + 0.15 * norm_emb
                            rec["gallery"][best_v_idx] /= (np.linalg.norm(rec["gallery"][best_v_idx]) + 1e-10)
                            db_dirty = True
                    elif len(rec["gallery"]) < 4:
                        rec["gallery"].append(norm_emb)
                        db_dirty = True

                    if db_dirty:
                        mean_v = np.mean(rec["gallery"], axis=0)
                        rec["vector"] = mean_v / (np.linalg.norm(mean_v) + 1e-10)
                        self._save_all_database_locked()
                        self._last_template_save_ts[rec["id"]] = now

                return rec["id"], rec["name"], sim_tag, rec.get("ignore", False)

            if not allow_register:
                return None, "Особа", "LOW", False

            new_id = self._next_unique_id_locked()
            new_name = f"Особа-{new_id[:8]}"
            new_record = {
                "id": new_id,
                "name": new_name,
                "ignore": False,
                "vector": norm_emb,
                "gallery": [norm_emb],
            }
            self.known_templates.append(new_record)
            self._save_thumb_if_needed(new_id, crop_img, overwrite=True)
            self._save_all_database_locked()
            self._last_template_save_ts[new_id] = now

            print(f"[FaceEngine] Зареєстровано нове обличчя: {new_name} ({new_id}) [IGNORE=FALSE]")
            return new_id, new_name, "NEW", False


class FallbackFaceDB(_BaseFaceDB):
    """Резервний менеджер бази облич (коли жодна камера ще не підключила FaceRecognizer або бібліотеки відсутні)."""
    def __init__(self, db_path: str, base_dir: str = ""):
        super().__init__(db_file_path=db_path)
        self.base_dir = os.path.abspath(base_dir) if base_dir else ""
        self._lazy_embedder = None
        self._lazy_tried = False
        with self._db_lock:
            self._load_database_locked()

    def _sync_if_modified_locked(self):
        # Для легкої БД без 30FPS-циклу перечитуємо завжди при зміні або відсутності
        if not os.path.exists(self.db_path):
            self.known_templates.clear()
            self._db_sig = None
            return
        self._load_database_locked()

    def _get_embedder(self):
        if self._lazy_embedder is not None:
            return self._lazy_embedder
        if self._lazy_tried:
            return None
        self._lazy_tried = True
        _ensure_deps()
        if ort is None or (cv2 is None and _PILImage is None) or not self.base_dir:
            return None
        rec_path = _find_model_file(self.base_dir, RECOGNIZER_CANDIDATES)
        if not rec_path:
            return None
        try:
            self._lazy_embedder = FaceRecognizer(rec_path, self.db_path)
            return self._lazy_embedder
        except Exception:
            return None


class FaceTrack:
    def __init__(self, track_id, bbox, face_crop, frame, sharpness, yaw=0.0, pitch=0.0, max_pool_size=5):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen = time.time()
        self.first_seen = self.last_seen
        self.saved = False
        self.max_pool_size = max_pool_size

        self.face_pool = [(face_crop, sharpness)]
        self.best_face_img = face_crop
        self.best_frame = frame
        self.best_sharpness = sharpness
        self.best_yaw = abs(float(yaw))
        self.best_pitch = abs(float(pitch))
        self.frames_tracked = 1

        self.u_id = None
        self.face_name = "Особа"
        self.status_tag = ""
        self.is_ignored = False
        self.last_preview_ts = 0.0

    def update(self, bbox, face_crop, frame, sharpness, yaw=0.0, pitch=0.0):
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_tracked += 1

        if sharpness > self.best_sharpness:
            self.best_sharpness = sharpness
            self.best_face_img = face_crop
            self.best_frame = frame
            self.best_yaw = abs(float(yaw))
            self.best_pitch = abs(float(pitch))

        if len(self.face_pool) < self.max_pool_size:
            self.face_pool.append((face_crop, sharpness))
            self.face_pool.sort(key=lambda x: x[1], reverse=True)
        elif sharpness > self.face_pool[-1][1]:
            self.face_pool[-1] = (face_crop, sharpness)
            self.face_pool.sort(key=lambda x: x[1], reverse=True)


class FaceEngine:
    def __init__(
        self,
        base_dir,
        config,
        on_saved_callback=None,
        detector: SCRFDDetector = None,
        recognizer: FaceRecognizer = None,
        output_dir: str = None,
        db_file_path: str = None,
        providers=None,
    ):
        self.base_dir = base_dir
        self.config = dict(config or {})
        self.on_saved_callback = on_saved_callback

        self.faces_dir = output_dir or os.path.join(base_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        self.db_file = db_file_path or os.path.join(self.faces_dir, "faces.txt")

        self._reload_params()

        if detector is not None:
            self.detector = detector
        else:
            det_path = _find_model_file(base_dir, DETECTOR_CANDIDATES)
            if not det_path:
                raise FileNotFoundError("Детектор облич SCRFD не знайдено в папці models/")
            self.detector = SCRFDDetector(det_path, providers=providers)

        if recognizer is not None:
            self.recognizer = recognizer
        else:
            rec_path = _find_model_file(base_dir, RECOGNIZER_CANDIDATES)
            if not rec_path:
                raise FileNotFoundError("Модель розпізнавання облич не знайдено в папці models/")
            self.recognizer = FaceRecognizer(rec_path, self.db_file, providers=providers)

        self.active_tracks = {}
        self.next_track_id = 1
        self.track_timeout = 2.0
        self.recently_saved = {}
        self._state_lock = threading.Lock()

    def _reload_params(self):
        self.face_min_similarity = float(self.config.get("faces_min_similarity", self.config.get("min_similarity", 0.48)))
        self.face_det_thresh = float(self.config.get("faces_det_thresh", self.config.get("det_thresh", 0.45)))
        self.face_min_size = int(self.config.get("faces_min_size", self.config.get("min_size", 20)))

        raw_sleep = self.config.get("faces_sleep_time_after_save", self.config.get("sleep_time_after_save", None))
        if raw_sleep is not None and str(raw_sleep).strip() != "":
            self.face_sleep_time = max(0.0, float(raw_sleep))
        else:
            self.face_sleep_time = max(0.0, float(self.config.get("plates_sleep_time_after_save", 5.0)) * 2.0)

        raw_extend = self.config.get("faces_cooldown_extend_on_seen", self.config.get("cooldown_extend_on_seen", True))
        if isinstance(raw_extend, str):
            self.cooldown_extend_on_seen = raw_extend.strip().lower() in ("true", "1", "yes", "on")
        else:
            self.cooldown_extend_on_seen = bool(raw_extend)

    def update_config(self, config: dict):
        with self._state_lock:
            self.config = dict(config or {})
            self._reload_params()

    def _prune_recently_saved_locked(self, now: float):
        if self.face_sleep_time <= 0:
            self.recently_saved.clear()
            return
        self.recently_saved = {
            uid: ts for uid, ts in self.recently_saved.items()
            if (now - ts) < self.face_sleep_time
        }

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            return []
        frame_h, frame_w = frame.shape[:2]
        face_detections = self.detector.detect(frame, conf_thresh=self.face_det_thresh)
        current_faces = []

        for f_info in face_detections:
            ok, ayaw, apitch = evaluate_frontal_face(
                f_info["landmarks"],
                f_info["box"],
                min_size=self.face_min_size,
                max_yaw=32.0,
                max_pitch=28.0,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            if not ok:
                continue

            aligned = align_face(frame, f_info["landmarks"], box=f_info["box"])
            f_sharpness = calculate_sharpness(aligned)

            current_faces.append({
                "box": f_info["box"],
                "crop": aligned,
                "sharpness": f_sharpness,
                "yaw": ayaw,
                "pitch": apitch,
            })

        now = time.time()
        unmatched_faces = list(range(len(current_faces)))

        with self._state_lock:
            self._prune_recently_saved_locked(now)

            for f_id, f_track in list(self.active_tracks.items()):
                best_idx = -1
                min_dist = float("inf")
                for idx in unmatched_faces:
                    det = current_faces[idx]
                    c1x = (f_track.bbox[0] + f_track.bbox[2]) / 2.0
                    c1y = (f_track.bbox[1] + f_track.bbox[3]) / 2.0
                    c2x = (det["box"][0] + det["box"][2]) / 2.0
                    c2y = (det["box"][1] + det["box"][3]) / 2.0
                    dist = math.hypot(c1x - c2x, c1y - c2y)
                    bw = max(10, det["box"][2] - det["box"][0])
                    bh = max(10, det["box"][3] - det["box"][1])
                    max_track_dist = max(40.0, max(bw, bh) * 2.0)
                    if dist < max_track_dist and dist < min_dist:
                        min_dist = dist
                        best_idx = idx

                if best_idx != -1:
                    det = current_faces[best_idx]
                    f_track.update(
                        det["box"], det["crop"], frame.copy(), det["sharpness"],
                        yaw=det["yaw"], pitch=det["pitch"]
                    )
                    unmatched_faces.remove(best_idx)

                    if f_track.u_id and self.cooldown_extend_on_seen and f_track.u_id in self.recently_saved:
                        self.recently_saved[f_track.u_id] = now

                    if not f_track.saved and f_track.frames_tracked >= 2 and (now - f_track.last_preview_ts) >= 0.8:
                        f_track.last_preview_ts = now
                        if f_track.best_sharpness >= 18.0:
                            emb = self.recognizer.compute_embedding(f_track.best_face_img)
                            p_uid, p_name, p_tag, p_ign = self.recognizer.identify_readonly(
                                emb, min_similarity=self.face_min_similarity
                            )
                            if p_uid is not None:
                                f_track.u_id = p_uid
                                f_track.face_name = p_name
                                f_track.status_tag = p_tag
                                f_track.is_ignored = p_ign

                    if not f_track.saved and f_track.best_sharpness >= 40.0 and f_track.frames_tracked >= 6:
                        self._save_verified_face_locked(f_track, is_flush=False)
                        f_track.saved = True

            for idx in unmatched_faces:
                det = current_faces[idx]
                f_tid = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[f_tid] = FaceTrack(
                    f_tid, det["box"], det["crop"], frame.copy(), det["sharpness"],
                    yaw=det["yaw"], pitch=det["pitch"]
                )

            for f_id, f_track in list(self.active_tracks.items()):
                if (now - f_track.last_seen) > self.track_timeout:
                    if not f_track.saved:
                        self._save_verified_face_locked(f_track, is_flush=False)
                    del self.active_tracks[f_id]

            display_boxes = [(t.bbox, t.face_name, t.status_tag) for t in self.active_tracks.values()]
        return display_boxes

    def flush_tracks(self):
        """Примусове збереження активних незавершених треків (при зупинці або межі дат)."""
        with self._state_lock:
            for f_id, f_track in list(self.active_tracks.items()):
                if not f_track.saved:
                    self._save_verified_face_locked(f_track, is_flush=True)
                del self.active_tracks[f_id]

    def _save_verified_face_locked(self, track: FaceTrack, is_flush: bool = False):
        """Multi-Shot Fusion + оновлення еталона та збереження на диск."""
        if track.best_face_img is None or track.best_sharpness < 25.0:
            return

        embeddings = []
        for face_crop, sh in track.face_pool:
            if sh >= 12.0:
                embeddings.append(self.recognizer.compute_embedding(face_crop))

        if not embeddings:
            embeddings.append(self.recognizer.compute_embedding(track.best_face_img))

        mean_vector = np.mean(embeddings, axis=0)
        fused_vector = mean_vector / (np.linalg.norm(mean_vector) + 1e-10)

        min_reg_frames = 2 if (is_flush and track.best_sharpness >= 45.0) else 3
        can_register_new = (
            track.frames_tracked >= min_reg_frames
            and track.best_sharpness >= 32.0
            and track.best_yaw <= 25.0
            and track.best_pitch <= 22.0
        )

        now = time.time()
        self._prune_recently_saved_locked(now)
        pre_in_cooldown = bool(track.u_id and (now - self.recently_saved.get(track.u_id, 0.0)) < self.face_sleep_time)
        allow_tpl_upd = not (track.is_ignored or pre_in_cooldown)

        u_id, f_name, status_tag, is_ignored = self.recognizer.identify_or_register(
            fused_vector,
            current_sharpness=track.best_sharpness,
            min_similarity=self.face_min_similarity,
            allow_register=can_register_new,
            allow_template_update=allow_tpl_upd,
            crop_img=track.best_face_img,
        )
        track.u_id = u_id
        track.face_name = f_name
        track.status_tag = status_tag
        track.is_ignored = is_ignored
        track.saved = True

        if u_id is None or is_ignored:
            return

        if self.face_sleep_time > 0 and (now - self.recently_saved.get(u_id, 0.0)) < self.face_sleep_time:
            if self.cooldown_extend_on_seen:
                self.recently_saved[u_id] = now
            return
        self.recently_saved[u_id] = now

        now_dt = datetime.now()
        day_folder = now_dt.strftime("%Y-%m-%d")
        time_prefix = now_dt.strftime("%H-%M-%S") + f"-{int(now_dt.microsecond / 1000):03d}"
        time_display = now_dt.strftime("%H:%M:%S")
        day_dir = os.path.join(self.faces_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        safe_name = sanitize_filename_part(f_name, fallback=u_id[:8])
        safe_tag = sanitize_filename_part(status_tag, fallback="ID")
        face_filename = f"{time_prefix}_face_{safe_name}_{safe_tag}.jpg"
        full_filename = f"{time_prefix}_full.jpg"
        face_path = os.path.join(day_dir, face_filename)
        full_path = os.path.join(day_dir, full_filename)
        meta_path = _sidecar_path_for_crop(face_path)

        meta_obj = {
            "person_uuid": str(u_id),
            "person_name": str(f_name),
            "tag": str(status_tag),
            "timestamp": float(now_dt.timestamp()),
            "day": day_folder,
            "time": time_display,
            "crop_file": face_filename,
            "full_file": full_filename,
            "sharpness": round(float(track.best_sharpness), 2),
        }

        crop_copy = track.best_face_img.copy() if track.best_face_img is not None else None
        full_copy = track.best_frame.copy() if track.best_frame is not None else None
        cb = self.on_saved_callback

        def _persist_event():
            try:
                _imwrite_bgr(face_path, crop_copy)
                if full_copy is not None:
                    _imwrite_bgr(full_path, full_copy)
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta_obj, mf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[FaceEngine Save Error]: {e}")
            if cb:
                try:
                    cb(day_folder, time_display, f_name, face_path)
                except Exception:
                    pass

        print(f"[ФІКСАЦІЯ ОБЛИЧЧЯ] {f_name} ({u_id}) [{status_tag}], різкість: {track.best_sharpness:.1f}")
        threading.Thread(target=_persist_event, daemon=True).start()


# ------------------------------------------------------------------
#  Контракт незалежного модуля для ModuleRegistry
# ------------------------------------------------------------------
def _resolve_db_path(base_dir: str, app_context: dict = None) -> str:
    ctx = app_context or {}
    dbp = ctx.get("faces_db_path")
    if dbp:
        return os.path.abspath(dbp)
    return os.path.join(os.path.abspath(base_dir), "archive", "faces_db", "faces.txt")


def _create_shared_resources(base_dir: str, providers: list, ai_threads: int, app_context: dict = None):
    det_path = _find_model_file(base_dir, DETECTOR_CANDIDATES)
    if not det_path:
        raise FileNotFoundError("Детектор облич SCRFD не знайдено в папці models/")
    rec_path = _find_model_file(base_dir, RECOGNIZER_CANDIDATES)
    if not rec_path:
        raise FileNotFoundError("Модель розпізнавання облич не знайдено в папці models/")

    db_path = _resolve_db_path(base_dir, app_context)
    det = SCRFDDetector(det_path, providers=providers, intra_op_threads=ai_threads)
    rec = FaceRecognizer(rec_path, db_path, providers=providers, intra_op_threads=ai_threads)
    return {
        "detector": det,
        "recognizer": rec,
        "rec_path": rec_path,
        "db_path": db_path,
        "providers": list(getattr(det, "active_providers", providers or ["CPUExecutionProvider"])),
    }


def _on_app_context_changed(shared_resources, app_context, base_dir: str, providers: list, ai_threads: int):
    if not isinstance(shared_resources, dict):
        return shared_resources
    new_db = _resolve_db_path(base_dir, app_context)
    cur_db = shared_resources.get("db_path") or ""
    if os.path.abspath(new_db) != os.path.abspath(cur_db):
        rec_path = shared_resources.get("rec_path") or _find_model_file(base_dir, RECOGNIZER_CANDIDATES)
        if rec_path:
            shared_resources["recognizer"] = FaceRecognizer(
                rec_path, new_db, providers=providers, intra_op_threads=ai_threads
            )
            shared_resources["db_path"] = new_db
    return shared_resources


_FALLBACK_DB_LOCK = threading.Lock()
_FALLBACK_DB_CACHE = {}


def _get_db_accessor(app_context: dict = None, shared_resources=None):
    if isinstance(shared_resources, dict) and shared_resources.get("recognizer") is not None:
        return shared_resources["recognizer"]
    base_dir = (app_context or {}).get("base_dir", "")
    db_path = _resolve_db_path(base_dir, app_context)
    cache_key = (os.path.abspath(db_path), os.path.abspath(base_dir) if base_dir else "")
    with _FALLBACK_DB_LOCK:
        inst = _FALLBACK_DB_CACHE.get(cache_key)
        if inst is None:
            inst = FallbackFaceDB(db_path, base_dir=base_dir)
            _FALLBACK_DB_CACHE[cache_key] = inst
        return inst


def _create_engine(base_dir, flat_cfg, output_dir, shared_resources=None, on_saved_callback=None, source=None, app_context=None):
    det = shared_resources.get("detector") if isinstance(shared_resources, dict) else None
    rec = shared_resources.get("recognizer") if isinstance(shared_resources, dict) else None
    db_path = _resolve_db_path(base_dir, app_context)
    return FaceEngine(
        base_dir=base_dir,
        config=flat_cfg,
        on_saved_callback=on_saved_callback,
        detector=det,
        recognizer=rec,
        output_dir=output_dir,
        db_file_path=db_path,
    )


def _update_engine_config(engine, flat_cfg, source=None, shared_resources=None, app_context=None):
    if engine is None:
        return
    if isinstance(shared_resources, dict):
        if shared_resources.get("detector") is not None:
            engine.detector = shared_resources["detector"]
        if shared_resources.get("recognizer") is not None:
            engine.recognizer = shared_resources["recognizer"]
        if shared_resources.get("db_path"):
            engine.db_file = shared_resources["db_path"]
    engine.update_config(flat_cfg)


def _process_and_pack(engine, frame, prev_state=None):
    boxes = engine.process_frame(frame)
    return {"boxes": boxes}


def _draw_overlays(frame, state_data: dict, mod_cfg: dict, now: float):
    if cv2 is None or frame is None or not (mod_cfg or {}).get("show_boxes", True):
        return
    for item in (state_data or {}).get("boxes") or []:
        (fx1, fy1, fx2, fy2), f_name, f_pct = item
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
        pct_str = str(f_pct).strip()
        if pct_str == "NEW":
            tag = f"{f_name} [NEW]"
        elif pct_str:
            tag = f"{f_name} {pct_str}" if pct_str.endswith("%") else f"{f_name} {pct_str}%"
        else:
            tag = f"{f_name}"
        cv2.putText(frame, tag, (fx1, max(18, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)


def _parse_archive_record(fpath: str, base_name: str, root_dir: str, rel_dir: str):
    fname = os.path.basename(fpath)
    lower = fname.lower()
    if "_full" in lower or lower.endswith(("_debug.txt", ".json")):
        return None
    if "_face_" in base_name:
        parts = base_name.split("_face_")
        time_disp = ":".join(parts[0].split("-")[:3])
        rest = parts[1].rsplit("_", 1)
        item_name = rest[0] if len(rest) == 2 else parts[1]
    else:
        time_disp = "--:--:--"
        item_name = base_name
    meta_p = _sidecar_path_for_crop(fpath)
    if os.path.exists(meta_p):
        try:
            with open(meta_p, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if isinstance(meta, dict):
                if meta.get("person_name"):
                    item_name = str(meta["person_name"])
                if meta.get("time"):
                    time_disp = str(meta["time"])
        except Exception:
            pass
    return {"time": time_disp, "name": item_name}


MODULE_SPEC = {
    "id": "faces",
    "version": "2.0.0",
    "order": 20,
    "kind": "ai",
    "name": "Обличчя",
    "icon": "👤",
    "short_badge": "О",
    "description": "Біометричне виявлення, оцінка фронтальності та розпізнавання облич зі спільною базою персон (faces.txt).",
    "on_help": "Увімкнути біометричне розпізнавання облич і роботу зі спільною базою персон (faces.txt) для цієї камери.",
    "provides": ["face_db"],
    "library_deps": ["opencv-python", "onnxruntime"],
    "settings_schema": [
        {
            "key": "show_boxes",
            "type": "bool",
            "default": True,
            "label": "Показувати рамки:",
            "bool_text": "Малювати блакитні рамки й підписи ID/Ім'я",
            "help": "Малювати блакитні рамки навколо облич та їхній поточний ID/ім'я і відсоток схожості.",
        },
        {
            "key": "min_similarity",
            "type": "float",
            "default": 0.48,
            "min": 0.20,
            "max": 0.90,
            "step": 0.02,
            "label": "Поріг схожості векторів:",
            "help": "Поріг схожості векторів облич (0.38–0.65; 0.48 оптимум). Вище — менше хибних збігів різних людей, нижче — краще впізнає під іншим кутом.",
        },
        {
            "key": "det_thresh",
            "type": "float",
            "default": 0.45,
            "min": 0.15,
            "max": 0.95,
            "step": 0.02,
            "label": "Поріг детектора SCRFD:",
            "help": "Поріг впевненості детектора облич SCRFD (0.35–0.70; 0.45 стандарт).",
        },
        {
            "key": "min_size",
            "type": "int",
            "default": 20,
            "min": 12,
            "max": 300,
            "step": 2,
            "label": "Мін. розмір обличчя (px):",
            "help": "Мінімальний розмір сторони обличчя в пікселях (20 для далеких камер, 40–50 для входу).",
        },
        {
            "key": "sleep_time_after_save",
            "type": "float",
            "default": 10.0,
            "min": 0.0,
            "max": 3600.0,
            "step": 1.0,
            "label": "Таймаут повторів (сек):",
            "help": "Персональний таймаут повторів збереження одного й того самого обличчя (сек). Більше не прив'язаний до номерів авто.",
        },
        {
            "key": "cooldown_extend_on_seen",
            "type": "bool",
            "default": True,
            "label": "Sliding Cooldown:",
            "bool_text": "Подовжувати таймаут, поки людина не вийде з кадру",
            "help": (
                "Режим присутності (Sliding Cooldown)\n"
                "Якщо увімкнено: поки людина залишається в полі зору камери, таймаут постійно продовжується і дублікат фото НЕ робиться, доки вона реально не зникне на N секунд.\n"
                "Якщо вимкнено: фото робитиметься фіксовано раз на кожні N секунд присутності."
            ),
        },
        {
            "key": "delete_older_days",
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 3650.0,
            "step": 1.0,
            "label": "Автовидалення старше (днів):",
            "help": "Автовидалення фото облич цієї камери, старших за N днів (0 = необмежено).",
        },
        {
            "key": "date_from",
            "key_to": "date_to",
            "type": "date_range",
            "default": "",
            "default_to": "",
            "label": "Період роботи (дати):",
            "help": "Початкова та кінцева дата роботи модуля облич (YYYY-MM-DD, порожньо = необмежено).",
        },
    ],
    "archive": {
        "has_archive": True,
        "button_label": "👤 Обличчя",
        "sidebar_title": "Історія облич",
        "item_preview_label": "Фото обличчя:",
        "icon": "👤",
        "record_type": "pair_image",
        "extensions": (".jpg", ".jpeg", ".png"),
        "parse_record": _parse_archive_record,
    },
    "refresh_capabilities": refresh_capabilities,
    "get_status_summary": get_status_summary,
    "create_shared_resources": _create_shared_resources,
    "on_app_context_changed": _on_app_context_changed,
    "get_db_accessor": _get_db_accessor,
    "create_engine": _create_engine,
    "update_engine_config": _update_engine_config,
    "process_and_pack": _process_and_pack,
    "draw_overlays": _draw_overlays,
    "flush_tracks": lambda eng: eng.flush_tracks() if eng is not None else None,
    "has_active_tracks": lambda eng: bool(eng is not None and getattr(eng, "active_tracks", None)),
}
