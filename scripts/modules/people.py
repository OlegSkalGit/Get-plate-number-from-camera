import os
import sys
import time
import importlib
import threading
from datetime import datetime
import numpy as np

from scripts.core.i18n import tr
from scripts.core.io_pool import submit_io

try:
    import cv2
except ImportError:
    cv2 = None


def _expand_cuda_provider(req_providers: list) -> list:
    out = []
    for p in req_providers:
        if p == "CUDAExecutionProvider":
            out.append(("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        else:
            out.append(p)
    return out


def _ensure_deps():
    global cv2
    if cv2 is None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            pass


def refresh_capabilities():
    """Перечитує cv2 / onnxruntime після встановлення або оновлення бібліотек."""
    global cv2
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


def is_ready(base_dir: str = "") -> bool:
    if cv2 is None:
        return False
    if base_dir:
        mp = os.path.join(base_dir, "models", "yolo11n.onnx")
        if not os.path.exists(mp):
            return False
    return True


def get_status_summary(base_dir: str = "") -> str:
    if cv2 is None:
        return tr("mod.common.libs_needed", "Required libraries: {libs}", libs="opencv-python")
    if base_dir:
        mp = os.path.join(base_dir, "models", "yolo11n.onnx")
        if not os.path.exists(mp):
            return tr("mod.people.status.missing_model", "Missing model: yolo11n.onnx")
    try:
        import onnxruntime  # noqa: F401
        rt = "ONNX Runtime"
    except ImportError:
        rt = "OpenCV DNN fallback"
    return tr("mod.people.status.ready", "Ready (YOLO11n, {rt})", rt=rt)


def ensure_resources(base_dir: str = "", download: bool = True, progress_cb=None) -> dict:
    """Перевіряє бібліотеки та ONNX-модель детектора людей/транспорту."""
    refresh_capabilities()
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if base_dir:
        mp = os.path.join(base_dir, "models", "yolo11n.onnx")
        if not os.path.exists(mp):
            missing.append("yolo11n.onnx")
    ok = len(missing) == 0
    return {
        "ok": ok,
        "downloaded": [],
        "missing": missing,
        "message": tr("mod.people.res.ready", "People/vehicles model (yolo11n.onnx) is ready.") if ok else tr(
            "mod.common.res_missing",
            "Missing resources: {items}",
            items=", ".join(missing),
        ),
    }


def async_write_image(path, img):
    """Асинхронний запис зображення на диск через спільний обмежений пул."""
    def _worker():
        try:
            if cv2 is not None and img is not None:
                cv2.imwrite(path, img)
        except Exception as e:
            print(f"[Помилка запису скріншота]: {e}")
    submit_io(_worker)


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
ALL_TARGET_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class PersonDetector:
    """Потокобезпечний детектор людей та транспорту на базі YOLO11n (ONNX)."""
    def __init__(self, model_path: str, detect_vehicles: bool = False, providers=None, intra_op_threads: int = 2):
        _ensure_deps()
        if cv2 is None:
            raise ImportError("Package 'opencv-python' is not installed for PersonDetector")
        self.model_path = model_path
        self.detect_vehicles = detect_vehicles
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
        except Exception as e:
            print(f"[PersonDetector ONNX Runtime fallback to cv2.dnn]: {e}")
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.use_ort = False
            self._supports_batch = False

    @staticmethod
    def _preprocess_one(frame: np.ndarray, imgsz: int):
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
    def _postprocess_one(raw_i, meta, conf_thresh: float, use_vehicles: bool, raw_person_hint_thresh: float = 0.12):
        w_orig, h_orig, r, dw, dh = meta
        preds = np.asarray(raw_i)
        if preds.ndim > 2:
            preds = np.squeeze(preds)
        if preds.ndim != 2:
            return [], True

        if preds.shape[0] > preds.shape[1]:
            preds = preds.T

        if preds.shape[0] < 5:
            return [], True

        person_hint = bool(np.any(preds[4, :] >= raw_person_hint_thresh))

        target_classes = ALL_TARGET_CLASSES if use_vehicles else PERSON_CLASSES
        target_rows = [4 + c for c in target_classes]
        class_scores = preds[target_rows, :]
        max_scores = np.max(class_scores, axis=0)
        best_class_indices = np.argmax(class_scores, axis=0)

        mask = max_scores >= conf_thresh
        if not np.any(mask):
            return [], person_hint

        valid_scores = max_scores[mask]
        valid_class_indices = best_class_indices[mask]
        valid_boxes = preds[:4, mask]

        cx = valid_boxes[0]
        cy = valid_boxes[1]
        bw = valid_boxes[2]
        bh = valid_boxes[3]
        hw = bw * 0.5
        hh = bh * 0.5

        ox1 = np.clip(np.round((cx - hw - dw) / r), 0, w_orig - 1).astype(np.int32)
        oy1 = np.clip(np.round((cy - hh - dh) / r), 0, h_orig - 1).astype(np.int32)
        ox2 = np.clip(np.round((cx + hw - dw) / r), 0, w_orig).astype(np.int32)
        oy2 = np.clip(np.round((cy + hh - dh) / r), 0, h_orig).astype(np.int32)

        box_w = ox2 - ox1
        box_h = oy2 - oy1
        keep = (box_w > 10) & (box_h > 10)
        if not np.any(keep):
            return [], person_hint

        ox1_k = ox1[keep]
        oy1_k = oy1[keep]
        ox2_k = ox2[keep]
        oy2_k = oy2[keep]
        bw_k = box_w[keep]
        bh_k = box_h[keep]
        scores_k = valid_scores[keep]
        cls_arr = np.asarray([target_classes[i] for i in valid_class_indices[keep]], dtype=np.int32)

        offsets = cls_arr * 4096
        boxes_for_nms = np.column_stack((ox1_k + offsets, oy1_k + offsets, bw_k, bh_k)).tolist()
        scores_list = scores_k.astype(np.float64).tolist()

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_list, score_threshold=conf_thresh, nms_threshold=0.45)
        if len(indices) == 0:
            return [], person_hint

        detections = []
        for idx in indices.flatten():
            c_id = int(cls_arr[idx])
            detections.append({
                "box": (int(ox1_k[idx]), int(oy1_k[idx]), int(ox2_k[idx]), int(oy2_k[idx])),
                "conf": float(scores_list[idx]),
                "class_id": c_id,
                "label": COCO_CLASSES.get(c_id, f"Class_{c_id}"),
            })
        return detections, person_hint

    def predict(self, frame: np.ndarray, conf_thresh: float = 0.45, imgsz: int = 640, detect_vehicles: bool = None):
        """Детекція людей (клас 0) та опційно транспорту (класи 1-8) без зміни спільного стану."""
        if cv2 is None or frame is None or frame.size == 0:
            return []

        use_vehicles = self.detect_vehicles if detect_vehicles is None else bool(detect_vehicles)
        chw, meta = self._preprocess_one(frame, imgsz)
        blob = chw[None, ...]

        with self._lock:
            if self.use_ort:
                raw = self.session.run(self.output_names, {self.input_name: blob})[0]
            else:
                self.net.setInput(blob)
                raw = self.net.forward()

        dets, _ = self._postprocess_one(raw[0] if raw.ndim >= 3 else raw, meta, conf_thresh, use_vehicles)
        return dets

    def predict_batch(
        self,
        frames: list,
        conf_threshs=0.45,
        detect_vehicles_flags=None,
        imgsz: int = 640,
        raw_person_hint_thresh: float = 0.12,
    ):
        n = len(frames)
        if n == 0 or cv2 is None:
            return [], []
        if isinstance(conf_threshs, (int, float)):
            thresh_list = [float(conf_threshs)] * n
        else:
            thresh_list = [float(x) for x in conf_threshs]
        if detect_vehicles_flags is None:
            veh_list = [self.detect_vehicles] * n
        elif isinstance(detect_vehicles_flags, bool):
            veh_list = [detect_vehicles_flags] * n
        else:
            veh_list = [bool(x) for x in detect_vehicles_flags]

        det_results = [[] for _ in range(n)]
        hint_results = [True] * n
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
            return det_results, hint_results

        if len(valid_indices) == 1 or not self.use_ort or not self._supports_batch:
            for k, orig_idx in enumerate(valid_indices):
                blob = chws[k][None, ...]
                with self._lock:
                    if self.use_ort:
                        raw = self.session.run(self.output_names, {self.input_name: blob})[0]
                    else:
                        self.net.setInput(blob)
                        raw = self.net.forward()
                dets, ph = self._postprocess_one(
                    raw[0] if raw.ndim >= 3 else raw,
                    metas[k],
                    thresh_list[orig_idx],
                    veh_list[orig_idx],
                    raw_person_hint_thresh=raw_person_hint_thresh,
                )
                det_results[orig_idx] = dets
                hint_results[orig_idx] = ph
            return det_results, hint_results

        batch_blob = np.stack(chws, axis=0)
        try:
            with self._lock:
                raw_batch = self.session.run(self.output_names, {self.input_name: batch_blob})[0]
            for k, orig_idx in enumerate(valid_indices):
                dets, ph = self._postprocess_one(
                    raw_batch[k],
                    metas[k],
                    thresh_list[orig_idx],
                    veh_list[orig_idx],
                    raw_person_hint_thresh=raw_person_hint_thresh,
                )
                det_results[orig_idx] = dets
                hint_results[orig_idx] = ph
            return det_results, hint_results
        except Exception:
            self._supports_batch = False
            for k, orig_idx in enumerate(valid_indices):
                blob = chws[k][None, ...]
                with self._lock:
                    raw = self.session.run(self.output_names, {self.input_name: blob})[0]
                dets, ph = self._postprocess_one(
                    raw[0] if raw.ndim >= 3 else raw,
                    metas[k],
                    thresh_list[orig_idx],
                    veh_list[orig_idx],
                    raw_person_hint_thresh=raw_person_hint_thresh,
                )
                det_results[orig_idx] = dets
                hint_results[orig_idx] = ph
            return det_results, hint_results


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


class PersonEngine:
    """Модуль детекції людей та транспорту (підтримує спільний детектор і per-camera стан)."""
    def __init__(
        self,
        base_dir: str,
        config: dict,
        on_saved_callback=None,
        detector: PersonDetector = None,
        output_dir: str = None,
        providers=None,
    ):
        self.base_dir = base_dir
        self.config = dict(config or {})
        self.on_saved_callback = on_saved_callback

        self.people_dir = output_dir or os.path.join(base_dir, "people")
        self._state_lock = threading.Lock()
        self.allow_disk_write = True
        self._reload_params()
        if self.allow_disk_write:
            os.makedirs(self.people_dir, exist_ok=True)

        if detector is not None:
            self.detector = detector
        else:
            model_path = os.path.join(base_dir, "models", "yolo11n.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.detector = PersonDetector(model_path, detect_vehicles=self.detect_vehicles, providers=providers)

        self.last_shot_time = 0.0

    def _reload_params(self):
        self.allow_disk_write = _parse_allow_disk_write(self.config)
        self.conf_thresh = float(self.config.get("people_det_thresh", self.config.get("det_thresh", 0.45)))
        self.timer_ms = int(self.config.get("people_timer_ms", self.config.get("timer_ms", 1000)))
        self.interval_sec = max(0.1, float(self.timer_ms) / 1000.0)

        val_veh = self.config.get("people_detect_transport", self.config.get("detect_transport", True))
        if isinstance(val_veh, str):
            self.detect_vehicles = val_veh.strip().lower() in ("true", "1", "yes", "on")
        else:
            self.detect_vehicles = bool(val_veh)

    def update_config(self, config: dict):
        with self._state_lock:
            self.config = dict(config or {})
            self._reload_params()

    def update_settings(self, conf_thresh: float = None, timer_ms: int = None, detect_transport: bool = None):
        with self._state_lock:
            if conf_thresh is not None:
                self.conf_thresh = float(conf_thresh)
            if timer_ms is not None:
                self.timer_ms = int(timer_ms)
                self.interval_sec = max(0.1, float(self.timer_ms) / 1000.0)
            if detect_transport is not None:
                self.detect_vehicles = bool(detect_transport)

    def _handle_detections(self, frame: np.ndarray, detections: list):
        if not detections:
            return []

        now = time.time()
        should_save = False
        with self._state_lock:
            allow_write = self.allow_disk_write
            interval = self.interval_sec
            if allow_write and (now - self.last_shot_time) >= interval:
                self.last_shot_time = now
                should_save = True

        if should_save:
            self._save_screenshot(frame, detections)

        display_boxes = [(det["box"], int(round(det["conf"] * 100)), det.get("label", "Person")) for det in detections]
        return display_boxes

    def process_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return []
        with self._state_lock:
            conf_t = self.conf_thresh
            det_veh = self.detect_vehicles

        detections = self.detector.predict(frame, conf_thresh=conf_t, detect_vehicles=det_veh)
        return self._handle_detections(frame, detections)

    def _save_screenshot(self, frame: np.ndarray, detections: list):
        if not self.allow_disk_write:
            return
        os.makedirs(self.people_dir, exist_ok=True)
        now_dt = datetime.now()
        day_folder = now_dt.strftime("%Y-%m-%d")
        day_dir = os.path.join(self.people_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        detected_labels = list(dict.fromkeys(det.get("label", "Person") for det in detections))
        summary_label = ", ".join(detected_labels)
        slug_label = "-".join(detected_labels)

        if self.timer_ms < 1000:
            time_prefix = now_dt.strftime("%H-%M-%S") + f"_{int(now_dt.microsecond / 1000):03d}"
        else:
            time_prefix = now_dt.strftime("%H-%M-%S")

        filename = f"{time_prefix}_{slug_label}.jpg"
        full_path = os.path.join(day_dir, filename)

        async_write_image(full_path, frame)

        time_display = now_dt.strftime("%H:%M:%S")
        print(f"[ФІКСАЦІЯ ОБ'ЄКТА ({summary_label})] Збережено скріншот: {full_path}")

        if self.on_saved_callback:
            self.on_saved_callback(day_folder, time_display, summary_label, full_path)


# ------------------------------------------------------------------
#  Контракт незалежного модуля для ModuleRegistry
# ------------------------------------------------------------------
def _create_shared_resources(base_dir: str, providers: list, ai_threads: int, app_context: dict = None):
    model_path = os.path.join(base_dir, "models", "yolo11n.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO11n model file not found: {model_path}")
    det = PersonDetector(model_path, detect_vehicles=True, providers=providers, intra_op_threads=ai_threads)
    return {
        "detector": det,
        "providers": list(getattr(det, "active_providers", providers or ["CPUExecutionProvider"])),
    }


def _create_engine(base_dir, flat_cfg, output_dir, shared_resources=None, on_saved_callback=None, source=None, app_context=None):
    det = shared_resources.get("detector") if isinstance(shared_resources, dict) else None
    cfg_copy = _merge_runtime_write_cfg(flat_cfg, app_context)
    return PersonEngine(
        base_dir=base_dir,
        config=cfg_copy,
        on_saved_callback=on_saved_callback,
        detector=det,
        output_dir=output_dir,
    )


def _update_engine_config(engine, flat_cfg, source=None, shared_resources=None, app_context=None):
    if engine is None:
        return
    if isinstance(shared_resources, dict) and shared_resources.get("detector") is not None:
        engine.detector = shared_resources["detector"]
    cfg_copy = _merge_runtime_write_cfg(flat_cfg, app_context)
    engine.update_config(cfg_copy)


def _process_and_pack(engine, frame, prev_state=None):
    boxes = engine.process_frame(frame)
    return {"boxes": boxes}


def _process_batch_and_pack(items: list, shared_resources=None) -> list:
    """Батч-детекція людей і транспорту across multiple cameras + виставлення підказки person_hint."""
    n = len(items)
    if n == 0:
        return []
    out_states = [None] * n

    groups = {}
    for idx, it in enumerate(items):
        eng = it["engine"]
        fr = it["frame"]
        if eng is None or fr is None or getattr(fr, "size", 0) == 0:
            out_states[idx] = {"boxes": []}
            continue
        groups.setdefault(id(eng.detector), []).append(idx)

    for idxs in groups.values():
        detector = items[idxs[0]]["engine"].detector
        frames_g = []
        threshs_g = []
        veh_flags_g = []
        for idx in idxs:
            eng = items[idx]["engine"]
            frames_g.append(items[idx]["frame"])
            with eng._state_lock:
                threshs_g.append(eng.conf_thresh)
                veh_flags_g.append(eng.detect_vehicles)

        if hasattr(detector, "predict_batch"):
            dets_list, hints_list = detector.predict_batch(
                frames_g,
                conf_threshs=threshs_g,
                detect_vehicles_flags=veh_flags_g,
                imgsz=640,
                raw_person_hint_thresh=0.12,
            )
        else:
            dets_list = [
                detector.predict(fr, conf_thresh=th, detect_vehicles=vf)
                for fr, th, vf in zip(frames_g, threshs_g, veh_flags_g)
            ]
            hints_list = [True] * len(frames_g)

        for local_i, idx in enumerate(idxs):
            it = items[idx]
            eng = it["engine"]
            meta = it.get("meta")
            if isinstance(meta, dict):
                meta["person_hint"] = bool(hints_list[local_i])
            boxes = eng._handle_detections(frames_g[local_i], dets_list[local_i])
            out_states[idx] = {"boxes": boxes}

    return out_states


def _draw_overlays(frame, state_data: dict, mod_cfg: dict, now: float):
    if cv2 is None or frame is None or not (mod_cfg or {}).get("show_boxes", True):
        return
    for item in (state_data or {}).get("boxes") or []:
        if len(item) == 3:
            (px1, py1, px2, py2), p_pct, p_label = item
        else:
            (px1, py1, px2, py2), p_pct = item
            p_label = "Person"
        box_color = (0, 165, 255) if str(p_label).lower() == "person" else (0, 215, 255)
        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 1)
        cv2.putText(frame, f"{p_label} {p_pct}%", (px1, max(18, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)


def _parse_archive_record(fpath: str, base_name: str, root_dir: str, rel_dir: str):
    time_parts = base_name.split("_")[0].split("-")
    time_disp = ":".join(time_parts[:3]) if len(time_parts) >= 3 else "--:--:--"
    parts = base_name.split("_")
    default_person = tr("mod.people.archive.person_in_frame", "Person in frame")
    if len(parts) >= 3 and parts[1].isdigit():
        time_disp = f"{time_disp}.{parts[1]}"
        raw_label = "_".join(parts[2:])
        item_name = raw_label.replace("-", ", ") if raw_label and not raw_label.isdigit() else default_person
    elif len(parts) >= 2 and not parts[1].isdigit():
        raw_label = "_".join(parts[1:])
        item_name = raw_label.replace("-", ", ")
    else:
        item_name = default_person
    return {"time": time_disp, "name": item_name}


MODULE_SPEC = {
    "id": "people",
    "version": "2.0.0",
    "order": 30,
    "kind": "ai",
    "name": "People / Vehicles",
    "icon": "🚶",
    "short_badge": "PPL",
    "description": "YOLO11n detector for capturing people (and optionally vehicles) with periodic clean screenshots.",
    "on_help": "Enable YOLO11n detector for capturing people (and optionally vehicles) with periodic clean screenshots.",
    "provides": ["person_detector"],
    "library_deps": ["opencv-python", "onnxruntime"],
    "settings_schema": [
        {
            "key": "show_boxes",
            "type": "bool",
            "default": True,
            "label": "Show Boxes:",
            "bool_text": "Show boxes around people and vehicles",
            "help": "Show bounding boxes around people and vehicles in preview (screenshots saved to disk are always clean).",
        },
        {
            "key": "detect_transport",
            "type": "bool",
            "default": True,
            "label": "Detect Vehicles:",
            "bool_text": "Also detect cars, buses, trucks, motorcycles",
            "help": "Also detect and capture cars, buses, trucks, motorcycles, and bicycles.",
        },
        {
            "key": "det_thresh",
            "type": "float",
            "default": 0.45,
            "min": 0.15,
            "max": 0.95,
            "step": 0.02,
            "label": "YOLO11n Detector Threshold:",
            "help": "Confidence threshold for YOLO11n detector (0.30–0.75; 0.45 optimal).",
        },
        {
            "key": "timer_ms",
            "type": "int",
            "default": 1000,
            "min": 100,
            "max": 60000,
            "step": 100,
            "disk_write": True,
            "label": "Screenshot Interval (ms):",
            "help": "Minimum interval in milliseconds between saved screenshots while an object is in frame (1000 ms = max 1 photo/sec).",
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
            "help": "Automatically delete screenshots from this camera older than N days (0 = keep forever).",
        },
        {
            "key": "date_from",
            "key_to": "date_to",
            "type": "date_range",
            "default": "",
            "default_to": "",
            "label": "Active Schedule (Dates):",
            "help": "Start and end dates for the people module (YYYY-MM-DD, empty = unlimited).",
        },
    ],
    "archive": {
        "has_archive": True,
        "button_label": lambda mod_cfg: (
            tr("mod.people.archive.button_transport", "🚶🚗 People/Vehicles")
            if (mod_cfg or {}).get("detect_transport", True)
            else tr("mod.people.archive.button_people", "🚶 People")
        ),
        "sidebar_title": lambda mod_cfg: (
            tr("mod.people.archive.sidebar_transport", "History: People & Vehicles")
            if (mod_cfg or {}).get("detect_transport", True)
            else tr("mod.people.archive.sidebar_people", "History: People (screenshots)")
        ),
        "item_preview_label": "Screenshot:",
        "icon": "🚶",
        "record_type": "single_image",
        "extensions": (".jpg", ".jpeg", ".png"),
        "parse_record": _parse_archive_record,
    },
    "is_ready": is_ready,
    "refresh_capabilities": refresh_capabilities,
    "get_status_summary": get_status_summary,
    "ensure_resources": ensure_resources,
    "create_shared_resources": _create_shared_resources,
    "on_app_context_changed": lambda res, ctx, *a, **kw: res,
    "create_engine": _create_engine,
    "update_engine_config": _update_engine_config,
    "process_and_pack": _process_and_pack,
    "process_batch_and_pack": _process_batch_and_pack,
    "draw_overlays": _draw_overlays,
}
