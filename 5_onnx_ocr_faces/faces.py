import os
import time
import math
import threading
from datetime import datetime
import cv2
import numpy as np
import onnxruntime as ort

REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Кончик носа
    [-30.0, -30.0, -30.0],    # Левый глаз
    [30.0, -30.0, -30.0],     # Правый глаз
    [-20.0, 40.0, -20.0],     # Левый угол рта
    [20.0, 40.0, -20.0]       # Правый угол рта
], dtype=np.float64)


def async_write_image(path, img):
    def _worker():
        try:
            cv2.imwrite(path, img)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def calculate_sharpness(img):
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def align_face(img, landmarks, box=None):
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
    if landmarks is None or len(landmarks) != 5:
        return None, None, None

    pts_2d = np.array([
        landmarks[2],
        landmarks[0],
        landmarks[1],
        landmarks[3],
        landmarks[4]
    ], dtype=np.float64)

    focal_length = float(frame_w)
    center = (float(frame_w) / 2.0, float(frame_h) / 2.0)
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


def is_valid_frontal_face(landmarks, box, min_size=45, max_yaw=32.0, max_pitch=28.0):
    if landmarks is None or len(landmarks) != 5:
        return False

    bx1, by1, bx2, by2 = box
    bw, bh = bx2 - bx1, by2 - by1
    if bw < min_size or bh < min_size:
        return False

    yaw, pitch, _ = estimate_head_pose_pnp(landmarks)
    if yaw is None:
        return False

    if abs(yaw) > max_yaw or abs(pitch) > max_pitch:
        return False

    left_eye, right_eye = landmarks[0], landmarks[1]
    eye_dist = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    if eye_dist < 18.0:
        return False

    return True


class SCRFDDetector:
    def __init__(self, model_path, input_size=(640, 640)):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
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


class FaceRecognizer:
    def __init__(self, model_path, db_file_path):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.db_path = db_file_path
        self.known_templates = []
        self._load_database()

    def _load_database(self):
        """
        Завантажує базу faces.txt.
        Формат: UniqueFaceID,FaceName,Ignore,Vector
        (Підтримує також старий формат з 3 колонок, ставлячи ignore=False)
        """
        self.known_templates.clear()
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                f.write("# UniqueFaceID,FaceName,Ignore,Vector\n")
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) >= 4:
                    u_id = parts[0].strip()
                    name = parts[1].strip()
                    ign_str = parts[2].strip().lower()
                    ignore_flag = ign_str in ("true", "1", "yes")
                    vec_str = parts[3].strip()
                elif len(parts) == 3:
                    u_id = parts[0].strip()
                    name = parts[1].strip()
                    ignore_flag = False  # За замовчуванням false
                    vec_str = parts[2].strip()
                else:
                    continue

                vec = np.fromstring(vec_str, sep=" ", dtype=np.float32)
                norm = np.linalg.norm(vec)
                self.known_templates.append({
                    "id": u_id,
                    "name": name,
                    "ignore": ignore_flag,
                    "vector": vec / (norm + 1e-10)
                })

    def _save_all_database(self):
        """Повний перезапис файлу бази при оновленні вектора на якісніший."""
        with open(self.db_path, "w", encoding="utf-8") as f:
            f.write("# UniqueFaceID,FaceName,Ignore,Vector\n")
            for r in self.known_templates:
                ign_str = "true" if r.get("ignore", False) else "false"
                vec_str = " ".join(f"{x:.6f}" for x in r["vector"])
                f.write(f"{r['id']},{r['name']},{ign_str},{vec_str}\n")

    def compute_embedding(self, face_112_bgr):
        rgb = cv2.cvtColor(face_112_bgr, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) - 127.5) / 128.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        outs = self.session.run(None, {self.input_name: tensor})
        emb = outs[0][0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-10)

    def identify_or_register(self, fused_embedding, current_sharpness=0.0, min_similarity=0.50):
        """
        Порівнює з базою:
        - Якщо особа впізнана: перевіряє умову перезапису на чіткіший вектор (In-place).
        - Якщо особа нова: створює рівно один запис із Ignore=False.
        Повертає: (UniqueFaceID, FaceName, status_tag, is_ignored: bool)
        """
        norm_emb = fused_embedding / (np.linalg.norm(fused_embedding) + 1e-10)

        best_idx = -1
        best_sim = -1.0

        for idx, record in enumerate(self.known_templates):
            sim = float(np.dot(norm_emb, record["vector"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        # 1. Особа впізнана
        if best_sim >= min_similarity and best_idx != -1:
            rec = self.known_templates[best_idx]
            sim_pct_val = int(round(best_sim * 100))
            sim_tag = f"{sim_pct_val}%"

            # Самовдосконалення: якщо схожість висока (>= 82%) і чіткість відмінна (>= 45.0)
            if best_sim >= 0.82 and current_sharpness >= 45.0:
                rec["vector"] = norm_emb
                self._save_all_database()
                print(f"[FaceEngine] Оновлено еталонний вектор для {rec['name']} ({rec['id']}) на чіткіший (різкість {current_sharpness:.1f})")

            return rec["id"], rec["name"], sim_tag, rec.get("ignore", False)

        # 2. Нова особа (за замовчуванням Ignore = False)
        unique_ids = {r["id"] for r in self.known_templates}
        new_id = f"ID_{len(unique_ids) + 1:04d}"
        new_name = new_id
        ignore_flag = False

        new_record = {
            "id": new_id,
            "name": new_name,
            "ignore": ignore_flag,
            "vector": norm_emb
        }
        self.known_templates.append(new_record)

        vec_str = " ".join(f"{x:.6f}" for x in norm_emb)
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(f"{new_id},{new_name},false,{vec_str}\n")

        print(f"[FaceEngine] Зареєстровано нове обличчя: {new_id} [IGNORE=FALSE]")
        return new_id, new_name, "NEW", False


class FaceTrack:
    def __init__(self, track_id, bbox, face_crop, frame, sharpness, max_pool_size=5):
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
        self.frames_tracked = 1

        self.face_name = "Особа"
        self.status_tag = ""
        self.is_ignored = False

    def update(self, bbox, face_crop, frame, sharpness):
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_tracked += 1

        if sharpness > self.best_sharpness:
            self.best_sharpness = sharpness
            self.best_face_img = face_crop
            self.best_frame = frame

        if len(self.face_pool) < self.max_pool_size:
            self.face_pool.append((face_crop, sharpness))
            self.face_pool.sort(key=lambda x: x[1], reverse=True)
        else:
            if sharpness > self.face_pool[-1][1]:
                self.face_pool[-1] = (face_crop, sharpness)
                self.face_pool.sort(key=lambda x: x[1], reverse=True)


class FaceEngine:
    def __init__(self, base_dir, config, on_saved_callback=None):
        self.base_dir = base_dir
        self.config = config
        self.on_saved_callback = on_saved_callback

        self.face_min_similarity = float(self.config.get("face_min_similarity", 0.50))
        self.face_det_thresh = float(self.config.get("face_det_thresh", 0.40))

        self.faces_dir = os.path.join(base_dir, "faces")
        os.makedirs(self.faces_dir, exist_ok=True)
        self.db_file = os.path.join(self.faces_dir, "faces.txt")

        candidates_det = [
            "scrfd_face_detector.onnx", "det_500m.onnx", "scrfd_2.5g_kps.onnx", "scrfd_500m_kps.onnx"
        ]
        det_path = None
        for c in candidates_det:
            p = os.path.join(base_dir, "model", c)
            if os.path.exists(p):
                det_path = p
                break
        if not det_path:
            raise FileNotFoundError("Детектор облич SCRFD не знайдено в папці model/")

        candidates_rec = [
            "face_recognizer.onnx", "w600k_r50.onnx", "w600k_mbf.onnx", "adaface_mobilefacenet.onnx", "adaface_ir50.onnx"
        ]
        rec_path = None
        for c in candidates_rec:
            p = os.path.join(base_dir, "model", c)
            if os.path.exists(p):
                rec_path = p
                break
        if not rec_path:
            raise FileNotFoundError("Модель розпізнавання облич не знайдено в папці model/")

        self.detector = SCRFDDetector(det_path)
        self.recognizer = FaceRecognizer(rec_path, self.db_file)

        self.active_tracks = {}
        self.next_track_id = 1
        self.track_timeout = 1.2
        self.recently_saved = {}

    def process_frame(self, frame):
        face_detections = self.detector.detect(frame, conf_thresh=self.face_det_thresh)
        current_faces = []

        for f_info in face_detections:
            if not is_valid_frontal_face(f_info["landmarks"], f_info["box"], min_size=45, max_yaw=32.0, max_pitch=28.0):
                continue

            aligned = align_face(frame, f_info["landmarks"], box=f_info["box"])
            f_sharpness = calculate_sharpness(aligned)

            current_faces.append({
                "box": f_info["box"],
                "crop": aligned,
                "sharpness": f_sharpness
            })

        now = time.time()
        unmatched_faces = list(range(len(current_faces)))

        for f_id, f_track in list(self.active_tracks.items()):
            best_idx = -1
            min_dist = float("inf")
            for idx in unmatched_faces:
                det = current_faces[idx]
                c1x, c1y = (f_track.bbox[0] + f_track.bbox[2]) / 2.0, (f_track.bbox[1] + f_track.bbox[3]) / 2.0
                c2x, c2y = (det["box"][0] + det["box"][2]) / 2.0, (det["box"][1] + det["box"][3]) / 2.0
                dist = math.hypot(c1x - c2x, c1y - c2y)
                if dist < 140.0 and dist < min_dist:
                    min_dist = dist
                    best_idx = idx

            if best_idx != -1:
                det = current_faces[best_idx]
                f_track.update(det["box"], det["crop"], frame.copy(), det["sharpness"])
                unmatched_faces.remove(best_idx)

                # Пікове збереження: людина впевнено в кадрі і різкість висока
                if not f_track.saved and f_track.best_sharpness >= 40.0 and f_track.frames_tracked >= 6:
                    self._save_verified_face(f_track)
                    f_track.saved = True

        for idx in unmatched_faces:
            det = current_faces[idx]
            f_tid = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[f_tid] = FaceTrack(
                f_tid, det["box"], det["crop"], frame.copy(), det["sharpness"]
            )

        for f_id, f_track in list(self.active_tracks.items()):
            if (now - f_track.last_seen) > self.track_timeout:
                if not f_track.saved:
                    self._save_verified_face(f_track)
                del self.active_tracks[f_id]

        display_boxes = [(t.bbox, t.face_name, t.status_tag) for t in self.active_tracks.values()]
        return display_boxes

    def _save_verified_face(self, track: FaceTrack):
        """Multi-Shot Fusion + оновлення еталона та збереження на диск."""
        if track.best_face_img is None or track.best_sharpness < 25.0:
            return

        # 1. Multi-Shot Fusion
        embeddings = []
        for face_crop, sh in track.face_pool:
            if sh >= 12.0:
                embeddings.append(self.recognizer.compute_embedding(face_crop))

        if not embeddings:
            embeddings.append(self.recognizer.compute_embedding(track.best_face_img))

        mean_vector = np.mean(embeddings, axis=0)
        fused_vector = mean_vector / (np.linalg.norm(mean_vector) + 1e-10)

        # 2. Звірка/Реєстрація/Оновлення еталона
        u_id, f_name, status_tag, is_ignored = self.recognizer.identify_or_register(
            fused_vector,
            current_sharpness=track.best_sharpness,
            min_similarity=self.face_min_similarity
        )
        track.face_name = f_name
        track.status_tag = status_tag
        track.is_ignored = is_ignored
        track.saved = True

        # Якщо особа позначена як Ignore (true) — скріншот на диск не зберігається
        if is_ignored:
            return

        now = time.time()
        sleep_face_timeout = float(self.config.get("sleep_time_after_save", 5.0)) * 2.0
        if (now - self.recently_saved.get(f_name, 0.0)) < sleep_face_timeout:
            return
        self.recently_saved[f_name] = now

        # 3. Збереження фотографій на диск (для тих, у кого ignore == false)
        now_dt = datetime.now()
        day_folder = now_dt.strftime("%Y-%m-%d")
        time_prefix = now_dt.strftime("%H-%M-%S")
        day_dir = os.path.join(self.faces_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        face_filename = f"{time_prefix}_face_{f_name}_{status_tag}.jpg"
        full_filename = f"{time_prefix}_full.jpg"
        face_path = os.path.join(day_dir, face_filename)
        full_path = os.path.join(day_dir, full_filename)

        async_write_image(face_path, track.best_face_img)
        async_write_image(full_path, track.best_frame)

        print(f"[ФІКСАЦІЯ ОБЛИЧЧЯ] {f_name} [{status_tag}], різкість: {track.best_sharpness:.1f}")

        if self.on_saved_callback:
            time_display = now_dt.strftime("%H:%M:%S")
            self.on_saved_callback(day_folder, time_display, f_name, face_path)
