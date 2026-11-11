import os
import time
import threading
import numpy as np

from scripts.core.module_registry import is_date_range_active

try:
    import cv2
except ImportError:
    cv2 = None

STREAM_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://")


def parse_source_value(raw_source):
    s = str(raw_source or "").strip().strip('\'"')
    if s.isdigit():
        return int(s)
    return s


def is_network_stream_source(source) -> bool:
    return isinstance(source, str) and source.strip().lower().startswith(STREAM_PREFIXES)


def is_local_device_or_file(source) -> bool:
    return isinstance(source, int) or (bool(source) and not is_network_stream_source(source))


def open_video_capture(source, decode_mode: str = "cpu"):
    """Відкриває VideoCapture із підтримкою тайм-аутів та опційного апаратного декодування."""
    if cv2 is None:
        return None, "cpu"
    mode = str(decode_mode or "cpu").strip().lower()
    is_net = is_network_stream_source(source)

    if mode != "cpu" and not isinstance(source, int) and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
        accel_map = {
            "auto": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
            "d3d11va": getattr(cv2, "VIDEO_ACCELERATION_D3D11", 3),
            "cuda": getattr(cv2, "VIDEO_ACCELERATION_ANY", 1),
            "qsv": getattr(cv2, "VIDEO_ACCELERATION_MFX", 4),
        }
        accel_val = accel_map.get(mode, getattr(cv2, "VIDEO_ACCELERATION_ANY", 1))
        try:
            params = [cv2.CAP_PROP_HW_ACCELERATION, accel_val]
            if is_net and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000])
            if is_net and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            backend = cv2.CAP_FFMPEG if hasattr(cv2, "CAP_FFMPEG") else cv2.CAP_ANY
            cap = cv2.VideoCapture(source, backend, params)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap, mode
            if cap is not None:
                cap.release()
        except Exception:
            pass

    try:
        if is_net and hasattr(cv2, "CAP_FFMPEG") and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000]
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        else:
            cap = cv2.VideoCapture(source)
        if cap is not None and cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap, "cpu"
        if cap is not None:
            cap.release()
    except Exception:
        pass
    return None, "cpu"


def probe_stream_info(raw_source, decode_mode: str = "cpu"):
    """Швидкий тест підключення потоку для кнопки 'Тест' у налаштуваннях камери."""
    if cv2 is None:
        return False, "Бібліотеку OpenCV (cv2) не встановлено"
    src = parse_source_value(raw_source)
    if src == "":
        return False, "Джерело порожнє"
    t0 = time.time()
    cap, used_mode = open_video_capture(src, decode_mode=decode_mode)
    if cap is None or not cap.isOpened():
        return False, f"Не вдалося відкрити джерело ({raw_source})"
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, "Потік відкрився, але кадр не отримано"
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0 or fps > 120 or np.isnan(fps):
            fps_str = "авто"
        else:
            fps_str = f"{fps:.1f} FPS"
        ms = int((time.time() - t0) * 1000)
        return True, f"OK: {w}×{h}, {fps_str}, декодер={used_mode.upper()} ({ms} мс)"
    except Exception as e:
        return False, f"Помилка читання: {e}"
    finally:
        try:
            cap.release()
        except Exception:
            pass


def _resize_to_rgb(bgr_img: np.ndarray, target_w: int, target_h: int, interp_mode: str = "linear"):
    if cv2 is None or bgr_img is None or bgr_img.size == 0:
        return None
    oh, ow = bgr_img.shape[:2]
    tw = max(16, int(target_w))
    th = max(16, int(target_h))
    scale = min(tw / float(ow), th / float(oh))
    dw = max(1, int(round(ow * scale)))
    dh = max(1, int(round(oh * scale)))

    if interp_mode == "nearest":
        interp = cv2.INTER_NEAREST
    elif interp_mode == "lanczos":
        interp = cv2.INTER_LANCZOS4
    else:
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    resized = cv2.resize(bgr_img, (dw, dh), interpolation=interp)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class CameraUnit:
    """
    Автономний блок однієї камери: sub-потік для сітки, main-потік для запису/AI/детального вікна.
    Усі AI та recorder двигуни підключаються динамічно через ModuleRegistry.
    """
    def __init__(
        self,
        cam_cfg: dict,
        archive_dir: str,
        inference_pool,
        registry,
        hardware_cfg: dict,
        on_saved_event=None,
    ):
        self.cam_cfg = dict(cam_cfg)
        self.uuid = str(cam_cfg["uuid"])
        self.archive_dir = archive_dir
        self.inference_pool = inference_pool
        self.registry = registry
        self.hardware_cfg = dict(hardware_cfg or {})
        self.on_saved_event = on_saved_event

        os.makedirs(self.archive_dir, exist_ok=True)
        self._write_camera_meta()

        self.lock = threading.Lock()
        self.running = False
        self.detail_subscribers = 0

        self._latest_main_bgr = None
        self._latest_main_seq = 0
        self._latest_main_ts = 0.0

        self._latest_grid_rgb = None
        self._latest_grid_seq = 0
        self._grid_target_size = (360, 210)
        self._grid_last_push_ts = 0.0

        # Динамічний стан оверлеїв: mod_id -> state_dict
        self._overlay_lock = threading.Lock()
        self._overlay_state = {}

        self._sub_status = "disabled"
        self._main_status = "standby"
        self._sub_fps = 0.0
        self._main_fps = 0.0

        self._pending_ai_frame = None
        self._pending_ai_ts = 0.0
        self._ai_busy = False
        self._last_ai_submit_ts = 0.0

        # Динамічні двигуни модулів: mod_id -> engine instance
        self._engines_lock = threading.Lock()
        self.engines = {}
        self._engine_sigs = {}
        self._engine_specs = {}  # mod_id -> spec (для безпечної зупинки видалених модулів)
        self._module_runtime_errors = {}

        self._sub_thread = None
        self._main_thread = None

        self._init_engines()

    def _write_camera_meta(self):
        try:
            meta_path = os.path.join(self.archive_dir, "camera_info.txt")
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"UUID: {self.uuid}\n")
                f.write(f"Name: {self.cam_cfg.get('name', '')}\n")
                f.write(f"Main: {self.cam_cfg.get('main_source', '')}\n")
                f.write(f"Sub: {self.cam_cfg.get('sub_source', '')}\n")
        except Exception:
            pass

    @staticmethod
    def _flatten_cfg(cam_cfg: dict) -> dict:
        mods = (cam_cfg or {}).get("modules") or {}
        flat = {}
        for mod_name, sec in mods.items():
            if isinstance(sec, dict):
                for k, v in sec.items():
                    flat[f"{mod_name}_{k}"] = v
        return flat

    def _make_saved_cb(self, mod_name: str):
        def _cb(day_str, time_str, label, fpath):
            if self.on_saved_event:
                self.on_saved_event(self.uuid, mod_name, day_str, time_str, label, fpath)
        return _cb

    def _teardown_engine_locked(self, mid: str):
        eng = self.engines.pop(mid, None)
        spec = self._engine_specs.pop(mid, None)
        self._engine_sigs.pop(mid, None)
        if eng is None:
            return
        if spec:
            stop_fn = spec.get("stop_engine")
            if callable(stop_fn):
                try:
                    stop_fn(eng, wait=True)
                except Exception:
                    pass
            flush_fn = spec.get("flush_tracks")
            if callable(flush_fn):
                try:
                    flush_fn(eng)
                except Exception:
                    pass
        with self._overlay_lock:
            self._overlay_state.pop(mid, None)

    def _init_engines(self, force_reload_ids=None):
        force_set = set(force_reload_ids or [])
        flat_cfg = self._flatten_cfg(self.cam_cfg)
        mods = self.cam_cfg.get("modules") or {}
        main_src = parse_source_value(self.cam_cfg.get("main_source", ""))
        entries = self.registry.list_modules()
        active_ids = {e["id"] for e in entries}

        with self._engines_lock:
            # 1. Прибираємо двигуни модулів, яких більше немає в реєстрі або які вимкнені
            for old_mid in list(self.engines.keys()):
                if old_mid not in active_ids:
                    self._teardown_engine_locked(old_mid)

            # 2. Ініціалізуємо / оновлюємо активні модулі
            for ent in entries:
                mid = ent["id"]
                spec = ent["spec"]
                m_cfg = mods.get(mid) or {}
                default_on = bool(spec.get("default_enabled", True))
                is_on = bool(m_cfg.get("on", default_on))

                if not is_on:
                    if mid in self.engines:
                        self._teardown_engine_locked(mid)
                    self._module_runtime_errors.pop(mid, None)
                    continue

                sig_now = ent.get("sig")
                must_recreate = (
                    (mid not in self.engines)
                    or (mid in force_set)
                    or (self._engine_sigs.get(mid) != sig_now)
                )

                try:
                    shared_res = self.inference_pool.get_module_resources(mid)
                    mod_out_dir = os.path.join(self.archive_dir, mid)
                    if not must_recreate:
                        upd_fn = spec.get("update_engine_config")
                        if callable(upd_fn):
                            upd_fn(
                                self.engines[mid],
                                flat_cfg,
                                source=main_src,
                                shared_resources=shared_res,
                                app_context=self.inference_pool.app_context,
                            )
                        self._engine_specs[mid] = spec
                    else:
                        if mid in self.engines:
                            self._teardown_engine_locked(mid)
                        create_fn = spec.get("create_engine")
                        if callable(create_fn):
                            eng = create_fn(
                                self.inference_pool.base_dir,
                                flat_cfg,
                                mod_out_dir,
                                shared_resources=shared_res,
                                on_saved_callback=self._make_saved_cb(mid),
                                source=main_src,
                                app_context=self.inference_pool.app_context,
                            )
                            self.engines[mid] = eng
                            self._engine_sigs[mid] = sig_now
                            self._engine_specs[mid] = spec
                    self._module_runtime_errors.pop(mid, None)
                except Exception as e:
                    self._module_runtime_errors[mid] = str(e)
                    print(f"[Module Engine Error '{mid}' @ {self.cam_cfg.get('name', self.uuid)}]: {e}")

    def reload_models_from_pool(self):
        """Оновлює посилання на моделі після зміни CPU/GPU бекенду в InferencePool."""
        self._init_engines(force_reload_ids=None)

    def force_reload_modules(self, mod_ids=None):
        """Перестворює двигуни оновлених/нових модулів і запускає нові recorder-и, якщо камера активна."""
        self._init_engines(force_reload_ids=mod_ids)
        if self.running:
            self._sync_recorders_state()

    def _sync_recorders_state(self):
        mods = self.cam_cfg.get("modules") or {}
        for ent in self.registry.list_modules(kind="recorder"):
            mid = ent["id"]
            spec = ent["spec"]
            with self._engines_lock:
                eng = self.engines.get(mid)
            if eng is None:
                continue
            m_cfg = mods.get(mid) or {}
            allowed = bool(m_cfg.get("on", spec.get("default_enabled", True))) and is_date_range_active(
                m_cfg.get("date_from", ""), m_cfg.get("date_to", "")
            )
            set_allowed = spec.get("set_allowed")
            if callable(set_allowed):
                try:
                    set_allowed(eng, allowed)
                except Exception:
                    pass
            if allowed:
                st_fn = spec.get("start_engine")
                if callable(st_fn):
                    try:
                        st_fn(eng)
                    except Exception:
                        pass
                rst_fn = spec.get("restart_if_mode_changed")
                if callable(rst_fn):
                    try:
                        rst_fn(eng)
                    except Exception:
                        pass

    def update_config(self, cam_cfg: dict, hardware_cfg: dict):
        old_main = parse_source_value(self.cam_cfg.get("main_source", ""))
        old_sub = parse_source_value(self.cam_cfg.get("sub_source", ""))
        old_enabled = bool(self.cam_cfg.get("enabled", True))
        old_hw_dec = str(self.hardware_cfg.get("video_decode_mode", "cpu"))

        self.cam_cfg = dict(cam_cfg)
        self.hardware_cfg = dict(hardware_cfg or {})
        self._write_camera_meta()

        new_main = parse_source_value(self.cam_cfg.get("main_source", ""))
        new_sub = parse_source_value(self.cam_cfg.get("sub_source", ""))
        new_enabled = bool(self.cam_cfg.get("enabled", True))
        new_hw_dec = str(self.hardware_cfg.get("video_decode_mode", "cpu"))

        sources_changed = (old_main != new_main) or (old_sub != new_sub) or (old_enabled != new_enabled) or (old_hw_dec != new_hw_dec)
        if sources_changed and self.running:
            self.stop()
            self._init_engines()
            if new_enabled:
                self.start()
        else:
            self._init_engines()
            if self.running:
                self._sync_recorders_state()

    def start(self):
        if self.running or not self.cam_cfg.get("enabled", True):
            return
        self.running = True

        main_src = parse_source_value(self.cam_cfg.get("main_source", ""))
        sub_src = parse_source_value(self.cam_cfg.get("sub_source", ""))

        has_dedicated_sub = bool(sub_src != "" and not (is_local_device_or_file(main_src) and sub_src == main_src))

        if has_dedicated_sub:
            self._sub_status = "connecting"
            self._sub_thread = threading.Thread(target=self._sub_capture_loop, args=(sub_src,), daemon=True)
            self._sub_thread.start()
        else:
            self._sub_status = "fallback_main"

        self._sync_recorders_state()

        self._main_status = "standby"
        self._main_thread = threading.Thread(
            target=self._main_capture_loop,
            args=(main_src, has_dedicated_sub),
            daemon=True,
        )
        self._main_thread.start()

    def stop(self):
        self.running = False
        with self._engines_lock:
            rec_items = [
                (mid, self.engines.get(mid), self._engine_specs.get(mid) or self.registry.get_spec(mid))
                for mid in list(self.engines.keys())
            ]
        for _mid, eng, spec in rec_items:
            if eng is not None and spec and spec.get("kind") == "recorder":
                stop_fn = spec.get("stop_engine")
                if callable(stop_fn):
                    try:
                        stop_fn(eng, wait=False)
                    except Exception:
                        pass

        if self._sub_thread is not None and self._sub_thread.is_alive():
            self._sub_thread.join(timeout=2.0)
        if self._main_thread is not None and self._main_thread.is_alive():
            self._main_thread.join(timeout=2.0)

        with self._engines_lock:
            all_items = [
                (mid, self.engines.get(mid), self._engine_specs.get(mid) or self.registry.get_spec(mid))
                for mid in list(self.engines.keys())
            ]
        for _mid, eng, spec in all_items:
            if eng is None or not spec:
                continue
            flush_fn = spec.get("flush_tracks")
            if callable(flush_fn):
                try:
                    flush_fn(eng)
                except Exception:
                    pass
            if spec.get("kind") == "recorder":
                stop_fn = spec.get("stop_engine")
                if callable(stop_fn):
                    try:
                        stop_fn(eng, wait=True)
                    except Exception:
                        pass

        with self.lock:
            self._sub_status = "disabled"
            self._main_status = "disabled"
            self._sub_fps = 0.0
            self._main_fps = 0.0

    def add_detail_subscriber(self):
        with self.lock:
            self.detail_subscribers += 1

    def remove_detail_subscriber(self):
        with self.lock:
            self.detail_subscribers = max(0, self.detail_subscribers - 1)

    def set_grid_target_size(self, w: int, h: int):
        with self.lock:
            self._grid_target_size = (max(64, int(w)), max(48, int(h)))

    def _get_recorder_status(self):
        rec_on = False
        rec_enc = ""
        for ent in self.registry.list_modules(kind="recorder"):
            mid = ent["id"]
            spec = ent["spec"]
            with self._engines_lock:
                eng = self.engines.get(mid)
            if eng is None:
                continue
            is_rec_fn = spec.get("is_recording")
            if callable(is_rec_fn):
                try:
                    if is_rec_fn(eng):
                        rec_on = True
                        enc_fn = spec.get("get_active_encoder_label")
                        if callable(enc_fn):
                            rec_enc = str(enc_fn(eng) or "")
                        break
                except Exception:
                    pass
        return rec_on, rec_enc

    def is_archive_file_busy(self, mod_id: str, fpath: str) -> bool:
        with self._engines_lock:
            eng = self.engines.get(mod_id)
            spec = self._engine_specs.get(mod_id) or self.registry.get_spec(mod_id)
        if eng is not None and spec:
            fn = spec.get("is_file_busy")
            if callable(fn):
                try:
                    return bool(fn(eng, fpath))
                except Exception:
                    pass
        return False

    def get_grid_snapshot(self):
        """Повертає готовий зменшений RGB-кадр для сітки та динамічні метадані статусу."""
        with self.lock:
            rgb = self._latest_grid_rgb
            seq = self._latest_grid_seq
            sub_st = self._sub_status
            main_st = self._main_status
            sub_fps = self._sub_fps
            subs = self.detail_subscribers
        mods = self.cam_cfg.get("modules") or {}
        modules_on = {
            ent["id"]: bool((mods.get(ent["id"]) or {}).get("on", False))
            for ent in self.registry.list_modules()
        }
        rec_on, rec_enc = self._get_recorder_status()
        return {
            "uuid": self.uuid,
            "name": self.cam_cfg.get("name", self.uuid),
            "enabled": bool(self.cam_cfg.get("enabled", True)),
            "rgb": rgb,
            "seq": seq,
            "sub_status": sub_st,
            "main_status": main_st,
            "fps": sub_fps,
            "recording": rec_on,
            "encoder": rec_enc,
            "detail_open": subs > 0,
            "modules_on": modules_on,
        }

    def get_detail_snapshot(self):
        """Повертає свіжий кадр основного потоку + динамічний кеш AI-оверлеїв усіх модулів."""
        with self.lock:
            frame = self._latest_main_bgr
            seq = self._latest_main_seq
            main_st = self._main_status
            main_fps = self._main_fps
        with self._overlay_lock:
            overlay = {mid: dict(st) for mid, st in self._overlay_state.items() if isinstance(st, dict)}
        rec_on, rec_enc = self._get_recorder_status()
        return {
            "frame": frame,
            "seq": seq,
            "main_status": main_st,
            "fps": main_fps,
            "recording": rec_on,
            "encoder": rec_enc,
            "overlay": overlay,
        }

    def _sleep_interruptible(self, seconds: float):
        end_ts = time.time() + seconds
        while self.running and time.time() < end_ts:
            time.sleep(min(0.2, max(0.01, end_ts - time.time())))

    def _needs_main_decode(self, has_dedicated_sub: bool, main_src) -> bool:
        mods = self.cam_cfg.get("modules") or {}
        if not has_dedicated_sub:
            return True
        with self.lock:
            if self.detail_subscribers > 0:
                return True

        for ent in self.registry.list_modules(kind="ai"):
            mc = mods.get(ent["id"]) or {}
            if mc.get("on", False) and is_date_range_active(mc.get("date_from", ""), mc.get("date_to", "")):
                return True

        for ent in self.registry.list_modules(kind="recorder"):
            mid = ent["id"]
            spec = ent["spec"]
            mc = mods.get(mid) or {}
            if mc.get("on", False) and is_date_range_active(mc.get("date_from", ""), mc.get("date_to", "")):
                with self._engines_lock:
                    eng = self.engines.get(mid)
                if eng is not None:
                    need_fn = spec.get("needs_decoded_frames")
                    if not callable(need_fn) or need_fn(eng, mc, main_src):
                        return True
        return False

    def _any_ai_active_now(self) -> bool:
        mods = self.cam_cfg.get("modules") or {}
        for ent in self.registry.list_modules(kind="ai"):
            mc = mods.get(ent["id"]) or {}
            if mc.get("on", False) and is_date_range_active(mc.get("date_from", ""), mc.get("date_to", "")):
                return True
        return False

    def _push_to_grid_cache(self, bgr_frame: np.ndarray, force: bool = False):
        grid_fps = max(2.0, min(30.0, float(self.hardware_cfg.get("display_grid_fps", 12.0))))
        now = time.time()
        if not force and (now - self._grid_last_push_ts) < (1.0 / grid_fps):
            return
        self._grid_last_push_ts = now
        with self.lock:
            tw, th = self._grid_target_size
        interp = str(self.hardware_cfg.get("display_resize_method", "linear"))
        rgb = _resize_to_rgb(bgr_frame, tw, th, interp_mode=interp)
        if rgb is not None:
            with self.lock:
                self._latest_grid_rgb = rgb
                self._latest_grid_seq += 1

    def _sub_capture_loop(self, sub_src):
        if cv2 is None:
            with self.lock:
                self._sub_status = "offline"
            return
        is_stream = is_network_stream_source(sub_src) or isinstance(sub_src, int)
        is_file = not is_stream
        cap = None
        consec_fails = 0
        first_fail_ts = 0.0
        frame_delay = 1.0 / 15.0
        fps_cnt = 0
        fps_t0 = time.time()

        try:
            while self.running:
                if cap is None or not cap.isOpened():
                    dec_mode = str(self.hardware_cfg.get("video_decode_mode", "cpu"))
                    with self.lock:
                        self._sub_status = "connecting"
                    cap, _ = open_video_capture(sub_src, decode_mode=dec_mode)
                    if cap is None or not cap.isOpened():
                        with self.lock:
                            self._sub_status = "offline"
                            self._sub_fps = 0.0
                        self._sleep_interruptible(2.0)
                        continue

                    tfps = cap.get(cv2.CAP_PROP_FPS) or 15.0
                    if tfps <= 0 or tfps > 120 or np.isnan(tfps):
                        tfps = 15.0
                    frame_delay = 1.0 / tfps
                    consec_fails = 0
                    with self.lock:
                        self._sub_status = "active"

                t0 = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    consec_fails += 1
                    if is_file:
                        if consec_fails >= 10:
                            cap.release()
                            cap = None
                            consec_fails = 0
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.02)
                        continue
                    if consec_fails == 1:
                        first_fail_ts = t0
                    if consec_fails >= 15 or (t0 - first_fail_ts) >= 2.0:
                        cap.release()
                        cap = None
                        consec_fails = 0
                        with self.lock:
                            self._sub_status = "reconnecting"
                            self._sub_fps = 0.0
                        self._sleep_interruptible(2.0)
                    else:
                        time.sleep(0.04)
                    continue

                consec_fails = 0
                fps_cnt += 1
                if (t0 - fps_t0) >= 1.0:
                    with self.lock:
                        self._sub_fps = fps_cnt / max(0.001, t0 - fps_t0)
                        self._sub_status = "active"
                    fps_cnt = 0
                    fps_t0 = t0

                self._push_to_grid_cache(frame)

                if is_file:
                    sleep_d = frame_delay - (time.time() - t0)
                    if sleep_d > 0:
                        time.sleep(sleep_d)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _main_capture_loop(self, main_src, has_dedicated_sub: bool):
        is_stream = is_network_stream_source(main_src) or isinstance(main_src, int)
        is_file = not is_stream
        cap = None
        consec_fails = 0
        first_fail_ts = 0.0
        frame_delay = 1.0 / 30.0
        fps_cnt = 0
        fps_t0 = time.time()

        try:
            while self.running:
                mods = self.cam_cfg.get("modules") or {}
                active_rec_feed = []

                for ent in self.registry.list_modules(kind="recorder"):
                    mid = ent["id"]
                    spec = ent["spec"]
                    with self._engines_lock:
                        eng = self.engines.get(mid)
                    if eng is None:
                        continue
                    mc = mods.get(mid) or {}
                    allowed = bool(mc.get("on", False)) and is_date_range_active(
                        mc.get("date_from", ""), mc.get("date_to", "")
                    )
                    set_fn = spec.get("set_allowed")
                    if callable(set_fn):
                        try:
                            set_fn(eng, allowed)
                        except Exception:
                            pass
                    if allowed:
                        if not getattr(eng, "is_running", True):
                            st_fn = spec.get("start_engine")
                            if callable(st_fn):
                                try:
                                    st_fn(eng)
                                except Exception:
                                    pass
                        need_fn = spec.get("needs_decoded_frames")
                        if not callable(need_fn) or need_fn(eng, mc, main_src):
                            push_fn = spec.get("push_frame")
                            if callable(push_fn):
                                active_rec_feed.append((eng, push_fn))

                need_decode = self._needs_main_decode(has_dedicated_sub, main_src)
                if not need_decode or cv2 is None:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                    with self.lock:
                        self._main_status = "offline" if cv2 is None else "standby"
                        self._main_fps = 0.0
                    time.sleep(0.25)
                    continue

                if cap is None or not cap.isOpened():
                    dec_mode = str(self.hardware_cfg.get("video_decode_mode", "cpu"))
                    with self.lock:
                        self._main_status = "connecting"
                    cap, _ = open_video_capture(main_src, decode_mode=dec_mode)
                    if cap is None or not cap.isOpened():
                        with self.lock:
                            self._main_status = "offline"
                            self._main_fps = 0.0
                        if not has_dedicated_sub:
                            with self.lock:
                                self._sub_status = "offline"
                        self._sleep_interruptible(2.0)
                        continue

                    tfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    if tfps <= 0 or tfps > 120 or np.isnan(tfps):
                        tfps = 30.0
                    frame_delay = 1.0 / tfps
                    consec_fails = 0
                    with self.lock:
                        self._main_status = "active"
                    if not has_dedicated_sub:
                        with self.lock:
                            self._sub_status = "active"

                t0 = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    consec_fails += 1
                    if is_file:
                        if consec_fails >= 15:
                            cap.release()
                            cap = None
                            consec_fails = 0
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.02)
                        continue
                    if consec_fails == 1:
                        first_fail_ts = t0
                    if consec_fails >= 15 or (t0 - first_fail_ts) >= 2.0:
                        cap.release()
                        cap = None
                        consec_fails = 0
                        with self.lock:
                            self._main_status = "reconnecting"
                            self._main_fps = 0.0
                        if not has_dedicated_sub:
                            with self.lock:
                                self._sub_status = "reconnecting"
                                self._sub_fps = 0.0
                        self._sleep_interruptible(2.0)
                    else:
                        time.sleep(0.04)
                    continue

                consec_fails = 0
                fps_cnt += 1
                if (t0 - fps_t0) >= 1.0:
                    cur_fps = fps_cnt / max(0.001, t0 - fps_t0)
                    with self.lock:
                        self._main_fps = cur_fps
                        self._main_status = "active"
                    if not has_dedicated_sub:
                        with self.lock:
                            self._sub_fps = cur_fps
                            self._sub_status = "active"
                    fps_cnt = 0
                    fps_t0 = t0

                # 1. Запис чистого основного кадру усіма активними recorder-модулями (якщо потребують декодування)
                for rec_eng, push_fn in active_rec_feed:
                    try:
                        push_fn(rec_eng, frame.copy(), allowed=True)
                    except Exception:
                        pass

                # 2. Якщо sub_source відсутній — оновлюємо плитку сітки
                if not has_dedicated_sub:
                    self._push_to_grid_cache(frame)

                # 3. Буфер для незалежного вікна повного перегляду
                with self.lock:
                    self._latest_main_bgr = frame
                    self._latest_main_seq += 1
                    self._latest_main_ts = t0

                # 4. Передача свіжого кадру в AI-чергу з урахуванням ліміту AI FPS на камеру
                ai_fps_limit = max(1.0, min(30.0, float(self.hardware_cfg.get("ai_max_fps_per_cam", 8.0))))
                if self._any_ai_active_now() and (t0 - self._last_ai_submit_ts) >= (1.0 / ai_fps_limit):
                    with self.lock:
                        if not self._ai_busy:
                            self._pending_ai_frame = frame.copy()
                            self._pending_ai_ts = t0
                            self._last_ai_submit_ts = t0

                if is_file:
                    sleep_d = frame_delay - (time.time() - t0)
                    if sleep_d > 0:
                        time.sleep(sleep_d)
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def pop_pending_ai_frame(self):
        with self.lock:
            if self._ai_busy or self._pending_ai_frame is None:
                return None
            self._ai_busy = True
            fr = self._pending_ai_frame
            self._pending_ai_frame = None
            return fr

    def run_ai_step(self, frame: np.ndarray):
        """Виконує AI-обробку одного кадру через усі підключені модулі (викликається з AIWorkerPool)."""
        try:
            mods = self.cam_cfg.get("modules") or {}
            ai_entries = self.registry.list_modules(kind="ai")

            with self._engines_lock:
                eng_map = {ent["id"]: self.engines.get(ent["id"]) for ent in ai_entries}

            with self._overlay_lock:
                prev_states = {mid: dict(st) for mid, st in self._overlay_state.items() if isinstance(st, dict)}

            new_states = {}
            for ent in ai_entries:
                mid = ent["id"]
                spec = ent["spec"]
                eng = eng_map.get(mid)
                if eng is None:
                    continue
                mc = mods.get(mid) or {}
                if mc.get("on", False) and is_date_range_active(mc.get("date_from", ""), mc.get("date_to", "")):
                    proc_fn = spec.get("process_and_pack")
                    if callable(proc_fn):
                        try:
                            new_states[mid] = proc_fn(eng, frame, prev_states.get(mid)) or {}
                        except Exception as me:
                            print(f"[AI Module Error '{mid}' @ {self.cam_cfg.get('name')}]: {me}")
                            new_states[mid] = prev_states.get(mid, {})
                else:
                    has_tr = spec.get("has_active_tracks")
                    flush_fn = spec.get("flush_tracks")
                    if callable(flush_fn) and (not callable(has_tr) or has_tr(eng)):
                        try:
                            flush_fn(eng)
                        except Exception:
                            pass
                    new_states[mid] = {}

            with self._overlay_lock:
                self._overlay_state = new_states
        except Exception as e:
            print(f"[AI Camera Error {self.cam_cfg.get('name')}]: {e}")
        finally:
            with self.lock:
                self._ai_busy = False


class CameraManager:
    """Координує всі камери за UUID, спільний пул AI-воркерів, гаряче оновлення модулів і автоочищення."""
    def __init__(self, config_manager, inference_pool, registry, on_saved_event=None, on_cleanup_event=None):
        self.cfg_mgr = config_manager
        self.inference_pool = inference_pool
        self.registry = registry
        self.on_saved_event = on_saved_event
        self.on_cleanup_event = on_cleanup_event

        self._lock = threading.Lock()
        self.units = {}  # uuid -> CameraUnit
        self._running = False
        self._ai_workers = []
        self._rr_index = 0
        self._cleanup_running = False
        self._cleanup_timer = None

    def sync_from_config(self):
        snap = self.cfg_mgr.get_snapshot()
        hw_cfg = snap.get("hardware") or {}
        cams = snap.get("cameras") or []

        try:
            use_ocl = bool(hw_cfg.get("use_opencv_opencl", False))
            if cv2 is not None and hasattr(cv2, "ocl"):
                cv2.ocl.setUseOpenCL(use_ocl and cv2.ocl.haveOpenCL())
        except Exception:
            pass

        self.inference_pool.configure(
            ai_backend=hw_cfg.get("ai_backend", "cpu"),
            ai_threads=int(hw_cfg.get("ai_threads", 2)),
            app_context={"faces_db_path": self.cfg_mgr.get_faces_db_path()},
        )

        new_map = {c["uuid"]: c for c in cams}
        with self._lock:
            for uid in list(self.units.keys()):
                if uid not in new_map:
                    unit = self.units.pop(uid)
                    unit.stop()

            for c in cams:
                uid = c["uuid"]
                arch_dir = self.cfg_mgr.get_camera_archive_dir(uid)
                if uid in self.units:
                    self.units[uid].update_config(c, hw_cfg)
                else:
                    unit = CameraUnit(
                        c,
                        archive_dir=arch_dir,
                        inference_pool=self.inference_pool,
                        registry=self.registry,
                        hardware_cfg=hw_cfg,
                        on_saved_event=self.on_saved_event,
                    )
                    self.units[uid] = unit
                    if self._running and c.get("enabled", True):
                        unit.start()

        if self._running:
            self._ensure_ai_workers()

    def reload_from_registry(self, changed_mod_ids=None, removed_mod_ids=None, force_all: bool = False):
        """Підхоплює додані/оновлені/видалені файли модулів на льоту без зайвого рестарту інших модулів."""
        full_reload = bool(force_all or (changed_mod_ids is None and not removed_mod_ids))
        if full_reload:
            self.inference_pool.invalidate_all()
        else:
            for mid in list(changed_mod_ids or []) + list(removed_mod_ids or []):
                self.inference_pool.invalidate_module(mid)

        self.sync_from_config()
        if full_reload:
            all_ids = [ent["id"] for ent in self.registry.list_modules()]
            with self._lock:
                units_list = list(self.units.values())
            for u in units_list:
                u.force_reload_modules(all_ids)

    def start_all(self):
        if self._running:
            return
        self._running = True
        self.sync_from_config()
        with self._lock:
            units_list = list(self.units.values())
        for u in units_list:
            if u.cam_cfg.get("enabled", True):
                u.start()
        self._ensure_ai_workers()
        self._schedule_cleanup(delay_sec=2.5)

    def stop_all(self):
        self._running = False
        if self._cleanup_timer is not None:
            try:
                self._cleanup_timer.cancel()
            except Exception:
                pass
            self._cleanup_timer = None

        for t in self._ai_workers:
            if t.is_alive():
                t.join(timeout=1.5)
        self._ai_workers.clear()

        with self._lock:
            units_list = list(self.units.values())
        for u in units_list:
            u.stop()

    def _ensure_ai_workers(self):
        self._ai_workers = [t for t in self._ai_workers if t.is_alive()]
        hw_cfg = (self.cfg_mgr.get_snapshot().get("hardware") or {})
        desired = max(1, min(4, int(hw_cfg.get("ai_threads", 2))))
        if hw_cfg.get("ai_backend", "cpu") in ("cuda", "directml"):
            desired = min(desired, 2)
        while len(self._ai_workers) < desired and self._running:
            t = threading.Thread(target=self._ai_scheduler_loop, daemon=True)
            t.start()
            self._ai_workers.append(t)

    def _ai_scheduler_loop(self):
        while self._running:
            with self._lock:
                units_list = list(self.units.values())
                n = len(units_list)
                if n == 0:
                    units_list = []
                else:
                    start_i = self._rr_index % n
                    self._rr_index = (self._rr_index + 1) % n
                    units_list = units_list[start_i:] + units_list[:start_i]

            worked = False
            for u in units_list:
                if not self._running:
                    break
                if not u.running:
                    continue
                fr = u.pop_pending_ai_frame()
                if fr is not None:
                    u.run_ai_step(fr)
                    worked = True
                    break

            if not worked:
                time.sleep(0.015)

    def get_unit(self, cam_uuid: str):
        with self._lock:
            return self.units.get(cam_uuid)

    def get_ordered_enabled_units(self):
        cams = self.cfg_mgr.get_cameras()
        with self._lock:
            res = []
            for c in cams:
                if c.get("enabled", True) and c["uuid"] in self.units:
                    res.append(self.units[c["uuid"]])
            return res

    def get_capability_accessor(self, cap_name: str):
        return self.inference_pool.get_capability_accessor(cap_name)

    def _schedule_cleanup(self, delay_sec: float = 86400.0):
        if not self._running:
            return
        if self._cleanup_timer is not None:
            try:
                self._cleanup_timer.cancel()
            except Exception:
                pass
        t = threading.Timer(delay_sec, self._run_cleanup_tick)
        t.daemon = True
        self._cleanup_timer = t
        t.start()

    def _run_cleanup_tick(self):
        if not self._running:
            return
        self.trigger_cleanup_now()
        self._schedule_cleanup(delay_sec=86400.0)

    def trigger_cleanup_now(self):
        with self._lock:
            if self._cleanup_running:
                return
            self._cleanup_running = True

        def _worker():
            deleted_total = 0
            try:
                cams = self.cfg_mgr.get_cameras()
                now = time.time()
                reg_mod_ids = [ent["id"] for ent in self.registry.list_modules()]
                for c in cams:
                    uid = c["uuid"]
                    cam_dir = self.cfg_mgr.get_camera_archive_dir(uid)
                    mods = c.get("modules") or {}
                    with self._lock:
                        unit = self.units.get(uid)

                    candidate_mods = list(dict.fromkeys(reg_mod_ids + list(mods.keys())))
                    for mod_name in candidate_mods:
                        sec = mods.get(mod_name)
                        if not isinstance(sec, dict):
                            continue
                        days = float(sec.get("delete_older_days", 0.0) or 0.0)
                        if days <= 0:
                            continue
                        folder = os.path.join(cam_dir, mod_name)
                        if not os.path.exists(folder):
                            continue
                        cutoff_ts = now - (days * 86400.0)
                        for root_dir, dirs, files in os.walk(folder, topdown=False):
                            if not self._running:
                                return
                            for fname in files:
                                fpath = os.path.join(root_dir, fname)
                                if unit and unit.is_archive_file_busy(mod_name, fpath):
                                    continue
                                if "_staging" in root_dir:
                                    continue
                                try:
                                    if os.path.getmtime(fpath) < cutoff_ts:
                                        os.remove(fpath)
                                        deleted_total += 1
                                except Exception:
                                    pass
                            for dname in dirs:
                                dpath = os.path.join(root_dir, dname)
                                if dname == "_staging":
                                    continue
                                try:
                                    if not os.listdir(dpath):
                                        os.rmdir(dpath)
                                except Exception:
                                    pass
                if deleted_total > 0:
                    print(f"[Автоочищення] Видалено застарілих файлів: {deleted_total}")
                    if self.on_cleanup_event:
                        self.on_cleanup_event(deleted_total)
            finally:
                with self._lock:
                    self._cleanup_running = False

        threading.Thread(target=_worker, daemon=True).start()
