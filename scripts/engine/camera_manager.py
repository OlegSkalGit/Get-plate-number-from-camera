import os
import time
import threading
import numpy as np

from scripts.core.i18n import tr
from scripts.core.io_pool import flush_io
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


def open_video_capture(source, decode_mode: str = "cpu", decode_threads: int = 2):
    """Відкриває VideoCapture із підтримкою тайм-аутів, обмеження потоків декодера та опційного апаратного декодування."""
    if cv2 is None:
        return None, "cpu"
    mode = str(decode_mode or "cpu").strip().lower()
    is_net = is_network_stream_source(source)
    n_thr = max(1, int(decode_threads or 2))

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
            if hasattr(cv2, "CAP_PROP_N_THREADS"):
                params.extend([cv2.CAP_PROP_N_THREADS, n_thr])
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
        if not isinstance(source, int) and hasattr(cv2, "CAP_FFMPEG"):
            params = []
            if is_net and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000])
            if is_net and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            if hasattr(cv2, "CAP_PROP_N_THREADS"):
                params.extend([cv2.CAP_PROP_N_THREADS, n_thr])
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if (cap is None or not cap.isOpened()) and not is_net:
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(source)
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
        return False, tr("cam.probe.no_cv2", "OpenCV (cv2) library is not installed")
    src = parse_source_value(raw_source)
    if src == "":
        return False, tr("cam.probe.empty_source", "Source is empty")
    t0 = time.time()
    cap, used_mode = open_video_capture(src, decode_mode=decode_mode, decode_threads=1)
    if cap is None or not cap.isOpened():
        return False, tr("cam.probe.open_failed", "Failed to open source ({src})", src=raw_source)
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, tr("cam.probe.no_frame", "Stream opened, but no frame was received")
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0 or fps > 120 or np.isnan(fps):
            fps_str = tr("cam.probe.fps_auto", "auto")
        else:
            fps_str = f"{fps:.1f} FPS"
        ms = int((time.time() - t0) * 1000)
        return True, tr(
            "cam.probe.ok",
            "OK: {w}×{h}, {fps_str}, decoder={dec} ({ms} ms)",
            w=w,
            h=h,
            fps_str=fps_str,
            dec=used_mode.upper(),
            ms=ms,
        )
    except Exception as e:
        return False, tr("cam.probe.read_error", "Read error: {err}", err=e)
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
        allow_disk_write: bool = True,
    ):
        self.cam_cfg = dict(cam_cfg)
        self.uuid = str(cam_cfg["uuid"])
        self.archive_dir = archive_dir
        self.inference_pool = inference_pool
        self.registry = registry
        self.hardware_cfg = dict(hardware_cfg or {})
        self.on_saved_event = on_saved_event
        self.allow_disk_write = bool(allow_disk_write)

        self._write_camera_meta()

        self.lock = threading.Lock()
        self.running = False
        self.detail_subscribers = 0
        self._visible_to_grid = True

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
        self._motion_ref_gray = None
        self._motion_frames_done = 0
        self._last_full_ai_ts = 0.0
        self._last_ai_sig = None

        # Динамічні двигуни модулів: mod_id -> engine instance
        self._engines_lock = threading.Lock()
        self.engines = {}
        self._engine_sigs = {}
        self._engine_specs = {}  # mod_id -> spec (для безпечної зупинки видалених модулів)
        self._module_runtime_errors = {}

        self._sub_thread = None
        self._main_thread = None

        self._init_engines()

    def _effective_allow_disk_write(self) -> bool:
        ctx = getattr(self.inference_pool, "app_context", None) or {}
        return bool(self.allow_disk_write) and bool(ctx.get("allow_disk_write", True))

    def _write_camera_meta(self):
        if not self._effective_allow_disk_write():
            return
        try:
            os.makedirs(self.archive_dir, exist_ok=True)
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
        eff_allow_write = self._effective_allow_disk_write()
        ctx_base = getattr(self.inference_pool, "app_context", None) or {}
        eff_ctx = dict(ctx_base)
        eff_ctx["allow_disk_write"] = eff_allow_write
        eff_ctx["operation_mode"] = "server" if eff_allow_write else "operator"
        flat_cfg_eff = dict(flat_cfg)
        flat_cfg_eff["allow_disk_write"] = eff_allow_write
        flat_cfg_eff["operation_mode"] = eff_ctx["operation_mode"]

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
                blocked_by_operator = (spec.get("kind") == "recorder" or bool(spec.get("writes_to_disk"))) and (not eff_allow_write)

                if not is_on or blocked_by_operator:
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
                                flat_cfg_eff,
                                source=main_src,
                                shared_resources=shared_res,
                                app_context=eff_ctx,
                            )
                        self._engine_specs[mid] = spec
                    else:
                        if mid in self.engines:
                            self._teardown_engine_locked(mid)
                        create_fn = spec.get("create_engine")
                        if callable(create_fn):
                            eng = create_fn(
                                self.inference_pool.base_dir,
                                flat_cfg_eff,
                                mod_out_dir,
                                shared_resources=shared_res,
                                on_saved_callback=self._make_saved_cb(mid),
                                source=main_src,
                                app_context=eff_ctx,
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
        eff_allow_write = self._effective_allow_disk_write()
        mods = self.cam_cfg.get("modules") or {}
        for ent in self.registry.list_modules(kind="recorder"):
            mid = ent["id"]
            spec = ent["spec"]
            with self._engines_lock:
                eng = self.engines.get(mid)
            if eng is None:
                continue
            m_cfg = mods.get(mid) or {}
            allowed = (
                eff_allow_write
                and bool(m_cfg.get("on", spec.get("default_enabled", True)))
                and is_date_range_active(m_cfg.get("date_from", ""), m_cfg.get("date_to", ""))
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
            else:
                stop_fn = spec.get("stop_engine")
                if callable(stop_fn):
                    try:
                        stop_fn(eng, wait=False)
                    except Exception:
                        pass

    def update_config(self, cam_cfg: dict, hardware_cfg: dict, allow_disk_write=None):
        old_main = parse_source_value(self.cam_cfg.get("main_source", ""))
        old_sub = parse_source_value(self.cam_cfg.get("sub_source", ""))
        old_enabled = bool(self.cam_cfg.get("enabled", True))
        old_hw_dec = str(self.hardware_cfg.get("video_decode_mode", "cpu"))

        self.cam_cfg = dict(cam_cfg)
        self.hardware_cfg = dict(hardware_cfg or {})
        if allow_disk_write is not None:
            self.allow_disk_write = bool(allow_disk_write)
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
            with self.lock:
                self._motion_frames_done = 0
                self._last_ai_sig = None
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
            self._pending_ai_frame = None
            self._ai_busy = False
            self._motion_ref_gray = None
            self._motion_frames_done = 0
            self._last_full_ai_ts = 0.0
            self._last_ai_sig = None

    def add_detail_subscriber(self):
        with self.lock:
            self.detail_subscribers += 1

    def remove_detail_subscriber(self):
        with self.lock:
            self.detail_subscribers = max(0, self.detail_subscribers - 1)

    def set_grid_target_size(self, w: int, h: int):
        with self.lock:
            self._grid_target_size = (max(64, int(w)), max(48, int(h)))

    def set_visible_to_grid(self, visible: bool):
        with self.lock:
            self._visible_to_grid = bool(visible)

    def _get_recorder_status(self):
        if not self._effective_allow_disk_write():
            return False, ""
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
        eff_allow_write = self._effective_allow_disk_write()
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
            "allow_disk_write": eff_allow_write,
            "operation_mode": "server" if eff_allow_write else "operator",
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
        eff_allow_write = self._effective_allow_disk_write()
        rec_on, rec_enc = self._get_recorder_status()
        return {
            "frame": frame,
            "seq": seq,
            "main_status": main_st,
            "fps": main_fps,
            "recording": rec_on,
            "encoder": rec_enc,
            "allow_disk_write": eff_allow_write,
            "operation_mode": "server" if eff_allow_write else "operator",
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

        if not self._effective_allow_disk_write():
            return False

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
        with self.lock:
            vis = self._visible_to_grid
            tw, th = self._grid_target_size
            last_push = self._grid_last_push_ts
            if not force and ((not vis) or (now - last_push) < (1.0 / grid_fps)):
                return
            self._grid_last_push_ts = now
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
                    dec_threads = max(1, int(self.hardware_cfg.get("video_decode_threads") or self.hardware_cfg.get("_decode_threads") or 2))
                    with self.lock:
                        self._sub_status = "connecting"
                    cap, _ = open_video_capture(sub_src, decode_mode=dec_mode, decode_threads=dec_threads)
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
                with self.lock:
                    vis = self._visible_to_grid
                    has_grid_img = (self._latest_grid_rgb is not None)
                    last_grid_ts = self._grid_last_push_ts
                grid_fps = max(2.0, min(30.0, float(self.hardware_cfg.get("display_grid_fps", 12.0))))
                sub_due = (not has_grid_img) or (vis and (t0 - last_grid_ts) >= (1.0 / grid_fps))

                if hasattr(cap, "grab") and hasattr(cap, "retrieve"):
                    ret = cap.grab()
                    if ret and sub_due:
                        ret, frame = cap.retrieve()
                    else:
                        frame = None
                else:
                    sub_due = True
                    ret, frame = cap.read()

                if (not ret) or (sub_due and frame is None):
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

                if sub_due and frame is not None:
                    self._push_to_grid_cache(frame, force=not has_grid_img)

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
        next_rec_push_ts = 0.0

        try:
            while self.running:
                eff_allow_write = self._effective_allow_disk_write()
                mods = self.cam_cfg.get("modules") or {}
                active_rec_feed = []
                max_rec_fps = 0.0

                for ent in self.registry.list_modules(kind="recorder"):
                    mid = ent["id"]
                    spec = ent["spec"]
                    with self._engines_lock:
                        eng = self.engines.get(mid)
                    if eng is None:
                        continue
                    mc = mods.get(mid) or {}
                    allowed = (
                        eff_allow_write
                        and bool(mc.get("on", False))
                        and is_date_range_active(mc.get("date_from", ""), mc.get("date_to", ""))
                    )
                    set_fn = spec.get("set_allowed")
                    if callable(set_fn):
                        try:
                            set_fn(eng, allowed)
                        except Exception:
                            pass
                    if not eff_allow_write:
                        stop_fn = spec.get("stop_engine")
                        if callable(stop_fn):
                            try:
                                stop_fn(eng, wait=False)
                            except Exception:
                                pass
                    elif allowed:
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
                                try:
                                    rf = float(mc.get("fps", 15.0) or 15.0)
                                except Exception:
                                    rf = 15.0
                                if rf > max_rec_fps:
                                    max_rec_fps = rf

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
                    dec_threads = max(1, int(self.hardware_cfg.get("video_decode_threads") or self.hardware_cfg.get("_decode_threads") or 2))
                    with self.lock:
                        self._main_status = "connecting"
                    cap, _ = open_video_capture(main_src, decode_mode=dec_mode, decode_threads=dec_threads)
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
                    next_rec_push_ts = 0.0
                    with self.lock:
                        self._main_status = "active"
                    if not has_dedicated_sub:
                        with self.lock:
                            self._sub_status = "active"

                t0 = time.time()
                hw = self.hardware_cfg
                ai_fps_limit = max(1.0, min(30.0, float(hw.get("ai_max_fps_per_cam", 8.0))))
                grid_fps = max(2.0, min(30.0, float(hw.get("display_grid_fps", 12.0))))
                ai_active_now = self._any_ai_active_now()

                with self.lock:
                    want_first = (self._latest_main_bgr is None)
                    want_detail = (self.detail_subscribers > 0)
                    vis_grid = self._visible_to_grid
                    has_grid_img = (self._latest_grid_rgb is not None)
                    last_grid_ts = self._grid_last_push_ts
                    ai_due = ai_active_now and (not self._ai_busy) and (self._pending_ai_frame is None) and ((t0 - self._last_ai_submit_ts) >= (1.0 / ai_fps_limit))

                want_grid = (not has_dedicated_sub) and (
                    (not has_grid_img) or (vis_grid and (t0 - last_grid_ts) >= (1.0 / grid_fps))
                )
                want_rec = False
                if active_rec_feed:
                    rec_target_fps = min(60.0, max(1.0, max_rec_fps) + 2.0)
                    rec_interval = 1.0 / rec_target_fps
                    if next_rec_push_ts <= 0.0 or t0 >= next_rec_push_ts:
                        want_rec = True
                        if next_rec_push_ts <= 0.0 or (t0 - next_rec_push_ts) > rec_interval * 2.0:
                            next_rec_push_ts = t0 + rec_interval
                        else:
                            next_rec_push_ts += rec_interval

                want_decode = want_first or want_detail or want_grid or ai_due or want_rec
                if hasattr(cap, "grab") and hasattr(cap, "retrieve"):
                    ret = cap.grab()
                    if ret and want_decode:
                        ret, frame = cap.retrieve()
                    else:
                        frame = None
                else:
                    want_decode = True
                    ret, frame = cap.read()

                if (not ret) or (want_decode and frame is None):
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

                if want_decode and frame is not None:
                    # 1. Запис чистого основного кадру усіма активними recorder-модулями (frame read-only ref)
                    for rec_eng, push_fn in active_rec_feed:
                        try:
                            push_fn(rec_eng, frame, allowed=True)
                        except Exception:
                            pass

                    # 2. Якщо sub_source відсутній — оновлюємо плитку сітки
                    if not has_dedicated_sub:
                        self._push_to_grid_cache(frame, force=not has_grid_img)

                    # 3. Буфер для незалежного вікна повного перегляду
                    with self.lock:
                        self._latest_main_bgr = frame
                        self._latest_main_seq += 1
                        self._latest_main_ts = t0

                    # 4. Передача свіжого кадру в AI-чергу з урахуванням ліміту AI FPS на камеру (frame read-only)
                    if ai_due or (ai_active_now and (t0 - self._last_ai_submit_ts) >= (1.0 / ai_fps_limit)):
                        with self.lock:
                            if not self._ai_busy and self._pending_ai_frame is None:
                                self._pending_ai_frame = frame
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

    def clear_ai_busy(self):
        with self.lock:
            self._ai_busy = False

    def _compute_motion_small(self, frame: np.ndarray):
        if cv2 is None or frame is None or getattr(frame, "size", 0) == 0:
            return None
        try:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return cv2.GaussianBlur(gray, (3, 3), 0)
        except Exception:
            return None

    def _prepare_wave_item(self, frame: np.ndarray, ai_entries: list, now: float = None) -> dict:
        """
        Готує контекст кадру для AI-хвилі:
        - робить flush/очищення вимкнених або поза-розкладових модулів;
        - застосовує легкий motion gating (160x90) без жодної втрати якості/трекінгу;
        - повертає словник плану для подальшого батч- або поодинокого виконання.
        """
        if now is None:
            now = time.time()
        mods = self.cam_cfg.get("modules") or {}

        with self._engines_lock:
            eng_map = {ent["id"]: self.engines.get(ent["id"]) for ent in ai_entries}

        with self._overlay_lock:
            prev_states = {mid: dict(st) for mid, st in self._overlay_state.items() if isinstance(st, dict)}

        base_states = {}
        active_items = []
        any_active_tracks = False
        any_prev_boxes = False
        sig_parts = []

        for ent in ai_entries:
            mid = ent["id"]
            spec = ent["spec"]
            eng = eng_map.get(mid)
            mc = mods.get(mid) or {}
            is_on = (
                eng is not None
                and bool(mc.get("on", False))
                and is_date_range_active(mc.get("date_from", ""), mc.get("date_to", ""))
            )
            sig_parts.append((mid, id(eng), is_on))
            if is_on:
                has_tr = spec.get("has_active_tracks")
                if callable(has_tr):
                    try:
                        if has_tr(eng):
                            any_active_tracks = True
                    except Exception:
                        pass
                prev_st = prev_states.get(mid)
                if prev_st and prev_st.get("boxes"):
                    any_prev_boxes = True
                active_items.append((mid, spec, eng, prev_st))
            else:
                if eng is not None:
                    has_tr = spec.get("has_active_tracks")
                    flush_fn = spec.get("flush_tracks")
                    if callable(flush_fn) and (not callable(has_tr) or has_tr(eng)):
                        try:
                            flush_fn(eng)
                        except Exception:
                            pass
                base_states[mid] = {}

        if not active_items:
            with self._overlay_lock:
                self._overlay_state = base_states
            with self.lock:
                self._last_ai_sig = tuple(sig_parts)
                self._motion_frames_done = 0
                self._motion_ref_gray = None
            return {"skip": True}

        ai_sig = tuple(sig_parts)
        small_gray = self._compute_motion_small(frame)

        with self.lock:
            if ai_sig != self._last_ai_sig:
                self._last_ai_sig = ai_sig
                self._motion_frames_done = 0
                self._motion_ref_gray = None

            must_run = (
                small_gray is None
                or self._motion_frames_done < 3
                or any_active_tracks
                or any_prev_boxes
                or (now - self._last_full_ai_ts) >= 1.0
                or self._motion_ref_gray is None
                or self._motion_ref_gray.shape != small_gray.shape
            )

            if not must_run:
                diff = cv2.absdiff(small_gray, self._motion_ref_gray)
                max_d = int(diff.max()) if diff.size > 0 else 0
                if max_d < 10:
                    is_static = True
                elif max_d < 22 and int(np.count_nonzero(diff >= 12)) < 8:
                    is_static = True
                else:
                    is_static = False

                if is_static:
                    skipped_states = dict(base_states)
                    for mid, _spec, _eng, prev_st in active_items:
                        st = dict(prev_st or {})
                        st["boxes"] = []
                        skipped_states[mid] = st
                    with self._overlay_lock:
                        self._overlay_state = skipped_states
                    return {"skip": True}

            self._motion_frames_done += 1
            self._last_full_ai_ts = now
            self._motion_ref_gray = small_gray

        return {
            "skip": False,
            "unit": self,
            "frame": frame,
            "base_states": base_states,
            "prev_states": prev_states,
            "active_items": active_items,
            "meta": {},
        }

    def run_ai_step(self, frame: np.ndarray):
        """Виконує AI-обробку одного кадру через усі підключені модулі (викликається з AIWorkerPool або напряму)."""
        try:
            all_ai = self.registry.list_modules(kind="ai")
            ai_entries = _sort_ai_entries_for_cascade(all_ai)
            plan = self._prepare_wave_item(frame, ai_entries, now=time.time())
            if plan.get("skip"):
                return

            new_states = dict(plan["base_states"])
            unit_meta = plan["meta"]
            for mid, spec, eng, prev_st in plan["active_items"]:
                batch_fn = spec.get("process_batch_and_pack")
                proc_fn = spec.get("process_and_pack")
                is_builtin = mid in ("plates", "people", "faces")
                fr_arg = frame if is_builtin else frame.copy()
                try:
                    if callable(batch_fn):
                        shared_res = self.inference_pool.get_module_resources(mid) if self.inference_pool is not None else None
                        res = batch_fn(
                            [{
                                "engine": eng,
                                "frame": fr_arg,
                                "prev_state": prev_st,
                                "meta": unit_meta,
                            }],
                            shared_res,
                        )
                        st = res[0] if res else {}
                        new_states[mid] = st if isinstance(st, dict) else {}
                    elif callable(proc_fn):
                        new_states[mid] = proc_fn(eng, fr_arg, prev_st) or {}
                except Exception as me:
                    print(f"[AI Module Error '{mid}' @ {self.cam_cfg.get('name')}]: {me}")
                    new_states[mid] = dict(prev_st or {})

            with self._overlay_lock:
                self._overlay_state = new_states
        except Exception as e:
            print(f"[AI Camera Error {self.cam_cfg.get('name')}]: {e}")
        finally:
            self.clear_ai_busy()


def _sort_ai_entries_for_cascade(ai_entries: list) -> list:
    """Гарантує виконання person_detector/people перед іншими модулями та faces наприкінці для безпечного каскаду SCRFD."""
    if len(ai_entries) <= 1:
        return list(ai_entries)

    def _order_key(ent):
        spec = ent.get("spec") or {}
        provides = spec.get("provides") or []
        mid = ent.get("id")
        if mid == "faces" or ("face_db" in provides):
            return 2
        if mid == "people" or ("person_detector" in provides):
            return 0
        return 1

    return sorted(ai_entries, key=_order_key)


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
        self._visible_grid_uuids = None
        self._cleanup_running = False
        self._cleanup_timer = None

    def set_visible_grid_uuids(self, visible_uuids):
        vis_set = set(visible_uuids or ())
        with self._lock:
            self._visible_grid_uuids = vis_set
            units_list = list(self.units.values())
        for u in units_list:
            if hasattr(u, "set_visible_to_grid"):
                u.set_visible_to_grid(u.uuid in vis_set)

    def _ordered_ai_modules_for_inference(self) -> list:
        """Гарантує виконання person_detector/people перед faces для безпечного каскаду SCRFD."""
        return _sort_ai_entries_for_cascade(self.registry.list_modules(kind="ai"))

    def sync_from_config(self):
        snap = self.cfg_mgr.get_snapshot()
        gen_cfg = snap.get("general") or {}
        op_mode = str(gen_cfg.get("operation_mode", "server")).strip().lower()
        if op_mode not in ("server", "operator"):
            op_mode = "server"
        allow_write = (op_mode == "server")
        hw_cfg = snap.get("hardware") or {}
        cams = snap.get("cameras") or []
        hw_cfg_eff = dict(hw_cfg)
        if not hw_cfg_eff.get("video_decode_threads"):
            enabled_cnt = sum(1 for c in cams if c.get("enabled", True))
            hw_cfg_eff["_decode_threads"] = 1 if enabled_cnt >= 8 else 2

        try:
            use_ocl = bool(hw_cfg_eff.get("use_opencv_opencl", False))
            if cv2 is not None and hasattr(cv2, "ocl"):
                cv2.ocl.setUseOpenCL(use_ocl and cv2.ocl.haveOpenCL())
        except Exception:
            pass

        self.inference_pool.configure(
            ai_backend=hw_cfg_eff.get("ai_backend", "cpu"),
            ai_threads=int(hw_cfg_eff.get("ai_threads", 2)),
            app_context={
                "operation_mode": op_mode,
                "allow_disk_write": allow_write,
                "faces_db_path": self.cfg_mgr.get_faces_db_path(create=allow_write),
            },
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
                    self.units[uid].update_config(c, hw_cfg_eff, allow_disk_write=allow_write)
                else:
                    unit = CameraUnit(
                        c,
                        archive_dir=arch_dir,
                        inference_pool=self.inference_pool,
                        registry=self.registry,
                        hardware_cfg=hw_cfg_eff,
                        on_saved_event=self.on_saved_event,
                        allow_disk_write=allow_write,
                    )
                    if self._visible_grid_uuids is not None:
                        unit.set_visible_to_grid(uid in self._visible_grid_uuids)
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
        flush_io(timeout=3.0)

    def _ensure_ai_workers(self):
        self._ai_workers = [t for t in self._ai_workers if t.is_alive()]
        hw_cfg = (self.cfg_mgr.get_snapshot().get("hardware") or {})
        provs = list(getattr(self.inference_pool, "providers", []) or [])
        is_gpu = any(p in provs for p in ("CUDAExecutionProvider", "DmlExecutionProvider"))
        if is_gpu:
            desired = 1
        else:
            desired = 1 if len(self.units) <= 2 else max(1, min(2, int(hw_cfg.get("ai_threads", 2))))
        while len(self._ai_workers) < desired and self._running:
            t = threading.Thread(target=self._ai_scheduler_loop, daemon=True)
            t.start()
            self._ai_workers.append(t)

    def _ai_scheduler_loop(self):
        while self._running:
            provs = list(getattr(self.inference_pool, "providers", []) or [])
            is_gpu = any(p in provs for p in ("CUDAExecutionProvider", "DmlExecutionProvider"))
            max_batch = 8 if is_gpu else 2

            with self._lock:
                units_all = list(self.units.values())
                n = len(units_all)
                if n == 0:
                    ordered = []
                else:
                    start_i = self._rr_index % n
                    ordered = units_all[start_i:] + units_all[:start_i]

            popped = []
            popped_indices = set()
            active_cams = 0
            last_offset = 0

            for idx, u in enumerate(ordered):
                if not self._running:
                    break
                if not u.running:
                    continue
                active_cams += 1
                if len(popped) < max_batch:
                    fr = u.pop_pending_ai_frame()
                    if fr is not None:
                        popped.append((u, fr))
                        popped_indices.add(idx)
                        last_offset = idx + 1

            # Коротке вікно коалесингу (3 мс), щоб зібрати ще активні камери у спільну хвилю GPU/CPU-батчу
            if popped and len(popped) < max_batch and active_cams > len(popped) and self._running:
                time.sleep(0.003)
                for idx, u in enumerate(ordered):
                    if not self._running or len(popped) >= max_batch:
                        break
                    if idx in popped_indices or not u.running:
                        continue
                    fr = u.pop_pending_ai_frame()
                    if fr is not None:
                        popped.append((u, fr))
                        popped_indices.add(idx)
                        if (idx + 1) > last_offset:
                            last_offset = idx + 1

            if n > 0:
                with self._lock:
                    self._rr_index = (self._rr_index + max(1, last_offset if popped else 1)) % n

            if not popped:
                time.sleep(0.012)
                continue

            try:
                now = time.time()
                ai_entries = self._ordered_ai_modules_for_inference()
                entries_by_id = {ent["id"]: ent for ent in ai_entries}

                wave_plans = []
                module_buckets = {}
                for u, fr in popped:
                    try:
                        plan = u._prepare_wave_item(fr, ai_entries, now=now)
                    except Exception as pe:
                        print(f"[AI Wave Prepare Error @ {u.cam_cfg.get('name')}]: {pe}")
                        u.clear_ai_busy()
                        continue

                    if plan.get("skip"):
                        u.clear_ai_busy()
                        continue

                    plan["new_states"] = dict(plan["base_states"])
                    wave_plans.append(plan)
                    for mid, _spec, eng, prev_st in plan["active_items"]:
                        module_buckets.setdefault(mid, []).append({
                            "plan": plan,
                            "engine": eng,
                            "frame": plan["frame"],
                            "prev_state": prev_st,
                            "meta": plan["meta"],
                        })

                for ent in ai_entries:
                    mid = ent["id"]
                    bucket = module_buckets.get(mid)
                    if not bucket:
                        continue
                    spec = entries_by_id[mid]["spec"]
                    batch_fn = spec.get("process_batch_and_pack")
                    proc_fn = spec.get("process_and_pack")
                    is_builtin = mid in ("plates", "people", "faces")

                    if callable(batch_fn):
                        try:
                            shared_res = self.inference_pool.get_module_resources(mid)
                            batch_items = [
                                {
                                    "engine": b["engine"],
                                    "frame": b["frame"] if is_builtin else b["frame"].copy(),
                                    "prev_state": b["prev_state"],
                                    "meta": b["meta"],
                                }
                                for b in bucket
                            ]
                            res_states = batch_fn(batch_items, shared_res)
                            if not isinstance(res_states, (list, tuple)) or len(res_states) != len(bucket):
                                raise RuntimeError("invalid batch result length")
                            for b, st in zip(bucket, res_states):
                                b["plan"]["new_states"][mid] = st if isinstance(st, dict) else {}
                        except Exception as be:
                            print(f"[AI Batch Module Error '{mid}']: {be}")
                            for b in bucket:
                                b["plan"]["new_states"][mid] = dict(b["prev_state"] or {})
                    elif callable(proc_fn):
                        for b in bucket:
                            try:
                                fr_arg = b["frame"] if is_builtin else b["frame"].copy()
                                b["plan"]["new_states"][mid] = proc_fn(b["engine"], fr_arg, b["prev_state"]) or {}
                            except Exception as me:
                                cname = b["plan"]["unit"].cam_cfg.get("name")
                                print(f"[AI Module Error '{mid}' @ {cname}]: {me}")
                                b["plan"]["new_states"][mid] = dict(b["prev_state"] or {})

                for plan in wave_plans:
                    u = plan["unit"]
                    with u._overlay_lock:
                        u._overlay_state = plan["new_states"]
            except Exception as e:
                print(f"[AI Wave Scheduler Error]: {e}")
            finally:
                for u, _fr in popped:
                    u.clear_ai_busy()

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
        if self.cfg_mgr.is_operator_mode():
            return
        with self._lock:
            if self._cleanup_running:
                return
            self._cleanup_running = True

        def _worker():
            deleted_total = 0
            try:
                if self.cfg_mgr.is_operator_mode():
                    return
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
