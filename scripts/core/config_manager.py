import os
import copy
import json
import uuid
import shutil
import threading
import configparser

from scripts.core.i18n import tr, normalize_language_code, DEFAULT_LANGUAGE


def make_default_camera_modules(registry=None) -> dict:
    """Формує типові налаштування модулів камери динамічно з ModuleRegistry."""
    if registry is not None:
        return registry.build_all_default_camera_modules()
    return {}


def make_camera_entry(
    name=None,
    main_source="0",
    sub_source="",
    enabled=True,
    cam_uuid=None,
    modules=None,
    registry=None,
):
    if modules is not None:
        mods = copy.deepcopy(modules)
    else:
        mods = make_default_camera_modules(registry=registry)

    default_name = tr("camera.default_numbered", "Camera {num}", num=1) if name is None else str(name or "").strip()
    if not default_name:
        default_name = tr("camera.default_single", "Camera")

    return {
        "uuid": str(cam_uuid or uuid.uuid4()),
        "name": default_name,
        "enabled": bool(enabled),
        "main_source": str(main_source or "").strip(),
        "sub_source": str(sub_source or "").strip(),
        "window": {
            "left": 80,
            "top": 60,
            "width": 1160,
            "height": 720,
            "maximized": False,
        },
        "roi_window": {
            "left": 40,
            "top": 40,
            "width": 520,
            "height": 360,
            "maximized": False,
        },
        "modules": mods,
    }


def _deep_merge(base: dict, override: dict) -> dict:
    res = copy.deepcopy(base)
    if not isinstance(override, dict):
        return res
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(res.get(k), dict):
            res[k] = _deep_merge(res[k], v)
        else:
            res[k] = copy.deepcopy(v)
    return res


SETTINGS_HELP = {
    # General
    "general.language": (
        "Interface Language\n"
        "Selects the application language loaded from JSON files in the lang/ folder (<code>.json).\n"
        "Default on first launch is English (en). You can also switch language from the top bar in the main window."
    ),
    "general.operation_mode": (
        "System Operation Mode\n"
        "• server (Server Mode) — all options, archive, video recording, saving plate/face/people snapshots, face DB updates, and auto-cleanup are enabled.\n"
        "• operator (Operator Mode) — full live viewing and on-screen AI analysis (plates, recognizing already known faces, people/vehicles) WITHOUT any disk writes (recording, screenshots, auto-registering faces, cleanup, and window position auto-save are disabled)."
    ),
    "general.archive_root": (
        "Archive Root Folder\n"
        "Disk directory (a spacious or fast drive is recommended for the video archive) "
        "where per-camera folders (<UUID>/<module>) and the shared faces database (faces_db/faces.txt) are stored."
    ),
    "general.autostart_os_and_play": (
        "Windows Autostart & Automatic Monitoring\n"
        "Adds the application to the current Windows user's startup (HKCU\\...\\Run) "
        "and automatically starts polling all enabled cameras on launch."
    ),
    "general.kiosk_mode": (
        "Kiosk Mode (Lock Exit)\n"
        "When enabled: blocks Alt+F4 / Esc / close button on the main window; exit is only possible with Ctrl+Alt+Shift+Q.\n"
        "When disabled (default): all windows can be freely moved across monitors, resized, and closed normally."
    ),

    # Grid
    "grid.mode": (
        "Camera Grid Mode\n"
        "• auto — automatically calculates rows and columns for the active cameras and window aspect ratio (1x1, 2x1, 2x2, 3x2, 3x3, etc.).\n"
        "• manual — fixed grid (Rows × Columns). If there are more cameras than slots, a ◀ / ▶ page switcher appears."
    ),
    "grid.rows": "Number of rows in manual grid mode (1–8).",
    "grid.cols": "Number of columns in manual grid mode (1–8).",
    "grid.border_color": "Default grid dividing line color (HEX, e.g. #00e5ff).",
    "grid.active_border_color": "Border color of the camera tile currently open in a detailed view window (e.g. #00ff88).",
    "grid.offline_border_color": "Border color of a tile when connection to the camera stream is lost (e.g. #ff4444).",
    "grid.border_width": "Single grid line thickness in pixels between cells and along outer grid edges (1–10 px).",
    "grid.tile_gap": "Outer margin around the entire camera grid in pixels (0–20 px).",
    "grid.show_labels": "Show the camera name/description in the top-left corner of each grid tile.",
    "grid.show_status": "Show status badges on each tile: ● REC (recording), active modules, and connection state.",
    "grid.show_fps": "Show current sub-stream FPS on each tile in the grid.",

    # Hardware & Graphics
    "hardware.ai_backend": (
        "Neural Network Accelerator (ONNX Runtime)\n"
        "• cpu — maximum compatibility on any PC (default).\n"
        "• auto — automatically picks the fastest available accelerator (CUDA -> DirectML -> OpenVINO -> CPU).\n"
        "• cuda — NVIDIA CUDA GPU (highest speed on GeForce/RTX cards; requires onnxruntime-gpu and CUDA/cuDNN).\n"
        "• directml — DirectX 12 GPU (works on NVIDIA, AMD, and Intel without CUDA Toolkit; requires onnxruntime-directml).\n"
        "• openvino — acceleration on Intel CPUs and integrated graphics."
    ),
    "hardware.ai_threads": (
        "Threads per ONNX Session & AI Workers (1–8)\n"
        "1–2 workers are usually enough for GPU. For CPU, increase cautiously while leaving cores for decoding, video encoding, and UI."
    ),
    "hardware.ai_max_fps_per_cam": (
        "Max AI Analysis Rate per Camera (FPS, 1–30)\n"
        "Limits how many times per second each camera sends frames to neural networks. 6–10 FPS per camera "
        "is enough for reliable plate and face recognition while saving 60–80% CPU/GPU resources with multiple cameras."
    ),
    "hardware.video_decode_mode": (
        "Input Stream Decoding Mode\n"
        "• cpu — standard OpenCV VideoCapture (most reliable for webcams and local files).\n"
        "• auto / d3d11va / cuda / qsv — FFmpeg GPU hardware decoding for RTSP/HTTP network streams (with automatic CPU fallback if unsupported)."
    ),
    "hardware.display_grid_fps": "Max refresh rate for grid tiles (3–30 FPS). Using 10–15 FPS drastically lowers UI and CPU load with 8–16 cameras.",
    "hardware.display_detail_fps": "Max refresh rate in the detailed view window of a selected camera (5–60 FPS).",
    "hardware.display_resize_method": (
        "Frame Scaling Algorithm for Display\n"
        "• nearest — fastest (lowest CPU usage);\n"
        "• linear — optimal balance of quality and speed;\n"
        "• lanczos — sharpest fine details (higher CPU usage)."
    ),
    "hardware.use_opencv_opencl": (
        "OpenCV OpenCL (Transparent API / UMat)\n"
        "Enables GPU-accelerated frame resizing and image processing in OpenCV when an OpenCL driver is available and stable."
    ),

    # Camera
    "camera.uuid": "Unique camera identifier (UUID v4). Generated automatically on creation and binds the archive folder so you can rename the camera without losing history.",
    "camera.name": "Friendly camera name (e.g., 'Gate Entrance', 'North Warehouse'). Used in the grid, window title, and search.",
    "camera.enabled": "Enable or completely disable this camera (disabled cameras do not connect to streams and do not take a grid slot).",
    "camera.main_source": (
        "Main Stream (High Quality)\n"
        "Always used for video recording, AI recognition, and detailed view. "
        "Accepts: rtsp://user:pass@ip:554/stream1, http://..., webcam index (0, 1), or local .mp4 file."
    ),
    "camera.sub_source": (
        "Compressed Sub-Stream (Low Resolution for Grid)\n"
        "Displayed ONLY in the main grid tiles (e.g., rtsp://.../stream2 or channel=1&subtype=1) to avoid decoding 4K/FullHD for dozens of thumbnails. "
        "If left empty (e.g., webcam 0 or local file), the grid automatically scales frames from the main stream."
    ),
}


class ConfigManager:
    def __init__(self, base_dir: str, registry=None):
        self.base_dir = os.path.abspath(base_dir)
        self.registry = registry
        self.config_json_path = os.path.join(self.base_dir, "config.json")
        self.legacy_ini_path = os.path.join(self.base_dir, "config.ini")
        self._lock = threading.Lock()
        self._version = 0
        self._fast_cache_ver = -1
        self._fast_cache_snap = None
        self.data = self._load_or_migrate()

    def _get_registry_default_modules(self) -> dict:
        if self.registry is not None:
            return self.registry.build_all_default_camera_modules()
        return {}

    def _make_defaults(self) -> dict:
        reg_mods = self._get_registry_default_modules()
        return {
            "version": 2,
            "general": {
                "language": DEFAULT_LANGUAGE,
                "operation_mode": "server",
                "archive_root": os.path.join(self.base_dir, "archive"),
                "autostart_os_and_play": False,
                "kiosk_mode": False,
            },
            "windows": {
                "grid": {"left": 40, "top": 30, "width": 1280, "height": 760, "maximized": False},
                "settings": {"left": 100, "top": 60, "width": 1080, "height": 740, "maximized": False},
                "people": {"left": 120, "top": 70, "width": 1180, "height": 760, "maximized": False},
            },
            "grid": {
                "mode": "auto",
                "rows": 2,
                "cols": 2,
                "border_color": "#00e5ff",
                "active_border_color": "#00ff88",
                "offline_border_color": "#ff4444",
                "border_width": 2,
                "tile_gap": 4,
                "show_labels": True,
                "show_status": True,
                "show_fps": True,
            },
            "hardware": {
                "ai_backend": "cpu",
                "ai_threads": 2,
                "ai_max_fps_per_cam": 8.0,
                "video_decode_mode": "cpu",
                "display_grid_fps": 12.0,
                "display_detail_fps": 25.0,
                "display_resize_method": "linear",
                "use_opencv_opencl": False,
            },
            "default_camera_modules": copy.deepcopy(reg_mods),
            "cameras": [],
        }

    def _migrate_from_ini(self, cfg: dict):
        """Міграція старого config.ini та локальної faces/faces.txt при першому запуску."""
        reg_mods = self._get_registry_default_modules()
        parser = configparser.ConfigParser(interpolation=None)
        if not os.path.exists(self.legacy_ini_path):
            cfg["cameras"] = [make_camera_entry(name="Camera 1", main_source="0", sub_source="", modules=reg_mods)]
            return cfg

        try:
            parser.read(self.legacy_ini_path, encoding="utf-8-sig")
        except Exception:
            cfg["cameras"] = [make_camera_entry(name="Camera 1", main_source="0", sub_source="", modules=reg_mods)]
            return cfg

        def _g_bool(key, default):
            if parser.has_option("SETTINGS", key):
                try:
                    raw = parser.get("SETTINGS", key).strip()
                    if not raw:
                        return default
                    return parser.getboolean("SETTINGS", key, fallback=default)
                except Exception:
                    return default
            return default

        def _g_float(key, default):
            if parser.has_option("SETTINGS", key):
                try:
                    raw = parser.get("SETTINGS", key).strip()
                    return float(raw) if raw else default
                except Exception:
                    return default
            return default

        def _g_int(key, default):
            if parser.has_option("SETTINGS", key):
                try:
                    raw = parser.get("SETTINGS", key).strip()
                    return int(round(float(raw))) if raw else default
                except Exception:
                    return default
            return default

        def _g_str(key, default):
            if parser.has_option("SETTINGS", key):
                try:
                    return parser.get("SETTINGS", key, fallback=default).strip().strip('\'"')
                except Exception:
                    return default
            return default

        cfg["general"]["autostart_os_and_play"] = _g_bool("autostart_os_and_play", False)
        mods = copy.deepcopy(reg_mods)

        mods.setdefault("plates", {}).update({
            "on": _g_bool("plates_on", True),
            "show_boxes": _g_bool("plates_show_boxes", True),
            "debug": _g_bool("plates_debug", False),
            "det_thresh": _g_float("plates_det_thresh", 0.18),
            "min_percent_to_save": _g_int("plates_min_percent_to_save", 80),
            "sleep_time_after_save": _g_float("plates_sleep_time_after_save", 5.0),
            "view_time": _g_int("plates_view_time", 30),
            "rect_scale": _g_float("plates_rect_scale", 5.0),
            "delete_older_days": max(0.0, _g_float("plates_delete_older_days", 0.0)),
            "date_from": _g_str("plates_date_from", ""),
            "date_to": _g_str("plates_date_to", ""),
        })

        face_sleep = _g_float("faces_sleep_time_after_save", mods["plates"]["sleep_time_after_save"] * 2.0)
        mods.setdefault("faces", {}).update({
            "on": _g_bool("faces_on", True),
            "show_boxes": _g_bool("faces_show_boxes", True),
            "min_similarity": _g_float("faces_min_similarity", 0.48),
            "det_thresh": _g_float("faces_det_thresh", 0.45),
            "min_size": _g_int("faces_min_size", 20),
            "sleep_time_after_save": max(0.0, face_sleep),
            "cooldown_extend_on_seen": _g_bool("faces_cooldown_extend_on_seen", True),
            "delete_older_days": max(0.0, _g_float("faces_delete_older_days", 0.0)),
            "date_from": _g_str("faces_date_from", ""),
            "date_to": _g_str("faces_date_to", ""),
        })

        mods.setdefault("people", {}).update({
            "on": _g_bool("people_on", True),
            "show_boxes": _g_bool("people_show_boxes", True),
            "detect_transport": _g_bool("people_detect_transport", True),
            "det_thresh": _g_float("people_det_thresh", 0.45),
            "timer_ms": _g_int("people_timer_ms", 1000),
            "delete_older_days": max(0.0, _g_float("people_delete_older_days", 0.0)),
            "date_from": _g_str("people_date_from", ""),
            "date_to": _g_str("people_date_to", ""),
        })

        mods.setdefault("video", {}).update({
            "on": _g_bool("video_on", True),
            "duration_min": max(0.05, _g_float("video_duration_min", 5.0)),
            "fps": max(1.0, min(60.0, _g_float("video_fps", 15.0))),
            "codec": (_g_str("video_codec", "libx264").lower() or "libx264"),
            "preset": (_g_str("video_preset", "veryfast").lower() or "veryfast"),
            "crf": max(0, min(51, _g_int("video_crf", 30))),
            "keyframe_sec": max(0.5, min(60.0, _g_float("video_keyframe_sec", 6.0))),
            "max_bitrate_kbps": max(0, _g_int("video_max_bitrate_kbps", 900)),
            "max_height": max(0, _g_int("video_max_height", 0)),
            "delete_older_days": max(0.0, _g_float("video_delete_older_days", 0.0)),
            "date_from": _g_str("video_date_from", ""),
            "date_to": _g_str("video_date_to", ""),
        })

        cam = make_camera_entry(
            name="Camera 1",
            main_source=_g_str("source", "0"),
            sub_source="",
            enabled=True,
            modules=mods,
        )
        cam["roi_window"] = {
            "top": _g_int("viewer_top", 34),
            "left": _g_int("viewer_left", 47),
            "width": max(260, _g_int("viewer_width", 927)),
            "height": max(180, _g_int("viewer_height", 535)),
            "maximized": False,
        }
        cfg["default_camera_modules"] = copy.deepcopy(mods)
        cfg["cameras"] = [cam]

        old_faces_db = os.path.join(self.base_dir, "faces", "faces.txt")
        new_faces_db_dir = os.path.join(cfg["general"]["archive_root"], "faces_db")
        new_faces_db = os.path.join(new_faces_db_dir, "faces.txt")
        if os.path.exists(old_faces_db) and not os.path.exists(new_faces_db):
            try:
                os.makedirs(new_faces_db_dir, exist_ok=True)
                shutil.copy2(old_faces_db, new_faces_db)
            except Exception:
                pass
        return cfg

    def _normalize_loaded_config(self, raw: dict) -> dict:
        defaults = self._make_defaults()
        reg_mods = self._get_registry_default_modules()
        cfg = _deep_merge(defaults, raw)
        gen = cfg.setdefault("general", {})
        raw_lang = normalize_language_code(gen.get("language")) or DEFAULT_LANGUAGE
        gen["language"] = raw_lang
        raw_mode = str(gen.get("operation_mode", "server") or "server").strip().lower()
        gen["operation_mode"] = "operator" if raw_mode == "operator" else "server"
        if not gen.get("archive_root"):
            gen["archive_root"] = defaults["general"]["archive_root"]

        # default_camera_modules: зливаємо схеми активних модулів і зберігаємо невідомі/тимчасово вимкнені
        raw_tpl_mods = (raw or {}).get("default_camera_modules") or {}
        cfg["default_camera_modules"] = _deep_merge(reg_mods, raw_tpl_mods)

        norm_cams = []
        seen_uuids = set()
        for idx, c in enumerate(cfg.get("cameras") or []):
            if not isinstance(c, dict):
                continue
            uid = str(c.get("uuid") or "").strip()
            if not uid or uid in seen_uuids:
                uid = str(uuid.uuid4())
            seen_uuids.add(uid)

            # Акуратно зливаємо відомі модулі з реєстру, не чіпаючи налаштування тимчасово відсутніх модулів
            raw_mods = c.get("modules") or {}
            merged_mods = _deep_merge(reg_mods, raw_mods)

            base_cam = make_camera_entry(
                name=c.get("name") or f"Camera {idx + 1}",
                main_source=c.get("main_source", ""),
                sub_source=c.get("sub_source", ""),
                enabled=c.get("enabled", True),
                cam_uuid=uid,
                modules=merged_mods,
            )
            base_cam["window"] = _deep_merge(base_cam["window"], c.get("window") or {})
            base_cam["roi_window"] = _deep_merge(base_cam["roi_window"], c.get("roi_window") or {})
            norm_cams.append(base_cam)

        if not norm_cams:
            norm_cams = [make_camera_entry(name="Camera 1", main_source="0", sub_source="", modules=reg_mods)]
        cfg["cameras"] = norm_cams
        return cfg

    def sync_with_registry(self, registry=None) -> bool:
        """
        Оновлює дефолтні поля щойно знайдених модулів у конфігурації всіх камер,
        не перезаписуючи існуючі значення користувача та зберігаючи невідомі секції.
        """
        if registry is not None:
            self.registry = registry
        with self._lock:
            before = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
            norm = self._normalize_loaded_config(self.data)
            after = json.dumps(norm, sort_keys=True, ensure_ascii=False)
            if after != before:
                self.data = norm
                self._version += 1
                if not self._is_operator_mode_locked():
                    self._save_dict_atomic(norm)
                return True
            return False

    def _load_or_migrate(self) -> dict:
        if os.path.exists(self.config_json_path):
            try:
                with open(self.config_json_path, "r", encoding="utf-8-sig") as f:
                    raw = json.load(f)
                return self._normalize_loaded_config(raw)
            except Exception as e:
                print(f"[ConfigManager] Помилка читання config.json ({e}), перестворення...")

        cfg = self._make_defaults()
        cfg = self._migrate_from_ini(cfg)
        self._save_dict_atomic(cfg)
        return cfg

    def _save_dict_atomic(self, cfg_dict: dict):
        tmp_path = f"{self.config_json_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp_path, self.config_json_path)

    def get_snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self.data)

    def get_ui_fast_config(self) -> dict:
        """Повертає кешований read-only знімок конфігурації (оновлюється лише при зміні _version)."""
        with self._lock:
            if self._fast_cache_snap is None or self._fast_cache_ver != self._version:
                self._fast_cache_snap = copy.deepcopy(self.data)
                self._fast_cache_ver = self._version
            return self._fast_cache_snap

    def save_data(self, new_data: dict):
        with self._lock:
            norm = self._normalize_loaded_config(new_data)
            self.data = norm
            self._version += 1
            self._save_dict_atomic(norm)

    def save(self):
        with self._lock:
            self._version += 1
            self._save_dict_atomic(self.data)

    def get_language(self) -> str:
        with self._lock:
            raw = (self.data.get("general") or {}).get("language", DEFAULT_LANGUAGE)
            return normalize_language_code(raw) or DEFAULT_LANGUAGE

    def set_language(self, lang_code: str) -> str:
        norm = normalize_language_code(lang_code) or DEFAULT_LANGUAGE
        with self._lock:
            self.data.setdefault("general", {})["language"] = norm
            self._version += 1
            self._save_dict_atomic(self.data)
        return norm

    def _is_operator_mode_locked(self) -> bool:
        raw_mode = str((self.data.get("general") or {}).get("operation_mode", "server") or "server").strip().lower()
        return raw_mode == "operator"

    def get_operation_mode(self) -> str:
        with self._lock:
            return "operator" if self._is_operator_mode_locked() else "server"

    def is_operator_mode(self) -> bool:
        return self.get_operation_mode() == "operator"

    def is_server_mode(self) -> bool:
        return self.get_operation_mode() == "server"

    def get_archive_root(self) -> str:
        with self._lock:
            root = self.data.get("general", {}).get("archive_root") or os.path.join(self.base_dir, "archive")
            return os.path.abspath(root)

    def get_faces_db_path(self, create: bool = None) -> str:
        with self._lock:
            root = self.data.get("general", {}).get("archive_root") or os.path.join(self.base_dir, "archive")
            if create is None:
                should_create = not self._is_operator_mode_locked()
            else:
                should_create = bool(create)
        db_dir = os.path.join(os.path.abspath(root), "faces_db")
        if should_create:
            os.makedirs(db_dir, exist_ok=True)
        return os.path.join(db_dir, "faces.txt")

    def get_camera_archive_dir(self, cam_uuid: str) -> str:
        return os.path.join(self.get_archive_root(), "cameras", str(cam_uuid))

    def get_cameras(self):
        with self._lock:
            return copy.deepcopy(self.data.get("cameras", []))

    def get_camera(self, cam_uuid: str):
        with self._lock:
            for c in self.data.get("cameras", []):
                if c.get("uuid") == cam_uuid:
                    return copy.deepcopy(c)
        return None

    def get_window_geometry(self, win_key: str) -> dict:
        with self._lock:
            return copy.deepcopy(self.data.get("windows", {}).get(win_key, {}))

    def set_window_geometry(self, win_key: str, geom: dict):
        with self._lock:
            if self._is_operator_mode_locked():
                return
            self.data.setdefault("windows", {})[win_key] = copy.deepcopy(geom)
            self._version += 1
            self._save_dict_atomic(self.data)

    def get_camera_window_geometry(self, cam_uuid: str, kind: str = "window") -> dict:
        with self._lock:
            for c in self.data.get("cameras", []):
                if c.get("uuid") == cam_uuid:
                    return copy.deepcopy(c.get(kind, {}))
        return {}

    def set_camera_window_geometry(self, cam_uuid: str, kind: str, geom: dict):
        with self._lock:
            if self._is_operator_mode_locked():
                return
            for c in self.data.get("cameras", []):
                if c.get("uuid") == cam_uuid:
                    c[kind] = copy.deepcopy(geom)
                    self._version += 1
                    self._save_dict_atomic(self.data)
                    return

    @staticmethod
    def flatten_camera_module_config(cam_cfg: dict) -> dict:
        """Формує плоский словник виду <module_id>_<param> динамічно для всіх модулів камери."""
        mods = (cam_cfg or {}).get("modules") or {}
        flat = {}
        for mod_name, sec in mods.items():
            if isinstance(sec, dict):
                for k, v in sec.items():
                    flat[f"{mod_name}_{k}"] = v
        return flat

    @staticmethod
    def get_help_text(help_key: str, registry=None) -> str:
        if registry is not None:
            get_help_fn = getattr(registry, "get_all_help_entries", None)
            if callable(get_help_fn):
                mod_help = get_help_fn() or {}
                if help_key in mod_help:
                    return mod_help[help_key]
        if help_key in SETTINGS_HELP:
            return tr(f"help.{help_key}", SETTINGS_HELP[help_key])
        return tr("help.missing", "Additional description for this parameter is unavailable.")
