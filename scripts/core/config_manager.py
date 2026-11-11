import os
import copy
import json
import uuid
import shutil
import threading
import configparser


def make_default_camera_modules(registry=None) -> dict:
    """Формує типові налаштування модулів камери динамічно з ModuleRegistry."""
    if registry is not None:
        return registry.build_all_default_camera_modules()
    return {}


def make_camera_entry(
    name="Камера 1",
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

    return {
        "uuid": str(cam_uuid or uuid.uuid4()),
        "name": str(name or "Камера"),
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
    # Загальні
    "general.archive_root": (
        "Коренева папка архіву\n"
        "Шлях на диску (рекомендовано місткий або швидкий диск для відеоархіву), "
        "куди зберігаються підпапки камер (<UUID>/<модуль>) та спільна база облич (faces_db/faces.txt)."
    ),
    "general.autostart_os_and_play": (
        "Автозапуск із Windows та автоматичний старт моніторингу\n"
        "Додає програму в автозавантаження поточного користувача Windows (HKCU\\...\\Run) "
        "та автоматично запускає опитування усіх увімкнених камер після старту."
    ),
    "general.kiosk_mode": (
        "Режим кіоску (блокування виходу)\n"
        "Коли увімкнено: блокує Alt+F4 / Esc / хрестик вікна в головному вікні; вихід можливий лише комбінацією Ctrl+Alt+Shift+Q.\n"
        "Коли вимкнено (типово): усі вікна можна вільно перетягувати на інші монітори, змінювати розмір та закривати стандартно."
    ),

    # Сітка
    "grid.mode": (
        "Режим сітки камер\n"
        "• auto — автоматично підбирає кількість рядків і колонок під число активних камер і пропорції вікна (1x1, 2x1, 2x2, 3x2, 3x3 тощо).\n"
        "• manual — фіксована сітка (Рядки × Колонки). Якщо камер більше, з'являється перемикач сторінок ◀ / ▶."
    ),
    "grid.rows": "Кількість рядків у ручному режимі сітки (1–8).",
    "grid.cols": "Кількість колонок у ручному режимі сітки (1–8).",
    "grid.border_color": "Базовий колір рамки кожної плитки камери у сітці (HEX, наприклад #00e5ff).",
    "grid.active_border_color": "Колір рамки тієї камери, для якої зараз відкрите вікно повного перегляду (наприклад #00ff88).",
    "grid.offline_border_color": "Колір рамки плитки, коли втрачено зв'язок із потоком камери (наприклад #ff4444).",
    "grid.border_width": "Товщина рамки плиток у пікселях (1–10 px).",
    "grid.tile_gap": "Проміжок між плитками камер у сітці в пікселях (0–20 px).",
    "grid.show_labels": "Показувати назву/опис камери у верхньому лівому куті плитки сітки.",
    "grid.show_status": "Показувати індикатори стану на плитці: ● REC (запис), активні модулі та статус підключення.",
    "grid.show_fps": "Показувати поточний FPS субпотоку на плитці в сітці.",

    # Залізо та графіка
    "hardware.ai_backend": (
        "Процесор/прискорювач нейромереж (ONNX Runtime)\n"
        "• cpu — максимальна сумісність на будь-якому ПК (використовується за замовчуванням).\n"
        "• auto — автоматично обирає найшвидший доступний прискорювач (CUDA -> DirectML -> OpenVINO -> CPU).\n"
        "• cuda — NVIDIA CUDA GPU (найвища швидкість на картах GeForce/RTX; потребує пакет onnxruntime-gpu і CUDA/cuDNN).\n"
        "• directml — DirectX 12 GPU (працює на NVIDIA, AMD та Intel без встановлення CUDA Toolkit; потребує пакет onnxruntime-directml).\n"
        "• openvino — прискорення на процесорах та інтегрованій графіці Intel."
    ),
    "hardware.ai_threads": (
        "Кількість потоків на ONNX-сесію та AI-воркерів (1–8)\n"
        "Для GPU зазвичай вистачає 1–2 воркери. Для CPU збільшуйте обережно, залишаючи ядра під декодування, кодування відео та інтерфейс."
    ),
    "hardware.ai_max_fps_per_cam": (
        "Макс. частота AI-аналізу на 1 камеру (кадр/сек, 1–30)\n"
        "Обмежує, скільки разів за секунду кожна камера відправляє кадр у нейромережі. Значення 6–10 к/с на камеру "
        "повністю достатньо для впевненого читання номерів і облич, при цьому економить 60–80% ресурсів CPU/GPU при кількох камерах."
    ),
    "hardware.video_decode_mode": (
        "Спосіб декодування вхідних потоків\n"
        "• cpu — стандартний OpenCV VideoCapture (найстабільніше для вебкамер і файлів).\n"
        "• auto / d3d11va / cuda / qsv — апаратне декодування FFmpeg GPU для мережевих RTSP/HTTP потоків (з автоматичним fallback на CPU, якщо камера або драйвер не підтримує)."
    ),
    "hardware.display_grid_fps": "Макс. частота оновлення плиток у сітці (3–30 FPS). Значення 10–15 FPS різко знижує навантаження на UI та CPU на 8–16 камерах.",
    "hardware.display_detail_fps": "Макс. частота оновлення у вікні повного перегляду обраної камери (5–60 FPS).",
    "hardware.display_resize_method": (
        "Алгоритм масштабування кадрів для відображення\n"
        "• nearest — найшвидший (мінімум CPU);\n"
        "• linear — оптимальний баланс якості й швидкості;\n"
        "• lanczos — найвища чіткість для дрібних деталей (вище навантаження CPU)."
    ),
    "hardware.use_opencv_opencl": (
        "OpenCV OpenCL (Transparent API / UMat)\n"
        "Дозволяє прискорювати ресайз/обробку кадрів у OpenCV засобами GPU (якщо драйвер OpenCL доступний і стабільний)."
    ),

    # Камера
    "camera.uuid": "Унікальний ідентифікатор камери (UUID v4). Генерується автоматично при створенні й прив'язує папку архіву — можна міняти назву камери без втрати історії.",
    "camera.name": "Зручний опис камери (наприклад: 'В'їзд КПП', 'Склад північ'). Використовується у сітці, заголовку вікна та пошуку.",
    "camera.enabled": "Увімкнути чи повністю вимкнути цю камеру (вимкнена камера не підключається до потоку і не займає слот у сітці).",
    "camera.main_source": (
        "Основний потік (висока якість)\n"
        "Завжди використовується для відеозапису, AI-розпізнавання та у вікні повного перегляду. "
        "Може бути: rtsp://логін:пароль@ip:554/stream1, http://..., індекс вебкамери (0, 1) або локальний .mp4 файл."
    ),
    "camera.sub_source": (
        "Стиснутий субпотік (низька роздільність для сітки)\n"
        "Відображається ТІЛЬКИ в плитці загальної сітки (наприклад rtsp://.../stream2 або channel=1&subtype=1), щоб не декодувати 4K/FullHD для десятків мініатюр. "
        "Якщо залишити порожнім (для вебкамери 0 або локального файлу), сітка автоматично масштабує кадр з основного потоку."
    ),
}


class ConfigManager:
    def __init__(self, base_dir: str, registry=None):
        self.base_dir = os.path.abspath(base_dir)
        self.registry = registry
        self.config_json_path = os.path.join(self.base_dir, "config.json")
        self.legacy_ini_path = os.path.join(self.base_dir, "config.ini")
        self._lock = threading.Lock()
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
            cfg["cameras"] = [make_camera_entry(name="Камера 1", main_source="0", sub_source="", modules=reg_mods)]
            return cfg

        try:
            parser.read(self.legacy_ini_path, encoding="utf-8-sig")
        except Exception:
            cfg["cameras"] = [make_camera_entry(name="Камера 1", main_source="0", sub_source="", modules=reg_mods)]
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
            name="Камера 1",
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
        if not cfg.get("general", {}).get("archive_root"):
            cfg["general"]["archive_root"] = defaults["general"]["archive_root"]

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
                name=c.get("name") or f"Камера {idx + 1}",
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
            norm_cams = [make_camera_entry(name="Камера 1", main_source="0", sub_source="", modules=reg_mods)]
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

    def save_data(self, new_data: dict):
        with self._lock:
            norm = self._normalize_loaded_config(new_data)
            self.data = norm
            self._save_dict_atomic(norm)

    def save(self):
        with self._lock:
            self._save_dict_atomic(self.data)

    def get_archive_root(self) -> str:
        with self._lock:
            root = self.data.get("general", {}).get("archive_root") or os.path.join(self.base_dir, "archive")
            return os.path.abspath(root)

    def get_faces_db_path(self) -> str:
        db_dir = os.path.join(self.get_archive_root(), "faces_db")
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
            self.data.setdefault("windows", {})[win_key] = copy.deepcopy(geom)
            self._save_dict_atomic(self.data)

    def get_camera_window_geometry(self, cam_uuid: str, kind: str = "window") -> dict:
        with self._lock:
            for c in self.data.get("cameras", []):
                if c.get("uuid") == cam_uuid:
                    return copy.deepcopy(c.get(kind, {}))
        return {}

    def set_camera_window_geometry(self, cam_uuid: str, kind: str, geom: dict):
        with self._lock:
            for c in self.data.get("cameras", []):
                if c.get("uuid") == cam_uuid:
                    c[kind] = copy.deepcopy(geom)
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
        if help_key in SETTINGS_HELP:
            return SETTINGS_HELP[help_key]
        if registry is not None:
            mod_help = registry.get_all_help_entries()
            if help_key in mod_help:
                return mod_help[help_key]
        return "Додатковий опис для цього параметра відсутній."
