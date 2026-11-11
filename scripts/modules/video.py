import os
import re
import sys
import time
import shutil
import importlib
import threading
import subprocess
from datetime import datetime
import numpy as np

from scripts.core.i18n import tr

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

ALLOWED_X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)

KNOWN_ENCODER_CATALOG = (
    ("copy", "copy — Direct RTSP/HTTP stream recording (0% CPU, no re-encoding)"),
    ("libx264", "libx264 — Software CPU H.264 compression (size/quality balance)"),
    ("libx265", "libx265 — Software CPU H.265 / HEVC compression (smaller size)"),
    ("h264_nvenc", "h264_nvenc — Hardware GPU NVIDIA NVENC H.264"),
    ("hevc_nvenc", "hevc_nvenc — Hardware GPU NVIDIA NVENC H.265 / HEVC"),
    ("av1_nvenc", "av1_nvenc — Hardware GPU NVIDIA NVENC AV1"),
    ("h264_qsv", "h264_qsv — Hardware Intel QuickSync H.264"),
    ("hevc_qsv", "hevc_qsv — Hardware Intel QuickSync H.265 / HEVC"),
    ("h264_amf", "h264_amf — Hardware AMD Radeon AMF H.264"),
    ("hevc_amf", "hevc_amf — Hardware AMD Radeon AMF H.265 / HEVC"),
    ("mp4v", "mp4v — Fallback OpenCV codec"),
)

BASE_CODEC_KEYS = ("copy", "libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf", "mp4v")

ALLOWED_VIDEO_CODECS = tuple(k for k, _ in KNOWN_ENCODER_CATALOG)

STREAM_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://", "srt://")

_CAP_LOCK = threading.Lock()
_CACHED_FFMPEG_ENCODERS = None


def _creation_flags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


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


def _is_network_stream(source) -> bool:
    return isinstance(source, str) and source.strip().lower().startswith(STREAM_PREFIXES)


def _get_ffmpeg_exe():
    if imageio_ffmpeg is not None:
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and os.path.exists(exe):
                return exe
        except Exception:
            pass
    return shutil.which("ffmpeg")


def probe_available_ffmpeg_encoders(force: bool = False) -> set:
    """Динамічно визначає перелік відеокодеків у поточному FFmpeg."""
    global _CACHED_FFMPEG_ENCODERS
    with _CAP_LOCK:
        if _CACHED_FFMPEG_ENCODERS is not None and not force:
            return set(_CACHED_FFMPEG_ENCODERS)

    found = set()
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe:
        try:
            res = subprocess.run(
                [ffmpeg_exe, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=_creation_flags(),
            )
            out_txt = res.stdout or ""
            for key, _ in KNOWN_ENCODER_CATALOG:
                if key in ("copy", "mp4v"):
                    continue
                if key in out_txt:
                    found.add(key)
        except Exception:
            pass

    with _CAP_LOCK:
        _CACHED_FFMPEG_ENCODERS = set(found)
    return found


def get_codec_choices() -> list:
    """Повертає актуальний список (key, label) з урахуванням оновленого FFmpeg / OpenCV."""
    detected = probe_available_ffmpeg_encoders(force=False)
    choices = []
    for key, label in KNOWN_ENCODER_CATALOG:
        if key in BASE_CODEC_KEYS or key in detected:
            choices.append((key, tr(f"mod.video.field.codec.choice.{key}", label)))
    return choices


def refresh_capabilities():
    """Перечитує cv2 / imageio_ffmpeg після встановлення або оновлення бібліотек."""
    global cv2, imageio_ffmpeg
    importlib.invalidate_caches()
    for mod_name in ("cv2", "imageio_ffmpeg"):
        try:
            if mod_name in sys.modules:
                mod = importlib.reload(sys.modules[mod_name])
            else:
                mod = importlib.import_module(mod_name)
            if mod_name == "cv2":
                cv2 = mod
            else:
                imageio_ffmpeg = mod
        except Exception:
            pass
    probe_available_ffmpeg_encoders(force=True)


def is_ready(base_dir: str = "") -> bool:
    return bool(_get_ffmpeg_exe()) and (cv2 is not None)


def get_status_summary(base_dir: str = "") -> str:
    ffmpeg_exe = _get_ffmpeg_exe()
    if ffmpeg_exe and cv2 is not None:
        encs = probe_available_ffmpeg_encoders(force=False)
        hw = [e for e in ("h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf") if e in encs]
        hw_str = f", HW: {', '.join(hw)}" if hw else ""
        return tr("mod.video.status.ready", "Ready (FFmpeg + copy{hw_str})", hw_str=hw_str)
    if cv2 is not None:
        return tr("mod.video.status.fallback_opencv", "Fallback mode (OpenCV mp4v only, no FFmpeg)")
    if ffmpeg_exe:
        return tr("mod.video.status.partial_copy", "Partial (FFmpeg copy only, no cv2)")
    return tr("mod.video.status.missing_deps", "Requires imageio-ffmpeg and opencv-python")


def ensure_resources(base_dir: str = "", download: bool = True, progress_cb=None) -> dict:
    """Перевіряє наявність FFmpeg та OpenCV для модуля відеозапису."""
    refresh_capabilities()
    missing = []
    if not _get_ffmpeg_exe():
        missing.append("imageio-ffmpeg (ffmpeg)")
    if cv2 is None:
        missing.append("opencv-python")
    ok = len(missing) == 0
    return {
        "ok": ok,
        "downloaded": [],
        "missing": missing,
        "message": get_status_summary(base_dir) if ok else tr(
            "mod.video.res.missing",
            "Missing video components: {items}",
            items=", ".join(missing),
        ),
    }


def recommend_settings(caps: dict, cam_cfg: dict = None, cur_mod_cfg: dict = None, active_cam_count: int = 1) -> dict:
    """
    Підбирає найоптимальніший кодек запису для камери або шаблону нових камер:
    - для мережевих RTSP/HTTP-камер з FFmpeg: 'copy' (~0% CPU/GPU);
    - для локальних файлів / вебкамер / шаблону: найкращий тестований GPU-кодек (NVENC -> QSV -> AMF) або libx264 / mp4v.
    """
    encoders = (caps or {}).get("video_encoders") or {}

    def _enc_supported(key: str) -> bool:
        if key in encoders:
            return bool((encoders.get(key) or {}).get("supported"))
        if key == "copy":
            return bool(_get_ffmpeg_exe())
        if key == "mp4v":
            return cv2 is not None
        return key in probe_available_ffmpeg_encoders(force=False)

    main_src = str((cam_cfg or {}).get("main_source") or "").strip()
    is_net = _is_network_stream(main_src) if cam_cfg is not None else False

    if is_net and _enc_supported("copy"):
        chosen = "copy"
        reason = tr("mod.video.rec_reason.copy", "direct RTSP/HTTP stream recording without re-encoding (~0% CPU)")
    elif _enc_supported("h264_nvenc"):
        chosen = "h264_nvenc"
        reason = tr("mod.video.rec_reason.nvenc", "hardware NVIDIA NVENC H.264 encoding")
    elif _enc_supported("h264_qsv"):
        chosen = "h264_qsv"
        reason = tr("mod.video.rec_reason.qsv", "hardware Intel QuickSync H.264 encoding")
    elif _enc_supported("h264_amf"):
        chosen = "h264_amf"
        reason = tr("mod.video.rec_reason.amf", "hardware AMD AMF H.264 encoding")
    elif _enc_supported("libx264"):
        chosen = "libx264"
        reason = tr("mod.video.rec_reason.libx264", "software H.264 (FFmpeg libx264)")
    else:
        chosen = "mp4v"
        reason = tr("mod.video.rec_reason.mp4v", "fallback OpenCV mp4v")

    return {
        "updates": {"codec": chosen},
        "notes": [tr("mod.video.rec_note", "Codec: {chosen} ({reason})", chosen=chosen, reason=reason)],
    }


class VideoEngine:
    """
    Автономний модуль циклічного відеозапису фрагментами заданої тривалості.
    Підтримує:
      - прямий запис RTSP-потоку без перекодування ('copy'),
      - апаратне кодування GPU ('h264_nvenc', 'hevc_nvenc', 'av1_nvenc', 'h264_qsv', 'hevc_qsv', 'h264_amf', 'hevc_amf') з fallback на 'libx264',
      - програмне кодування CPU ('libx264', 'libx265') через imageio-ffmpeg,
      - резервний OpenCV 'mp4v'.
    """
    def __init__(
        self,
        base_dir: str,
        config: dict,
        on_saved_callback=None,
        output_dir: str = None,
        source=None,
    ):
        self.base_dir = base_dir
        self.config = dict(config or {})
        self.on_saved_callback = on_saved_callback
        self.source = source

        self.video_dir = output_dir or os.path.join(self.base_dir, "video")
        self._staging_dir = os.path.join(self.video_dir, "_staging")
        self.allow_disk_write = _parse_allow_disk_write(self.config)
        if self.allow_disk_write:
            os.makedirs(self.video_dir, exist_ok=True)

        self.ffmpeg_exe = _get_ffmpeg_exe()
        if not self.ffmpeg_exe:
            print("[VideoEngine] imageio-ffmpeg / ffmpeg недоступний, fallback на OpenCV mp4v")

        self.lock = threading.Lock()
        self.is_running = False
        self.latest_frame = None
        self.latest_frame_ts = 0.0
        self.recording_allowed = self.allow_disk_write

        self.writer_thread = None
        self.finalizer_threads = []

        self.active_filepath = None
        self.finalizing_paths = set()

        self.ffmpeg_proc = None
        self.cv_writer = None
        self.segment_start_ts = 0.0
        self.segment_w = 0
        self.segment_h = 0
        self.output_w = 0
        self.output_h = 0
        self.current_day_folder = ""
        self.current_time_display = ""

        self._active_encoder_label = ""
        self._hw_fallback_triggered = False
        self._copy_restart_requested = False

        self._reload_params()

    def _reload_params(self):
        self.ffmpeg_exe = _get_ffmpeg_exe()
        self.allow_disk_write = _parse_allow_disk_write(self.config)
        self.duration_min = max(0.05, float(self.config.get("video_duration_min", self.config.get("duration_min", 5.0))))
        self.segment_duration_sec = self.duration_min * 60.0
        self.fps = max(1.0, min(60.0, float(self.config.get("video_fps", self.config.get("fps", 15.0)))))
        raw_codec = str(self.config.get("video_codec", self.config.get("codec", "libx264")) or "libx264").strip().lower()
        self.codec = raw_codec if raw_codec in ALLOWED_VIDEO_CODECS else "libx264"

        raw_preset = str(self.config.get("video_preset", self.config.get("preset", "veryfast")) or "veryfast").strip().lower()
        if raw_preset in ALLOWED_X264_PRESETS:
            self.preset = raw_preset
        else:
            self.preset = "veryfast"

        self.crf = max(0, min(51, int(round(float(self.config.get("video_crf", self.config.get("crf", 30)))))))
        self.keyframe_sec = max(0.5, min(60.0, float(self.config.get("video_keyframe_sec", self.config.get("keyframe_sec", 6.0)))))
        self.max_bitrate_kbps = max(0, int(round(float(self.config.get("video_max_bitrate_kbps", self.config.get("max_bitrate_kbps", 900))))))
        self.max_height = max(0, int(round(float(self.config.get("video_max_height", self.config.get("max_height", 0))))))
        self._hw_fallback_triggered = False

    def is_direct_copy_active(self) -> bool:
        """Чи працює зараз прямий запис потоку без декодування/перекодування."""
        return (
            self.codec == "copy"
            and self.ffmpeg_exe is not None
            and _is_network_stream(self.source)
            and not self._hw_fallback_triggered
        )

    def get_active_encoder_label(self) -> str:
        with self.lock:
            return self._active_encoder_label or self.codec

    def is_currently_recording(self) -> bool:
        with self.lock:
            if not self.is_running or not self.recording_allowed or not self.allow_disk_write:
                return False
            return (self.ffmpeg_proc is not None) or (self.cv_writer is not None)

    def update_config(self, config: dict, source=None):
        with self.lock:
            old_sig = (
                self.codec, self.preset, self.crf, self.fps,
                self.segment_duration_sec, self.keyframe_sec,
                self.max_bitrate_kbps, self.max_height, self.source, self.allow_disk_write,
            )
            self.config = dict(config or {})
            if source is not None:
                self.source = source
            self._reload_params()
            if not self.allow_disk_write:
                self.recording_allowed = False
            new_sig = (
                self.codec, self.preset, self.crf, self.fps,
                self.segment_duration_sec, self.keyframe_sec,
                self.max_bitrate_kbps, self.max_height, self.source, self.allow_disk_write,
            )
            if old_sig != new_sig and self.is_running:
                self._copy_restart_requested = True

    def start(self):
        """Запуск фонового потоку запису відео."""
        old_thread = None
        with self.lock:
            if not self.allow_disk_write:
                self.recording_allowed = False
                return
            if self.is_running:
                return
            if self.writer_thread is not None and self.writer_thread.is_alive():
                old_thread = self.writer_thread

        if old_thread is not None:
            old_thread.join(timeout=3.0)

        with self.lock:
            if not self.allow_disk_write:
                self.recording_allowed = False
                return
            if self.is_running:
                return
            self.is_running = True
            self.latest_frame = None
            self.latest_frame_ts = 0.0
            self._hw_fallback_triggered = False
            self._copy_restart_requested = False
            target_fn = self._direct_copy_loop if self.is_direct_copy_active() else self._writer_loop
            self.writer_thread = threading.Thread(target=target_fn, daemon=True)
            self.writer_thread.start()

    def stop(self, wait: bool = True):
        """Зупинка запису та коректне закриття активного відеофрагмента."""
        with self.lock:
            self.is_running = False
            w_thread = self.writer_thread

        if wait and w_thread is not None and w_thread.is_alive():
            w_thread.join(timeout=6.0)
            with self.lock:
                if self.writer_thread is w_thread:
                    self.writer_thread = None

        if wait:
            with self.lock:
                f_threads = [t for t in self.finalizer_threads if t.is_alive()]
            for t in f_threads:
                t.join(timeout=10.0)

    def restart_if_mode_changed(self):
        """Перезапускає воркер при перемиканні між 'copy' і покадровим кодуванням."""
        with self.lock:
            was_running = self.is_running
            allow_write = self.allow_disk_write
        if was_running:
            self.stop(wait=True)
            if allow_write:
                self.start()

    def push_frame(self, clean_frame: np.ndarray, allowed: bool = True):
        """Передача чистого кадру (без рамок детекції) у буфер запису."""
        if clean_frame is None:
            return
        with self.lock:
            eff_allowed = bool(allowed) and bool(self.allow_disk_write)
            self.recording_allowed = eff_allowed
            if eff_allowed and not self.is_direct_copy_active():
                self.latest_frame = clean_frame
                self.latest_frame_ts = time.time()

    def set_allowed(self, allowed: bool):
        """Оновлення дозволу запису (наприклад, за межами дат або пауза)."""
        with self.lock:
            self.recording_allowed = bool(allowed) and bool(self.allow_disk_write)

    def is_file_busy(self, fpath: str) -> bool:
        """Перевірка, чи файл ще записується або фіналізується."""
        norm = os.path.normpath(fpath)
        if os.path.normpath(self._staging_dir) in norm:
            return True
        with self.lock:
            if self.active_filepath and os.path.normpath(self.active_filepath) == norm:
                return True
            if norm in self.finalizing_paths:
                return True
        return False

    # ------------------------------------------------------------------
    #  1. Режим прямого копіювання RTSP/URL потоку ('copy') без перекодування
    # ------------------------------------------------------------------
    def _flush_staging_files(self, exclude_path: str = None):
        if not os.path.isdir(self._staging_dir):
            return
        try:
            names = sorted(os.listdir(self._staging_dir))
        except Exception:
            return
        for fname in names:
            if not fname.lower().endswith(".mp4"):
                continue
            src = os.path.join(self._staging_dir, fname)
            if exclude_path and os.path.normpath(src) == os.path.normpath(exclude_path):
                continue
            self._finalize_copied_segment(src)

    def _finalize_copied_segment(self, src_path: str):
        if not self.allow_disk_write:
            return
        if not src_path or not os.path.exists(src_path):
            return
        try:
            size_bytes = os.path.getsize(src_path)
        except Exception:
            return
        if size_bytes < 1024:
            try:
                os.remove(src_path)
            except Exception:
                pass
            return

        base = os.path.basename(src_path)
        stem, ext = os.path.splitext(base)
        if "_" in stem and len(stem.split("_")[0]) == 10:
            day_folder, time_prefix = stem.split("_", 1)
        else:
            now_dt = datetime.now()
            day_folder = now_dt.strftime("%Y-%m-%d")
            time_prefix = now_dt.strftime("%H-%M-%S")

        day_dir = os.path.join(self.video_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)
        dest_path = os.path.join(day_dir, f"{time_prefix}{ext}")
        idx = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(day_dir, f"{time_prefix}_{idx}{ext}")
            idx += 1

        try:
            shutil.move(src_path, dest_path)
        except Exception as e:
            print(f"[VideoEngine Copy Move Error]: {e}")
            return

        time_display = time_prefix.replace("-", ":")
        print(f"[ВІДЕОЗАПИС COPY ЗБЕРЕЖЕНО] {dest_path}")
        if self.on_saved_callback:
            try:
                self.on_saved_callback(day_folder, time_display, os.path.basename(dest_path), dest_path)
            except Exception:
                pass

    def _direct_copy_loop(self):
        with self.lock:
            allow_write_init = self.allow_disk_write
        if not allow_write_init:
            return
        os.makedirs(self._staging_dir, exist_ok=True)
        self._flush_staging_files()
        creationflags = _creation_flags()

        try:
            while True:
                with self.lock:
                    running = self.is_running
                    allowed = self.recording_allowed
                    allow_write = self.allow_disk_write
                    src_url = str(self.source or "").strip()
                    seg_sec = max(3.0, float(self.segment_duration_sec))
                    self._copy_restart_requested = False

                if not running or not allow_write:
                    break
                if not allowed or not _is_network_stream(src_url) or self.codec != "copy" or not self.ffmpeg_exe:
                    if self.codec != "copy":
                        break
                    time.sleep(0.25)
                    continue

                out_pattern = os.path.join(self._staging_dir, "%Y-%m-%d_%H-%M-%S.mp4")
                cmd = [self.ffmpeg_exe, "-y"]
                if src_url.lower().startswith(("rtsp://", "rtsps://")):
                    cmd.extend(["-rtsp_transport", "tcp"])
                cmd.extend([
                    "-i", src_url,
                    "-an",
                    "-c:v", "copy",
                    "-f", "segment",
                    "-segment_time", f"{seg_sec:.2f}",
                    "-reset_timestamps", "1",
                    "-segment_format", "mp4",
                    "-segment_format_options", "movflags=+frag_keyframe+empty_moov+default_base_moof",
                    "-segment_list", "pipe:1",
                    "-segment_list_type", "flat",
                    "-strftime", "1",
                    out_pattern,
                ])

                proc = None
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        creationflags=creationflags,
                    )
                    with self.lock:
                        self.ffmpeg_proc = proc
                        self._active_encoder_label = "copy"
                    print(f"[ВІДЕОЗАПИС COPY СТАРТ] {src_url} (сегмент {self.duration_min:g} хв)")
                except Exception as e:
                    print(f"[VideoEngine Copy Launch Error]: {e}")
                    time.sleep(2.0)
                    continue

                start_ts = time.time()

                def _read_stdout():
                    try:
                        for line in proc.stdout:
                            clean_line = line.strip()
                            if clean_line:
                                seg_path = (
                                    clean_line
                                    if os.path.isabs(clean_line)
                                    else os.path.join(self._staging_dir, clean_line)
                                )
                                self._finalize_copied_segment(seg_path)
                    except Exception:
                        pass

                reader_t = threading.Thread(target=_read_stdout, daemon=True)
                reader_t.start()

                while proc.poll() is None:
                    with self.lock:
                        r_now = self.is_running
                        a_now = self.recording_allowed
                        w_now = self.allow_disk_write
                        rst_now = self._copy_restart_requested
                        codec_now = self.codec
                    if (not r_now) or (not a_now) or (not w_now) or rst_now or (codec_now != "copy"):
                        try:
                            if proc.stdin:
                                proc.stdin.write("q\n")
                                proc.stdin.flush()
                        except Exception:
                            pass
                        try:
                            proc.wait(timeout=4.0)
                        except Exception:
                            try:
                                proc.terminate()
                                proc.wait(timeout=2.0)
                            except Exception:
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                        break
                    time.sleep(0.25)

                reader_t.join(timeout=2.0)
                with self.lock:
                    if self.ffmpeg_proc is proc:
                        self.ffmpeg_proc = None
                    self._active_encoder_label = ""

                self._flush_staging_files()

                with self.lock:
                    still_running = self.is_running
                    still_allow = self.allow_disk_write
                    cur_codec = self.codec

                if not still_running or not still_allow:
                    break
                if cur_codec != "copy":
                    self._writer_loop()
                    return

                if (time.time() - start_ts) < 1.5:
                    time.sleep(2.0)
                else:
                    time.sleep(1.0)
        finally:
            self._flush_staging_files()

    # ------------------------------------------------------------------
    #  2. Покадрове кодування (GPU NVENC/QSV/AMF або CPU libx264/libx265 / mp4v)
    # ------------------------------------------------------------------
    def _build_ffmpeg_encoder_args(self, codec_name: str, even_w: int, even_h: int, out_w: int, out_h: int):
        gop_frames = max(1, int(round(self.fps * self.keyframe_sec)))
        args = []

        if codec_name in ("h264_nvenc", "hevc_nvenc", "av1_nvenc"):
            nv_preset_map = {
                "ultrafast": "p1",
                "superfast": "p2",
                "veryfast": "p3",
                "faster": "p4",
                "fast": "p4",
                "medium": "p5",
                "slow": "p6",
                "slower": "p7",
                "veryslow": "p7",
            }
            nv_preset = nv_preset_map.get(self.preset, "p3")
            args.extend([
                "-vcodec", codec_name,
                "-preset", nv_preset,
                "-rc", "vbr",
                "-cq", str(self.crf),
                "-b:v", "0",
                "-g", str(gop_frames),
            ])
        elif codec_name in ("h264_qsv", "hevc_qsv"):
            qsv_preset = "veryfast" if self.preset in ("ultrafast", "superfast", "veryfast") else "medium"
            args.extend([
                "-vcodec", codec_name,
                "-preset", qsv_preset,
                "-global_quality", str(self.crf),
                "-g", str(gop_frames),
            ])
        elif codec_name in ("h264_amf", "hevc_amf"):
            amf_quality = "speed" if self.preset in ("ultrafast", "superfast", "veryfast", "faster") else "balanced"
            args.extend([
                "-vcodec", codec_name,
                "-quality", amf_quality,
                "-rc", "cqp",
                "-qp_i", str(self.crf),
                "-qp_p", str(self.crf),
                "-g", str(gop_frames),
            ])
        elif codec_name == "libx265":
            args.extend([
                "-vcodec", "libx265",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-g", str(gop_frames),
            ])
        else:
            args.extend([
                "-vcodec", "libx264",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-g", str(gop_frames),
            ])

        if self.max_bitrate_kbps > 0:
            args.extend([
                "-maxrate", f"{self.max_bitrate_kbps}k",
                "-bufsize", f"{self.max_bitrate_kbps * 2}k",
            ])
        if (out_w, out_h) != (even_w, even_h):
            args.extend(["-vf", f"scale={out_w}:{out_h}"])
        args.extend([
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ])
        return args

    def _open_segment(self, width: int, height: int):
        if not self.allow_disk_write:
            return False
        even_w = width - (width % 2)
        even_h = height - (height % 2)
        if even_w < 16 or even_h < 16:
            return False

        out_w, out_h = even_w, even_h
        if self.max_height > 0 and even_h > self.max_height:
            out_h = max(16, self.max_height - (self.max_height % 2))
            scaled_w = int(round(even_w * (out_h / float(even_h))))
            out_w = max(16, scaled_w - (scaled_w % 2))

        now_dt = datetime.now()
        day_folder = now_dt.strftime("%Y-%m-%d")
        time_prefix = now_dt.strftime("%H-%M-%S")
        time_display = now_dt.strftime("%H:%M:%S")

        day_dir = os.path.join(self.video_dir, day_folder)
        os.makedirs(day_dir, exist_ok=True)

        out_path = os.path.join(day_dir, f"{time_prefix}.mp4")
        idx = 1
        while os.path.exists(out_path):
            out_path = os.path.join(day_dir, f"{time_prefix}_{idx}.mp4")
            idx += 1

        target_codec = "libx264" if (self.codec == "copy" or self._hw_fallback_triggered) else self.codec
        use_ffmpeg = (self.ffmpeg_exe is not None) and ("mp4v" not in target_codec)
        used_label = "mp4v"

        if use_ffmpeg:
            creationflags = _creation_flags()
            codecs_to_try = [target_codec]
            if target_codec != "libx264":
                codecs_to_try.append("libx264")

            opened = False
            for enc in codecs_to_try:
                cmd = [
                    self.ffmpeg_exe,
                    "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{even_w}x{even_h}",
                    "-r", f"{self.fps:.3f}",
                    "-i", "-",
                    "-an",
                ]
                cmd.extend(self._build_ffmpeg_encoder_args(enc, even_w, even_h, out_w, out_h))
                cmd.append(out_path)
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=creationflags,
                    )
                    time.sleep(0.08)
                    if proc.poll() is not None:
                        raise RuntimeError(f"encoder {enc} exited with code {proc.returncode}")
                    self.ffmpeg_proc = proc
                    self.cv_writer = None
                    used_label = enc
                    opened = True
                    if enc != target_codec:
                        self._hw_fallback_triggered = True
                    break
                except Exception as e:
                    print(f"[VideoEngine] Не вдалося запустити '{enc}' ({e}), пробуємо резерв")
                    self.ffmpeg_proc = None
            if not opened:
                use_ffmpeg = False

        if not use_ffmpeg:
            if cv2 is None:
                print("[VideoEngine] OpenCV недоступний для резервного mp4v запису")
                return False
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, self.fps, (out_w, out_h))
            if not writer.isOpened():
                print(f"[VideoEngine] Не вдалося відкрити VideoWriter для {out_path}")
                return False
            self.cv_writer = writer
            self.ffmpeg_proc = None
            used_label = "mp4v"

        with self.lock:
            self.active_filepath = out_path
            self._active_encoder_label = used_label
        self.segment_start_ts = time.time()
        self.segment_w = even_w
        self.segment_h = even_h
        self.output_w = out_w
        self.output_h = out_h
        self.current_day_folder = day_folder
        self.current_time_display = time_display

        res_info = f"{out_w}x{out_h}" if (out_w, out_h) == (even_w, even_h) else f"{even_w}x{even_h}->{out_w}x{out_h}"
        print(f"[ВІДЕОЗАПИС СТАРТ] {out_path} ({res_info} @ {self.fps:g} fps, {used_label})")
        return True

    def _close_segment_async(self, block: bool = False):
        proc = self.ffmpeg_proc
        writer = self.cv_writer
        fpath = self.active_filepath
        day_folder = self.current_day_folder
        time_display = self.current_time_display

        self.ffmpeg_proc = None
        self.cv_writer = None
        with self.lock:
            self.active_filepath = None
            self._active_encoder_label = ""
            if fpath:
                self.finalizing_paths.add(os.path.normpath(fpath))

        if proc is None and writer is None:
            return

        def _finalize():
            try:
                if proc is not None:
                    try:
                        if proc.stdin:
                            proc.stdin.flush()
                            proc.stdin.close()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=15)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                if writer is not None:
                    try:
                        writer.release()
                    except Exception:
                        pass
            finally:
                if fpath:
                    norm = os.path.normpath(fpath)
                    with self.lock:
                        self.finalizing_paths.discard(norm)
                    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                        print(f"[ВІДЕОЗАПИС ЗБЕРЕЖЕНО] {fpath}")
                        if self.on_saved_callback:
                            try:
                                self.on_saved_callback(day_folder, time_display, os.path.basename(fpath), fpath)
                            except Exception:
                                pass

        if block:
            _finalize()
        else:
            t = threading.Thread(target=_finalize, daemon=True)
            with self.lock:
                self.finalizer_threads = [ft for ft in self.finalizer_threads if ft.is_alive()]
                self.finalizer_threads.append(t)
            t.start()

    def _writer_loop(self):
        interval = 1.0 / max(1.0, self.fps)
        next_tick = time.perf_counter()

        try:
            while True:
                with self.lock:
                    running = self.is_running
                    allowed = self.recording_allowed
                    allow_write = self.allow_disk_write
                    frame = self.latest_frame
                    frame_ts = self.latest_frame_ts
                    restart_req = self._copy_restart_requested
                    self._copy_restart_requested = False

                if not running or not allow_write:
                    break

                if self.is_direct_copy_active():
                    self._close_segment_async(block=True)
                    self._direct_copy_loop()
                    return

                now_wall = time.time()
                if not allowed or frame is None or (now_wall - frame_ts) > 3.0:
                    if self.ffmpeg_proc is not None or self.cv_writer is not None:
                        self._close_segment_async(block=False)
                    time.sleep(0.1)
                    next_tick = time.perf_counter()
                    continue

                fh, fw = frame.shape[:2]
                even_w = fw - (fw % 2)
                even_h = fh - (fh % 2)
                today_str = datetime.now().strftime("%Y-%m-%d")

                need_rotate = (
                    restart_req
                    or (self.ffmpeg_proc is None and self.cv_writer is None)
                    or (even_w != self.segment_w or even_h != self.segment_h)
                    or ((now_wall - self.segment_start_ts) >= self.segment_duration_sec)
                    or (today_str != self.current_day_folder)
                )

                if need_rotate:
                    if self.ffmpeg_proc is not None or self.cv_writer is not None:
                        self._close_segment_async(block=False)
                    interval = 1.0 / max(1.0, self.fps)
                    if not self._open_segment(fw, fh):
                        time.sleep(0.5)
                        next_tick = time.perf_counter()
                        continue

                frame_to_write = frame[:self.segment_h, :self.segment_w]

                write_ok = True
                if self.ffmpeg_proc is not None:
                    try:
                        self.ffmpeg_proc.stdin.write(np.ascontiguousarray(frame_to_write).tobytes())
                    except Exception as e:
                        print(f"[VideoEngine Write Error]: {e}")
                        if self._active_encoder_label != "libx264":
                            self._hw_fallback_triggered = True
                        write_ok = False
                elif self.cv_writer is not None and cv2 is not None:
                    try:
                        out_frame = frame_to_write
                        if (self.output_w, self.output_h) != (self.segment_w, self.segment_h):
                            out_frame = cv2.resize(frame_to_write, (self.output_w, self.output_h), interpolation=cv2.INTER_AREA)
                        self.cv_writer.write(out_frame)
                    except Exception as e:
                        print(f"[VideoEngine OpenCV Write Error]: {e}")
                        write_ok = False

                if not write_ok:
                    self._close_segment_async(block=False)

                next_tick += interval
                sleep_dur = next_tick - time.perf_counter()
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
                elif sleep_dur < -interval * 3:
                    next_tick = time.perf_counter()
        finally:
            self._close_segment_async(block=True)


# ------------------------------------------------------------------
#  Контракт незалежного модуля для ModuleRegistry
# ------------------------------------------------------------------
def _create_engine(base_dir, flat_cfg, output_dir, shared_resources=None, on_saved_callback=None, source=None, app_context=None):
    eff_cfg = _merge_runtime_write_cfg(flat_cfg, app_context)
    return VideoEngine(
        base_dir=base_dir,
        config=eff_cfg,
        on_saved_callback=on_saved_callback,
        output_dir=output_dir,
        source=source,
    )


def _update_engine_config(engine, flat_cfg, source=None, shared_resources=None, app_context=None):
    if engine is not None:
        eff_cfg = _merge_runtime_write_cfg(flat_cfg, app_context)
        engine.update_config(eff_cfg, source=source)
        if not engine.allow_disk_write:
            engine.set_allowed(False)
            engine.stop(wait=False)


def _parse_archive_record(fpath: str, base_name: str, root_dir: str, rel_dir: str) -> dict:
    time_parts = base_name.split("_")[0].split("-")
    time_disp = ":".join(time_parts[:3]) if len(time_parts) >= 3 else "--:--:--"
    try:
        mb = os.path.getsize(fpath) / (1024.0 * 1024.0)
    except Exception:
        mb = 0.0
    return {
        "time": time_disp,
        "name": tr("mod.video.archive.record_name", "Video ({mb:.1f} MB)", mb=mb),
    }


_DAY_FOLDER_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SEG_TIME_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})")
_VIDEO_DUR_LOCK = threading.Lock()
_VIDEO_DUR_CACHE = {}  # (fpath, size_bytes, mtime_ns) -> float duration_sec


def _probe_video_duration_sec(fpath: str, size_bytes: int, mtime_ns: int, start_ts: float, mtime_sec: float) -> float:
    key = (os.path.normpath(fpath), int(size_bytes), int(mtime_ns))
    with _VIDEO_DUR_LOCK:
        cached = _VIDEO_DUR_CACHE.get(key)
    if cached is not None:
        return cached

    dur = None
    if cv2 is not None and size_bytes > 0:
        cap = None
        try:
            cap = cv2.VideoCapture(fpath)
            if cap is not None and cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                cnt = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if 0.1 < fps <= 240.0 and cnt > 0.0:
                    dur = cnt / fps
        except Exception:
            dur = None
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    if dur is None or dur <= 0.0 or np.isnan(dur):
        diff = mtime_sec - start_ts
        dur = diff if (1.0 <= diff <= 14400.0) else 300.0

    dur = max(0.5, float(dur))
    with _VIDEO_DUR_LOCK:
        if len(_VIDEO_DUR_CACHE) > 2048:
            _VIDEO_DUR_CACHE.clear()
        _VIDEO_DUR_CACHE[key] = dur
    return dur


def list_video_segments(cam_archive_dir: str, target_timestamps=None, margin_sec: float = 2.0) -> list:
    """
    Повертає список відеофрагментів камери з точним часом початку/кінця (start_ts, end_ts).
    Якщо передано target_timestamps, ефективно фільтрує й детально аналізує лише сегменти,
    які перетинають хоча б одну з вказаних міток часу.
    """
    video_root = os.path.join(str(cam_archive_dir or ""), "video")
    if not video_root or not os.path.isdir(video_root):
        return []

    ts_list = [float(t) for t in target_timestamps] if target_timestamps is not None else None
    if ts_list is not None and not ts_list:
        return []

    out = []
    for root_dir, _, files in os.walk(video_root):
        if "_staging" in root_dir:
            continue
        rel_dir = os.path.relpath(root_dir, video_root)
        day_candidate = rel_dir.split(os.sep)[0] if rel_dir != "." else ""
        for fname in files:
            if not fname.lower().endswith((".mp4", ".avi", ".mkv")):
                continue
            fpath = os.path.join(root_dir, fname)
            try:
                st = os.stat(fpath)
            except Exception:
                continue
            if st.st_size < 256:
                continue

            mtime_sec = float(st.st_mtime)
            day_str = day_candidate if _DAY_FOLDER_RE.match(day_candidate) else datetime.fromtimestamp(mtime_sec).strftime("%Y-%m-%d")
            stem = os.path.splitext(fname)[0]
            m_t = _SEG_TIME_RE.match(stem)
            if m_t:
                hh, mm, ss = m_t.group(1), m_t.group(2), m_t.group(3)
                time_str = f"{hh}:{mm}:{ss}"
                try:
                    start_ts = float(datetime.strptime(f"{day_str} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S").timestamp())
                except Exception:
                    start_ts = mtime_sec
            else:
                dt_m = datetime.fromtimestamp(mtime_sec)
                day_str = dt_m.strftime("%Y-%m-%d")
                time_str = dt_m.strftime("%H:%M:%S")
                start_ts = mtime_sec

            if ts_list is not None:
                coarse_end = mtime_sec if mtime_sec > (start_ts + 1.0) else (start_ts + 1800.0)
                if not any((start_ts - margin_sec) <= t <= (coarse_end + margin_sec + 5.0) for t in ts_list):
                    continue

            dur_sec = _probe_video_duration_sec(fpath, st.st_size, st.st_mtime_ns, start_ts, mtime_sec)
            end_ts = start_ts + dur_sec

            hits = []
            if ts_list is not None:
                hits = sorted(t for t in ts_list if (start_ts - margin_sec) <= t <= (end_ts + margin_sec))
                if not hits:
                    continue

            out.append({
                "path": fpath,
                "filename": fname,
                "day": day_str,
                "time": time_str,
                "start_ts": float(start_ts),
                "end_ts": float(end_ts),
                "duration_sec": float(dur_sec),
                "size_mb": round(st.st_size / (1024.0 * 1024.0), 2),
                "hit_timestamps": hits,
            })

    out.sort(key=lambda s: (s["start_ts"], s["filename"]))
    return out


MODULE_SPEC = {
    "id": "video",
    "version": "2.0.0",
    "order": 40,
    "kind": "recorder",
    "writes_to_disk": True,
    "name": "Video Recording",
    "icon": "🎬",
    "short_badge": "",
    "description": "Continuous cyclic background video recording of the main stream (main_source) without overlays.",
    "on_help": "Enable continuous cyclic background video recording of this camera's main stream (main_source) without overlays.",
    "provides": ["video_recorder"],
    "library_deps": ["imageio-ffmpeg", "opencv-python"],
    "settings_schema": [
        {
            "key": "codec",
            "type": "choice",
            "default": "libx264",
            "label": "Encoder / Codec:",
            "choices_provider": get_codec_choices,
            "help": (
                "Video Recording Processor & Codec\n"
                "• copy — DIRECT RTSP/HTTP STREAM RECORDING without re-encoding! Almost 0% CPU and GPU load. "
                "Ignores CRF/FPS/Preset/Scale and splits on camera keyframes (not suitable for USB webcams like 0).\n"
                "• h264_nvenc / hevc_nvenc / av1_nvenc — NVIDIA GPU hardware encoding (minimal CPU).\n"
                "• h264_qsv / hevc_qsv — Intel QuickSync hardware encoding.\n"
                "• h264_amf / hevc_amf — AMD Radeon hardware encoding.\n"
                "• libx264 / libx265 — software CPU compression.\n"
                "• mp4v — fallback OpenCV codec."
            ),
        },
        {
            "key": "duration_min",
            "type": "float",
            "default": 5.0,
            "min": 0.05,
            "max": 180.0,
            "step": 1.0,
            "label": "Segment Duration (min):",
            "help": "Duration of a single video segment file in minutes (default 5 min; 1 or 10 supported).",
        },
        {
            "key": "preset",
            "type": "choice",
            "default": "veryfast",
            "label": "Compression Speed Preset:",
            "choices": [(p, p) for p in ALLOWED_X264_PRESETS],
            "disable_when": {"codec": "copy"},
            "help": "Compression speed/efficiency (veryfast is optimal; ultrafast yields 4–5× larger files). Inactive in 'copy' mode.",
        },
        {
            "key": "crf",
            "type": "int",
            "default": 30,
            "min": 0,
            "max": 51,
            "step": 1,
            "label": "Quality CRF / CQ (0–51):",
            "disable_when": {"codec": "copy"},
            "help": "Video quality (0–51; lower = sharper and larger file). 28–31 is optimal for CCTV. Inactive in 'copy' mode.",
        },
        {
            "key": "fps",
            "type": "float",
            "default": 15.0,
            "min": 1.0,
            "max": 60.0,
            "step": 1.0,
            "label": "Frame Rate (FPS):",
            "disable_when": {"codec": "copy"},
            "help": "Target frame rate of archived MP4 (1–60; 12–15 fps saves disk and CPU). Inactive in 'copy' mode.",
        },
        {
            "key": "keyframe_sec",
            "type": "float",
            "default": 6.0,
            "min": 0.5,
            "max": 60.0,
            "step": 0.5,
            "label": "Keyframe Interval (sec):",
            "disable_when": {"codec": "copy"},
            "help": "Interval between keyframes in seconds (5–8 s optimal). More frequent keyframes allow faster seeking (+25% size). Inactive in 'copy'.",
        },
        {
            "key": "max_bitrate_kbps",
            "type": "int",
            "default": 900,
            "min": 0,
            "max": 50000,
            "step": 100,
            "label": "Max Bitrate (kbps, 0=unlimited):",
            "disable_when": {"codec": "copy"},
            "help": "Bitrate cap (kbps, 0 = unlimited). Prevents night sensor noise from inflating file size. Inactive in 'copy'.",
        },
        {
            "key": "max_height",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 4320,
            "step": 120,
            "label": "Max Height (px, 0=original):",
            "disable_when": {"codec": "copy"},
            "help": "Maximum height of archived MP4 (px; 0 = original camera height; 720 or 480 lowers size and CPU without affecting AI). Inactive in 'copy'.",
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
            "help": "Automatically delete video clips from this camera older than N days (0 = keep forever).",
        },
        {
            "key": "date_from",
            "key_to": "date_to",
            "type": "date_range",
            "default": "",
            "default_to": "",
            "label": "Active Schedule (Dates):",
            "help": "Start and end dates for video recording (YYYY-MM-DD, empty = unlimited).",
        },
    ],
    "notices": [
        {
            "after_key": "codec",
            "when": {"codec": "copy"},
            "text": (
                "⚡ DIRECT STREAM COPY ACTIVE (copy): FFmpeg saves the incoming RTSP/HTTP stream without decoding or re-encoding (~0% CPU/GPU). "
                "Preset, CRF, FPS, Keyframe, Max Bitrate, and Max Height do not apply. Do not use 'copy' for webcams (0)."
            ),
            "fg": "#00e5ff",
            "bg": "#1b2c3b",
        }
    ],
    "archive": {
        "has_archive": True,
        "button_label": "🎬 Video",
        "sidebar_title": "Video Recording Archive",
        "icon": "🎬",
        "record_type": "video",
        "extensions": (".mp4", ".avi", ".mkv"),
        "parse_record": _parse_archive_record,
        "list_segments": list_video_segments,
    },
    "is_ready": is_ready,
    "refresh_capabilities": refresh_capabilities,
    "get_status_summary": get_status_summary,
    "ensure_resources": ensure_resources,
    "recommend_settings": recommend_settings,
    "create_shared_resources": None,
    "create_engine": _create_engine,
    "update_engine_config": _update_engine_config,
    "start_engine": lambda eng: eng.start() if eng is not None else None,
    "stop_engine": lambda eng, wait=True: eng.stop(wait=wait) if eng is not None else None,
    "restart_if_mode_changed": lambda eng: eng.restart_if_mode_changed() if eng is not None else None,
    "set_allowed": lambda eng, allowed: eng.set_allowed(allowed) if eng is not None else None,
    "push_frame": lambda eng, frame, allowed=True: eng.push_frame(frame, allowed=allowed) if eng is not None else None,
    "needs_decoded_frames": lambda eng, cfg, src: not (eng is not None and eng.is_direct_copy_active()),
    "is_recording": lambda eng: bool(eng is not None and eng.is_currently_recording()),
    "get_active_encoder_label": lambda eng: eng.get_active_encoder_label() if eng is not None else "",
    "is_file_busy": lambda eng, fpath: bool(eng is not None and eng.is_file_busy(fpath)),
}
