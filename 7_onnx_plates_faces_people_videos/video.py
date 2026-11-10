import os
import sys
import time
import threading
import subprocess
from datetime import datetime
import cv2
import numpy as np

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


class VideoEngine:
    """
    Автономний модуль циклічного відеозапису фрагментами заданої тривалості.
    Використовує imageio-ffmpeg (libx264 із налаштованими preset, CRF, keyframe-інтервалом
    та опційним обмеженням бітрейту/висоти кадру) у фоновому потоці
    з резервним fallback на OpenCV mp4v, щоб не навантажувати основний потік камери.
    Записує чисті кадри без рамок детекції у папку video/YYYY-MM-DD/HH-MM-SS.mp4.
    """
    def __init__(self, base_dir: str, config: dict, on_saved_callback=None):
        self.base_dir = base_dir
        self.config = config
        self.on_saved_callback = on_saved_callback

        self.video_dir = os.path.join(self.base_dir, "video")
        os.makedirs(self.video_dir, exist_ok=True)

        self.duration_min = max(0.05, float(self.config.get("video_duration_min", 5.0)))
        self.segment_duration_sec = self.duration_min * 60.0
        self.fps = max(1.0, min(60.0, float(self.config.get("video_fps", 15.0))))
        self.codec = str(self.config.get("video_codec", "libx264")).strip().lower() or "libx264"

        raw_preset = str(self.config.get("video_preset", "veryfast")).strip().lower()
        if raw_preset in ALLOWED_X264_PRESETS:
            self.preset = raw_preset
        else:
            print(f"[VideoEngine] Невідомий preset '{raw_preset}', використано 'veryfast'")
            self.preset = "veryfast"

        self.crf = max(0, min(51, int(round(float(self.config.get("video_crf", 30))))))
        self.keyframe_sec = max(0.5, min(60.0, float(self.config.get("video_keyframe_sec", 6.0))))
        self.max_bitrate_kbps = max(0, int(round(float(self.config.get("video_max_bitrate_kbps", 900)))))
        self.max_height = max(0, int(round(float(self.config.get("video_max_height", 0)))))

        self.ffmpeg_exe = None
        if imageio_ffmpeg is not None:
            try:
                self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception as e:
                print(f"[VideoEngine] imageio-ffmpeg недоступний ({e}), fallback на OpenCV mp4v")

        self.lock = threading.Lock()
        self.is_running = False
        self.latest_frame = None
        self.latest_frame_ts = 0.0
        self.recording_allowed = True

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

    def start(self):
        """Запуск фонового потоку запису відео."""
        old_thread = None
        with self.lock:
            if self.is_running:
                return
            if self.writer_thread is not None and self.writer_thread.is_alive():
                old_thread = self.writer_thread

        if old_thread is not None:
            old_thread.join(timeout=3.0)

        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self.latest_frame = None
            self.latest_frame_ts = 0.0
            self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self.writer_thread.start()

    def stop(self, wait: bool = True):
        """Зупинка запису та коректне закриття активного відеофрагмента."""
        with self.lock:
            self.is_running = False
            w_thread = self.writer_thread

        if wait and w_thread is not None and w_thread.is_alive():
            w_thread.join(timeout=5.0)
            with self.lock:
                if self.writer_thread is w_thread:
                    self.writer_thread = None

        # Дочекатися завершення фіналізації останніх файлів
        if wait:
            with self.lock:
                f_threads = [t for t in self.finalizer_threads if t.is_alive()]
            for t in f_threads:
                t.join(timeout=10.0)

    def push_frame(self, clean_frame: np.ndarray, allowed: bool = True):
        """
        Передача чистого кадру (без рамок детекції) у буфер запису.
        Виконується миттєво без блокування відеопотоку.
        """
        if clean_frame is None:
            return
        with self.lock:
            self.recording_allowed = allowed
            if allowed:
                self.latest_frame = clean_frame
                self.latest_frame_ts = time.time()

    def set_allowed(self, allowed: bool):
        """Оновлення дозволу запису (наприклад, за межами дат)."""
        with self.lock:
            self.recording_allowed = allowed

    def is_file_busy(self, fpath: str) -> bool:
        """Перевірка, чи файл ще записується або фіналізується."""
        norm = os.path.normpath(fpath)
        with self.lock:
            if self.active_filepath and os.path.normpath(self.active_filepath) == norm:
                return True
            if norm in self.finalizing_paths:
                return True
        return False

    def _open_segment(self, width: int, height: int):
        """Відкриває новий файл фрагмента відео у папці поточного дня."""
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
        # Захист від колізії імен при швидкому рестарті в межах однієї секунди
        idx = 1
        while os.path.exists(out_path):
            out_path = os.path.join(day_dir, f"{time_prefix}_{idx}.mp4")
            idx += 1

        gop_frames = max(1, int(round(self.fps * self.keyframe_sec)))
        use_ffmpeg = (self.ffmpeg_exe is not None) and ("mp4v" not in self.codec)
        if use_ffmpeg:
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
                "-vcodec", "libx264",
                "-preset", self.preset,
                "-crf", str(self.crf),
                "-g", str(gop_frames),
            ]
            if self.max_bitrate_kbps > 0:
                cmd.extend([
                    "-maxrate", f"{self.max_bitrate_kbps}k",
                    "-bufsize", f"{self.max_bitrate_kbps * 2}k",
                ])
            if (out_w, out_h) != (even_w, even_h):
                cmd.extend(["-vf", f"scale={out_w}:{out_h}"])
            cmd.extend([
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ])
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags
                )
                self.ffmpeg_proc = proc
                self.cv_writer = None
            except Exception as e:
                print(f"[VideoEngine] Помилка запуску ffmpeg ({e}), перехід на OpenCV mp4v")
                self.ffmpeg_proc = None
                use_ffmpeg = False

        if not use_ffmpeg:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, self.fps, (out_w, out_h))
            if not writer.isOpened():
                print(f"[VideoEngine] Не вдалося відкрити VideoWriter для {out_path}")
                return False
            self.cv_writer = writer
            self.ffmpeg_proc = None

        with self.lock:
            self.active_filepath = out_path
        self.segment_start_ts = time.time()
        self.segment_w = even_w
        self.segment_h = even_h
        self.output_w = out_w
        self.output_h = out_h
        self.current_day_folder = day_folder
        self.current_time_display = time_display
        res_info = f"{out_w}x{out_h}" if (out_w, out_h) == (even_w, even_h) else f"{even_w}x{even_h}->{out_w}x{out_h}"
        if use_ffmpeg:
            cap_info = f", cap={self.max_bitrate_kbps}k" if self.max_bitrate_kbps > 0 else ""
            enc_info = f"libx264 preset={self.preset}, crf={self.crf}, kf={self.keyframe_sec:g}s{cap_info}"
        else:
            enc_info = "mp4v"
        print(f"[ВІДЕОЗАПИС СТАРТ] {out_path} ({res_info} @ {self.fps:g} fps, {enc_info})")
        return True

    def _close_segment_async(self, block: bool = False):
        """Закриває поточний фрагмент і фіналізує файл у фоні без затримки наступного сегмента."""
        proc = self.ffmpeg_proc
        writer = self.cv_writer
        fpath = self.active_filepath
        day_folder = self.current_day_folder
        time_display = self.current_time_display

        self.ffmpeg_proc = None
        self.cv_writer = None
        with self.lock:
            self.active_filepath = None
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
        """Основний цикл запису з точним витримуванням інтервалу кадрів (CFR)."""
        interval = 1.0 / self.fps
        next_tick = time.perf_counter()

        try:
            while True:
                with self.lock:
                    running = self.is_running
                    allowed = self.recording_allowed
                    frame = self.latest_frame
                    frame_ts = self.latest_frame_ts

                if not running:
                    break

                now_wall = time.time()
                # Якщо запис вимкнено по датах або кадрів немає понад 3 с — фіналізуємо сегмент
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
                    (self.ffmpeg_proc is None and self.cv_writer is None)
                    or (even_w != self.segment_w or even_h != self.segment_h)
                    or ((now_wall - self.segment_start_ts) >= self.segment_duration_sec)
                    or (today_str != self.current_day_folder)
                )

                if need_rotate:
                    if self.ffmpeg_proc is not None or self.cv_writer is not None:
                        self._close_segment_async(block=False)
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
                        write_ok = False
                elif self.cv_writer is not None:
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
