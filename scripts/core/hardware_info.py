import os
import sys
import ctypes
import shutil
import string
import importlib
import subprocess
import threading
import importlib.metadata
from ctypes import wintypes

try:
    import psutil
except ImportError:
    psutil = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    import cv2
except ImportError:
    cv2 = None


# Відстеження версій завантажених у пам'ять бібліотек vs версій на диску
# (для виявлення оновлень через pip або .pyd/.dll, що потребують перезапуску)
_TRACKED_LIBS = [
    ("onnxruntime", ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-openvino"), True),
    ("cv2", ("opencv-python", "opencv-contrib-python", "opencv-python-headless"), True),
    ("imageio_ffmpeg", ("imageio-ffmpeg",), False),
    ("fast_alpr", ("fast-alpr",), False),
    ("fast_plate_ocr", ("fast-plate-ocr",), False),
    ("psutil", ("psutil",), True),
    ("numpy", ("numpy",), True),
    ("PIL", ("pillow",), True),
    ("yaml", ("pyyaml",), False),
]

_LOADED_MEM_STATE = {}  # imp_name -> (disk_pkg_name, disk_version)
_MEM_STATE_LOCK = threading.Lock()


def _creation_flags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _get_disk_dists_map() -> dict:
    """Повертає {lower_name: version_str} з диск-метаданих Python-пакетів."""
    out = {}
    try:
        importlib.invalidate_caches()
        for d in importlib.metadata.distributions():
            name = (d.metadata.get("Name") or "").strip().lower()
            ver = (d.version or "").strip()
            if name:
                out[name] = ver
    except Exception:
        pass
    return out


def observe_loaded_libraries():
    """Фіксує версію кожного відстежуваного пакета на момент його першого потрапляння в sys.modules."""
    dists = None
    with _MEM_STATE_LOCK:
        for imp_name, dist_cands, _is_native in _TRACKED_LIBS:
            if imp_name in sys.modules and imp_name not in _LOADED_MEM_STATE:
                if dists is None:
                    dists = _get_disk_dists_map()
                matched_pkg = ""
                matched_ver = getattr(sys.modules[imp_name], "__version__", "") or ""
                for dc in dist_cands:
                    if dc in dists:
                        matched_pkg = dc
                        matched_ver = matched_ver or dists[dc]
                        break
                _LOADED_MEM_STATE[imp_name] = (matched_pkg or imp_name, str(matched_ver))


observe_loaded_libraries()


def _reset_pure_module_loaded_state(imp_name: str, pkg_name: str, ver: str):
    with _MEM_STATE_LOCK:
        _LOADED_MEM_STATE[imp_name] = (pkg_name, str(ver or ""))


def get_library_versions(extra_pkgs=None) -> list:
    """
    Повертає список стану ключових Python-бібліотек (та залежностей із ModuleRegistry):
    disk_pkg, disk_version, loaded_pkg, loaded_version, in_memory, needs_restart, status.
    """
    observe_loaded_libraries()
    dists = _get_disk_dists_map()
    seen_groups = set()
    rows = []

    with _MEM_STATE_LOCK:
        mem_snap = dict(_LOADED_MEM_STATE)

    for imp_name, dist_cands, is_native in _TRACKED_LIBS:
        seen_groups.update(dist_cands)
        cur_pkg = ""
        cur_ver = ""
        for dc in dist_cands:
            if dc in dists:
                cur_pkg = dc
                cur_ver = dists[dc]
                break

        in_mem = imp_name in sys.modules
        l_pkg, l_ver = mem_snap.get(imp_name, ("", ""))

        # Визначаємо, чи змінився пакет на диску відносно вже завантаженого у RAM C-розширення
        needs_restart = False
        if in_mem and is_native and (l_pkg or l_ver):
            if not cur_pkg:
                needs_restart = True
            elif (cur_pkg != l_pkg) or (l_ver and cur_ver and cur_ver != l_ver):
                needs_restart = True

        if not cur_pkg:
            status = "Не встановлено (у пам'яті стара копія)" if in_mem else "Не встановлено"
        elif needs_restart:
            status = f"Оновлено на диску ({l_pkg} {l_ver} -> {cur_pkg} {cur_ver}) — потрібен перезапуск"
        elif in_mem:
            status = f"Активна (v{cur_ver or l_ver})"
        else:
            status = f"На диску (v{cur_ver})"

        rows.append({
            "import_name": imp_name,
            "name": cur_pkg or dist_cands[0],
            "disk_version": cur_ver or "—",
            "loaded_version": f"{l_pkg} {l_ver}".strip() if in_mem else "ще не імпортована",
            "in_memory": in_mem,
            "native_binary": is_native,
            "needs_restart": needs_restart,
            "status": status,
        })

    # Додаткові бібліотеки, заявлені зовнішніми модулями у library_deps
    for ep in (extra_pkgs or []):
        low = str(ep or "").strip().lower()
        if not low or low in seen_groups:
            continue
        seen_groups.add(low)
        cur_ver = dists.get(low, "")
        imp_guess = low.replace("-", "_")
        in_mem = imp_guess in sys.modules
        if not cur_ver:
            st = "Не встановлено"
        elif in_mem:
            st = f"Активна (v{cur_ver})"
        else:
            st = f"На диску (v{cur_ver})"
        rows.append({
            "import_name": imp_guess,
            "name": low,
            "disk_version": cur_ver or "—",
            "loaded_version": getattr(sys.modules.get(imp_guess), "__version__", "активна") if in_mem else "ще не імпортована",
            "in_memory": in_mem,
            "native_binary": False,
            "needs_restart": False,
            "status": st,
        })

    return rows


def _read_cpu_name_windows() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val:
                    return " ".join(str(val).split())
        except Exception:
            pass
    import platform
    return platform.processor() or "CPU"


def _winapi_memory_status():
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("sullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        total_gb = ms.ullTotalPhys / (1024.0 ** 3)
        free_gb = ms.ullAvailPhys / (1024.0 ** 3)
        used_gb = max(0.0, total_gb - free_gb)
        pct = float(ms.dwMemoryLoad)
        return total_gb, used_gb, free_gb, pct
    return 0.0, 0.0, 0.0, 0.0


def _detect_disk_media_map_windows() -> dict:
    """Повертає словник {'C': 'NVMe SSD', 'D': 'HDD', ...} через PowerShell CIM."""
    media_map = {}
    if sys.platform != "win32":
        return media_map

    ps_script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$disks=@{};"
        "Get-PhysicalDisk | ForEach-Object { $disks[[string]$_.DeviceId] = @($_.MediaType, $_.BusType) };"
        "Get-Partition | Where-Object { $_.DriveLetter } | ForEach-Object {"
        "  $d = $disks[[string]$_.DiskNumber];"
        "  if ($d) { Write-Output ($_.DriveLetter + '|' + $d[0] + '|' + $d[1]) }"
        "}"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_creation_flags(),
        )
        if res.returncode == 0 and res.stdout:
            for line in res.stdout.splitlines():
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3 and parts[0]:
                    letter = parts[0].upper().rstrip(":")
                    mtype = parts[1].upper()
                    btype = parts[2].upper()
                    if "NVME" in btype or btype == "17":
                        label = "NVMe SSD"
                    elif "SSD" in mtype or mtype == "4":
                        label = "SSD"
                    elif "HDD" in mtype or mtype == "3":
                        label = "HDD"
                    elif "SCM" in mtype or mtype == "5":
                        label = "SCM"
                    elif btype in ("USB", "7"):
                        label = "USB"
                    else:
                        label = mtype if mtype not in ("", "UNSPECIFIED", "0") else "Диск"
                    media_map[letter] = label
    except Exception:
        pass
    return media_map


def _detect_gpus_nvidia_smi():
    gpus = []
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.encoder,utilization.decoder",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=_creation_flags(),
        )
        if res.returncode != 0 or not res.stdout.strip():
            return []

        def _num(s):
            s = str(s or "").strip()
            if s in ("", "N/A", "[N/A]", "[Not Supported]"):
                return None
            try:
                return float(s)
            except Exception:
                return None

        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            v_tot = _num(parts[3]) or 0.0
            v_used = _num(parts[4]) or 0.0
            v_free = _num(parts[5]) or max(0.0, v_tot - v_used)
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "vendor": "NVIDIA",
                "driver": parts[2],
                "vram_total_mb": v_tot,
                "vram_used_mb": v_used,
                "vram_free_mb": v_free,
                "gpu_util_pct": _num(parts[6]),
                "enc_util_pct": _num(parts[7]) if len(parts) > 7 else None,
                "dec_util_pct": _num(parts[8]) if len(parts) > 8 else None,
                "live_metrics": True,
            })
    except Exception:
        pass
    return gpus


def _detect_gpus_wmi():
    gpus = []
    if sys.platform != "win32":
        return gpus
    ps_script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-CimInstance Win32_VideoController | ForEach-Object {"
        "  Write-Output ($_.Name + '|' + $_.DriverVersion + '|' + $_.AdapterRAM)"
        "}"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=_creation_flags(),
        )
        if res.returncode == 0 and res.stdout:
            for idx, line in enumerate(res.stdout.splitlines()):
                parts = [p.strip() for p in line.split("|")]
                if not parts or not parts[0]:
                    continue
                name = parts[0]
                drv = parts[1] if len(parts) > 1 else ""
                try:
                    ram_bytes = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                except Exception:
                    ram_bytes = 0
                upper = name.upper()
                if "NVIDIA" in upper or "GEFORCE" in upper or "RTX" in upper or "GTX" in upper:
                    vendor = "NVIDIA"
                elif "AMD" in upper or "RADEON" in upper:
                    vendor = "AMD"
                elif "INTEL" in upper or "ARC" in upper or "UHD" in upper or "IRIS" in upper:
                    vendor = "Intel"
                else:
                    vendor = "Other"
                gpus.append({
                    "index": str(idx),
                    "name": name,
                    "vendor": vendor,
                    "driver": drv,
                    "vram_total_mb": round(ram_bytes / (1024.0 * 1024.0), 1) if ram_bytes > 0 else None,
                    "vram_used_mb": None,
                    "vram_free_mb": None,
                    "gpu_util_pct": None,
                    "enc_util_pct": None,
                    "dec_util_pct": None,
                    "live_metrics": False,
                })
    except Exception:
        pass
    return gpus


def get_installed_ort_package() -> str:
    """Визначає, який саме пакет onnxruntime* встановлено в поточному середовищі."""
    installed = set(_get_disk_dists_map().keys())
    for pkg in ("onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino", "onnxruntime"):
        if pkg in installed:
            return pkg
    return "не встановлено"


def install_ort_package(package_name: str, progress_cb=None):
    """Перевстановлює вибраний пакет onnxruntime у активний Python (.venv)."""
    if getattr(sys, "frozen", False):
        return False, "Перемикання pip-пакетів недоступне всередині скомпільованого EXE. Скористайтеся 1_SETUP.bat."
    allowed = {"onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"}
    if package_name not in allowed:
        return False, f"Невідомий пакет: {package_name}"

    # Зафіксуємо версію onnxruntime у RAM до того, як pip перезапише диск
    observe_loaded_libraries()

    py_exe = sys.executable
    if os.path.basename(py_exe).lower() == "pythonw.exe":
        cand = os.path.join(os.path.dirname(py_exe), "python.exe")
        if os.path.exists(cand):
            py_exe = cand

    try:
        if progress_cb:
            progress_cb("Видалення попереднього варіанта onnxruntime...")
        subprocess.run(
            [py_exe, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-directml", "onnxruntime-gpu"],
            capture_output=True,
            text=True,
            timeout=120,
            creationflags=_creation_flags(),
        )
        if progress_cb:
            progress_cb(f"Встановлення {package_name}...")
        res = subprocess.run(
            [py_exe, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=_creation_flags(),
        )
        if res.returncode == 0:
            importlib.invalidate_caches()
            ort_loaded = "onnxruntime" in sys.modules
            if ort_loaded:
                return True, f"Пакет '{package_name}' встановлено на диск. Оскільки C-розширення onnxruntime вже завантажено в пам'ять, перезапустіть програму."
            return True, f"Пакет '{package_name}' успішно встановлено."
        err_msg = (res.stderr or res.stdout or "").strip()[-400:]
        return False, f"Помилка pip install:\n{err_msg}"
    except Exception as e:
        return False, f"Виняток під час встановлення: {e}"


class HardwareMonitor:
    """Монітор апаратних ресурсів ПК та реальний тестер графічних прискорювачів."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._lock = threading.Lock()
        self._cpu_name = _read_cpu_name_windows()
        self._cpu_phys = psutil.cpu_count(logical=False) if psutil else None
        self._cpu_log = os.cpu_count() or 1
        self._disk_media_map = {}
        self._wmi_gpus = None
        self._last_caps = None
        self._proc = psutil.Process(os.getpid()) if psutil else None
        if self._proc is not None:
            try:
                self._proc.cpu_percent(interval=None)
            except Exception:
                pass
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        self._init_static_async()

    def _init_static_async(self):
        def _bg():
            dmap = _detect_disk_media_map_windows()
            wgpus = _detect_gpus_wmi()
            with self._lock:
                self._disk_media_map = dmap
                self._wmi_gpus = wgpus
        threading.Thread(target=_bg, daemon=True).start()

    def refresh_static(self):
        dmap = _detect_disk_media_map_windows()
        wgpus = _detect_gpus_wmi()
        with self._lock:
            self._disk_media_map = dmap
            self._wmi_gpus = wgpus

    def refresh_all(self, force: bool = True) -> dict:
        """Перечитує бібліотеки з диска, оновлює метадані дисків/GPU і перетестовує прискорювачі."""
        global psutil, imageio_ffmpeg, cv2
        importlib.invalidate_caches()
        observe_loaded_libraries()

        if psutil is None:
            try:
                psutil = importlib.import_module("psutil")
                self._proc = psutil.Process(os.getpid())
                self._cpu_phys = psutil.cpu_count(logical=False)
                observe_loaded_libraries()
            except Exception:
                pass

        if imageio_ffmpeg is None:
            try:
                imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
                observe_loaded_libraries()
            except Exception:
                pass
        else:
            try:
                imageio_ffmpeg = importlib.reload(imageio_ffmpeg)
                ver = _get_disk_dists_map().get("imageio-ffmpeg", "")
                _reset_pure_module_loaded_state("imageio_ffmpeg", "imageio-ffmpeg", ver)
            except Exception:
                pass

        if cv2 is None:
            try:
                cv2 = importlib.import_module("cv2")
                observe_loaded_libraries()
            except Exception:
                pass

        self.refresh_static()
        return self.probe_acceleration_capabilities(force=force)

    def get_live_metrics(self) -> dict:
        with self._lock:
            dmap = dict(self._disk_media_map)
            wmi_gpus = list(self._wmi_gpus) if self._wmi_gpus is not None else []

        # CPU
        cpu_pct = 0.0
        proc_cpu_pct = 0.0
        cpu_freq_mhz = None
        if psutil is not None:
            try:
                cpu_pct = float(psutil.cpu_percent(interval=None))
            except Exception:
                pass
            if self._proc is not None:
                try:
                    proc_cpu_pct = float(self._proc.cpu_percent(interval=None)) / max(1, self._cpu_log)
                except Exception:
                    pass
            try:
                fr = psutil.cpu_freq()
                if fr and fr.current:
                    cpu_freq_mhz = round(float(fr.current), 0)
            except Exception:
                pass

        # RAM
        proc_ram_mb = 0.0
        if psutil is not None:
            vm = psutil.virtual_memory()
            ram_total = vm.total / (1024.0 ** 3)
            ram_avail = vm.available / (1024.0 ** 3)
            ram_used = max(0.0, ram_total - ram_avail)
            ram_pct = float(vm.percent)
            if self._proc is not None:
                try:
                    proc_ram_mb = self._proc.memory_info().rss / (1024.0 * 1024.0)
                except Exception:
                    pass
        else:
            ram_total, ram_used, ram_avail, ram_pct = _winapi_memory_status()

        # GPUs (NVIDIA live + WMI для інтегрованих/AMD)
        nv_gpus = _detect_gpus_nvidia_smi()
        gpus = list(nv_gpus)
        nv_names = {g["name"].lower() for g in nv_gpus}
        for wg in wmi_gpus:
            if wg["name"].lower() not in nv_names and wg["vendor"] != "NVIDIA":
                gpus.append(wg)
        if not gpus and wmi_gpus:
            gpus = wmi_gpus

        # Диски
        disks = []
        if psutil is not None:
            try:
                parts = psutil.disk_partitions(all=False)
                seen_mounts = set()
                for p in parts:
                    if "cdrom" in p.opts.lower() or not p.fstype:
                        continue
                    mount = p.mountpoint
                    if mount in seen_mounts:
                        continue
                    seen_mounts.add(mount)
                    try:
                        usage = psutil.disk_usage(mount)
                    except Exception:
                        continue
                    letter = mount.replace(":", "").replace("\\", "").strip().upper()
                    disks.append({
                        "mount": mount,
                        "fs": p.fstype,
                        "media": dmap.get(letter, "Диск"),
                        "total_gb": usage.total / (1024.0 ** 3),
                        "used_gb": usage.used / (1024.0 ** 3),
                        "free_gb": usage.free / (1024.0 ** 3),
                        "percent": float(usage.percent),
                    })
            except Exception:
                pass

        if not disks and sys.platform == "win32":
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    try:
                        usage = shutil.disk_usage(drive)
                        disks.append({
                            "mount": drive,
                            "fs": "",
                            "media": dmap.get(letter, "Диск"),
                            "total_gb": usage.total / (1024.0 ** 3),
                            "used_gb": usage.used / (1024.0 ** 3),
                            "free_gb": usage.free / (1024.0 ** 3),
                            "percent": round((usage.used / float(max(1, usage.total))) * 100.0, 1),
                        })
                    except Exception:
                        pass

        return {
            "cpu": {
                "name": self._cpu_name,
                "physical_cores": self._cpu_phys,
                "logical_threads": self._cpu_log,
                "usage_pct": cpu_pct,
                "process_pct": proc_cpu_pct,
                "freq_mhz": cpu_freq_mhz,
            },
            "ram": {
                "total_gb": ram_total,
                "used_gb": ram_used,
                "free_gb": ram_avail,
                "percent": ram_pct,
                "process_mb": proc_ram_mb,
            },
            "gpus": gpus,
            "disks": disks,
        }

    def probe_acceleration_capabilities(self, force: bool = False) -> dict:
        with self._lock:
            if not force and self._last_caps is not None:
                return self._last_caps

        metrics = self.get_live_metrics()
        gpu_vendors = {g["vendor"] for g in metrics["gpus"]}
        has_nvidia = "NVIDIA" in gpu_vendors
        has_intel_gpu = "Intel" in gpu_vendors or "Intel" in metrics["cpu"]["name"]
        has_amd = "AMD" in gpu_vendors

        ort_pkg = get_installed_ort_package()
        ort_providers = []
        try:
            import onnxruntime as ort
            observe_loaded_libraries()
            ort_providers = ort.get_available_providers()
        except Exception:
            ort = None

        probe_model = None
        for cand in ("yolo11n.onnx", "scrfd_face_detector.onnx", "best.onnx", "cct_xs_v2_global.onnx"):
            mp = os.path.join(self.base_dir, "models", cand)
            if os.path.exists(mp):
                probe_model = mp
                break

        def _test_ort_provider(prov_name: str):
            if ort is None:
                return False, "Модуль onnxruntime не знайдено"
            if prov_name not in ort_providers:
                return False, "Немає серед available_providers()"
            if probe_model is None:
                return True, "Доступний у onnxruntime"
            try:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                sess = ort.InferenceSession(probe_model, opts, providers=[prov_name])
                used = sess.get_providers()
                del sess
                if prov_name in used:
                    return True, f"Перевірено на {os.path.basename(probe_model)}"
                return False, f"Відкотився на {used}"
            except Exception as e:
                return False, f"Помилка ініціалізації: {str(e).splitlines()[0][:110]}"

        ai_backends = {}
        ok_cpu, msg_cpu = _test_ort_provider("CPUExecutionProvider")
        ai_backends["cpu"] = {
            "label": "CPU (ONNX CPUExecutionProvider)",
            "supported": ok_cpu,
            "status": "Працює" if ok_cpu else "Недоступно",
            "reason": msg_cpu,
        }

        if "CUDAExecutionProvider" in ort_providers:
            ok_cuda, msg_cuda = _test_ort_provider("CUDAExecutionProvider")
            ai_backends["cuda"] = {
                "label": "NVIDIA CUDA GPU (CUDAExecutionProvider)",
                "supported": ok_cuda,
                "status": "Працює" if ok_cuda else "Не запускається",
                "reason": msg_cuda,
            }
        else:
            ai_backends["cuda"] = {
                "label": "NVIDIA CUDA GPU (CUDAExecutionProvider)",
                "supported": False,
                "status": "Потребує onnxruntime-gpu + CUDA" if has_nvidia else "Немає NVIDIA GPU",
                "reason": (
                    f"Зараз встановлено '{ort_pkg}'. Для CUDA встановіть 'onnxruntime-gpu' (і CUDA/cuDNN)."
                    if has_nvidia else "Відеокарту NVIDIA не виявлено."
                ),
            }

        if "TensorrtExecutionProvider" in ort_providers:
            ok_trt, msg_trt = _test_ort_provider("TensorrtExecutionProvider")
            ai_backends["tensorrt"] = {
                "label": "NVIDIA TensorRT (TensorrtExecutionProvider)",
                "supported": ok_trt,
                "status": "Працює" if ok_trt else "Не запускається",
                "reason": msg_trt,
            }
        else:
            ai_backends["tensorrt"] = {
                "label": "NVIDIA TensorRT",
                "supported": False,
                "status": "Не встановлено" if has_nvidia else "Немає NVIDIA GPU",
                "reason": "Доступний у onnxruntime-gpu за наявності бібліотек TensorRT." if has_nvidia else "Відеокарту NVIDIA не виявлено.",
            }

        if "DmlExecutionProvider" in ort_providers:
            ok_dml, msg_dml = _test_ort_provider("DmlExecutionProvider")
            ai_backends["directml"] = {
                "label": "DirectX 12 GPU (DirectML — NVIDIA/AMD/Intel)",
                "supported": ok_dml,
                "status": "Працює" if ok_dml else "Не запускається",
                "reason": msg_dml,
            }
        else:
            ai_backends["directml"] = {
                "label": "DirectX 12 GPU (DirectML — NVIDIA/AMD/Intel)",
                "supported": False,
                "status": "Потребує onnxruntime-directml",
                "reason": f"Зараз встановлено '{ort_pkg}'. Для DirectML перемкніть пакет на 'onnxruntime-directml'.",
            }

        if "OpenVINOExecutionProvider" in ort_providers:
            ok_ov, msg_ov = _test_ort_provider("OpenVINOExecutionProvider")
            ai_backends["openvino"] = {
                "label": "Intel OpenVINO (CPU / iGPU)",
                "supported": ok_ov,
                "status": "Працює" if ok_ov else "Не запускається",
                "reason": msg_ov,
            }
        else:
            ai_backends["openvino"] = {
                "label": "Intel OpenVINO (CPU / iGPU)",
                "supported": False,
                "status": "Не встановлено" if has_intel_gpu else "Опційно для Intel",
                "reason": f"Зараз встановлено '{ort_pkg}'. Потребує пакет 'onnxruntime-openvino'.",
            }

        ffmpeg_exe = None
        if imageio_ffmpeg is not None:
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass

        encoders_out = ""
        hwaccels_out = ""
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            try:
                encoders_out = subprocess.run(
                    [ffmpeg_exe, "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=5, creationflags=_creation_flags()
                ).stdout
            except Exception:
                pass
            try:
                hwaccels_out = subprocess.run(
                    [ffmpeg_exe, "-hide_banner", "-hwaccels"],
                    capture_output=True, text=True, timeout=5, creationflags=_creation_flags()
                ).stdout
            except Exception:
                pass

        def _test_ffmpeg_encoder(enc_name: str):
            if not ffmpeg_exe:
                return False, "FFmpeg (imageio-ffmpeg) недоступний"
            if enc_name not in encoders_out:
                return False, "Кодек відсутній у збірці FFmpeg"
            dummy_frame = bytes(128 * 128 * 3)
            cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                "-s", "128x128", "-r", "15", "-i", "-",
                "-frames:v", "1", "-an", "-vcodec", enc_name, "-pix_fmt", "yuv420p",
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1",
            ]
            try:
                res = subprocess.run(
                    cmd, input=dummy_frame, capture_output=True, timeout=6, creationflags=_creation_flags()
                )
                if res.returncode == 0 and len(res.stdout) > 0:
                    return True, "Апаратний тест пройдено"
                err = res.stderr.decode("utf-8", errors="ignore").strip().splitlines()
                return False, (err[-1][:110] if err else f"Код виходу {res.returncode}")
            except Exception as e:
                return False, str(e)[:110]

        video_encoders = {}
        for enc_id, enc_title in (
            ("copy", "Прямий запис потоку RTSP (copy — без перекодування)"),
            ("libx264", "CPU H.264 (libx264)"),
            ("h264_nvenc", "NVIDIA NVENC H.264 (GPU)"),
            ("hevc_nvenc", "NVIDIA NVENC H.265/HEVC (GPU)"),
            ("h264_qsv", "Intel QuickSync H.264 (QSV)"),
            ("h264_amf", "AMD AMF H.264 (GPU)"),
        ):
            if enc_id == "copy":
                ok_e = bool(ffmpeg_exe)
                msg_e = "Працює для мережевих RTSP/HTTP (H.264/H.265) з ~0% CPU" if ok_e else "FFmpeg недоступний"
            else:
                ok_e, msg_e = _test_ffmpeg_encoder(enc_id)
            video_encoders[enc_id] = {
                "label": enc_title,
                "supported": ok_e,
                "status": "Працює" if ok_e else "Недоступно",
                "reason": msg_e,
            }

        hw_decoders = []
        for line in hwaccels_out.splitlines():
            w = line.strip().lower()
            if w and w not in ("hardware", "acceleration", "methods:"):
                hw_decoders.append(w)

        ocl_available = False
        ocl_device = ""
        cv_cuda_cnt = 0
        if cv2 is not None:
            try:
                ocl_available = bool(cv2.ocl.haveOpenCL())
                if ocl_available:
                    ocl_device = cv2.ocl.Device.getDefault().name() or "OpenCL GPU"
            except Exception:
                pass
            try:
                if hasattr(cv2, "cuda"):
                    cv_cuda_cnt = int(cv2.cuda.getCudaEnabledDeviceCount())
            except Exception:
                pass

        if ai_backends["cuda"]["supported"]:
            rec_ai = "cuda"
            rec_ai_text = "CUDA (NVIDIA GPU) — максимальна швидкість уже готова до роботи."
        elif ai_backends["directml"]["supported"]:
            rec_ai = "directml"
            rec_ai_text = "DirectML (DX12 GPU) — готове до роботи на вашій відеокарті."
        elif ai_backends["openvino"]["supported"]:
            rec_ai = "openvino"
            rec_ai_text = "Intel OpenVINO — готове до роботи."
        elif has_nvidia:
            rec_ai = "directml"
            rec_ai_text = (
                "Виявлено NVIDIA GPU, але поточний пакет лише CPU ('onnxruntime'). "
                "Рекомендовано встановити 'onnxruntime-directml' (1 клік без CUDA Toolkit) або 'onnxruntime-gpu' (для CUDA)."
            )
        elif has_amd or has_intel_gpu:
            rec_ai = "directml"
            rec_ai_text = "Виявлено GPU/iGPU. Для прискорення встановіть 'onnxruntime-directml'."
        else:
            rec_ai = "cpu"
            rec_ai_text = "Оптимальний режим для поточного заліза — CPU."

        if video_encoders["h264_nvenc"]["supported"]:
            rec_enc = "h264_nvenc"
            rec_enc_text = "Для IP-камер: 'copy' (0% CPU) або 'h264_nvenc' (стиснення на RTX GPU)."
        elif video_encoders["h264_qsv"]["supported"]:
            rec_enc = "h264_qsv"
            rec_enc_text = "Для IP-камер: 'copy' (0% CPU) або 'h264_qsv' (Intel QuickSync)."
        elif video_encoders["h264_amf"]["supported"]:
            rec_enc = "h264_amf"
            rec_enc_text = "Для IP-камер: 'copy' (0% CPU) або 'h264_amf' (AMD GPU)."
        else:
            rec_enc = "libx264"
            rec_enc_text = "Для IP-камер: 'copy' (0% CPU) або 'libx264' (CPU)."

        caps = {
            "ort_package": ort_pkg,
            "ort_providers": ort_providers,
            "ai_backends": ai_backends,
            "video_encoders": video_encoders,
            "hw_decoders": hw_decoders,
            "opencv_opencl": {"supported": ocl_available, "device": ocl_device},
            "opencv_cuda": {"supported": cv_cuda_cnt > 0, "count": cv_cuda_cnt},
            "ui_renderer": "Tkinter GDI (CPU) + фоновий ресайз потоків",
            "recommendations": {
                "ai_backend": rec_ai,
                "ai_text": rec_ai_text,
                "video_encoder": rec_enc,
                "video_text": rec_enc_text,
            },
        }
        with self._lock:
            self._last_caps = caps
        return caps
