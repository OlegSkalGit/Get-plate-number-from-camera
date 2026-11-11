import os
import re
import ast
import sys
import copy
import json
import glob
import time
import ctypes
import shutil
import string
import hashlib
import zipfile
import tempfile
import importlib
import importlib.machinery
import subprocess
import threading
import urllib.request
import importlib.metadata
from ctypes import wintypes

try:
    from scripts.core.i18n import tr
except ImportError:
    try:
        from i18n import tr
    except ImportError:
        def tr(key, default=None, **kwargs):
            text = key if default is None else str(default)
            return text.format(**kwargs) if kwargs else text

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


def _set_ort_quiet(ort_mod):
    if ort_mod is None:
        return
    try:
        ort_mod.set_default_logger_severity(3)
    except Exception:
        pass


def _norm_pkg_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name or "").strip().lower())


CORE_REQUIRED_PACKAGES = (
    "opencv-python",
    "numpy",
    "pillow",
    "pyyaml",
    "fast-alpr",
    "fast-plate-ocr",
    "imageio-ffmpeg",
    "psutil",
    "onnxruntime",
)

_PACKAGE_ALIASES = {
    "opencv-python": ("opencv-python", "opencv-contrib-python", "opencv-python-headless"),
    "onnxruntime": ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"),
}

ORT_VARIANT_PACKAGES = (
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-openvino",
    "onnxruntime",
)


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


def _get_normalized_disk_dists_map() -> dict:
    """Повертає {norm_hyphen_name: version_str} для надійного порівняння пакетів."""
    out = {}
    for k, v in _get_disk_dists_map().items():
        nk = _norm_pkg_name(k)
        if nk and nk not in out:
            out[nk] = v
    return out


def _find_disk_ort_package_dir() -> str:
    """Знаходить каталог активного пакета onnxruntime на диску (без звернення до sys.modules)."""
    if getattr(sys, "frozen", False):
        return ""
    try:
        importlib.invalidate_caches()
        spec = importlib.machinery.PathFinder.find_spec("onnxruntime")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                if loc and os.path.isdir(loc):
                    return loc
    except Exception:
        pass
    for p in list(sys.path):
        if not p:
            continue
        cand = os.path.join(p, "onnxruntime")
        if os.path.isdir(os.path.join(cand, "capi")):
            return cand
    return ""


def _read_disk_ort_package_info() -> tuple:
    """
    Читає фактичну назву варіанта та версію з активних файлів onnxruntime/ на диску
    (capi/build_and_package_info.py або нативні маркери в capi/), не торкаючись sys.modules.
    Повертає (pkg_name, version, cuda_version).
    """
    ort_dir = _find_disk_ort_package_dir()
    if not ort_dir:
        return ("", "", "")
    capi_dir = os.path.join(ort_dir, "capi")
    pkg_name = ""
    ver_str = ""
    cuda_ver = ""
    bpi_path = os.path.join(capi_dir, "build_and_package_info.py")
    if os.path.isfile(bpi_path):
        try:
            with open(bpi_path, "r", encoding="utf-8", errors="ignore") as fh:
                tree = ast.parse(fh.read(), filename=bpi_path)
            for node in getattr(tree, "body", ()):
                if isinstance(node, ast.Assign):
                    val_node = node.value
                    if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name):
                                if tgt.id == "package_name":
                                    pkg_name = _norm_pkg_name(val_node.value)
                                elif tgt.id == "__version__":
                                    ver_str = val_node.value.strip()
                                elif tgt.id == "cuda_version":
                                    cuda_ver = val_node.value.strip()
        except Exception:
            pass

    if not ver_str:
        v_path = os.path.join(ort_dir, "package_info.py")
        if not os.path.isfile(v_path):
            v_path = os.path.join(ort_dir, "_version.py")
        if os.path.isfile(v_path):
            try:
                with open(v_path, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                m = re.search(r"""__version__\s*=\s*['"]([^'"]+)['"]""", txt)
                if m:
                    ver_str = m.group(1).strip()
            except Exception:
                pass

    if not pkg_name and os.path.isdir(capi_dir):
        try:
            files = {f.lower() for f in os.listdir(capi_dir)}
        except Exception:
            files = set()
        if "onnxruntime_providers_cuda.dll" in files:
            pkg_name = "onnxruntime-gpu"
        elif "directml.dll" in files:
            pkg_name = "onnxruntime-directml"
        elif "onnxruntime_providers_openvino.dll" in files or "onnxruntime_providers_openvino_wrapper.dll" in files:
            pkg_name = "onnxruntime-openvino"
        elif any(f.startswith("onnxruntime_pybind11_state") for f in files):
            pkg_name = "onnxruntime"

    return (pkg_name, ver_str, cuda_ver)


def get_installed_ort_conflicts() -> dict:
    """Повертає фактичний активний пакет onnxruntime на диску та всі знайдені ORT .dist-info."""
    disk_pkg, disk_ver, _ = _read_disk_ort_package_info()
    dists = _get_normalized_disk_dists_map()
    dist_pkgs = [p for p in ORT_VARIANT_PACKAGES if p in dists]
    if not disk_pkg and dist_pkgs:
        disk_pkg = dist_pkgs[0]
        disk_ver = dists.get(disk_pkg, "")
    elif disk_pkg and not disk_ver:
        disk_ver = dists.get(disk_pkg, "")
    stale = [p for p in dist_pkgs if disk_pkg and p != disk_pkg]
    return {
        "active_disk_pkg": disk_pkg,
        "active_disk_version": disk_ver,
        "dist_packages": dist_pkgs,
        "stale_dist_packages": stale,
    }


def _detect_in_memory_ort_info() -> tuple:
    """Визначає (pkg_name, version) саме тієї копії onnxruntime, яка завантажена в sys.modules."""
    ort_mod = sys.modules.get("onnxruntime")
    if ort_mod is None:
        return ("", "")
    pn = ""
    ver = str(getattr(ort_mod, "__version__", "") or "").strip()

    # build_and_package_info кешовано у sys.modules на момент першого import onnxruntime
    bpi_mod = sys.modules.get("onnxruntime.capi.build_and_package_info")
    if bpi_mod is not None:
        try:
            bp_name = str(getattr(bpi_mod, "package_name", "") or "").strip()
            bp_ver = str(getattr(bpi_mod, "__version__", "") or "").strip()
            if bp_name:
                pn = _norm_pkg_name(bp_name)
            if bp_ver and not ver:
                ver = bp_ver
        except Exception:
            pass

    if not pn or pn == "onnxruntime":
        try:
            provs = ort_mod.get_available_providers()
            if "CUDAExecutionProvider" in provs:
                pn = "onnxruntime-gpu"
            elif "DmlExecutionProvider" in provs:
                pn = "onnxruntime-directml"
            elif "OpenVINOExecutionProvider" in provs:
                pn = "onnxruntime-openvino"
        except Exception:
            pass

    if not pn:
        val_mod = sys.modules.get("onnxruntime.capi.onnxruntime_validation")
        if val_mod is not None:
            try:
                vp, vv, _ = val_mod.get_package_name_and_version_info()
                if vp:
                    pn = _norm_pkg_name(vp)
                if vv and not ver:
                    ver = str(vv).strip()
            except Exception:
                pass

    return (pn or "onnxruntime", ver)


def observe_loaded_libraries():
    """Фіксує версію кожного відстежуваного пакета на момент його першого потрапляння в sys.modules."""
    dists = None
    with _MEM_STATE_LOCK:
        for imp_name, dist_cands, _is_native in _TRACKED_LIBS:
            if imp_name in sys.modules and imp_name not in _LOADED_MEM_STATE:
                if imp_name == "onnxruntime":
                    mem_pkg, mem_ver = _detect_in_memory_ort_info()
                    if not mem_ver:
                        if dists is None:
                            dists = _get_normalized_disk_dists_map()
                        mem_ver = dists.get(mem_pkg, "")
                    _LOADED_MEM_STATE[imp_name] = (mem_pkg or imp_name, str(mem_ver))
                    continue

                if dists is None:
                    dists = _get_normalized_disk_dists_map()
                matched_pkg = ""
                matched_ver = ""
                for dc in dist_cands:
                    ndc = _norm_pkg_name(dc)
                    if ndc in dists:
                        matched_pkg = dc
                        matched_ver = dists[ndc]
                        break
                if not matched_ver:
                    matched_ver = getattr(sys.modules[imp_name], "__version__", "") or ""
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
    norm_dists = _get_normalized_disk_dists_map()
    seen_groups = set()
    rows = []

    disk_ort_pkg, disk_ort_ver, _ = _read_disk_ort_package_info()
    disk_ort_dir_present = bool(disk_ort_pkg or _find_disk_ort_package_dir())

    with _MEM_STATE_LOCK:
        mem_snap = dict(_LOADED_MEM_STATE)

    for imp_name, dist_cands, is_native in _TRACKED_LIBS:
        seen_groups.update(dist_cands)
        cur_pkg = ""
        cur_ver = ""
        if imp_name == "onnxruntime":
            if disk_ort_pkg:
                cur_pkg = disk_ort_pkg
                cur_ver = disk_ort_ver or norm_dists.get(disk_ort_pkg, "")
            elif getattr(sys, "frozen", False):
                cur_pkg = get_installed_ort_package()
                if cur_pkg:
                    cur_ver = norm_dists.get(cur_pkg, "") or str(getattr(sys.modules.get("onnxruntime"), "__version__", "") or "")
            elif disk_ort_dir_present:
                for dc in dist_cands:
                    ndc = _norm_pkg_name(dc)
                    if ndc in norm_dists:
                        cur_pkg = dc
                        cur_ver = norm_dists[ndc]
                        break
        else:
            for dc in dist_cands:
                ndc = _norm_pkg_name(dc)
                if ndc in norm_dists:
                    cur_pkg = dc
                    cur_ver = norm_dists[ndc]
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
            status_kind = "missing"
            status = (
                tr("hw.lib.status.missing_in_mem", "Not installed (stale copy in memory)")
                if in_mem
                else tr("hw.lib.status.missing", "Not installed")
            )
        elif needs_restart:
            status_kind = "restart"
            status = tr(
                "hw.lib.status.restart",
                "Updated on disk ({l_pkg} {l_ver} -> {cur_pkg} {cur_ver}) — restart required",
                l_pkg=l_pkg,
                l_ver=l_ver,
                cur_pkg=cur_pkg,
                cur_ver=cur_ver,
            )
        elif in_mem:
            status_kind = "active"
            status = tr("hw.lib.status.active", "Active (v{ver})", ver=cur_ver or l_ver)
        else:
            status_kind = "disk_only"
            status = tr("hw.lib.status.disk_only", "On disk (v{ver})", ver=cur_ver)

        rows.append({
            "import_name": imp_name,
            "name": cur_pkg or dist_cands[0],
            "disk_version": cur_ver or "—",
            "loaded_version": f"{l_pkg} {l_ver}".strip() if in_mem else tr("hw.lib.loaded.not_imported", "not imported yet"),
            "in_memory": in_mem,
            "native_binary": is_native,
            "needs_restart": needs_restart,
            "status_kind": status_kind,
            "status": status,
        })

    # Додаткові бібліотеки, заявлені зовнішніми модулями у library_deps
    norm_seen = {_norm_pkg_name(s) for s in seen_groups}
    for ep in (extra_pkgs or []):
        low = _norm_pkg_name(ep)
        if not low or low in norm_seen:
            continue
        norm_seen.add(low)
        cur_ver = norm_dists.get(low, "")
        imp_guess = low.replace("-", "_")
        in_mem = imp_guess in sys.modules
        if not cur_ver:
            sk = "missing"
            st = tr("hw.lib.status.missing", "Not installed")
        elif in_mem:
            sk = "active"
            st = tr("hw.lib.status.active", "Active (v{ver})", ver=cur_ver)
        else:
            sk = "disk_only"
            st = tr("hw.lib.status.disk_only", "On disk (v{ver})", ver=cur_ver)
        rows.append({
            "import_name": imp_guess,
            "name": low,
            "disk_version": cur_ver or "—",
            "loaded_version": (
                getattr(sys.modules.get(imp_guess), "__version__", tr("hw.lib.loaded.active_fallback", "active"))
                if in_mem
                else tr("hw.lib.loaded.not_imported", "not imported yet")
            ),
            "in_memory": in_mem,
            "native_binary": False,
            "needs_restart": False,
            "status_kind": sk,
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


def _detect_disk_info_map_windows() -> dict:
    """
    Повертає словник {'C': {'media': 'NVMe SSD', 'phys_id': 'disk:2', 'phys_model': '...', 'phys_size_gb': ...}, ...}
    через PowerShell CIM (Get-PhysicalDisk + Get-Disk + Get-Partition).
    """
    info_map = {}
    if sys.platform != "win32":
        return info_map

    ps_script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$pd=@{};"
        "Get-PhysicalDisk | ForEach-Object { $pd[[string]$_.DeviceId] = $_ };"
        "$gd=@{};"
        "Get-Disk | ForEach-Object { $gd[[string]$_.Number] = $_ };"
        "Get-Partition | Where-Object { $_.DriveLetter -and [char]::IsLetter([char]$_.DriveLetter) } | ForEach-Object {"
        "  $dn = [string]$_.DiskNumber;"
        "  $p = $pd[$dn];"
        "  $g = $gd[$dn];"
        "  $mt = if ($p -and $null -ne $p.MediaType) { [string]$p.MediaType } else { '' };"
        "  $bt = if ($p -and $null -ne $p.BusType -and [string]$p.BusType -ne '') { [string]$p.BusType } elseif ($g -and $null -ne $g.BusType) { [string]$g.BusType } else { '' };"
        "  $fn = if ($g -and $g.FriendlyName) { [string]$g.FriendlyName } elseif ($p -and $p.FriendlyName) { [string]$p.FriendlyName } else { '' };"
        "  $sz = if ($g -and $g.Size) { [string]$g.Size } elseif ($p -and $p.Size) { [string]$p.Size } else { '0' };"
        "  $fn = ($fn -replace '[\\r\\n\\|]+', ' ').Trim();"
        "  Write-Output ($_.DriveLetter + '|' + $dn + '|' + $mt + '|' + $bt + '|' + $fn + '|' + $sz)"
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
                    disk_num = parts[1] if len(parts) > 1 else ""
                    mtype = (parts[2] if len(parts) > 2 else "").upper()
                    btype = (parts[3] if len(parts) > 3 else "").upper()
                    model = parts[4].strip() if len(parts) > 4 else ""
                    size_raw = parts[5].strip() if len(parts) > 5 else "0"

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
                        label = mtype if mtype not in ("", "UNSPECIFIED", "0") else "Disk"

                    try:
                        sz_bytes = float(size_raw) if size_raw else 0.0
                        phys_size_gb = round(sz_bytes / (1024.0 ** 3), 1) if sz_bytes > 0 else None
                    except Exception:
                        phys_size_gb = None

                    phys_id = f"disk:{disk_num}" if disk_num else f"letter:{letter}"
                    info_map[letter] = {
                        "media": label,
                        "phys_id": phys_id,
                        "phys_model": model,
                        "phys_size_gb": phys_size_gb,
                    }
    except Exception:
        pass
    return info_map


def _detect_disk_media_map_windows() -> dict:
    """Повертає словник {'C': 'NVMe SSD', 'D': 'HDD', ...} через PowerShell CIM."""
    return {k: v.get("media", "Disk") for k, v in _detect_disk_info_map_windows().items()}


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
    """Визначає, який саме пакет onnxruntime* фактично встановлено на диску в поточному середовищі."""
    disk_pkg, _disk_ver, _disk_cuda = _read_disk_ort_package_info()
    if disk_pkg:
        return disk_pkg

    if not getattr(sys, "frozen", False):
        # Якщо каталог onnxruntime/ у site-packages відсутній, не видаємо осиротілий .dist-info чи RAM за наявний на диску
        if not _find_disk_ort_package_dir():
            return ""
        installed = set(_get_normalized_disk_dists_map().keys())
        for pkg in ORT_VARIANT_PACKAGES:
            if pkg in installed:
                return pkg
        return ""

    # Фолбек для замороженого EXE без dist-info
    try:
        from onnxruntime.capi import onnxruntime_validation
        pn, _pv, _cv = onnxruntime_validation.get_package_name_and_version_info()
        if pn:
            return _norm_pkg_name(pn)
    except Exception:
        pass
    if "onnxruntime" in sys.modules:
        try:
            provs = sys.modules["onnxruntime"].get_available_providers()
            if "CUDAExecutionProvider" in provs:
                return "onnxruntime-gpu"
            if "DmlExecutionProvider" in provs:
                return "onnxruntime-directml"
            if "OpenVINOExecutionProvider" in provs:
                return "onnxruntime-openvino"
            return "onnxruntime"
        except Exception:
            pass
    return ""


def get_ort_loaded_package() -> str:
    """Повертає назву пакета onnxruntime*, фактично завантаженого у RAM (або '', якщо ще не імпортовано)."""
    if "onnxruntime" not in sys.modules:
        return ""
    observe_loaded_libraries()
    mem_pkg, _mem_ver = _detect_in_memory_ort_info()
    if mem_pkg and mem_pkg != "onnxruntime":
        return mem_pkg
    with _MEM_STATE_LOCK:
        mem_ent = _LOADED_MEM_STATE.get("onnxruntime")
    if mem_ent and mem_ent[0]:
        return str(mem_ent[0])
    return mem_pkg or "onnxruntime"


def is_onnxruntime_gpu_installed() -> bool:
    """Повертає True, якщо в середовищі встановлено саме GPU-білд onnxruntime-gpu (CUDA)."""
    pkg = get_installed_ort_package()
    if pkg:
        return pkg == "onnxruntime-gpu"
    if getattr(sys, "frozen", False):
        try:
            from onnxruntime.capi import onnxruntime_validation
            pn, _pv, cv = onnxruntime_validation.get_package_name_and_version_info()
            if _norm_pkg_name(pn) == "onnxruntime-gpu" or bool(cv):
                return True
        except Exception:
            pass
    return False


def has_nvidia_gpu() -> bool:
    """Швидка перевірка наявності відеокарти NVIDIA у системі (nvidia-smi -> WMI)."""
    if _detect_gpus_nvidia_smi():
        return True
    try:
        return any(g.get("vendor") == "NVIDIA" for g in _detect_gpus_wmi())
    except Exception:
        return False


def _cleanup_ort_swap_trash(site_dir: str = ""):
    """Видаляє старі тимчасові каталоги .ort_swap_trash_* після попереднього перемикання ONNX Runtime у Windows."""
    if getattr(sys, "frozen", False):
        return
    dirs_to_scan = []
    if site_dir and os.path.isdir(site_dir):
        dirs_to_scan.append(site_dir)
    else:
        ort_dir = _find_disk_ort_package_dir()
        if ort_dir:
            dirs_to_scan.append(os.path.dirname(ort_dir))
    for parent in dirs_to_scan:
        try:
            for name in os.listdir(parent):
                if name.startswith(".ort_swap_trash_"):
                    shutil.rmtree(os.path.join(parent, name), ignore_errors=True)
        except Exception:
            pass


_cleanup_ort_swap_trash()


def install_ort_package(package_name: str, progress_cb=None):
    """Встановлює або перемикає пакет onnxruntime у .venv (включно з безпечною заміною завантажених у RAM DLL у Windows)."""
    if getattr(sys, "frozen", False):
        return False, tr(
            "hw.ort.err_frozen",
            "Switching pip packages is unavailable inside a compiled EXE. Use 1_SETUP.bat.",
        )
    allowed = {"onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-openvino"}
    if package_name not in allowed:
        return False, tr("hw.ort.err_unknown_pkg", "Unknown package: {pkg}", pkg=package_name)

    # Зафіксуємо версію onnxruntime у RAM до того, як pip перезапише диск
    observe_loaded_libraries()

    disk_dists = _get_normalized_disk_dists_map()
    installed_orts = [p for p in ORT_VARIANT_PACKAGES if p in disk_dists]
    other_orts = [p for p in installed_orts if p != package_name]
    target_already_on_disk = package_name in installed_orts
    actual_disk_ort = get_installed_ort_package()
    _cleanup_ort_swap_trash()

    if target_already_on_disk and not other_orts and (not actual_disk_ort or actual_disk_ort == package_name):
        if progress_cb:
            progress_cb(
                tr(
                    "hw.ort.progress_kept",
                    "Package '{pkg}' is already installed — existing module kept without reinstalling.",
                    pkg=package_name,
                )
            )
        ort_loaded = "onnxruntime" in sys.modules
        loaded_pkg = get_ort_loaded_package() if ort_loaded else ""
        if ort_loaded and loaded_pkg and loaded_pkg != package_name:
            return True, tr(
                "hw.ort.already_installed_restart",
                "Package '{pkg}' is already installed on disk. Since '{loaded_pkg}' is still active in memory, restart the application.",
                pkg=package_name,
                loaded_pkg=loaded_pkg,
            )
        return True, tr(
            "hw.ort.already_installed",
            "Package '{pkg}' is already installed (reinstall not needed).",
            pkg=package_name,
        )

    py_exe = sys.executable
    if os.path.basename(py_exe).lower() == "pythonw.exe":
        cand = os.path.join(os.path.dirname(py_exe), "python.exe")
        if os.path.exists(cand):
            py_exe = cand

    ort_in_mem = ("onnxruntime" in sys.modules)
    moved_pairs = []
    trash_dir = ""

    try:
        with tempfile.TemporaryDirectory(prefix="ort_dl_") as dl_dir:
            find_links_args = []
            if sys.platform == "win32" and ort_in_mem:
                # У Windows завантажені в RAM .dll/.pyd блокують видалення чи перезапис (WinError 5),
                # але дозволяють атомарне перейменування на тому самому диску NTFS.
                # Спочатку завантажуємо wheel локально, щоб не зламати .venv за відсутності мережі.
                if progress_cb:
                    progress_cb(tr("hw.ort.progress_installing", "Installing {pkg}...", pkg=package_name))
                dl_res = subprocess.run(
                    [py_exe, "-m", "pip", "download", "--no-deps", "-d", dl_dir, package_name],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    creationflags=_creation_flags(),
                )
                if dl_res.returncode != 0:
                    err_msg = (dl_res.stderr or dl_res.stdout or "").strip()[-400:]
                    return False, tr("hw.ort.err_pip", "pip install error:\n{err}", err=err_msg)
                find_links_args = ["--find-links", dl_dir]

                ort_dir = _find_disk_ort_package_dir()
                if ort_dir and os.path.isdir(ort_dir):
                    site_dir = os.path.dirname(ort_dir)
                    trash_dir = os.path.join(site_dir, f".ort_swap_trash_{os.getpid()}_{int(time.time() * 1000)}")
                    for walk_root, _dirs, files in os.walk(ort_dir):
                        for fn in files:
                            if fn.lower().endswith((".dll", ".pyd")):
                                src = os.path.join(walk_root, fn)
                                rel = os.path.relpath(src, ort_dir)
                                dst = os.path.join(trash_dir, rel)
                                os.makedirs(os.path.dirname(dst), exist_ok=True)
                                os.replace(src, dst)
                                moved_pairs.append((src, dst))

            if other_orts:
                if progress_cb:
                    progress_cb(
                        tr(
                            "hw.ort.progress_uninstall_conflict",
                            "Removing conflicting variant ({pkgs})...",
                            pkgs=", ".join(other_orts),
                        )
                    )
                subprocess.run(
                    [py_exe, "-m", "pip", "uninstall", "-y", *other_orts],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    creationflags=_creation_flags(),
                )

            if progress_cb and not (sys.platform == "win32" and ort_in_mem):
                progress_cb(tr("hw.ort.progress_installing", "Installing {pkg}...", pkg=package_name))

            install_cmd = [py_exe, "-m", "pip", "install", *find_links_args]
            if target_already_on_disk and (other_orts or actual_disk_ort != package_name):
                # Якщо вилучався конфліктний пакет або файли onnxruntime/ на диску не відповідають .dist-info,
                # примусово відновлюємо файли цільового пакета без перевстановлення інших залежностей.
                install_cmd.extend(["--force-reinstall", "--no-deps"])
            install_cmd.append(package_name)
            res = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                timeout=600,
                creationflags=_creation_flags(),
            )
            if res.returncode == 0:
                if trash_dir and os.path.isdir(trash_dir):
                    shutil.rmtree(trash_dir, ignore_errors=True)
                importlib.invalidate_caches()
                ort_loaded = "onnxruntime" in sys.modules
                loaded_pkg = get_ort_loaded_package() if ort_loaded else ""
                if ort_loaded and loaded_pkg and loaded_pkg != package_name:
                    return True, tr(
                        "hw.ort.installed_restart",
                        "Package '{pkg}' installed on disk. Since the onnxruntime C-extension is already loaded in memory, restart the application.",
                        pkg=package_name,
                    )
                return True, tr(
                    "hw.ort.installed_ok",
                    "Package '{pkg}' installed successfully.",
                    pkg=package_name,
                )

            # Відкат переміщених бінарників у разі помилки pip install
            for src, dst in moved_pairs:
                if os.path.exists(dst) and not os.path.exists(src):
                    try:
                        os.makedirs(os.path.dirname(src), exist_ok=True)
                        os.replace(dst, src)
                    except Exception:
                        pass
            err_msg = (res.stderr or res.stdout or "").strip()[-400:]
            return False, tr("hw.ort.err_pip", "pip install error:\n{err}", err=err_msg)
    except Exception as e:
        for src, dst in moved_pairs:
            if os.path.exists(dst) and not os.path.exists(src):
                try:
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    os.replace(dst, src)
                except Exception:
                    pass
        return False, tr("hw.ort.err_exception", "Exception during installation: {err}", err=e)


# ------------------------------------------------------------------
#  Перевірка та автоматичне завантаження бібліотек CUDA / cuDNN у папку cuda/
# ------------------------------------------------------------------
_CUDA_ENV_LOCK = threading.Lock()
_CUDA_DL_LOCK = threading.Lock()
_CUDA_DLL_DIR_HANDLES = []
_CUDA_REGISTERED_DIRS = set()
_CUDA_PYPI_JSON_CACHE = {}


def get_cuda_dir(base_dir: str) -> str:
    root = os.path.abspath(base_dir or os.getcwd())
    return os.path.join(root, "cuda")


def _parse_ver_tuple(ver_str: str) -> tuple:
    nums = [int(p) for p in re.findall(r"\d+", str(ver_str or ""))[:4]]
    return tuple(nums) if nums else (0,)


def _read_cuda_manifest(cuda_dir: str) -> dict:
    mpath = os.path.join(cuda_dir, "cuda_manifest.json")
    if os.path.isfile(mpath):
        try:
            with open(mpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _write_cuda_manifest(cuda_dir: str, data: dict):
    os.makedirs(cuda_dir, exist_ok=True)
    mpath = os.path.join(cuda_dir, "cuda_manifest.json")
    tmp = f"{mpath}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, mpath)


def _fetch_pypi_json(url: str, timeout_sec: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "ANPR-MultiCam-NVR/2.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8"))


def detect_target_cuda_major(base_dir: str = "", allow_network: bool = False) -> str:
    """
    Визначає потрібну мажорну версію CUDA ('12' або '13') для встановленого/цільового onnxruntime-gpu.
    Перевіряє спочатку метадані встановленого на диску onnxruntime-gpu, потім onnxruntime_validation,
    потім локальний cuda_manifest.json або PyPI JSON.
    """
    importlib.invalidate_caches()
    conflicts = get_installed_ort_conflicts()
    active_disk_pkg = conflicts.get("active_disk_pkg") or ""
    active_disk_ver = conflicts.get("active_disk_version") or ""

    # 1. Метадані встановленого на диску onnxruntime-gpu (лише якщо активний саме він, а не застарілий .dist-info)
    if active_disk_pkg == "onnxruntime-gpu":
        try:
            dist = importlib.metadata.distribution("onnxruntime-gpu")
            reqs = dist.requires or []
            for r in reqs:
                low = r.lower()
                if "cu12" in low or ("nvidia-cuda-runtime" in low and "12" in low):
                    return "12"
                if "cu13" in low or ("nvidia-cuda-runtime" in low and "13" in low) or ("nvidia-cudnn-cu13" in low):
                    return "13"
            if _parse_ver_tuple(active_disk_ver or dist.version) >= (1, 27):
                return "13"
            return "12"
        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception:
            pass

    # 2. Якщо працює усередині зібраного EXE або вже завантажено onnxruntime-gpu в RAM
    if "onnxruntime" in sys.modules or getattr(sys, "frozen", False):
        try:
            from onnxruntime.capi import onnxruntime_validation
            pkg_name, ver_str, cuda_ver = onnxruntime_validation.get_package_name_and_version_info()
            if cuda_ver and (_norm_pkg_name(pkg_name) == "onnxruntime-gpu" or getattr(sys, "frozen", False)):
                maj = str(cuda_ver).split(".")[0].strip()
                if maj.isdigit():
                    return maj
        except Exception:
            pass

    # 3. Версія активного onnxruntime-gpu на диску (якщо інший варіант — питаємо актуальний реліз на PyPI)
    disk_ort_ver = active_disk_ver if active_disk_pkg == "onnxruntime-gpu" else ""
    if allow_network:
        try:
            url = (
                f"https://pypi.org/pypi/onnxruntime-gpu/{disk_ort_ver}/json"
                if disk_ort_ver else "https://pypi.org/pypi/onnxruntime-gpu/json"
            )
            data = _CUDA_PYPI_JSON_CACHE.get(url)
            if data is None:
                data = _fetch_pypi_json(url, timeout_sec=8.0)
                _CUDA_PYPI_JSON_CACHE[url] = data
            reqs = (data.get("info") or {}).get("requires_dist") or []
            for r in reqs:
                low = r.lower()
                if "cu12" in low or ("nvidia-cuda-runtime" in low and "12" in low):
                    return "12"
                if "cu13" in low or ("nvidia-cuda-runtime" in low and "13" in low) or ("nvidia-cudnn-cu13" in low):
                    return "13"
        except Exception:
            pass

    # 4. Локальний маніфест у cuda/
    if base_dir:
        man = _read_cuda_manifest(get_cuda_dir(base_dir))
        maj = str(man.get("cuda_major") or "").strip()
        if maj.isdigit():
            return maj

    # 5. Офлайн-евристика за гілкою ORT (1.27+ -> CUDA 13, до 1.26 -> CUDA 12)
    if disk_ort_ver and _parse_ver_tuple(disk_ort_ver) < (1, 27):
        return "12"
    return "13"


def get_cuda_components_spec(cuda_major: str) -> list:
    maj = str(cuda_major or "13").strip()
    if maj.isdigit() and int(maj) >= 13:
        return [
            {
                "key": "cuda_runtime",
                "label": "CUDA Runtime (cudart)",
                "dlls": ("cudart64_13.dll",),
                "prefixes": ("cudart64_13",),
                "pypi": "nvidia-cuda-runtime",
                "compat_prefix": "13.",
                "required": True,
            },
            {
                "key": "cublas",
                "label": "cuBLAS / cuBLASLt",
                "dlls": ("cublasLt64_13.dll", "cublas64_13.dll"),
                "prefixes": ("cublasLt64_13", "cublas64_13"),
                "pypi": "nvidia-cublas",
                "compat_prefix": "13.",
                "required": True,
            },
            {
                "key": "cufft",
                "label": "cuFFT",
                "dlls": ("cufft64_12.dll",),
                "prefixes": ("cufft64_12",),
                "pypi": "nvidia-cufft",
                "compat_prefix": "12.",
                "required": True,
            },
            {
                "key": "cudnn",
                "label": tr("hw.cuda.label.cudnn", "cuDNN 9 (neural network primitives)"),
                "dlls": (
                    "cudnn64_9.dll",
                    "cudnn_ops64_9.dll",
                    "cudnn_graph64_9.dll",
                    "cudnn_adv64_9.dll",
                    "cudnn_heuristic64_9.dll",
                    "cudnn_engines_precompiled64_9.dll",
                    "cudnn_engines_runtime_compiled64_9.dll",
                    "cudnn_engines_tensor_ir64_9.dll",
                ),
                "prefixes": (),
                "pypi": "nvidia-cudnn-cu13",
                "compat_prefix": "9.",
                "required": True,
            },
            {
                "key": "curand",
                "label": tr("hw.cuda.label.curand", "cuRAND (optional)"),
                "dlls": ("curand64_10.dll",),
                "prefixes": ("curand64_10",),
                "pypi": "nvidia-curand",
                "compat_prefix": "10.",
                "required": False,
            },
            {
                "key": "nvrtc",
                "label": tr("hw.cuda.label.nvrtc", "NVRTC JIT (optional)"),
                "dlls": ("nvrtc64_130_0.dll",),
                "prefixes": ("nvrtc64_13",),
                "pypi": "nvidia-cuda-nvrtc",
                "compat_prefix": "13.",
                "required": False,
            },
        ]

    # CUDA 12.x
    return [
        {
            "key": "cuda_runtime",
            "label": "CUDA Runtime 12 (cudart)",
            "dlls": ("cudart64_12.dll",),
            "prefixes": ("cudart64_12",),
            "pypi": "nvidia-cuda-runtime-cu12",
            "compat_prefix": "12.",
            "required": True,
        },
        {
            "key": "cublas",
            "label": "cuBLAS / cuBLASLt 12",
            "dlls": ("cublasLt64_12.dll", "cublas64_12.dll"),
            "prefixes": ("cublasLt64_12", "cublas64_12"),
            "pypi": "nvidia-cublas-cu12",
            "compat_prefix": "12.",
            "required": True,
        },
        {
            "key": "cufft",
            "label": "cuFFT 11",
            "dlls": ("cufft64_11.dll",),
            "prefixes": ("cufft64_11",),
            "pypi": "nvidia-cufft-cu12",
            "compat_prefix": "11.",
            "required": True,
        },
        {
            "key": "cudnn",
            "label": tr("hw.cuda.label.cudnn", "cuDNN 9 (neural network primitives)"),
            "dlls": (
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_graph64_9.dll",
                "cudnn_adv64_9.dll",
                "cudnn_heuristic64_9.dll",
                "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll",
                "cudnn_engines_tensor_ir64_9.dll",
            ),
            "prefixes": (),
            "pypi": "nvidia-cudnn-cu12",
            "compat_prefix": "9.",
            "required": True,
        },
        {
            "key": "curand",
            "label": tr("hw.cuda.label.curand_10", "cuRAND 10 (optional)"),
            "dlls": ("curand64_10.dll",),
            "prefixes": ("curand64_10",),
            "pypi": "nvidia-curand-cu12",
            "compat_prefix": "10.",
            "required": False,
        },
        {
            "key": "nvrtc",
            "label": tr("hw.cuda.label.nvrtc_12", "NVRTC JIT 12 (optional)"),
            "dlls": ("nvrtc64_120_0.dll",),
            "prefixes": ("nvrtc64_12",),
            "pypi": "nvidia-cuda-nvrtc-cu12",
            "compat_prefix": "12.",
            "required": False,
        },
    ]


def _search_win_path_dll(dll_name: str) -> str:
    if sys.platform != "win32":
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.SearchPathW(None, str(dll_name), None, 1024, buf, None)
        if n > 0 and buf.value and os.path.isfile(buf.value):
            return buf.value
    except Exception:
        pass
    return ""


def _find_cuda_candidate_dirs(base_dir: str):
    """Повертає (local_cuda_dir, [(source_tag, dir_path), ...])."""
    local_cuda = get_cuda_dir(base_dir)
    extra = []
    seen = {os.path.normcase(os.path.normpath(local_cuda))}

    def _add(tag, path):
        if not path or not os.path.isdir(path):
            return
        norm = os.path.normcase(os.path.normpath(path))
        if norm in seen:
            return
        seen.add(norm)
        extra.append((tag, path))

    # 1. Встановлені pip-пакети nvidia/* або torch/lib
    for sp_root in list(sys.path):
        if not sp_root or not os.path.isdir(sp_root):
            continue
        nv_root = os.path.join(sp_root, "nvidia")
        if os.path.isdir(nv_root):
            for pat in (
                os.path.join(nv_root, "*", "bin"),
                os.path.join(nv_root, "*", "bin", "x86_64"),
            ):
                for p in glob.glob(pat):
                    _add("site_packages", p)
        torch_lib = os.path.join(sp_root, "torch", "lib")
        _add("site_packages", torch_lib)

    # 2. Системний CUDA Toolkit та cuDNN у Windows
    if sys.platform == "win32":
        for k, v in os.environ.items():
            if k.upper() == "CUDA_PATH" or k.upper().startswith("CUDA_PATH_V"):
                _add("system", os.path.join(v, "bin"))
                _add("system", os.path.join(v, "bin", "x64"))
            elif k.upper().startswith("CUDNN_PATH"):
                _add("system", os.path.join(v, "bin"))

        for pat in (
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\x64",
            r"C:\Program Files\NVIDIA\CUDNN\v*\bin",
            r"C:\Program Files\NVIDIA\CUDNN\v*\bin\*",
        ):
            for p in sorted(glob.glob(pat), reverse=True):
                _add("system", p)

        for p in (os.environ.get("PATH") or "").split(os.pathsep):
            p = p.strip()
            if p:
                _add("system", p)

    return local_cuda, extra


def _locate_dll_with_prefixes(dll_name: str, prefixes: tuple, local_cuda_dir: str, extra_dirs: list):
    """Повертає (source_tag, full_path) або ('missing', '')."""
    def _match_in_dir(dpath: str):
        cand = os.path.join(dpath, dll_name)
        if os.path.isfile(cand):
            return cand
        if prefixes and os.path.isdir(dpath):
            try:
                low_prefs = tuple(p.lower() for p in prefixes)
                for fn in os.listdir(dpath):
                    lfn = fn.lower()
                    if lfn.endswith(".dll") and any(lfn.startswith(pr) for pr in low_prefs):
                        fp = os.path.join(dpath, fn)
                        if os.path.isfile(fp):
                            return fp
            except Exception:
                pass
        return ""

    if os.path.isdir(local_cuda_dir):
        hit = _match_in_dir(local_cuda_dir)
        if hit:
            return "local_cuda", hit

    for tag, dpath in extra_dirs:
        hit = _match_in_dir(dpath)
        if hit:
            return tag, hit

    sys_hit = _search_win_path_dll(dll_name)
    if sys_hit:
        return "system", sys_hit
    return "missing", ""


def _find_tensorrt_runtime(base_dir: str = "", ort_mod=None) -> tuple:
    """
    Перевіряє наявність локального/системного TensorRT (onnxruntime_providers_tensorrt.dll + nvinfer*.dll),
    щоб не провокувати спам помилок C++ provider_bridge_ort.cc при відсутності TensorRT SDK.
    Повертає (has_provider_dll, nvinfer_path).
    """
    has_provider_dll = False
    if ort_mod is not None and getattr(ort_mod, "__file__", None):
        capi_dir = os.path.join(os.path.dirname(os.path.abspath(ort_mod.__file__)), "capi")
        has_provider_dll = os.path.isfile(os.path.join(capi_dir, "onnxruntime_providers_tensorrt.dll"))

    local_cuda, extra_dirs = _find_cuda_candidate_dirs(base_dir or os.getcwd())
    trt_dirs = list(extra_dirs)
    seen = {os.path.normcase(os.path.normpath(d)) for _, d in trt_dirs}
    if sys.platform == "win32":
        for k, v in os.environ.items():
            if "TENSORRT" in k.upper() and v:
                for sub in (v, os.path.join(v, "lib"), os.path.join(v, "bin")):
                    if os.path.isdir(sub):
                        norm = os.path.normcase(os.path.normpath(sub))
                        if norm not in seen:
                            seen.add(norm)
                            trt_dirs.append(("system", sub))
        for pat in (
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT*\lib",
            r"C:\Program Files\NVIDIA\TensorRT*\lib",
            r"C:\Program Files\TensorRT*\lib",
        ):
            for p in sorted(glob.glob(pat), reverse=True):
                norm = os.path.normcase(os.path.normpath(p))
                if norm not in seen:
                    seen.add(norm)
                    trt_dirs.append(("system", p))

    nvinfer_path = ""
    for dll_cand in ("nvinfer_10.dll", "nvinfer_8.dll", "nvinfer.dll"):
        _tag, hit = _locate_dll_with_prefixes(dll_cand, (), local_cuda, trt_dirs)
        if hit:
            nvinfer_path = hit
            break
    return has_provider_dll, nvinfer_path


def _check_msvc_runtime() -> bool:
    if sys.platform != "win32":
        return True
    return bool(_search_win_path_dll("vcruntime140.dll") and _search_win_path_dll("msvcp140.dll"))


def check_cuda_libraries_status(base_dir: str, cuda_major: str = None, allow_network: bool = False) -> dict:
    """
    Перевіряє наявність необхідних бібліотек CUDA та cuDNN у папці <base_dir>/cuda/,
    у site-packages/nvidia/ та у системі Windows (PATH / CUDA Toolkit).
    """
    maj = str(cuda_major or detect_target_cuda_major(base_dir, allow_network=allow_network))
    specs = get_cuda_components_spec(maj)
    local_cuda, extra_dirs = _find_cuda_candidate_dirs(base_dir)
    manifest = _read_cuda_manifest(local_cuda)

    comp_rows = []
    missing_req = []
    missing_any = []
    all_req_local = True

    for spec in specs:
        dlls = list(spec["dlls"])
        prefixes = tuple(spec.get("prefixes") or ())
        found_sources = []
        found_paths = []
        missing_dlls = []

        for dname in dlls:
            src, fpath = _locate_dll_with_prefixes(dname, prefixes, local_cuda, extra_dirs)
            if fpath:
                found_sources.append(src)
                found_paths.append(fpath)
            else:
                missing_dlls.append(dname)

        if not missing_dlls:
            status = "ready"
            if all(s == "local_cuda" for s in found_sources):
                source = "local_cuda"
            elif "local_cuda" in found_sources:
                source = "mixed"
            elif all(s == "site_packages" for s in found_sources):
                source = "site_packages"
            else:
                source = "system"
        else:
            status = "missing"
            source = "partial" if found_paths else "missing"
            missing_any.append(spec["key"])
            if spec["required"]:
                missing_req.append(spec["key"])

        if spec["required"] and (status != "ready" or source != "local_cuda"):
            all_req_local = False

        comp_rows.append({
            "key": spec["key"],
            "label": spec["label"],
            "pypi": spec["pypi"],
            "compat_prefix": spec["compat_prefix"],
            "required": bool(spec["required"]),
            "dlls": dlls,
            "prefixes": list(prefixes),
            "status": status,
            "source": source,
            "found_paths": found_paths,
            "missing_dlls": missing_dlls,
        })

    return {
        "cuda_dir": local_cuda,
        "cuda_major": maj,
        "ready": len(missing_req) == 0,
        "all_required_in_local": all_req_local,
        "missing_required": missing_req,
        "missing_any": missing_any,
        "components": comp_rows,
        "msvc_ok": _check_msvc_runtime(),
        "manifest": manifest,
    }


def prepare_cuda_dll_environment(base_dir: str, preload: bool = True) -> dict:
    """
    Реєструє локальну папку <base_dir>/cuda/ (та системні папки CUDA) у Windows DLL пошуку
    (os.add_dll_directory + PATH) і за потреби викликає ort.preload_dlls().
    """
    local_cuda, extra_dirs = _find_cuda_candidate_dirs(base_dir)
    registered_now = []

    dirs_to_register = []
    if os.path.isdir(local_cuda):
        dirs_to_register.append(local_cuda)
    for tag, dpath in extra_dirs:
        if tag in ("site_packages", "system") and (
            "nvidia" in dpath.lower() or "cuda" in dpath.lower() or "cudnn" in dpath.lower()
        ):
            dirs_to_register.append(dpath)

    with _CUDA_ENV_LOCK:
        cur_path_parts = [p for p in (os.environ.get("PATH") or "").split(os.pathsep) if p]
        cur_norm = {os.path.normcase(os.path.normpath(p)) for p in cur_path_parts}
        to_prepend = []

        for dpath in dirs_to_register:
            if not os.path.isdir(dpath):
                continue
            norm = os.path.normcase(os.path.normpath(dpath))
            if norm not in _CUDA_REGISTERED_DIRS:
                _CUDA_REGISTERED_DIRS.add(norm)
                if hasattr(os, "add_dll_directory"):
                    try:
                        h = os.add_dll_directory(dpath)
                        _CUDA_DLL_DIR_HANDLES.append(h)
                        registered_now.append(dpath)
                    except Exception:
                        pass
            if norm not in cur_norm:
                to_prepend.append(dpath)
                cur_norm.add(norm)

        if to_prepend:
            os.environ["PATH"] = os.pathsep.join(to_prepend + cur_path_parts)

        preloaded = False
        if preload and "onnxruntime" in sys.modules:
            try:
                ort = sys.modules["onnxruntime"]
                has_local_dlls = os.path.isdir(local_cuda) and any(
                    fn.lower().endswith(".dll") for fn in os.listdir(local_cuda)
                )
                # Попереднє завантаження cublasLt/cublas/cudart/cudnn у правильному порядку
                if has_local_dlls:
                    for ordered in (
                        "cublasLt64_13.dll", "cublas64_13.dll", "cudart64_13.dll", "cufft64_12.dll",
                        "cublasLt64_12.dll", "cublas64_12.dll", "cudart64_12.dll", "cufft64_11.dll",
                        "curand64_10.dll",
                        "cudnn_engines_runtime_compiled64_9.dll",
                        "cudnn_engines_precompiled64_9.dll",
                        "cudnn_heuristic64_9.dll",
                        "cudnn_ops64_9.dll",
                        "cudnn_adv64_9.dll",
                        "cudnn_graph64_9.dll",
                        "cudnn64_9.dll",
                        "cudnn_engines_tensor_ir64_9.dll",
                    ):
                        cand = os.path.join(local_cuda, ordered)
                        if os.path.isfile(cand):
                            try:
                                ctypes.CDLL(cand)
                            except Exception:
                                pass
                if hasattr(ort, "preload_dlls"):
                    # Викликаємо лише якщо це GPU-білд ORT (де cuda_version не порожня)
                    from onnxruntime.capi import onnxruntime_validation
                    _pn, _pv, cv = onnxruntime_validation.get_package_name_and_version_info()
                    if cv:
                        ort.preload_dlls(
                            cuda=True,
                            cudnn=True,
                            msvc=True,
                            directory=local_cuda if has_local_dlls else None,
                        )
                        preloaded = True
            except Exception:
                pass

    return {
        "cuda_dir": local_cuda,
        "registered_now": registered_now,
        "preloaded": preloaded,
    }


def _resolve_pypi_win_wheel(pkg_name: str, compat_prefix: str = "", timeout_sec: float = 12.0) -> dict:
    """Повертає URL, версію та SHA256 найновішого win_amd64 .whl на PyPI для пакета."""
    url = f"https://pypi.org/pypi/{pkg_name}/json"
    data = _fetch_pypi_json(url, timeout_sec=timeout_sec)
    releases = data.get("releases") or {}

    def _matches(ver_s: str) -> bool:
        if not compat_prefix:
            return True
        return str(ver_s).startswith(compat_prefix)

    # Перебираємо версії від найновішої до найстарішої (бо остання реліз-версія інколи не має win_amd64 wheel)
    versions = sorted(releases.keys(), key=_parse_ver_tuple, reverse=True)
    for ver in versions:
        if not _matches(ver):
            continue
        files = releases.get(ver) or []
        for fobj in files:
            fname = str(fobj.get("filename") or "")
            if fname.endswith(".whl") and "win_amd64" in fname and not fobj.get("yanked", False):
                return {
                    "package": pkg_name,
                    "version": ver,
                    "filename": fname,
                    "url": str(fobj.get("url") or ""),
                    "size": int(fobj.get("size") or 0),
                    "sha256": str((fobj.get("digests") or {}).get("sha256") or ""),
                }

    # Фолбек: перевірити urls для останнього релізу якщо releases порожні
    for fobj in (data.get("urls") or []):
        fname = str(fobj.get("filename") or "")
        if fname.endswith(".whl") and "win_amd64" in fname:
            return {
                "package": pkg_name,
                "version": str((data.get("info") or {}).get("version") or ""),
                "filename": fname,
                "url": str(fobj.get("url") or ""),
                "size": int(fobj.get("size") or 0),
                "sha256": str((fobj.get("digests") or {}).get("sha256") or ""),
            }

    raise RuntimeError(
        tr(
            "hw.cuda.err_no_wheel",
            "No win_amd64 wheel found on PyPI for '{pkg}' ({compat})",
            pkg=pkg_name,
            compat=compat_prefix or "any",
        )
    )


def download_cuda_libraries(
    base_dir: str,
    progress_cb=None,
    include_optional: bool = True,
    force_into_local: bool = False,
    cuda_major: str = None,
) -> dict:
    """
    Перевіряє наявність бібліотек CUDA/cuDNN і завантажує відсутні (або всі, якщо force_into_local=True)
    з офіційних коліс NVIDIA на PyPI безпосередньо у локальну папку <base_dir>/cuda/.
    Працює без pip, зокрема і всередині скомпільованого EXE.
    """
    with _CUDA_DL_LOCK:
        target_major = str(cuda_major or detect_target_cuda_major(base_dir, allow_network=True))
        status_before = check_cuda_libraries_status(base_dir, cuda_major=target_major, allow_network=True)
        local_cuda = status_before["cuda_dir"]

        to_fetch = []
        for comp in status_before["components"]:
            if not include_optional and not comp["required"]:
                continue
            if force_into_local:
                # Завантажуємо якщо хоч одна DLL цього компонента не лежить у локальній папці cuda/
                in_local = (comp["source"] == "local_cuda")
                if not in_local:
                    to_fetch.append(comp)
            else:
                if comp["status"] != "ready":
                    to_fetch.append(comp)

        if not to_fetch:
            prepare_cuda_dll_environment(base_dir, preload=True)
            return {
                "ok": True,
                "downloaded_packages": [],
                "extracted_dlls": [],
                "cuda_dir": local_cuda,
                "cuda_major": target_major,
                "status": status_before,
                "message": tr(
                    "hw.cuda.dl.already_present",
                    "All required CUDA / cuDNN libraries are already present.",
                ),
            }

        os.makedirs(local_cuda, exist_ok=True)
        tmp_dir = os.path.join(local_cuda, "_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        downloaded_pkgs = []
        extracted_dlls = []

        try:
            total_comps = len(to_fetch)
            for idx, comp in enumerate(to_fetch, start=1):
                pkg_name = comp["pypi"]
                if progress_cb:
                    progress_cb(
                        tr(
                            "hw.cuda.dl.searching",
                            "[{idx}/{total}] Searching PyPI for {pkg}...",
                            idx=idx,
                            total=total_comps,
                            pkg=pkg_name,
                        )
                    )
                winfo = _resolve_pypi_win_wheel(pkg_name, compat_prefix=comp["compat_prefix"])
                whl_path = os.path.join(tmp_dir, winfo["filename"])

                req = urllib.request.Request(winfo["url"], headers={"User-Agent": "ANPR-MultiCam-NVR/2.0"})
                hasher = hashlib.sha256()
                done_bytes = 0
                tot_bytes = max(0, int(winfo.get("size") or 0))
                last_rep = 0.0

                with urllib.request.urlopen(req, timeout=60) as resp, open(whl_path, "wb") as out_f:
                    if tot_bytes <= 0:
                        try:
                            tot_bytes = int(resp.headers.get("Content-Length") or 0)
                        except Exception:
                            tot_bytes = 0
                    while True:
                        chunk = resp.read(262144)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        hasher.update(chunk)
                        done_bytes += len(chunk)
                        now_t = time.time() if "time" in globals() else 0.0
                        if progress_cb and (now_t - last_rep >= 0.2 or done_bytes == tot_bytes):
                            last_rep = now_t
                            mb_done = done_bytes / (1024.0 * 1024.0)
                            if tot_bytes > 0:
                                pct = min(100, int(done_bytes * 100 / tot_bytes))
                                mb_tot = tot_bytes / (1024.0 * 1024.0)
                                progress_cb(
                                    tr(
                                        "hw.cuda.dl.progress_pct",
                                        "[{idx}/{total}] {pkg} {ver}: {done:.1f} / {tot:.1f} MB ({pct}%)",
                                        idx=idx,
                                        total=total_comps,
                                        pkg=pkg_name,
                                        ver=winfo["version"],
                                        done=mb_done,
                                        tot=mb_tot,
                                        pct=pct,
                                    )
                                )
                            else:
                                progress_cb(
                                    tr(
                                        "hw.cuda.dl.progress_mb",
                                        "[{idx}/{total}] {pkg} {ver}: {done:.1f} MB...",
                                        idx=idx,
                                        total=total_comps,
                                        pkg=pkg_name,
                                        ver=winfo["version"],
                                        done=mb_done,
                                    )
                                )

                exp_sha = winfo.get("sha256")
                if exp_sha and hasher.hexdigest().lower() != exp_sha.lower():
                    raise RuntimeError(
                        tr(
                            "hw.cuda.dl.sha_mismatch",
                            "SHA256 checksum mismatch for {filename}",
                            filename=winfo["filename"],
                        )
                    )

                if progress_cb:
                    progress_cb(
                        tr(
                            "hw.cuda.dl.extracting",
                            "[{idx}/{total}] Extracting DLLs from {filename} to cuda/...",
                            idx=idx,
                            total=total_comps,
                            filename=winfo["filename"],
                        )
                    )

                with zipfile.ZipFile(whl_path, "r") as zf:
                    for member in zf.namelist():
                        if not member.lower().endswith(".dll"):
                            continue
                        base_dll = os.path.basename(member)
                        if not base_dll:
                            continue
                        dest_dll = os.path.join(local_cuda, base_dll)
                        tmp_dll = f"{dest_dll}.part"
                        with zf.open(member, "r") as src_m, open(tmp_dll, "wb") as dst_f:
                            shutil.copyfileobj(src_m, dst_f, length=262144)
                        try:
                            os.replace(tmp_dll, dest_dll)
                        except PermissionError:
                            # Якщо DLL уже завантажена цим процесом у пам'ять і заблокована Windows
                            try:
                                os.remove(tmp_dll)
                            except Exception:
                                pass
                        extracted_dlls.append(base_dll)

                try:
                    os.remove(whl_path)
                except Exception:
                    pass

                downloaded_pkgs.append({
                    "package": pkg_name,
                    "version": winfo["version"],
                    "filename": winfo["filename"],
                })

            manifest = _read_cuda_manifest(local_cuda)
            manifest["cuda_major"] = target_major
            manifest["packages"] = {
                **(manifest.get("packages") or {}),
                **{p["package"]: p["version"] for p in downloaded_pkgs},
            }
            all_dlls = sorted(fn for fn in os.listdir(local_cuda) if fn.lower().endswith(".dll"))
            manifest["dll_count"] = len(all_dlls)
            _write_cuda_manifest(local_cuda, manifest)
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
                "downloaded_packages": downloaded_pkgs,
                "extracted_dlls": extracted_dlls,
                "cuda_dir": local_cuda,
                "cuda_major": target_major,
                "status": check_cuda_libraries_status(base_dir, cuda_major=target_major),
                "message": tr(
                    "hw.cuda.dl.err_download",
                    "Error downloading CUDA into cuda/: {err}",
                    err=e,
                ),
            }
        finally:
            try:
                if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                    os.rmdir(tmp_dir)
            except Exception:
                pass

        prepare_cuda_dll_environment(base_dir, preload=True)
        final_status = check_cuda_libraries_status(base_dir, cuda_major=target_major)
        return {
            "ok": bool(final_status["ready"]),
            "downloaded_packages": downloaded_pkgs,
            "extracted_dlls": sorted(set(extracted_dlls)),
            "cuda_dir": local_cuda,
            "cuda_major": target_major,
            "status": final_status,
            "message": (
                tr(
                    "hw.cuda.dl.ok_done",
                    "Downloaded {pkgs} package(s) ({dlls} DLLs) into cuda/.",
                    pkgs=len(downloaded_pkgs),
                    dlls=len(set(extracted_dlls)),
                )
                if final_status["ready"]
                else tr(
                    "hw.cuda.dl.missing_after",
                    "Download finished, but some required DLLs are missing: {missing}",
                    missing=", ".join(final_status["missing_required"]),
                )
            ),
        }


def ensure_cuda_libraries_cli(base_dir: str = ".") -> bool:
    """Зручний консольний виклик для 1_SETUP.bat."""
    st = check_cuda_libraries_status(base_dir, allow_network=True)
    if st["ready"]:
        print(
            tr(
                "hw.cuda.cli.already",
                "[OK] CUDA {maj}.x / cuDNN libraries are already present ({dir}).",
                maj=st["cuda_major"],
                dir=st["cuda_dir"],
            )
        )
        return True
    print(
        tr(
            "hw.cuda.cli.missing",
            "Missing CUDA {maj}.x libraries detected: {missing}",
            maj=st["cuda_major"],
            missing=", ".join(st["missing_required"]),
        )
    )
    print(
        tr(
            "hw.cuda.cli.auto_dl",
            "Auto-downloading to local folder: {dir} ...",
            dir=st["cuda_dir"],
        )
    )
    res = download_cuda_libraries(base_dir, progress_cb=lambda m: print(f"  {m}"), include_optional=True, force_into_local=False)
    print(res.get("message", ""))
    return bool(res.get("ok"))


class HardwareMonitor:
    """Монітор апаратних ресурсів ПК та реальний тестер графічних прискорювачів."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._lock = threading.Lock()
        self._cpu_name = _read_cpu_name_windows()
        self._cpu_phys = psutil.cpu_count(logical=False) if psutil else None
        self._cpu_log = os.cpu_count() or 1
        self._disk_media_map = {}
        self._disk_info_map = {}
        self._wmi_gpus = None
        self._last_caps = None
        self._last_caps_raw = None
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
            dinfo = _detect_disk_info_map_windows()
            wgpus = _detect_gpus_wmi()
            with self._lock:
                self._disk_info_map = dinfo
                self._disk_media_map = {k: v.get("media", "Disk") for k, v in dinfo.items()}
                self._wmi_gpus = wgpus
        threading.Thread(target=_bg, daemon=True).start()

    def refresh_static(self):
        dinfo = _detect_disk_info_map_windows()
        wgpus = _detect_gpus_wmi()
        with self._lock:
            self._disk_info_map = dinfo
            self._disk_media_map = {k: v.get("media", "Disk") for k, v in dinfo.items()}
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
            dinfo_map = dict(self._disk_info_map)
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
                    di = dinfo_map.get(letter) or {}
                    disks.append({
                        "mount": mount,
                        "fs": p.fstype,
                        "media": di.get("media") or dmap.get(letter, "Disk"),
                        "phys_id": di.get("phys_id") or f"mount:{mount}",
                        "phys_model": di.get("phys_model") or "",
                        "phys_size_gb": di.get("phys_size_gb"),
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
                        di = dinfo_map.get(letter) or {}
                        disks.append({
                            "mount": drive,
                            "fs": "",
                            "media": di.get("media") or dmap.get(letter, "Disk"),
                            "phys_id": di.get("phys_id") or f"mount:{drive}",
                            "phys_model": di.get("phys_model") or "",
                            "phys_size_gb": di.get("phys_size_gb"),
                            "total_gb": usage.total / (1024.0 ** 3),
                            "used_gb": usage.used / (1024.0 ** 3),
                            "free_gb": usage.free / (1024.0 ** 3),
                            "percent": round((usage.used / float(max(1, usage.total))) * 100.0, 1),
                        })
                    except Exception:
                        pass

        disks.sort(key=lambda item: str(item.get("mount") or "").upper())

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

    def _format_ort_probe_msg(self, res_tuple) -> tuple:
        ok, code, detail = res_tuple
        if code == "ok_model":
            return ok, tr("hw.ort.probe.ok_model", "Verified on {model}", model=detail)
        if code == "ok_available":
            return ok, tr("hw.ort.probe.ok_available", "Available in onnxruntime")
        if code == "no_module":
            return ok, tr("hw.ort.probe.no_module", "Module onnxruntime not found")
        if code == "not_in_providers":
            return ok, tr("hw.ort.probe.not_in_providers", "Not listed in available_providers()")
        if code == "fallback":
            return ok, tr("hw.ort.probe.fallback", "Fell back to {used}", used=detail)
        if code == "init_error":
            return ok, tr("hw.ort.probe.init_error", "Initialization error: {err}", err=detail)
        return ok, str(detail or "")

    def _format_enc_probe_msg(self, res_tuple) -> tuple:
        ok, code, detail = res_tuple
        if code == "ok_hw":
            return ok, tr("hw.enc.probe.ok_hw", "Hardware test passed")
        if code == "no_ffmpeg":
            return ok, tr("hw.enc.probe.no_ffmpeg", "FFmpeg (imageio-ffmpeg) unavailable")
        if code == "missing_codec":
            return ok, tr("hw.enc.probe.missing_codec", "Codec missing from FFmpeg build")
        if code == "exit_code":
            return ok, tr("hw.enc.probe.exit_code", "Exit code {code}", code=detail)
        return ok, str(detail or "")

    def _build_localized_caps(self, raw: dict) -> dict:
        has_nvidia = bool(raw.get("has_nvidia"))
        has_intel_gpu = bool(raw.get("has_intel_gpu"))
        has_amd = bool(raw.get("has_amd"))
        ort_pkg = str(raw.get("ort_pkg") or "")
        ort_pkg_disp = ort_pkg or tr("hw.ort_not_installed", "not installed")
        ort_loaded_pkg = str(raw.get("ort_loaded_pkg") or "")
        ort_restart_required = bool(raw.get("ort_restart_required"))
        ort_providers = list(raw.get("ort_providers") or [])

        cuda_libs_status = copy.deepcopy(raw.get("cuda_libs_status") or {})
        try:
            spec_labels = {
                s["key"]: s["label"]
                for s in get_cuda_components_spec(cuda_libs_status.get("cuda_major") or "13")
            }
            for comp in (cuda_libs_status.get("components") or []):
                if comp.get("key") in spec_labels:
                    comp["label"] = spec_labels[comp["key"]]
        except Exception:
            pass

        ort_probes = raw.get("ort_probes") or {}
        st_working = tr("hw.status.working", "Working")
        st_unavail = tr("hw.status.unavailable", "Unavailable")
        st_no_nv = tr("hw.status.no_nvidia", "No NVIDIA GPU")
        st_fail_start = tr("hw.status.failed_start", "Failed to start")
        st_need_restart = tr("hw.status.restart_required", "Restart required")

        ai_backends = {}
        ok_cpu, msg_cpu = self._format_ort_probe_msg(ort_probes.get("cpu", (False, "no_module", "")))
        ai_backends["cpu"] = {
            "label": "CPU (ONNX CPUExecutionProvider)",
            "supported": ok_cpu,
            "status": st_working if ok_cpu else st_unavail,
            "reason": msg_cpu,
        }

        miss_req_list = cuda_libs_status.get("missing_required") or []
        miss_str = ", ".join(miss_req_list)
        if "CUDAExecutionProvider" in ort_providers:
            if not has_nvidia:
                ok_cuda = False
                msg_cuda = tr("hw.cuda.no_nvidia_desc", "NVIDIA GPU not detected.")
            elif not cuda_libs_status.get("ready"):
                ok_cuda = False
                msg_cuda = tr(
                    "hw.cuda.missing_libs_desc",
                    "Missing CUDA {maj}.x libraries ({miss}) — download into cuda/ folder.",
                    maj=cuda_libs_status.get("cuda_major"),
                    miss=miss_str,
                )
            else:
                ok_cuda, msg_cuda = self._format_ort_probe_msg(ort_probes.get("cuda", (False, "init_error", "")))
            ai_backends["cuda"] = {
                "label": "NVIDIA CUDA GPU (CUDAExecutionProvider)",
                "supported": ok_cuda,
                "status": st_working if ok_cuda else (st_no_nv if not has_nvidia else st_fail_start),
                "reason": msg_cuda,
            }
        elif ort_restart_required and ort_pkg == "onnxruntime-gpu":
            if not has_nvidia:
                cuda_st_label = st_no_nv
                cuda_reason = tr(
                    "hw.cuda.restart_no_nvidia",
                    "Package 'onnxruntime-gpu' is installed, but no NVIDIA GPU was detected.",
                )
            elif cuda_libs_status.get("ready"):
                cuda_st_label = st_need_restart
                cuda_reason = tr(
                    "hw.cuda.restart_ready",
                    "Package 'onnxruntime-gpu' and CUDA DLLs are on disk, but the previous module is still active in memory. Restart the application.",
                )
            else:
                cuda_st_label = tr("hw.status.restart_plus_cuda", "Restart + CUDA DLL")
                cuda_reason = tr(
                    "hw.cuda.restart_missing_dlls",
                    "Package 'onnxruntime-gpu' is on disk (restart required), missing CUDA DLLs: {miss}.",
                    miss=miss_str,
                )
            ai_backends["cuda"] = {
                "label": "NVIDIA CUDA GPU (CUDAExecutionProvider)",
                "supported": False,
                "status": cuda_st_label,
                "reason": cuda_reason,
            }
        elif ort_pkg == "onnxruntime-gpu":
            cuda_note = (
                tr("hw.cuda.libs_ready_short", "CUDA libraries ready")
                if cuda_libs_status.get("ready")
                else tr("hw.cuda.libs_missing_short", "missing DLLs: {miss}", miss=miss_str)
            )
            ai_backends["cuda"] = {
                "label": "NVIDIA CUDA GPU (CUDAExecutionProvider)",
                "supported": False,
                "status": tr("hw.cuda.status_unavail", "CUDA unavailable"),
                "reason": (
                    tr(
                        "hw.cuda.not_init",
                        "Package 'onnxruntime-gpu' is installed, but CUDAExecutionProvider failed to initialize ({note}).",
                        note=cuda_note,
                    )
                    if has_nvidia
                    else tr("hw.cuda.no_nvidia_desc", "NVIDIA GPU not detected.")
                ),
            }
        else:
            cuda_note = (
                tr("hw.cuda.libs_ready_short", "CUDA libraries ready")
                if cuda_libs_status.get("ready")
                else tr("hw.cuda.libs_missing_short", "missing DLLs: {miss}", miss=miss_str)
            )
            ai_backends["cuda"] = {
                "label": "NVIDIA CUDA GPU (CUDAExecutionProvider)",
                "supported": False,
                "status": tr("hw.cuda.need_ort_cuda", "Needs onnxruntime-gpu + CUDA") if has_nvidia else st_no_nv,
                "reason": (
                    tr(
                        "hw.cuda.need_pkg_desc",
                        "Currently installed: '{pkg}'. CUDA requires 'onnxruntime-gpu' ({note}).",
                        pkg=ort_pkg_disp,
                        note=cuda_note,
                    )
                    if has_nvidia
                    else tr("hw.cuda.no_nvidia_desc", "NVIDIA GPU not detected.")
                ),
            }

        if raw.get("trt_ready"):
            ok_trt, msg_trt = self._format_ort_probe_msg(ort_probes.get("tensorrt", (False, "init_error", "")))
            ai_backends["tensorrt"] = {
                "label": "NVIDIA TensorRT (TensorrtExecutionProvider)",
                "supported": ok_trt,
                "status": st_working if ok_trt else st_fail_start,
                "reason": msg_trt,
            }
        else:
            ai_backends["tensorrt"] = {
                "label": "NVIDIA TensorRT",
                "supported": False,
                "status": tr("hw.status.not_installed", "Not installed") if has_nvidia else st_no_nv,
                "reason": (
                    tr(
                        "hw.trt.missing_sdk",
                        "Standalone NVIDIA TensorRT SDK (nvinfer_10.dll / TRT provider DLL) is absent — CUDA is used for GPU.",
                    )
                    if has_nvidia
                    else tr("hw.cuda.no_nvidia_desc", "NVIDIA GPU not detected.")
                ),
            }

        if "DmlExecutionProvider" in ort_providers:
            ok_dml, msg_dml = self._format_ort_probe_msg(ort_probes.get("directml", (False, "init_error", "")))
            ai_backends["directml"] = {
                "label": "DirectX 12 GPU (DirectML — NVIDIA/AMD/Intel)",
                "supported": ok_dml,
                "status": st_working if ok_dml else st_fail_start,
                "reason": msg_dml,
            }
        elif ort_restart_required and ort_pkg == "onnxruntime-directml":
            ai_backends["directml"] = {
                "label": "DirectX 12 GPU (DirectML — NVIDIA/AMD/Intel)",
                "supported": False,
                "status": st_need_restart,
                "reason": tr(
                    "hw.dml.restart_ready",
                    "Package 'onnxruntime-directml' is already installed on disk, but the previous module is still in memory. Restart the application.",
                ),
            }
        else:
            ai_backends["directml"] = {
                "label": "DirectX 12 GPU (DirectML — NVIDIA/AMD/Intel)",
                "supported": False,
                "status": tr("hw.dml.need_pkg", "Needs onnxruntime-directml"),
                "reason": tr(
                    "hw.dml.need_pkg_desc",
                    "Currently installed: '{pkg}'. Switch package to 'onnxruntime-directml' for DirectML.",
                    pkg=ort_pkg_disp,
                ),
            }

        if "OpenVINOExecutionProvider" in ort_providers:
            ok_ov, msg_ov = self._format_ort_probe_msg(ort_probes.get("openvino", (False, "init_error", "")))
            ai_backends["openvino"] = {
                "label": "Intel OpenVINO (CPU / iGPU)",
                "supported": ok_ov,
                "status": st_working if ok_ov else st_fail_start,
                "reason": msg_ov,
            }
        elif ort_restart_required and ort_pkg == "onnxruntime-openvino":
            ai_backends["openvino"] = {
                "label": "Intel OpenVINO (CPU / iGPU)",
                "supported": False,
                "status": st_need_restart,
                "reason": tr(
                    "hw.ov.restart_ready",
                    "Package 'onnxruntime-openvino' is already installed on disk. Restart the application.",
                ),
            }
        else:
            ai_backends["openvino"] = {
                "label": "Intel OpenVINO (CPU / iGPU)",
                "supported": False,
                "status": (
                    tr("hw.status.not_installed", "Not installed")
                    if has_intel_gpu
                    else tr("hw.ov.optional_intel", "Optional for Intel")
                ),
                "reason": tr(
                    "hw.ov.need_pkg_desc",
                    "Currently installed: '{pkg}'. Requires package 'onnxruntime-openvino'.",
                    pkg=ort_pkg_disp,
                ),
            }

        enc_titles = {
            "copy": tr("hw.enc.title.copy", "Direct RTSP stream copy (copy — no re-encoding)"),
            "libx264": "CPU H.264 (libx264)",
            "h264_nvenc": "NVIDIA NVENC H.264 (GPU)",
            "hevc_nvenc": "NVIDIA NVENC H.265/HEVC (GPU)",
            "h264_qsv": "Intel QuickSync H.264 (QSV)",
            "h264_amf": "AMD AMF H.264 (GPU)",
        }
        ffmpeg_exe_ok = bool(raw.get("ffmpeg_exe_ok"))
        enc_probes = raw.get("enc_probes") or {}
        video_encoders = {}
        for enc_id in ("copy", "libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf"):
            if enc_id == "copy":
                ok_e = ffmpeg_exe_ok
                msg_e = (
                    tr("hw.enc.copy_ok", "Works for network RTSP/HTTP (H.264/H.265) with ~0% CPU")
                    if ok_e
                    else tr("hw.enc.ffmpeg_unavail", "FFmpeg unavailable")
                )
            else:
                ok_e, msg_e = self._format_enc_probe_msg(enc_probes.get(enc_id, (False, "no_ffmpeg", "")))
            video_encoders[enc_id] = {
                "label": enc_titles[enc_id],
                "supported": ok_e,
                "status": st_working if ok_e else st_unavail,
                "reason": msg_e,
            }

        if ai_backends["cuda"]["supported"]:
            rec_ai = "cuda"
            rec_ai_text = tr("hw.rec.ai.cuda_ready", "CUDA (NVIDIA GPU) — maximum speed is ready.")
        elif ai_backends["directml"]["supported"]:
            rec_ai = "directml"
            rec_ai_text = tr("hw.rec.ai.dml_ready", "DirectML (DX12 GPU) — ready for your graphics card.")
        elif ai_backends["openvino"]["supported"]:
            rec_ai = "openvino"
            rec_ai_text = tr("hw.rec.ai.ov_ready", "Intel OpenVINO — ready.")
        elif ort_restart_required and ort_pkg == "onnxruntime-gpu" and has_nvidia:
            rec_ai = "cuda"
            if cuda_libs_status.get("ready"):
                rec_ai_text = tr(
                    "hw.rec.ai.cuda_restart",
                    "Packages 'onnxruntime-gpu' and CUDA are already installed! Restart the app to activate CUDA GPU.",
                )
            else:
                rec_ai_text = tr(
                    "hw.rec.ai.cuda_dl_restart",
                    "Package 'onnxruntime-gpu' is installed. Download CUDA into cuda/ and restart the app.",
                )
        elif ort_restart_required and ort_pkg == "onnxruntime-directml":
            rec_ai = "directml"
            rec_ai_text = tr(
                "hw.rec.ai.dml_restart",
                "Package 'onnxruntime-directml' is already installed! Restart the app to activate GPU.",
            )
        elif ort_restart_required and ort_pkg == "onnxruntime-openvino":
            rec_ai = "openvino"
            rec_ai_text = tr(
                "hw.rec.ai.ov_restart",
                "Package 'onnxruntime-openvino' is already installed! Restart the app to activate OpenVINO.",
            )
        elif has_nvidia:
            if ort_pkg == "onnxruntime-gpu":
                if not cuda_libs_status.get("ready"):
                    rec_ai = "cuda"
                    rec_ai_text = tr(
                        "hw.rec.ai.cuda_miss_libs",
                        "Installed 'onnxruntime-gpu', but CUDA libraries are missing ({miss}). Select CUDA mode or run Full Auto-Setup.",
                        miss=miss_str,
                    )
                else:
                    rec_ai = "directml"
                    rec_ai_text = tr(
                        "hw.rec.ai.cuda_init_fail",
                        "Installed 'onnxruntime-gpu' and CUDA DLLs exist, but CUDA provider failed to start (check NVIDIA driver or switch to DirectML).",
                    )
            else:
                rec_ai = "directml"
                rec_ai_text = tr(
                    "hw.rec.ai.nv_has_cpu_ort",
                    "NVIDIA GPU detected, but current package is CPU-only ('{pkg}'). Recommended: install 'onnxruntime-directml' (1-click without CUDA Toolkit) or 'onnxruntime-gpu' (for CUDA).",
                    pkg=ort_pkg_disp,
                )
        elif has_amd or has_intel_gpu:
            rec_ai = "directml"
            rec_ai_text = tr(
                "hw.rec.ai.igpu_dml",
                "GPU/iGPU detected. Install 'onnxruntime-directml' for hardware acceleration.",
            )
        else:
            rec_ai = "cpu"
            rec_ai_text = tr("hw.rec.ai.cpu_best", "Optimal mode for current hardware — CPU.")

        if video_encoders["h264_nvenc"]["supported"]:
            rec_enc = "h264_nvenc"
            rec_enc_text = tr(
                "hw.rec.enc.nvenc",
                "For IP cameras: 'copy' (0% CPU) or 'h264_nvenc' (RTX GPU compression).",
            )
        elif video_encoders["h264_qsv"]["supported"]:
            rec_enc = "h264_qsv"
            rec_enc_text = tr(
                "hw.rec.enc.qsv",
                "For IP cameras: 'copy' (0% CPU) or 'h264_qsv' (Intel QuickSync).",
            )
        elif video_encoders["h264_amf"]["supported"]:
            rec_enc = "h264_amf"
            rec_enc_text = tr(
                "hw.rec.enc.amf",
                "For IP cameras: 'copy' (0% CPU) or 'h264_amf' (AMD GPU).",
            )
        else:
            rec_enc = "libx264"
            rec_enc_text = tr(
                "hw.rec.enc.cpu",
                "For IP cameras: 'copy' (0% CPU) or 'libx264' (CPU).",
            )

        return {
            "ort_package": ort_pkg,
            "ort_loaded_package": ort_loaded_pkg,
            "ort_restart_required": ort_restart_required,
            "ort_providers": ort_providers,
            "ai_backends": ai_backends,
            "cuda_libraries": cuda_libs_status,
            "video_encoders": video_encoders,
            "hw_decoders": list(raw.get("hw_decoders") or []),
            "opencv_opencl": copy.deepcopy(raw.get("opencv_opencl") or {"supported": False, "device": ""}),
            "opencv_cuda": copy.deepcopy(raw.get("opencv_cuda") or {"supported": False, "count": 0}),
            "ui_renderer": tr("hw.ui_renderer", "Tkinter GDI (CPU) + background thread resizing"),
            "hardware_flags": copy.deepcopy(raw.get("hardware_flags") or {}),
            "recommendations": {
                "ai_backend": rec_ai,
                "ai_text": rec_ai_text,
                "video_encoder": rec_enc,
                "video_text": rec_enc_text,
            },
        }

    def relocalize_cached_caps(self):
        with self._lock:
            raw = copy.deepcopy(self._last_caps_raw) if self._last_caps_raw is not None else None
        if raw is None:
            return self._last_caps
        caps = self._build_localized_caps(raw)
        with self._lock:
            self._last_caps = caps
        return caps

    def probe_acceleration_capabilities(self, force: bool = False) -> dict:
        with self._lock:
            if not force and self._last_caps is not None:
                return self._last_caps

        metrics = self.get_live_metrics()
        gpu_vendors = {g["vendor"] for g in metrics["gpus"]}
        has_nvidia = "NVIDIA" in gpu_vendors
        has_intel_gpu = "Intel" in gpu_vendors or "Intel" in metrics["cpu"]["name"]
        has_amd = "AMD" in gpu_vendors

        prepare_cuda_dll_environment(self.base_dir, preload=False)
        ort_pkg = get_installed_ort_package()
        ort_providers = []
        try:
            import onnxruntime as ort
            _set_ort_quiet(ort)
            observe_loaded_libraries()
            prepare_cuda_dll_environment(self.base_dir, preload=True)
            ort_providers = ort.get_available_providers()
        except Exception:
            ort = None
        ort_loaded_pkg = get_ort_loaded_package()
        ort_restart_required = False
        if "onnxruntime" in sys.modules and ort_pkg:
            if ort_loaded_pkg and ort_pkg != ort_loaded_pkg:
                ort_restart_required = True
            else:
                try:
                    ort_restart_required = any(
                        lr.get("import_name") == "onnxruntime" and lr.get("needs_restart")
                        for lr in get_library_versions()
                    )
                except Exception:
                    ort_restart_required = False
        cuda_libs_status = check_cuda_libraries_status(self.base_dir, allow_network=False)

        probe_model = None
        for cand in ("yolo11n.onnx", "scrfd_face_detector.onnx", "best.onnx", "cct_xs_v2_global.onnx"):
            mp = os.path.join(self.base_dir, "models", cand)
            if os.path.exists(mp):
                probe_model = mp
                break

        def _test_ort_provider_raw(prov_name: str):
            if ort is None:
                return (False, "no_module", "")
            if prov_name not in ort_providers:
                return (False, "not_in_providers", "")
            if probe_model is None:
                return (True, "ok_available", "")
            try:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                sess = ort.InferenceSession(probe_model, opts, providers=[prov_name])
                used = sess.get_providers()
                del sess
                if prov_name in used:
                    return (True, "ok_model", os.path.basename(probe_model))
                return (False, "fallback", str(used))
            except Exception as e:
                return (False, "init_error", str(e).splitlines()[0][:110])

        ort_probes = {
            "cpu": _test_ort_provider_raw("CPUExecutionProvider"),
        }
        if "CUDAExecutionProvider" in ort_providers and has_nvidia and cuda_libs_status.get("ready"):
            ort_probes["cuda"] = _test_ort_provider_raw("CUDAExecutionProvider")

        has_trt_prov, trt_nvinfer = _find_tensorrt_runtime(self.base_dir, ort_mod=ort) if has_nvidia else (False, "")
        trt_ready = bool("TensorrtExecutionProvider" in ort_providers and has_nvidia and has_trt_prov and trt_nvinfer)
        if trt_ready:
            ort_probes["tensorrt"] = _test_ort_provider_raw("TensorrtExecutionProvider")

        if "DmlExecutionProvider" in ort_providers:
            ort_probes["directml"] = _test_ort_provider_raw("DmlExecutionProvider")

        if "OpenVINOExecutionProvider" in ort_providers:
            ort_probes["openvino"] = _test_ort_provider_raw("OpenVINOExecutionProvider")

        ffmpeg_exe = None
        if imageio_ffmpeg is not None:
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass

        encoders_out = ""
        hwaccels_out = ""
        ffmpeg_exe_ok = bool(ffmpeg_exe and os.path.exists(ffmpeg_exe))
        if ffmpeg_exe_ok:
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

        def _test_ffmpeg_encoder_raw(enc_name: str):
            if not ffmpeg_exe:
                return (False, "no_ffmpeg", "")
            if enc_name not in encoders_out:
                return (False, "missing_codec", "")
            dummy_frame = bytes(256 * 256 * 3)
            cmd = [
                ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                "-s", "256x256", "-r", "15", "-i", "-",
                "-frames:v", "1", "-an", "-vcodec", enc_name, "-pix_fmt", "yuv420p",
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1",
            ]
            try:
                res = subprocess.run(
                    cmd, input=dummy_frame, capture_output=True, timeout=6, creationflags=_creation_flags()
                )
                if res.returncode == 0 and len(res.stdout) > 0:
                    return (True, "ok_hw", "")
                err = res.stderr.decode("utf-8", errors="ignore").strip().splitlines()
                if err:
                    return (False, "raw_err", err[-1][:110])
                return (False, "exit_code", str(res.returncode))
            except Exception as e:
                return (False, "raw_err", str(e)[:110])

        enc_probes = {}
        for enc_id in ("libx264", "h264_nvenc", "hevc_nvenc", "h264_qsv", "h264_amf"):
            enc_probes[enc_id] = _test_ffmpeg_encoder_raw(enc_id)

        hw_decoders = []
        for line in hwaccels_out.splitlines():
            w = line.strip().lower()
            if w and " " not in w and not w.endswith(":") and w not in ("hardware", "acceleration", "methods:"):
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

        raw = {
            "has_nvidia": bool(has_nvidia),
            "has_amd": bool(has_amd),
            "has_intel_gpu": bool(has_intel_gpu),
            "ort_pkg": ort_pkg,
            "ort_loaded_pkg": ort_loaded_pkg,
            "ort_restart_required": ort_restart_required,
            "ort_providers": list(ort_providers),
            "cuda_libs_status": cuda_libs_status,
            "ort_probes": ort_probes,
            "trt_ready": trt_ready,
            "ffmpeg_exe_ok": bool(ffmpeg_exe),
            "enc_probes": enc_probes,
            "hw_decoders": hw_decoders,
            "opencv_opencl": {"supported": ocl_available, "device": ocl_device},
            "opencv_cuda": {"supported": cv_cuda_cnt > 0, "count": cv_cuda_cnt},
            "hardware_flags": {
                "has_nvidia": bool(has_nvidia),
                "has_amd": bool(has_amd),
                "has_intel": bool(has_intel_gpu),
                "cpu_name": str((metrics.get("cpu") or {}).get("name") or ""),
                "cpu_cores": (metrics.get("cpu") or {}).get("physical_cores"),
                "cpu_threads": (metrics.get("cpu") or {}).get("logical_threads"),
                "gpus": [
                    {"name": g.get("name", ""), "vendor": g.get("vendor", "Other")}
                    for g in (metrics.get("gpus") or [])
                ],
            },
        }
        caps = self._build_localized_caps(raw)
        with self._lock:
            self._last_caps_raw = raw
            self._last_caps = caps
        return caps


# ------------------------------------------------------------------
#  Повне автоналаштування (1 клік) і підбір оптимальних параметрів
# ------------------------------------------------------------------
def find_missing_python_packages(extra_deps=None, exclude_ort: bool = True) -> list:
    """Повертає список відсутніх Python-пакетів серед базових залежностей і залежностей модулів."""
    installed = _get_normalized_disk_dists_map()
    ort_norms = {"onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "onnxruntime-openvino"}

    candidates = []
    seen = set()
    for raw in list(CORE_REQUIRED_PACKAGES) + list(extra_deps or []):
        orig = str(raw or "").strip()
        norm = _norm_pkg_name(orig)
        if not norm or norm in seen:
            continue
        if exclude_ort and norm in ort_norms:
            continue
        seen.add(norm)
        candidates.append((orig, norm))

    missing = []
    for orig, norm in candidates:
        aliases = _PACKAGE_ALIASES.get(norm, (norm,))
        if not any(_norm_pkg_name(a) in installed for a in aliases):
            missing.append(orig)
    return missing


def ensure_python_packages(packages: list, progress_cb=None) -> dict:
    """Встановлює відсутні Python-пакети через pip у активне оточення (.venv), не чіпаючи вже встановлені."""
    raw_pkgs = [str(p).strip() for p in (packages or []) if str(p or "").strip()]
    installed_map = _get_normalized_disk_dists_map()
    pkgs = []
    seen_req = set()
    for raw in raw_pkgs:
        norm = _norm_pkg_name(raw)
        if not norm or norm in seen_req:
            continue
        seen_req.add(norm)
        aliases = _PACKAGE_ALIASES.get(norm, (norm,))
        if any(_norm_pkg_name(a) in installed_map for a in aliases):
            continue
        pkgs.append(raw)
    if not pkgs:
        return {
            "ok": True,
            "installed": [],
            "failed": [],
            "needs_restart": False,
            "message": tr("hw.pip.all_installed", "All required Python libraries are already installed."),
        }
    if getattr(sys, "frozen", False):
        return {
            "ok": False,
            "installed": [],
            "failed": pkgs,
            "needs_restart": False,
            "message": tr(
                "hw.pip.err_frozen",
                "Installing pip packages is unavailable in compiled EXE (run 1_SETUP.bat).",
            ),
        }

    observe_loaded_libraries()
    py_exe = sys.executable
    if os.path.basename(py_exe).lower() == "pythonw.exe":
        cand = os.path.join(os.path.dirname(py_exe), "python.exe")
        if os.path.exists(cand):
            py_exe = cand

    if progress_cb:
        progress_cb(
            tr("hw.pip.installing_multi", "Installing Python packages: {pkgs}...", pkgs=", ".join(pkgs))
        )

    installed = []
    failed = []
    err_details = []

    try:
        res = subprocess.run(
            [py_exe, "-m", "pip", "install", *pkgs],
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=_creation_flags(),
        )
        if res.returncode == 0:
            installed = list(pkgs)
        else:
            # Якщо пакетний виклик дав збій (напр. один із пакетів конфліктує), пробуємо поодинці
            for pkg in pkgs:
                if progress_cb:
                    progress_cb(tr("hw.pip.installing_one", "Installing {pkg}...", pkg=pkg))
                r_one = subprocess.run(
                    [py_exe, "-m", "pip", "install", pkg],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    creationflags=_creation_flags(),
                )
                if r_one.returncode == 0:
                    installed.append(pkg)
                else:
                    failed.append(pkg)
                    tail = (r_one.stderr or r_one.stdout or "").strip().splitlines()
                    if tail:
                        err_details.append(f"{pkg}: {tail[-1][:140]}")
    except Exception as e:
        failed = list(pkgs)
        err_details.append(str(e))

    importlib.invalidate_caches()
    needs_restart = False
    try:
        needs_restart = any(lr.get("needs_restart") for lr in get_library_versions(extra_pkgs=pkgs))
    except Exception:
        needs_restart = False

    ok = len(failed) == 0
    if ok:
        msg = tr("hw.pip.ok_done", "Installed Python packages: {pkgs}", pkgs=", ".join(installed))
    elif installed:
        msg = tr(
            "hw.pip.partial",
            "Installed: {inst}; failed: {fail} ({err})",
            inst=", ".join(installed),
            fail=", ".join(failed),
            err="; ".join(err_details),
        )
    else:
        msg = tr(
            "hw.pip.failed",
            "Failed to install: {fail} ({err})",
            fail=", ".join(failed),
            err="; ".join(err_details),
        )

    return {
        "ok": ok,
        "installed": installed,
        "failed": failed,
        "needs_restart": needs_restart,
        "message": msg,
    }


def determine_optimal_ort_package(live_metrics: dict = None) -> tuple:
    """
    Визначає найоптимальніший пакет ONNX Runtime для поточного заліза.
    Повертає (target_pkg, fallback_pkg, reason).
    """
    gpus = (live_metrics or {}).get("gpus") if isinstance(live_metrics, dict) else None
    if gpus is None:
        nv = _detect_gpus_nvidia_smi()
        wmi = _detect_gpus_wmi()
        gpus = list(nv)
        nv_names = {g["name"].lower() for g in nv}
        for wg in wmi:
            if wg["name"].lower() not in nv_names and wg["vendor"] != "NVIDIA":
                gpus.append(wg)
        if not gpus:
            gpus = wmi

    vendors = {g.get("vendor") for g in (gpus or [])}
    cpu_name = ((live_metrics or {}).get("cpu") or {}).get("name") or _read_cpu_name_windows()
    has_nv = "NVIDIA" in vendors
    has_amd = "AMD" in vendors
    has_intel_gpu = ("Intel" in vendors) or ("INTEL" in str(cpu_name).upper() and ("ARC" in str(cpu_name).upper() or any("Intel" in g.get("vendor", "") for g in (gpus or []))))

    if has_nv:
        return (
            "onnxruntime-gpu",
            "onnxruntime-directml",
            tr("hw.ort.optimal.nv", "NVIDIA GPU detected → optimal 'onnxruntime-gpu' (CUDA)"),
        )
    if has_amd or has_intel_gpu or len(gpus or []) > 0:
        return (
            "onnxruntime-directml",
            "onnxruntime",
            tr("hw.ort.optimal.dx12", "DX12 GPU detected → optimal 'onnxruntime-directml' (DirectML)"),
        )
    return (
        "onnxruntime",
        "onnxruntime",
        tr("hw.ort.optimal.cpu", "No discrete GPU detected → base 'onnxruntime' (CPU)"),
    )


def compute_optimal_hardware_config(
    caps: dict,
    live_metrics: dict = None,
    current_hw: dict = None,
    active_cameras_count: int = 1,
) -> dict:
    """Обчислює повний оптимальний словник секції config['hardware'] на основі caps і live_metrics."""
    caps = caps or {}
    ai_backends = caps.get("ai_backends") or {}
    cuda_libs = caps.get("cuda_libraries") or {}
    ort_pkg = str(caps.get("ort_package") or "")
    restart_req = bool(caps.get("ort_restart_required"))

    if (ai_backends.get("cuda") or {}).get("supported"):
        chosen_ai = "cuda"
    elif restart_req and ort_pkg == "onnxruntime-gpu" and cuda_libs.get("ready"):
        chosen_ai = "cuda"
    elif (ai_backends.get("directml") or {}).get("supported"):
        chosen_ai = "directml"
    elif restart_req and ort_pkg == "onnxruntime-directml":
        chosen_ai = "directml"
    elif (ai_backends.get("openvino") or {}).get("supported"):
        chosen_ai = "openvino"
    elif restart_req and ort_pkg == "onnxruntime-openvino":
        chosen_ai = "openvino"
    else:
        chosen_ai = "cpu"

    cpu_info = (live_metrics or {}).get("cpu") or {}
    phys = cpu_info.get("physical_cores")
    logi = int(cpu_info.get("logical_threads") or os.cpu_count() or 4)
    cam_cnt = max(1, int(active_cameras_count or 1))

    if chosen_ai != "cpu":
        chosen_threads = 2
    else:
        chosen_threads = min(4, max(2, int(phys or (logi // 2))))

    hw_decoders = caps.get("hw_decoders") or []
    if logi >= 6:
        # На багатопотокових ПК CPU-декод із вибірковим grab/retrieve уникає D3D11/GPU readback-затримок
        # і залишає 100% GPU для CUDA/DirectML інференсу та NVENC кодування
        chosen_decode = "cpu"
    elif chosen_ai in ("directml", "cuda"):
        if "d3d11va" in hw_decoders:
            chosen_decode = "d3d11va"
        elif chosen_ai == "cuda" and "cuda" in hw_decoders:
            chosen_decode = "cuda"
        else:
            chosen_decode = "cpu"
    else:
        chosen_decode = "cpu"

    chosen_opencl = False

    if chosen_ai in ("cuda", "directml"):
        if cam_cnt <= 4:
            ai_fps, grid_fps = 10.0, 15.0
        elif cam_cnt <= 8:
            ai_fps, grid_fps = 8.0, 12.0
        elif cam_cnt <= 16:
            ai_fps, grid_fps = 6.0, 10.0
        elif cam_cnt <= 32:
            ai_fps, grid_fps = 4.0, 8.0
        else:
            ai_fps, grid_fps = 3.0, 6.0
    else:
        if cam_cnt <= 2:
            ai_fps, grid_fps = 8.0, 15.0
        elif cam_cnt <= 6:
            ai_fps, grid_fps = 6.0, 12.0
        elif cam_cnt <= 16:
            ai_fps, grid_fps = 4.0, 8.0
        else:
            ai_fps, grid_fps = 2.0, 5.0

    detail_fps = 25.0
    resize_method = "linear" if (chosen_ai != "cpu" or logi >= 4) else "nearest"

    return {
        "ai_backend": chosen_ai,
        "ai_threads": int(chosen_threads),
        "ai_max_fps_per_cam": float(ai_fps),
        "video_decode_mode": chosen_decode,
        "display_grid_fps": float(grid_fps),
        "display_detail_fps": float(detail_fps),
        "display_resize_method": resize_method,
        "use_opencv_opencl": bool(chosen_opencl),
    }


def build_full_autosetup_plan(
    base_dir: str,
    hw_monitor: HardwareMonitor,
    current_cfg: dict,
    registry=None,
) -> dict:
    """
    Сканує ПК, бібліотеки, ./cuda/, модулі та поточний config без змін на диску
    і формує детальний план дій для вікна підтвердження перед 1-клік автоналаштуванням.
    """
    live_metrics = hw_monitor.get_live_metrics()
    caps = hw_monitor.probe_acceleration_capabilities(force=False)
    extra_deps = registry.collect_library_dependencies() if registry is not None else []
    missing_pkgs = find_missing_python_packages(extra_deps=extra_deps, exclude_ort=True)

    cur_ort = get_installed_ort_package()
    target_ort, fallback_ort, ort_reason = determine_optimal_ort_package(live_metrics)
    need_ort_switch = (cur_ort != target_ort)

    has_nv = any(g.get("vendor") == "NVIDIA" for g in (live_metrics.get("gpus") or []))
    cuda_st = check_cuda_libraries_status(base_dir, allow_network=False) if (has_nv and target_ort == "onnxruntime-gpu") else None
    need_cuda_dl = bool(cuda_st is not None and (not cuda_st.get("ready") or bool(cuda_st.get("missing_any"))))

    mod_checks = registry.check_modules_resources() if registry is not None else []

    active_cams = sum(1 for c in ((current_cfg or {}).get("cameras") or []) if c.get("enabled", True))
    # Прогнозуємо очікувані параметри з урахуванням майбутнього ORT (якщо перемикається)
    sim_caps = copy.deepcopy(caps)
    if target_ort == "onnxruntime-gpu" and has_nv:
        sim_caps.setdefault("ai_backends", {}).setdefault("cuda", {})["supported"] = True
    elif target_ort == "onnxruntime-directml":
        sim_caps.setdefault("ai_backends", {}).setdefault("directml", {})["supported"] = True
    pred_hw = compute_optimal_hardware_config(
        sim_caps,
        live_metrics=live_metrics,
        current_hw=(current_cfg or {}).get("hardware"),
        active_cameras_count=max(1, active_cams),
    )

    cam_recs = []
    if registry is not None:
        tpl_rec = registry.recommend_camera_modules(
            sim_caps,
            cam_cfg=None,
            cur_modules=(current_cfg or {}).get("default_camera_modules"),
            active_cam_count=max(1, active_cams),
        )
        for nitem in tpl_rec.get("notes") or []:
            cam_recs.append(
                tr(
                    "hw.plan.tpl_note",
                    "New cameras template ({mod}): {notes}",
                    mod=nitem["name"],
                    notes="; ".join(nitem["notes"]),
                )
            )
        for cam in ((current_cfg or {}).get("cameras") or []):
            crec = registry.recommend_camera_modules(
                sim_caps,
                cam_cfg=cam,
                cur_modules=cam.get("modules"),
                active_cam_count=max(1, active_cams),
            )
            cname = cam.get("name") or (cam.get("uuid") or "")[:8] or tr("camera.default_single", "Camera")
            for nitem in crec.get("notes") or []:
                cam_recs.append(f"{cname} ({nitem['name']}): {'; '.join(nitem['notes'])}")

    return {
        "frozen": bool(getattr(sys, "frozen", False)),
        "live_metrics": live_metrics,
        "caps": caps,
        "missing_packages": missing_pkgs,
        "cur_ort": cur_ort,
        "target_ort": target_ort,
        "fallback_ort": fallback_ort,
        "ort_reason": ort_reason,
        "need_ort_switch": need_ort_switch,
        "cuda_status": cuda_st,
        "need_cuda_dl": need_cuda_dl,
        "module_checks": mod_checks,
        "predicted_hw": pred_hw,
        "camera_recommendations": cam_recs,
    }


def run_full_autosetup(
    base_dir: str,
    hw_monitor: HardwareMonitor,
    draft_cfg: dict,
    registry=None,
    progress_cb=None,
    pip_install_pkgs_fn=None,
    ort_install_fn=None,
    cuda_dl_fn=None,
) -> dict:
    """
    Повне 1-клік автоналаштування:
      1) перевірка й встановлення відсутніх Python-бібліотек;
      2) автовибір та встановлення оптимального ONNX Runtime (CUDA -> DirectML -> CPU);
      3) перевірка й завантаження відсутніх CUDA / cuDNN DLL у ./cuda/;
      4) перевірка й завантаження ресурсів/моделей модулів + оновлення ModuleRegistry;
      5) передіагностика заліза та обчислення оптимальних hardware + модульних параметрів;
      6) формування звіту змін і стану restart_required.
    """
    logs = []
    errors = []
    warnings = []
    frozen = bool(getattr(sys, "frozen", False))

    def _emit(step: int, total: int, pct: int, msg: str):
        line = f"[{step}/{total}] {msg}"
        logs.append(line)
        if progress_cb:
            try:
                progress_cb({"step": step, "total": total, "pct": pct, "message": msg, "line": line})
            except Exception:
                pass

    total_steps = 6
    _emit(1, total_steps, 5, tr("hw.setup.step1.checking", "Checking required Python libraries..."))
    extra_deps = registry.collect_library_dependencies() if registry is not None else []
    missing_pkgs = find_missing_python_packages(extra_deps=extra_deps, exclude_ort=True)

    pip_res = {"ok": True, "installed": [], "failed": [], "needs_restart": False, "message": ""}
    if missing_pkgs:
        if frozen and pip_install_pkgs_fn is None:
            warnings.append(
                tr(
                    "hw.setup.step1.warn_frozen",
                    "Skipped installing Python packages in EXE build ({pkgs}).",
                    pkgs=", ".join(missing_pkgs),
                )
            )
            _emit(1, total_steps, 15, tr("hw.setup.step1.skip_frozen", "⚠ Skipped in EXE: {pkgs}", pkgs=", ".join(missing_pkgs)))
        else:
            fn_pip = pip_install_pkgs_fn or ensure_python_packages
            pip_res = fn_pip(missing_pkgs, progress_cb=lambda t: _emit(1, total_steps, 12, t))
            if not pip_res.get("ok"):
                errors.append(pip_res.get("message") or tr("hw.setup.step1.err", "Failed to install some Python libraries."))
            _emit(1, total_steps, 18, pip_res.get("message") or tr("hw.setup.step1.checked", "Libraries checked."))
    else:
        _emit(1, total_steps, 18, tr("hw.setup.step1.all_present", "✔ All base libraries and module dependencies are already installed."))

    # Крок 2: ONNX Runtime
    _emit(2, total_steps, 22, tr("hw.setup.step2.selecting", "Selecting optimal ONNX Runtime package for PC graphics hardware..."))
    live_metrics = hw_monitor.get_live_metrics()
    target_ort, fallback_ort, ort_reason = determine_optimal_ort_package(live_metrics)
    cur_ort = get_installed_ort_package()
    cur_ort_disp = cur_ort or tr("hw.ort_not_installed", "not installed")
    ort_switched = False
    active_ort_pkg = cur_ort

    if cur_ort != target_ort:
        if frozen and ort_install_fn is None:
            warnings.append(
                tr(
                    "hw.setup.step2.warn_frozen",
                    "Current ONNX Runtime in EXE: '{cur}' (recommended: '{target}' via 1_SETUP.bat).",
                    cur=cur_ort_disp,
                    target=target_ort,
                )
            )
            _emit(2, total_steps, 35, tr("hw.setup.step2.exe_mode", "ℹ EXE mode: using '{cur}'.", cur=cur_ort_disp))
        else:
            _emit(
                2,
                total_steps,
                26,
                tr(
                    "hw.setup.step2.switching",
                    "Switching ONNX Runtime: '{cur}' → '{target}' ({reason})...",
                    cur=cur_ort_disp,
                    target=target_ort,
                    reason=ort_reason,
                ),
            )
            fn_ort = ort_install_fn or install_ort_package
            ok_ort, msg_ort = fn_ort(target_ort, progress_cb=lambda t: _emit(2, total_steps, 32, t))
            if ok_ort:
                ort_switched = True
                active_ort_pkg = target_ort
                _emit(2, total_steps, 38, f"✔ {msg_ort}")
            else:
                warnings.append(
                    tr("hw.setup.step2.fail_target", "Failed to install '{target}': {err}", target=target_ort, err=msg_ort)
                )
                if fallback_ort and fallback_ort != target_ort and fallback_ort != cur_ort:
                    _emit(2, total_steps, 35, tr("hw.setup.step2.trying_fb", "Trying fallback '{fb}'...", fb=fallback_ort))
                    ok_fb, msg_fb = fn_ort(fallback_ort, progress_cb=lambda t: _emit(2, total_steps, 37, t))
                    if ok_fb:
                        ort_switched = True
                        active_ort_pkg = fallback_ort
                        _emit(2, total_steps, 38, tr("hw.setup.step2.ok_fb", "✔ Installed fallback: {msg}", msg=msg_fb))
                    else:
                        errors.append(
                            tr(
                                "hw.setup.step2.err_both",
                                "Failed to install ONNX Runtime ({target} / {fb}): {err}",
                                target=target_ort,
                                fb=fallback_ort,
                                err=msg_fb,
                            )
                        )
                else:
                    errors.append(msg_ort)
    else:
        _emit(
            2,
            total_steps,
            38,
            tr("hw.setup.step2.already", "✔ Optimal ONNX Runtime ('{cur}') is already installed.", cur=cur_ort_disp),
        )

    # Крок 3: CUDA / cuDNN DLL у ./cuda/
    _emit(3, total_steps, 42, tr("hw.setup.step3.checking", "Checking NVIDIA CUDA / cuDNN libraries in cuda/ folder..."))
    has_nv = any(g.get("vendor") == "NVIDIA" for g in (live_metrics.get("gpus") or [])) or has_nvidia_gpu()
    cuda_downloaded = False
    cur_ort_now = active_ort_pkg if ort_switched else get_installed_ort_package()

    if has_nv and cur_ort_now == "onnxruntime-gpu":
        cuda_st = check_cuda_libraries_status(base_dir, allow_network=True)
        if not cuda_st.get("ready") or bool(cuda_st.get("missing_any")):
            fn_cuda = cuda_dl_fn or download_cuda_libraries
            dl_res = fn_cuda(
                base_dir,
                progress_cb=lambda t: _emit(3, total_steps, 52, t),
                include_optional=True,
                force_into_local=False,
            )
            cuda_downloaded = bool(dl_res.get("downloaded_packages"))
            if dl_res.get("ok"):
                _emit(3, total_steps, 58, dl_res.get("message") or tr("hw.setup.step3.ready_local", "✔ CUDA libraries ready in ./cuda/."))
            else:
                # Якщо обов'язкові все одно не готові і не EXE -> м'який відкат на DirectML, якщо доступно
                st_after_dl = dl_res.get("status") or check_cuda_libraries_status(base_dir, allow_network=False)
                if st_after_dl.get("ready"):
                    warnings.append(
                        tr("hw.setup.step3.warn_opt", "Optional CUDA components failed to download: {msg}", msg=dl_res.get("message"))
                    )
                    _emit(3, total_steps, 58, tr("hw.setup.step3.req_ready", "✔ Required CUDA libraries ready."))
                else:
                    warnings.append(
                        tr("hw.setup.step3.warn_dl", "Failed to download CUDA DLLs: {msg}", msg=dl_res.get("message"))
                    )
                    if not frozen and ort_install_fn is None and fallback_ort == "onnxruntime-directml":
                        _emit(3, total_steps, 56, tr("hw.setup.step3.fallback_dml", "Attempting switch to DirectML (no CUDA DLLs)..."))
                        ok_dml, msg_dml = install_ort_package("onnxruntime-directml", progress_cb=lambda t: _emit(3, total_steps, 57, t))
                        if ok_dml:
                            ort_switched = True
                            active_ort_pkg = "onnxruntime-directml"
                            _emit(3, total_steps, 58, f"✔ {msg_dml}")
                    elif not frozen and ort_install_fn is not None and fallback_ort == "onnxruntime-directml":
                        ok_dml, msg_dml = ort_install_fn("onnxruntime-directml", progress_cb=lambda t: _emit(3, total_steps, 57, t))
                        if ok_dml:
                            ort_switched = True
                            active_ort_pkg = "onnxruntime-directml"
                            _emit(3, total_steps, 58, f"✔ {msg_dml}")
        else:
            prepare_cuda_dll_environment(base_dir, preload=True)
            _emit(
                3,
                total_steps,
                58,
                tr("hw.setup.step3.already", "✔ All CUDA {maj}.x / cuDNN 9 libraries are ready.", maj=cuda_st.get("cuda_major")),
            )
    else:
        _emit(3, total_steps, 58, tr("hw.setup.step3.not_needed", "✔ Additional CUDA DLLs are not required for selected provider."))

    # Крок 4: Ресурси / моделі модулів і оновлення реєстру
    _emit(4, total_steps, 62, tr("hw.setup.step4.checking", "Checking independent module resources and models..."))
    mod_resources = []
    if registry is not None:
        try:
            registry.scan_and_load(force=True)
        except Exception as e:
            warnings.append(tr("hw.setup.step4.reg_err", "Module registry refresh error: {err}", err=e))
        mod_resources = registry.ensure_modules_resources(progress_cb=lambda t: _emit(4, total_steps, 70, t))
        for mr in mod_resources:
            if mr.get("downloaded"):
                _emit(
                    4,
                    total_steps,
                    72,
                    tr("hw.setup.step4.downloaded", "✔ [{name}] downloaded: {items}", name=mr["name"], items=", ".join(mr["downloaded"])),
                )
            elif not mr.get("ok"):
                warnings.append(tr("hw.setup.step4.warn_mod", "Module \"{name}\": {msg}", name=mr["name"], msg=mr.get("message")))
                _emit(4, total_steps, 72, f"⚠ [{mr['name']}] {mr.get('message')}")
            else:
                _emit(4, total_steps, 72, f"✔ [{mr['name']}] {mr.get('message')}")
    else:
        _emit(4, total_steps, 72, tr("hw.setup.step4.no_reg", "Module registry not provided, step skipped."))

    # Крок 5: Передіагностика заліза та підбір оптимальних параметрів
    _emit(5, total_steps, 78, tr("hw.setup.step5.probing", "Testing GPU/FFmpeg/OpenCL and computing optimal settings..."))
    caps = hw_monitor.refresh_all(force=True)
    live_metrics = hw_monitor.get_live_metrics()
    new_cfg = copy.deepcopy(draft_cfg or {})

    cams = new_cfg.setdefault("cameras", [])
    active_cams = sum(1 for c in cams if c.get("enabled", True))
    cur_hw = copy.deepcopy(new_cfg.get("hardware") or {})
    opt_hw = compute_optimal_hardware_config(
        caps,
        live_metrics=live_metrics,
        current_hw=cur_hw,
        active_cameras_count=max(1, active_cams),
    )

    hw_labels = {
        "ai_backend": tr("hw.setup.hw_label.ai_backend", "AI Backend"),
        "ai_threads": tr("hw.setup.hw_label.ai_threads", "ONNX/AI Threads"),
        "ai_max_fps_per_cam": tr("hw.setup.hw_label.ai_fps", "Max AI FPS/camera"),
        "video_decode_mode": tr("hw.setup.hw_label.decode", "Stream Decoding"),
        "display_grid_fps": tr("hw.setup.hw_label.grid_fps", "Camera Grid FPS"),
        "display_detail_fps": tr("hw.setup.hw_label.detail_fps", "Detail View FPS"),
        "display_resize_method": tr("hw.setup.hw_label.resize", "UI Resize Method"),
        "use_opencv_opencl": tr("hw.setup.hw_label.opencl", "OpenCV OpenCL (GPU UMat)"),
    }
    hw_changes = []
    for k, nv in opt_hw.items():
        ov = cur_hw.get(k)
        changed = (ov != nv)
        hw_changes.append({"key": k, "label": hw_labels.get(k, k), "old": ov, "new": nv, "changed": changed})
    new_cfg.setdefault("hardware", {}).update(opt_hw)

    module_changes = []
    tpl_target_label = tr("hw.setup.tpl_target", "New cameras template")
    if registry is not None:
        # 5a. Шаблон для нових камер
        tpl_mods = new_cfg.setdefault("default_camera_modules", {})
        tpl_rec = registry.recommend_camera_modules(
            caps,
            cam_cfg=None,
            cur_modules=tpl_mods,
            active_cam_count=max(1, active_cams),
        )
        for mid, upd in (tpl_rec.get("updates_by_mod") or {}).items():
            msec = tpl_mods.setdefault(mid, {})
            for pk, pv in upd.items():
                old_pv = msec.get(pk)
                msec[pk] = copy.deepcopy(pv)
                module_changes.append({
                    "target": tpl_target_label,
                    "mod_id": mid,
                    "key": pk,
                    "old": old_pv,
                    "new": pv,
                    "changed": (old_pv != pv),
                })

        # 5b. Кожна камера (враховує rtsp:// / http:// -> copy vs файл/вебкамера -> HW/SW кодек)
        for cam in cams:
            cmods = cam.setdefault("modules", {})
            crec = registry.recommend_camera_modules(
                caps,
                cam_cfg=cam,
                cur_modules=cmods,
                active_cam_count=max(1, active_cams),
            )
            cname = str(cam.get("name") or (cam.get("uuid") or "")[:8] or tr("camera.default_single", "Camera"))
            for mid, upd in (crec.get("updates_by_mod") or {}).items():
                msec = cmods.setdefault(mid, {})
                for pk, pv in upd.items():
                    old_pv = msec.get(pk)
                    msec[pk] = copy.deepcopy(pv)
                    module_changes.append({
                        "target": cname,
                        "mod_id": mid,
                        "key": pk,
                        "old": old_pv,
                        "new": pv,
                        "changed": (old_pv != pv),
                    })

    _emit(
        5,
        total_steps,
        90,
        tr(
            "hw.setup.step5.profile",
            "✔ Optimal profile: AI={ai}, decode={dec}, resize={res}",
            ai=opt_hw["ai_backend"].upper(),
            dec=opt_hw["video_decode_mode"],
            res=opt_hw["display_resize_method"],
        ),
    )

    op_mode = str((new_cfg.get("general") or {}).get("operation_mode", "server") or "server").strip().lower()
    if op_mode == "operator":
        warnings.append(
            tr(
                "hw.setup.operator_warn",
                "Operator mode is active: disk recording remains disabled; live AI and decoding are optimized.",
            )
        )

    # Крок 6: Перевірка потреби перезапуску
    libs_restart = False
    try:
        libs_restart = any(lr.get("needs_restart") for lr in get_library_versions(extra_pkgs=extra_deps))
    except Exception:
        libs_restart = False
    restart_required = bool(caps.get("ort_restart_required") or libs_restart or pip_res.get("needs_restart"))

    if restart_required:
        _emit(
            6,
            total_steps,
            100,
            tr(
                "hw.setup.step6.restart",
                "⚠ Settings saved, but an application restart is required to activate updated libraries in memory!",
            ),
        )
    else:
        _emit(6, total_steps, 100, tr("hw.setup.step6.done", "✔ Full auto-setup completed successfully!"))

    return {
        "ok": len(errors) == 0,
        "config": new_cfg,
        "caps": caps,
        "live_metrics": live_metrics,
        "cuda_status": caps.get("cuda_libraries"),
        "installed_packages": list(pip_res.get("installed") or []),
        "failed_packages": list(pip_res.get("failed") or []),
        "ort_package": caps.get("ort_package"),
        "ort_switched": ort_switched,
        "cuda_downloaded": cuda_downloaded,
        "module_resources": mod_resources,
        "hw_changes": hw_changes,
        "module_changes": module_changes,
        "restart_required": restart_required,
        "logs": logs,
        "errors": errors,
        "warnings": warnings,
    }


def assess_ai_modes(base_dir: str, caps: dict = None, live_metrics: dict = None) -> dict:
    """
    Повертає структуровану оцінку всіх режимів AI ('auto', 'cuda', 'directml', 'openvino', 'cpu')
    для поточного ПК: апаратну підтримку, режим із найкращою швидкодією та точний список кроків
    (лише того, що справді потрібно довстановити для вибраного режиму, без перевстановлення наявного).
    """
    caps = caps if isinstance(caps, dict) else {}
    hw_flags = caps.get("hardware_flags") if isinstance(caps.get("hardware_flags"), dict) else {}
    ai_backends = caps.get("ai_backends") if isinstance(caps.get("ai_backends"), dict) else {}

    if isinstance(live_metrics, dict) and ("gpus" in live_metrics or "cpu" in live_metrics):
        gpus = list(live_metrics.get("gpus") or [])
        cpu_info = live_metrics.get("cpu") or {}
        cpu_name = str(cpu_info.get("name") or hw_flags.get("cpu_name") or _read_cpu_name_windows())
        phys = cpu_info.get("physical_cores") or hw_flags.get("cpu_cores")
        logi = cpu_info.get("logical_threads") or hw_flags.get("cpu_threads") or (os.cpu_count() or 4)
        has_nvidia = any(g.get("vendor") == "NVIDIA" for g in gpus)
        has_amd = any(g.get("vendor") == "AMD" for g in gpus)
        has_intel = any(g.get("vendor") == "Intel" for g in gpus) or ("INTEL" in cpu_name.upper())
    elif hw_flags:
        gpus = list(hw_flags.get("gpus") or [])
        cpu_name = str(hw_flags.get("cpu_name") or _read_cpu_name_windows())
        phys = hw_flags.get("cpu_cores")
        logi = hw_flags.get("cpu_threads") or (os.cpu_count() or 4)
        has_nvidia = bool(hw_flags.get("has_nvidia"))
        has_amd = bool(hw_flags.get("has_amd"))
        has_intel = bool(hw_flags.get("has_intel"))
    else:
        nv = _detect_gpus_nvidia_smi()
        wmi = _detect_gpus_wmi()
        gpus = list(nv)
        nv_names = {g["name"].lower() for g in nv}
        for wg in wmi:
            if wg["name"].lower() not in nv_names and wg["vendor"] != "NVIDIA":
                gpus.append(wg)
        if not gpus:
            gpus = wmi
        cpu_name = _read_cpu_name_windows()
        phys = psutil.cpu_count(logical=False) if psutil is not None else None
        logi = (psutil.cpu_count(logical=True) if psutil is not None else None) or (os.cpu_count() or 4)
        has_nvidia = any(g.get("vendor") == "NVIDIA" for g in gpus)
        has_amd = any(g.get("vendor") == "AMD" for g in gpus)
        has_intel = any(g.get("vendor") == "Intel" for g in gpus) or ("INTEL" in cpu_name.upper())

    if (ai_backends.get("cuda") or {}).get("supported"):
        has_nvidia = True

    nv_names = [g.get("name", "NVIDIA GPU") for g in gpus if g.get("vendor") == "NVIDIA"]
    gpu_names = [g.get("name", "GPU") for g in gpus]
    has_any_gpu = bool(gpus)

    ort_pkg_raw = str(caps.get("ort_package") or get_installed_ort_package() or "")
    if ort_pkg_raw in ("не встановлено", "not installed"):
        ort_pkg_raw = ""
    ort_pkg = ort_pkg_raw or tr("hw.ort_not_installed", "not installed")
    ort_loaded_pkg = str(caps.get("ort_loaded_package") or get_ort_loaded_package() or "")
    ort_restart_req = bool(caps.get("ort_restart_required"))
    frozen = bool(getattr(sys, "frozen", False))

    cuda_st = (
        caps.get("cuda_libraries")
        if isinstance(caps.get("cuda_libraries"), dict)
        else check_cuda_libraries_status(base_dir, allow_network=False)
    )
    cuda_major = str(cuda_st.get("cuda_major") or "13")
    cuda_ready = bool(cuda_st.get("ready"))
    cuda_local = bool(cuda_st.get("all_required_in_local"))
    miss_cuda_req = list(cuda_st.get("missing_required") or [])
    miss_cuda_opt = list(cuda_st.get("missing_optional") or [])
    msvc_ok = bool(cuda_st.get("msvc_ok", True))

    if has_nvidia:
        best_mode = "cuda"
    elif (sys.platform == "win32" and has_any_gpu) or (ai_backends.get("directml") or {}).get("supported"):
        best_mode = "directml"
    elif (ai_backends.get("openvino") or {}).get("supported"):
        best_mode = "openvino"
    else:
        best_mode = "cpu"

    if (ai_backends.get("cuda") or {}).get("supported"):
        eff_auto = "cuda"
    elif (ai_backends.get("directml") or {}).get("supported"):
        eff_auto = "directml"
    elif (ai_backends.get("openvino") or {}).get("supported"):
        eff_auto = "openvino"
    else:
        eff_auto = "cpu"

    modes = {}

    # 1. NVIDIA CUDA
    cuda_cap = ai_backends.get("cuda") or {}
    cuda_steps = []
    if not has_nvidia:
        cuda_steps.append({
            "done": False,
            "kind": "hw",
            "label": tr("hw.mode.cuda.step_no_gpu", "NVIDIA GPU not detected"),
            "detail": tr("hw.mode.cuda.step_no_gpu_det", "CUDA mode works only on NVIDIA GeForce / RTX / Quadro GPUs."),
        })
        modes["cuda"] = {
            "id": "cuda",
            "short_title": "🚀 CUDA",
            "title": "NVIDIA CUDA GPU",
            "role": tr("hw.mode.role.max_speed", "Max speed"),
            "hw_supported": False,
            "is_best": False,
            "state": "unavailable",
            "badge": tr("hw.mode.badge.no_gpu", "✖ No GPU"),
            "headline": tr("hw.mode.cuda.no_gpu_head", "No NVIDIA GPU was detected on this PC."),
            "steps": cuda_steps,
            "install_plan": {
                "target_mode": "cuda",
                "need_ort_pkg": None,
                "need_cuda_dlls": False,
                "needs_restart": False,
                "can_install": False,
                "action_label": "",
            },
        }
    else:
        cuda_steps.append({
            "done": True,
            "kind": "hw",
            "label": tr(
                "hw.mode.cuda.step_hw_ok",
                "NVIDIA GPU: {gpus}",
                gpus=", ".join(nv_names) if nv_names else tr("hw.mode.detected", "detected"),
            ),
            "detail": tr("hw.mode.cuda.step_hw_ok_det", "CUDA hardware acceleration is supported."),
        })
        need_gpu_ort = (ort_pkg_raw != "onnxruntime-gpu")
        if not need_gpu_ort:
            cuda_steps.append({
                "done": True,
                "kind": "ort",
                "label": tr("hw.mode.cuda.step_ort_ok", "ONNX Runtime package: onnxruntime-gpu already installed"),
                "detail": tr("hw.mode.ort_kept_det", "Existing package is preserved — no reinstall needed."),
            })
        else:
            cuda_steps.append({
                "done": False,
                "kind": "ort",
                "label": tr("hw.mode.cuda.step_ort_need", "Requires 'onnxruntime-gpu' (currently '{pkg}')", pkg=ort_pkg),
                "detail": (
                    tr("hw.mode.cuda.step_ort_need_det", "Switches only the ONNX Runtime package without reinstalling other modules.")
                    if not frozen
                    else tr("hw.mode.ort_exe_det", "In the EXE build, install via 1_SETUP.bat.")
                ),
            })

        need_cuda_dl = not cuda_ready
        if cuda_ready:
            loc_txt = (
                tr("hw.mode.cuda.loc_local", "in local cuda/ folder")
                if cuda_local
                else tr("hw.mode.cuda.loc_sys", "in system / cuda/")
            )
            opt_miss_note = (
                tr("hw.mode.cuda.opt_miss", " Optionally missing: {miss}.", miss=", ".join(miss_cuda_opt))
                if miss_cuda_opt
                else ""
            )
            cuda_steps.append({
                "done": True,
                "kind": "cuda_dll",
                "label": tr("hw.mode.cuda.step_dll_ok", "CUDA {maj}.x + cuDNN 9 libraries: ready ({loc})", maj=cuda_major, loc=loc_txt),
                "detail": tr("hw.mode.cuda.step_dll_ok_det", "All required DLLs are present.") + opt_miss_note,
            })
        else:
            miss_str = ", ".join(miss_cuda_req) if miss_cuda_req else "CUDA/cuDNN DLL"
            cuda_steps.append({
                "done": False,
                "kind": "cuda_dll",
                "label": tr("hw.mode.cuda.step_dll_need", "Missing CUDA {maj}.x / cuDNN 9 DLLs ({miss})", maj=cuda_major, miss=miss_str),
                "detail": tr("hw.mode.cuda.step_dll_need_det", "Automatically downloaded from PyPI into ./cuda/ (existing DLLs are not re-downloaded)."),
            })

        if not msvc_ok:
            cuda_steps.append({
                "done": False,
                "kind": "info",
                "label": tr("hw.mode.cuda.step_msvc", "Missing VC++ Redistributable (vcruntime140.dll)"),
                "detail": tr("hw.mode.cuda.step_msvc_det", "Installing Microsoft Visual C++ 2015-2022 Redistributable x64 is recommended."),
            })

        cuda_needs_restart = bool((not need_gpu_ort) and cuda_ready and ort_restart_req)
        if cuda_needs_restart:
            cuda_steps.append({
                "done": False,
                "kind": "restart",
                "label": tr("hw.mode.step_restart", "Application restart required ('{pkg}' is still in memory)", pkg=ort_loaded_pkg or "onnxruntime"),
                "detail": tr("hw.mode.cuda.step_restart_det", "Files on disk are ready — simply restart the application."),
            })

        can_inst_cuda = bool((need_gpu_ort and not frozen) or need_cuda_dl)
        if cuda_cap.get("supported"):
            c_state = "ready"
            c_badge = tr("hw.mode.badge.ready", "✔ Ready")
            c_head = tr("hw.mode.cuda.head_ready", "NVIDIA CUDA is fully ready — highest performance on this PC.")
            c_act = ""
        elif need_gpu_ort or need_cuda_dl:
            c_state = "needs_setup"
            c_badge = tr("hw.mode.cuda.badge_add", "⚙ Add CUDA")
            if need_gpu_ort and need_cuda_dl:
                c_act = tr("hw.mode.cuda.act_install", "⬇ Install CUDA")
                c_head = tr("hw.mode.cuda.head_need_both", "To run CUDA, install onnxruntime-gpu and download missing CUDA DLLs.")
            elif need_cuda_dl:
                c_act = tr("hw.mode.cuda.act_dl_dll", "⬇ Download CUDA DLLs")
                c_head = tr("hw.mode.cuda.head_need_dll", "onnxruntime-gpu is installed — just download missing DLLs into ./cuda/.")
            else:
                c_act = tr("hw.mode.cuda.act_install_ort", "⬇ Install onnxruntime-gpu")
                c_head = tr("hw.mode.cuda.head_need_ort", "CUDA DLLs are ready — just switch package to onnxruntime-gpu.")
        elif cuda_needs_restart:
            c_state = "restart_required"
            c_badge = tr("hw.mode.badge.restart", "⚠ Restart")
            c_head = tr("hw.mode.cuda.head_restart", "Everything needed is already on disk! Restart the app to activate CUDA.")
            c_act = ""
        else:
            c_state = "needs_setup"
            c_badge = tr("hw.mode.badge.check", "⚠ Check")
            c_head = str(cuda_cap.get("reason") or tr("hw.mode.cuda.head_fail", "Failed to initialize CUDA (check NVIDIA driver)."))
            c_act = ""

        modes["cuda"] = {
            "id": "cuda",
            "short_title": "🚀 CUDA",
            "title": "NVIDIA CUDA GPU",
            "role": tr("hw.mode.role.fastest", "Fastest") if best_mode == "cuda" else "NVIDIA GPU",
            "hw_supported": True,
            "is_best": (best_mode == "cuda"),
            "state": c_state,
            "badge": c_badge,
            "headline": c_head,
            "steps": cuda_steps,
            "install_plan": {
                "target_mode": "cuda",
                "need_ort_pkg": "onnxruntime-gpu" if need_gpu_ort else None,
                "need_cuda_dlls": need_cuda_dl,
                "needs_restart": cuda_needs_restart,
                "can_install": can_inst_cuda,
                "action_label": c_act,
            },
        }

    # 2. DirectML (DX12)
    dml_cap = ai_backends.get("directml") or {}
    dml_hw_ok = bool((sys.platform == "win32" and has_any_gpu) or dml_cap.get("supported"))
    dml_steps = []
    if not dml_hw_ok:
        dml_steps.append({
            "done": False,
            "kind": "hw",
            "label": tr("hw.mode.dml.step_no_gpu", "DirectX 12 GPU not detected (or non-Windows OS)"),
            "detail": tr("hw.mode.dml.step_no_gpu_det", "DirectML runs via DirectX 12 on NVIDIA / AMD / Intel GPUs on Windows."),
        })
        modes["directml"] = {
            "id": "directml",
            "short_title": "🎮 DirectML",
            "title": "DirectX 12 GPU (DirectML)",
            "role": "DX12 GPU",
            "hw_supported": False,
            "is_best": False,
            "state": "unavailable",
            "badge": tr("hw.mode.badge.no_gpu", "✖ No GPU"),
            "headline": tr("hw.mode.dml.no_gpu_head", "Hardware DirectX 12 GPU is unavailable."),
            "steps": dml_steps,
            "install_plan": {
                "target_mode": "directml",
                "need_ort_pkg": None,
                "need_cuda_dlls": False,
                "needs_restart": False,
                "can_install": False,
                "action_label": "",
            },
        }
    else:
        dml_steps.append({
            "done": True,
            "kind": "hw",
            "label": tr(
                "hw.mode.dml.step_hw_ok",
                "DX12 GPU: {gpus}",
                gpus=", ".join(gpu_names) if gpu_names else tr("hw.mode.detected", "detected"),
            ),
            "detail": tr("hw.mode.dml.step_hw_ok_det", "Requires no external CUDA libraries — uses DirectX 12 driver."),
        })
        need_dml_ort = (ort_pkg_raw != "onnxruntime-directml")
        if not need_dml_ort:
            dml_steps.append({
                "done": True,
                "kind": "ort",
                "label": tr("hw.mode.dml.step_ort_ok", "ONNX Runtime package: onnxruntime-directml already installed"),
                "detail": tr("hw.mode.ort_saved_det", "Existing package is preserved."),
            })
        else:
            dml_steps.append({
                "done": False,
                "kind": "ort",
                "label": tr("hw.mode.dml.step_ort_need", "Requires 'onnxruntime-directml' (currently '{pkg}')", pkg=ort_pkg),
                "detail": (
                    tr("hw.mode.dml.step_ort_nv_note", "Note: NVIDIA CUDA is usually faster than DirectML.")
                    if has_nvidia
                    else tr("hw.mode.dml.step_ort_no_cuda_note", "Installs in 1 click without CUDA Toolkit.")
                ),
            })

        dml_needs_restart = bool((not need_dml_ort) and ort_restart_req)
        if dml_needs_restart:
            dml_steps.append({
                "done": False,
                "kind": "restart",
                "label": tr("hw.mode.step_restart", "Application restart required ('{pkg}' is still in memory)", pkg=ort_loaded_pkg or "onnxruntime"),
                "detail": tr("hw.mode.dml.step_restart_det", "Restart the application to activate DirectML."),
            })

        if dml_cap.get("supported"):
            d_state, d_badge, d_act = "ready", tr("hw.mode.badge.ready", "✔ Ready"), ""
            d_head = tr("hw.mode.dml.head_ready", "DirectML GPU is ready.")
        elif need_dml_ort:
            d_state = "needs_setup"
            d_badge = tr("hw.mode.badge.install", "⚙ Install")
            d_head = tr("hw.mode.dml.head_need_ort", "Switch ONNX Runtime package to 'onnxruntime-directml'.")
            d_act = "" if frozen else tr("hw.mode.dml.act_install", "⬇ Install DirectML")
        elif dml_needs_restart:
            d_state, d_badge, d_act = "restart_required", tr("hw.mode.badge.restart", "⚠ Restart"), ""
            d_head = tr("hw.mode.dml.head_restart", "onnxruntime-directml is already on disk — restart the app.")
        else:
            d_state, d_badge, d_act = "needs_setup", tr("hw.mode.badge.check", "⚠ Check"), ""
            d_head = str(dml_cap.get("reason") or tr("hw.mode.dml.head_fail", "DmlExecutionProvider failed to initialize."))

        modes["directml"] = {
            "id": "directml",
            "short_title": "🎮 DirectML",
            "title": "DirectX 12 GPU (DirectML)",
            "role": tr("hw.mode.role.fastest", "Fastest") if best_mode == "directml" else "DX12 GPU",
            "hw_supported": True,
            "is_best": (best_mode == "directml"),
            "state": d_state,
            "badge": d_badge,
            "headline": d_head,
            "steps": dml_steps,
            "install_plan": {
                "target_mode": "directml",
                "need_ort_pkg": "onnxruntime-directml" if need_dml_ort else None,
                "need_cuda_dlls": False,
                "needs_restart": dml_needs_restart,
                "can_install": bool(need_dml_ort and not frozen),
                "action_label": d_act,
            },
        }

    # 3. Intel OpenVINO
    ov_cap = ai_backends.get("openvino") or {}
    ov_hw_ok = bool(has_intel or ov_cap.get("supported"))
    ov_steps = []
    if not ov_hw_ok:
        ov_steps.append({
            "done": False,
            "kind": "hw",
            "label": tr("hw.mode.ov.step_no_intel", "Non-Intel CPU/GPU ({cpu})", cpu=cpu_name),
            "detail": tr("hw.mode.ov.step_no_intel_det", "OpenVINO is optimized specifically for Intel processors and integrated/discrete graphics."),
        })
        modes["openvino"] = {
            "id": "openvino",
            "short_title": "🧠 OpenVINO",
            "title": "Intel OpenVINO",
            "role": "Intel AI",
            "hw_supported": False,
            "is_best": False,
            "state": "unavailable",
            "badge": tr("hw.mode.badge.intel_only", "✖ Intel only"),
            "headline": tr("hw.mode.ov.no_intel_head", "Intel hardware platform not detected."),
            "steps": ov_steps,
            "install_plan": {
                "target_mode": "openvino",
                "need_ort_pkg": None,
                "need_cuda_dlls": False,
                "needs_restart": False,
                "can_install": False,
                "action_label": "",
            },
        }
    else:
        ov_steps.append({
            "done": True,
            "kind": "hw",
            "label": tr("hw.mode.ov.step_hw_ok", "Intel platform: {cpu}", cpu=cpu_name),
            "detail": tr("hw.mode.ov.step_hw_ok_det", "Supports Intel OpenVINO acceleration."),
        })
        need_ov_ort = (ort_pkg_raw != "onnxruntime-openvino")
        if not need_ov_ort:
            ov_steps.append({
                "done": True,
                "kind": "ort",
                "label": tr("hw.mode.ov.step_ort_ok", "ONNX Runtime package: onnxruntime-openvino already installed"),
                "detail": tr("hw.mode.ort_saved_det", "Existing package is preserved."),
            })
        else:
            ov_steps.append({
                "done": False,
                "kind": "ort",
                "label": tr("hw.mode.ov.step_ort_need", "Requires 'onnxruntime-openvino' (currently '{pkg}')", pkg=ort_pkg),
                "detail": tr("hw.mode.ov.step_ort_need_det", "Optimizes execution on Intel CPU / iGPU."),
            })
        ov_needs_restart = bool((not need_ov_ort) and ort_restart_req)
        if ov_needs_restart:
            ov_steps.append({
                "done": False,
                "kind": "restart",
                "label": tr("hw.mode.step_restart", "Application restart required ('{pkg}' is still in memory)", pkg=ort_loaded_pkg or "onnxruntime"),
                "detail": tr("hw.mode.ov.step_restart_det", "Restart the application to activate OpenVINO."),
            })
        if ov_cap.get("supported"):
            o_state, o_badge, o_act = "ready", tr("hw.mode.badge.ready", "✔ Ready"), ""
            o_head = tr("hw.mode.ov.head_ready", "Intel OpenVINO is ready.")
        elif need_ov_ort:
            o_state, o_badge = "needs_setup", tr("hw.mode.badge.install", "⚙ Install")
            o_act = "" if frozen else tr("hw.mode.ov.act_install", "⬇ Install OpenVINO")
            o_head = tr("hw.mode.ov.head_need_ort", "Requires package 'onnxruntime-openvino'.")
        elif ov_needs_restart:
            o_state, o_badge, o_act = "restart_required", tr("hw.mode.badge.restart", "⚠ Restart"), ""
            o_head = tr("hw.mode.ov.head_restart", "onnxruntime-openvino is already on disk — restart the app.")
        else:
            o_state, o_badge, o_act = "needs_setup", tr("hw.mode.badge.check", "⚠ Check"), ""
            o_head = str(ov_cap.get("reason") or tr("hw.mode.ov.head_fail", "OpenVINOExecutionProvider failed to initialize."))

        modes["openvino"] = {
            "id": "openvino",
            "short_title": "🧠 OpenVINO",
            "title": "Intel OpenVINO",
            "role": tr("hw.mode.role.fastest", "Fastest") if best_mode == "openvino" else "Intel AI",
            "hw_supported": True,
            "is_best": (best_mode == "openvino"),
            "state": o_state,
            "badge": o_badge,
            "headline": o_head,
            "steps": ov_steps,
            "install_plan": {
                "target_mode": "openvino",
                "need_ort_pkg": "onnxruntime-openvino" if need_ov_ort else None,
                "need_cuda_dlls": False,
                "needs_restart": ov_needs_restart,
                "can_install": bool(need_ov_ort and not frozen),
                "action_label": o_act,
            },
        }

    # 4. CPU (ніколи не видаляє GPU-пакети!)
    has_any_ort = bool(ort_pkg_raw)
    cores_s = tr("hw.mode.cpu.cores_fmt", "{phys}c / {logi}t", phys=phys or "?", logi=logi or "?")
    cpu_steps = [
        {
            "done": True,
            "kind": "hw",
            "label": tr("hw.mode.cpu.step_hw", "Processor: {cpu} ({cores})", cpu=cpu_name, cores=cores_s),
            "detail": tr("hw.mode.cpu.step_hw_det", "Universal baseline mode for any computer."),
        }
    ]
    if has_any_ort:
        cpu_steps.append({
            "done": True,
            "kind": "ort",
            "label": tr("hw.mode.cpu.step_ort_ok", "Package '{pkg}' already supports CPUExecutionProvider", pkg=ort_pkg),
            "detail": tr("hw.mode.cpu.step_ort_ok_det", "No need to remove or reinstall GPU/DirectML libraries."),
        })
        if best_mode != "cpu":
            best_title = modes[best_mode]["title"]
            cpu_steps.append({
                "done": True,
                "kind": "info",
                "label": tr("hw.mode.cpu.step_tip", "Tip: {best} is faster on this PC", best=best_title),
                "detail": tr("hw.mode.cpu.step_tip_det", "CPU uses less GPU memory (VRAM) but processes frames slower than GPU."),
            })
        cpu_state, cpu_badge, cpu_act = ("ready", tr("hw.mode.badge.ready", "✔ Ready"), "")
        cpu_head = tr("hw.mode.cpu.head_ready", "CPU mode is ready (installed GPU packages remain intact).")
    else:
        cpu_steps.append({
            "done": False,
            "kind": "ort",
            "label": tr("hw.mode.cpu.step_ort_missing", "Package 'onnxruntime' is missing in environment"),
            "detail": tr("hw.mode.cpu.step_ort_missing_det", "Base onnxruntime package must be installed."),
        })
        cpu_state = "needs_setup"
        cpu_badge = tr("hw.mode.badge.install", "⚙ Install")
        cpu_act = "" if frozen else tr("hw.mode.cpu.act_install", "⬇ Install onnxruntime")
        cpu_head = tr("hw.mode.cpu.head_missing", "Base onnxruntime package is missing.")

    modes["cpu"] = {
        "id": "cpu",
        "short_title": tr("hw.mode.cpu.short_title", "🖥 CPU"),
        "title": tr("hw.mode.cpu.title", "Processor (CPU)"),
        "role": tr("hw.mode.role.fastest", "Fastest") if best_mode == "cpu" else tr("hw.mode.role.compatible", "Compatible"),
        "hw_supported": True,
        "is_best": (best_mode == "cpu"),
        "state": cpu_state,
        "badge": cpu_badge,
        "headline": cpu_head,
        "steps": cpu_steps,
        "install_plan": {
            "target_mode": "cpu",
            "need_ort_pkg": None if has_any_ort else "onnxruntime",
            "need_cuda_dlls": False,
            "needs_restart": False,
            "can_install": bool((not has_any_ort) and (not frozen)),
            "action_label": cpu_act,
        },
    }

    # 5. AUTO (автовибір найкращого доступного)
    best_info = modes[best_mode]
    best_plan = best_info["install_plan"]
    auto_steps = [
        {
            "done": True,
            "kind": "info",
            "label": tr("hw.mode.auto.step_cur", "AUTO currently uses: {cur}", cur=modes[eff_auto]["title"]),
            "detail": tr("hw.mode.auto.step_cur_det", "Priority: CUDA → DirectML → OpenVINO → CPU."),
        }
    ]
    if best_mode != eff_auto and best_info["state"] in ("needs_setup", "restart_required"):
        if best_info["state"] == "restart_required":
            auto_steps.append({
                "done": False,
                "kind": "restart",
                "label": tr("hw.mode.auto.step_restart", "Optimal {best} is installed — restart required", best=best_info["title"]),
                "detail": tr("hw.mode.auto.step_restart_det", "After restart, AUTO will automatically switch to it."),
            })
            auto_head = tr("hw.mode.auto.head_restart", "AUTO ready (currently {eff}, after restart — {best}).", eff=eff_auto.upper(), best=best_mode.upper())
            auto_badge = tr("hw.mode.badge.restart", "⚠ Restart")
            auto_state = "restart_required"
            auto_install_plan = {
                "target_mode": best_mode,
                "need_ort_pkg": None,
                "need_cuda_dlls": False,
                "needs_restart": True,
                "can_install": False,
                "action_label": "",
            }
        else:
            for st_item in best_info["steps"]:
                if not st_item.get("done"):
                    auto_steps.append(dict(st_item))
            auto_head = tr("hw.mode.auto.head_can_faster", "AUTO currently uses {eff}; you can enable faster {best} in 1 click.", eff=eff_auto.upper(), best=best_mode.upper())
            auto_badge = tr("hw.mode.badge.working", "✔ Working")
            auto_state = "ready"
            auto_install_plan = {
                **best_plan,
                "action_label": tr("hw.mode.auto.act_enable_best", "⬇ Enable {short} for AUTO", short=best_info["short_title"]) if best_plan.get("can_install") else "",
            }
    else:
        auto_steps.append({
            "done": True,
            "kind": "hw",
            "label": tr("hw.mode.auto.step_opt", "Optimal accelerator ({best}) is already active", best=best_info["title"]),
            "detail": tr("hw.mode.auto.step_opt_det", "Nothing extra needs to be installed or reinstalled."),
        })
        auto_head = tr("hw.mode.auto.head_opt", "AUTO is active with maximum speed ({eff}).", eff=eff_auto.upper())
        auto_badge = tr("hw.mode.badge.ready", "✔ Ready")
        auto_state = "ready"
        auto_install_plan = {
            "target_mode": best_mode,
            "need_ort_pkg": None,
            "need_cuda_dlls": False,
            "needs_restart": False,
            "can_install": False,
            "action_label": "",
        }

    modes["auto"] = {
        "id": "auto",
        "short_title": tr("hw.mode.auto.short_title", "⚡ Auto"),
        "title": tr("hw.mode.auto.title", "Auto-select (AUTO)"),
        "role": tr("hw.mode.auto.role", "Now {eff}", eff=eff_auto.upper()),
        "hw_supported": True,
        "is_best": False,
        "state": auto_state,
        "badge": auto_badge,
        "headline": auto_head,
        "steps": auto_steps,
        "install_plan": auto_install_plan,
    }

    concrete_order = ("cuda", "directml", "openvino", "cpu")
    sup_concrete = [k for k in concrete_order if (modes.get(k) or {}).get("hw_supported")]
    n_sup = len(sup_concrete)

    for k in concrete_order:
        m = modes[k]
        hw_ok = bool(m.get("hw_supported"))
        st = str(m.get("state") or "unavailable")
        plan = m.get("install_plan") or {}
        if not hw_ok:
            m["speed_rank"] = None
            m["speed_tile"] = tr("hw.tile.speed.unsupported", "🚫 Not supported")
            m["support_level"] = "none"
            m["support_label"] = tr("hw.tile.sup_label.none", "Not supported by this PC")
            if k == "cuda":
                m["support_badge"] = tr("hw.tile.sup_badge.no_nv", "✖ No NVIDIA")
                m["speed_detail"] = tr("hw.tile.cuda.speed_det_unavail", "Unavailable — requires discrete or mobile NVIDIA GPU")
                m["tile_hint"] = tr("hw.tile.cuda.hint_unavail", "NVIDIA GPUs only")
                m["pros"] = [
                    tr("hw.tile.cuda.pro_unavail", "On compatible NVIDIA PCs delivers top AI speed (≈5–15× faster than CPU)"),
                ]
                m["cons"] = [
                    tr("hw.tile.cuda.con_unavail_1", "Not supported on this PC: NVIDIA GPU not detected"),
                    tr("hw.tile.cuda.con_unavail_2", "Works exclusively with NVIDIA drivers and GPUs"),
                ]
            elif k == "directml":
                m["support_badge"] = tr("hw.tile.sup_badge.no_gpu", "✖ No GPU")
                m["speed_detail"] = tr("hw.tile.dml.speed_det_unavail", "Unavailable — DirectX 12 GPU not detected")
                m["tile_hint"] = tr("hw.tile.dml.hint_unavail", "DX12 GPU required")
                m["pros"] = [
                    tr("hw.tile.dml.pro_unavail", "On Windows GPUs provides 3–8× speedup without CUDA DLLs"),
                ]
                m["cons"] = [
                    tr("hw.tile.dml.con_unavail", "Not supported on this PC: DirectX 12 graphics adapter not found"),
                ]
            else:
                m["support_badge"] = tr("hw.tile.sup_badge.intel_only", "✖ Intel only")
                m["speed_detail"] = tr("hw.tile.ov.speed_det_unavail", "Unavailable — CPU/GPU is not Intel")
                m["tile_hint"] = tr("hw.tile.ov.hint_unavail", "Intel CPU/iGPU only")
                m["pros"] = [
                    tr("hw.tile.ov.pro_unavail", "Provides moderate speedup on Intel platforms without a discrete GPU"),
                ]
                m["cons"] = [
                    tr("hw.tile.ov.con_unavail", "Not supported on this PC: detected '{cpu}' (non-Intel)", cpu=cpu_name),
                ]
            continue

        rank = sup_concrete.index(k) + 1
        m["speed_rank"] = rank
        is_best_m = bool(m.get("is_best"))
        m["support_level"] = "full" if st == "ready" else ("restart" if st == "restart_required" else "partial")

        if st == "ready":
            m["support_badge"] = tr("hw.tile.sup_badge.full", "✔ Full support")
            m["support_label"] = tr("hw.tile.sup_label.full", "Full support — everything installed and ready")
        elif st == "restart_required":
            m["support_badge"] = tr("hw.mode.badge.restart", "⚠ Restart")
            m["support_label"] = tr("hw.tile.sup_label.restart", "All files are on disk — restart the app to activate")
        elif k == "cuda" and plan.get("need_ort_pkg") and plan.get("need_cuda_dlls"):
            m["support_badge"] = tr("hw.tile.sup_badge.pkg_cuda", "⚙ Pkg + CUDA")
            m["support_label"] = tr("hw.tile.sup_label.pkg_cuda", "Hardware supported — need onnxruntime-gpu and CUDA DLLs")
        elif k == "cuda" and plan.get("need_cuda_dlls"):
            m["support_badge"] = tr("hw.mode.cuda.badge_add", "⚙ Add CUDA")
            m["support_label"] = tr("hw.tile.sup_label.add_cuda", "Hardware supported — download CUDA / cuDNN DLLs to ./cuda/")
        elif plan.get("need_ort_pkg"):
            m["support_badge"] = tr("hw.tile.sup_badge.need_pkg", "⚙ Need pkg")
            m["support_label"] = tr("hw.tile.sup_label.need_pkg", "Hardware supported — requires '{pkg}' package", pkg=plan["need_ort_pkg"])
        else:
            m["support_badge"] = tr("hw.mode.badge.check", "⚠ Check")
            m["support_label"] = str(m.get("headline") or tr("hw.tile.sup_label.check_drv", "Requires driver check"))

        if k == "cuda":
            m["speed_tile"] = tr("hw.tile.speed.fastest", "🏆 #1 Fastest")
            m["speed_detail"] = tr("hw.tile.cuda.speed_det", "#1 Fastest mode (≈ 5–15× faster than CPU on RTX / GeForce GPU)")
            m["tile_hint"] = tr("hw.tile.cuda.hint", "+ Max FPS | − CUDA DLLs")
            m["pros"] = [
                tr("hw.tile.cuda.pro_1", "Highest frame processing speed on this PC (≈5–15× faster than CPU) and minimal latency"),
                tr("hw.tile.cuda.pro_2", "Nearly completely offloads CPU when running 2+ cameras and multiple AI modules"),
            ]
            m["cons"] = [
                tr("hw.tile.cuda.con_1", "Works only on NVIDIA GPUs"),
                (
                    tr("hw.tile.cuda.con_2_setup", "Requires onnxruntime-gpu and CUDA/cuDNN (~1.5 GB in ./cuda/) — auto-installed on selection")
                    if st != "ready"
                    else tr("hw.tile.cuda.con_2_ready", "Uses some VRAM and ~1.5 GB disk space for CUDA/cuDNN DLLs in ./cuda/")
                ),
            ]
        elif k == "directml":
            m["speed_tile"] = (
                tr("hw.tile.speed.fastest", "🏆 #1 Fastest")
                if is_best_m
                else tr("hw.tile.speed.fast", "⚡ #{rank} Fast", rank=rank)
            )
            m["speed_detail"] = (
                tr("hw.tile.dml.speed_det_best", "#1 Fastest mode on this PC (≈ 3–8× faster than CPU via DirectX 12)")
                if is_best_m
                else tr("hw.tile.dml.speed_det_rank", "#{rank} Fast on GPU (≈ 3–8× faster than CPU, ~15–35% slower than CUDA)", rank=rank)
            )
            m["tile_hint"] = (
                tr("hw.tile.dml.hint_nv", "+ No CUDA | − Slower than CUDA")
                if has_nvidia
                else tr("hw.tile.dml.hint_other", "+ No CUDA | + Fast GPU")
            )
            m["pros"] = [
                tr("hw.tile.dml.pro_1", "Runs via DirectX 12 on NVIDIA, AMD and Intel GPUs without downloading CUDA DLLs"),
                tr("hw.tile.dml.pro_2", "Much faster and lighter on CPU than processor-only mode"),
            ]
            m["cons"] = [
                (
                    tr("hw.tile.dml.con_1_nv", "On NVIDIA GPUs typically ~15–35% slower than CUDA")
                    if has_nvidia
                    else tr("hw.tile.dml.con_1_other", "Slightly slower than dedicated CUDA under heavy load")
                ),
                (
                    tr("hw.tile.dml.con_2_setup", "Switches package to 'onnxruntime-directml' (replaces current ORT package; restart required)")
                    if st != "ready"
                    else tr("hw.tile.dml.con_2_ready", "Uses shared Windows DirectX 12 pipeline (shares GPU with games or heavy video)")
                ),
            ]
        elif k == "openvino":
            m["speed_tile"] = (
                tr("hw.tile.speed.fastest", "🏆 #1 Fastest")
                if is_best_m
                else tr("hw.tile.speed.medium", "◐ #{rank} Medium", rank=rank)
            )
            m["speed_detail"] = (
                tr("hw.tile.ov.speed_det_best", "#1 Fastest on this PC (Intel CPU/iGPU optimization — ≈ 1.5–3× faster than plain CPU)")
                if is_best_m
                else tr("hw.tile.ov.speed_det_rank", "#{rank} Medium speed (≈ 1.5–3× faster than CPU, much slower than GPU CUDA / DirectML)", rank=rank)
            )
            m["tile_hint"] = tr("hw.tile.ov.hint", "+ No dGPU | − Slower than GPU")
            m["pros"] = [
                tr("hw.tile.ov.pro_1", "Accelerates models on Intel CPUs and Intel UHD/Iris/Arc integrated graphics without CUDA DLLs"),
                tr("hw.tile.ov.pro_2", "Lower CPU load compared to standard CPU mode"),
            ]
            m["cons"] = [
                (
                    tr("hw.tile.ov.con_1_gpu", "Noticeably slower than discrete GPUs via CUDA or DirectML")
                    if (has_nvidia or has_any_gpu)
                    else tr("hw.tile.ov.con_1_igpu", "Speed is limited by integrated Intel graphics")
                ),
                tr("hw.tile.ov.con_2", "Works only on Intel platforms and requires 'onnxruntime-openvino' package"),
            ]
        else:  # cpu
            m["speed_tile"] = (
                tr("hw.tile.speed.slowest", "🐢 #{rank} Slowest", rank=rank)
                if n_sup > 1
                else tr("hw.tile.speed.base_cpu", "🖥 Base CPU")
            )
            m["speed_detail"] = (
                tr("hw.tile.cpu.speed_det_rank", "#{rank} Slowest mode (baseline 1× speed — significantly behind GPU)", rank=rank)
                if n_sup > 1
                else tr("hw.tile.cpu.speed_det_only", "Baseline processor mode (1× — no hardware GPU acceleration on this PC)")
            )
            m["tile_hint"] = (
                tr("hw.tile.cpu.hint_multi", "+ Stable | − 1× slow")
                if n_sup > 1
                else tr("hw.tile.cpu.hint_only", "+ Stable | − No GPU")
            )
            m["pros"] = [
                tr("hw.tile.cpu.pro_1", "Maximum compatibility — runs reliably on any PC without GPU drivers"),
                tr("hw.tile.cpu.pro_2", "Uses zero VRAM; existing GPU packages and CUDA DLLs remain intact on disk"),
            ]
            m["cons"] = [
                (
                    tr("hw.tile.cpu.con_1_multi", "Slowest processing mode (baseline 1× speed)")
                    if n_sup > 1
                    else tr("hw.tile.cpu.con_1_only", "No GPU hardware acceleration on this PC (1× speed)")
                ),
                tr("hw.tile.cpu.con_2", "High CPU usage and lower FPS when running several cameras simultaneously"),
            ]

    eff_info = modes.get(eff_auto) or modes["cpu"]
    best_info = modes.get(best_mode) or modes["cpu"]
    eff_rank = eff_info.get("speed_rank") or 1
    auto_opt = (eff_auto == best_mode and best_info.get("state") == "ready")
    auto_m = modes["auto"]
    auto_m["speed_rank"] = eff_rank
    if auto_opt:
        if best_mode == "cpu" and n_sup <= 1:
            auto_m["speed_tile"] = tr("hw.tile.auto.speed_cpu", "→ CPU (base)")
            auto_m["speed_detail"] = tr("hw.tile.auto.speed_det_cpu", "Running via Processor (CPU) — no hardware GPU detected on this PC")
            auto_m["support_level"] = "full"
            auto_m["support_badge"] = tr("hw.tile.sup_badge.full", "✔ Full support")
            auto_m["support_label"] = tr("hw.tile.auto.sup_lbl_cpu", "Full support (CPU is the only available backend on this PC)")
            auto_m["tile_hint"] = tr("hw.tile.cpu.hint_only", "+ Stable | − No GPU")
        else:
            auto_m["speed_tile"] = f"→ {eff_auto.upper()} (#{eff_rank})"
            auto_m["speed_detail"] = tr(
                "hw.tile.auto.speed_det_opt",
                "Running via #{rank} Fastest ({title}) — maximum performance",
                rank=eff_rank,
                title=eff_info["title"],
            )
            auto_m["support_level"] = "full"
            auto_m["support_badge"] = tr("hw.tile.sup_badge.full", "✔ Full support")
            auto_m["support_label"] = tr("hw.tile.auto.sup_lbl_opt", "Full support — fastest backend for this PC selected automatically")
            auto_m["tile_hint"] = tr("hw.tile.auto.hint_opt", "+ Picks best GPU")
    elif auto_m["state"] == "restart_required":
        auto_m["speed_tile"] = f"→ {eff_auto.upper()} (#{eff_rank})"
        auto_m["speed_detail"] = tr(
            "hw.tile.auto.speed_det_restart",
            "Temporarily {eff} (#{rank}); after restart will switch to {best} (#1)",
            eff=eff_auto.upper(),
            rank=eff_rank,
            best=best_info["title"],
        )
        auto_m["support_level"] = "restart"
        auto_m["support_badge"] = tr("hw.mode.badge.restart", "⚠ Restart")
        auto_m["support_label"] = tr("hw.tile.auto.sup_lbl_restart", "Optimal mode installed on disk — restart required")
        auto_m["tile_hint"] = tr("hw.tile.auto.hint_restart", "Restart → {best}", best=best_mode.upper())
    else:
        auto_m["speed_tile"] = f"⚠ → {eff_auto.upper()} (#{eff_rank})"
        auto_m["speed_detail"] = tr(
            "hw.tile.auto.speed_det_partial",
            "Currently using {cur} (#{rank}), while faster {best} (#1) is available",
            cur=eff_info["title"],
            rank=eff_rank,
            best=best_info["title"],
        )
        auto_m["support_level"] = "partial"
        auto_m["support_badge"] = tr("hw.tile.auto.sup_badge_cur", "⚡ Now {eff}", eff=eff_auto.upper())
        auto_m["support_label"] = tr(
            "hw.tile.auto.sup_lbl_partial",
            "Running on {eff}; faster {best} is not activated yet",
            eff=eff_auto.upper(),
            best=best_mode.upper(),
        )
        auto_m["tile_hint"] = tr("hw.tile.auto.hint_partial", "− Can be faster ({best})", best=best_mode.upper())
    auto_m["pros"] = [
        tr("hw.tile.auto.pro_1", "Automatically runs the fastest ready accelerator by priority: CUDA → DirectML → OpenVINO → CPU"),
        tr("hw.tile.auto.pro_2", "Automatically falls back to CPU if the GPU provider is temporarily unavailable"),
    ]
    auto_m["cons"] = [
        (
            tr("hw.tile.auto.con_opt", "No downsides — already running in the fastest mode for your PC")
            if auto_opt
            else tr(
                "hw.tile.auto.con_partial",
                "AUTO itself does not download missing libraries — while '{pkg}' is installed, it uses {eff} instead of faster {best}",
                pkg=ort_pkg,
                eff=eff_auto.upper(),
                best=best_mode.upper(),
            )
        ),
    ]

    return {
        "best_mode": best_mode,
        "effective_auto_mode": eff_auto,
        "ort_package": ort_pkg,
        "ort_loaded_package": ort_loaded_pkg,
        "ort_restart_required": ort_restart_req,
        "frozen": frozen,
        "modes": modes,
    }


def ensure_ai_backend_dependencies(
    base_dir: str,
    mode: str,
    hw_monitor: HardwareMonitor = None,
    registry=None,
    progress_cb=None,
    ort_install_fn=None,
    cuda_dl_fn=None,
) -> dict:
    """
    Автоматично довстановлює ТІЛЬКИ відсутні компоненти для вибраного режиму ('cuda', 'directml',
    'openvino', 'cpu' або 'auto'), ніколи не видаляючи і не перевстановлюючи вже наявні модулі/пакети.
    """
    req_mode = str(mode or "cpu").strip().lower()
    if req_mode not in ("auto", "cuda", "directml", "openvino", "cpu"):
        req_mode = "cpu"

    if hw_monitor is not None:
        caps = hw_monitor.probe_acceleration_capabilities(force=False)
        live_metrics = hw_monitor.get_live_metrics()
    else:
        caps = {}
        live_metrics = None

    assessment = assess_ai_modes(base_dir, caps=caps, live_metrics=live_metrics)
    target_mode = assessment["best_mode"] if req_mode == "auto" else req_mode
    minfo = assessment["modes"].get(target_mode) or {}

    if not minfo.get("hw_supported"):
        return {
            "ok": False,
            "mode": req_mode,
            "target_mode": target_mode,
            "ort_changed": False,
            "cuda_downloaded": False,
            "restart_required": bool(assessment.get("ort_restart_required")),
            "caps": caps,
            "cuda_status": (caps or {}).get("cuda_libraries"),
            "message": minfo.get("headline") or tr(
                "hw.ensure.unsupported",
                "Mode {mode} is not supported on this PC.",
                mode=target_mode.upper(),
            ),
        }

    plan = minfo.get("install_plan") or {}
    need_ort_pkg = plan.get("need_ort_pkg")
    need_cuda_dlls = bool(plan.get("need_cuda_dlls"))

    messages = []
    ort_changed = False
    cuda_downloaded = False
    ok_all = True

    if need_ort_pkg:
        fn_ort = ort_install_fn or install_ort_package
        if progress_cb:
            progress_cb(
                tr(
                    "hw.ensure.prep_pkg",
                    "Preparing package '{pkg}' (other installed modules remain untouched)...",
                    pkg=need_ort_pkg,
                )
            )
        ok_ort, msg_ort = fn_ort(need_ort_pkg, progress_cb=progress_cb)
        messages.append(msg_ort)
        if ok_ort:
            ort_changed = True
        else:
            ok_all = False

    if ok_all and need_cuda_dlls:
        fn_cuda = cuda_dl_fn or download_cuda_libraries
        if progress_cb:
            progress_cb(tr("hw.ensure.dl_cuda", "Checking and downloading missing CUDA / cuDNN DLLs to ./cuda/..."))
        dl_res = fn_cuda(
            base_dir,
            progress_cb=progress_cb,
            include_optional=True,
            force_into_local=False,
        )
        cuda_downloaded = bool(dl_res.get("downloaded_packages"))
        if dl_res.get("message"):
            messages.append(dl_res["message"])
        if not dl_res.get("ok"):
            ok_all = False

    if target_mode == "cuda":
        try:
            prepare_cuda_dll_environment(base_dir, preload=True)
        except Exception:
            pass

    if (ort_changed or cuda_downloaded) and registry is not None:
        try:
            registry.refresh_libraries_capabilities()
        except Exception:
            pass

    if hw_monitor is not None:
        caps_after = hw_monitor.refresh_all(force=True)
    else:
        caps_after = caps

    cuda_st_after = (caps_after or {}).get("cuda_libraries") or check_cuda_libraries_status(base_dir, allow_network=False)
    restart_req = bool((caps_after or {}).get("ort_restart_required"))

    if not messages:
        if restart_req:
            messages.append(
                tr(
                    "hw.ensure.restart_req",
                    "Everything required is already on disk — restart the application to activate.",
                )
            )
        else:
            messages.append(
                tr(
                    "hw.ensure.all_ready",
                    "All components for {mode} are already installed (reinstall not needed).",
                    mode=target_mode.upper(),
                )
            )

    return {
        "ok": ok_all,
        "mode": req_mode,
        "target_mode": target_mode,
        "ort_changed": ort_changed,
        "cuda_downloaded": cuda_downloaded,
        "restart_required": restart_req,
        "caps": caps_after,
        "cuda_status": cuda_st_after,
        "message": " ".join(m.strip() for m in messages if str(m or "").strip()),
    }


if __name__ == "__main__":
    if "--ensure-cuda" in sys.argv:
        try:
            idx = sys.argv.index("--ensure-cuda")
            target_dir = sys.argv[idx + 1] if (idx + 1) < len(sys.argv) and not sys.argv[idx + 1].startswith("-") else "."
        except Exception:
            target_dir = "."
        ok = ensure_cuda_libraries_cli(target_dir)
        sys.exit(0 if ok else 1)
