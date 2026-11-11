import os
import sys
import copy
import importlib
import importlib.util
import threading
import traceback
from datetime import datetime


def parse_date_str(val):
    s = str(val or "").strip()
    if not s or s.lower() in ("none", "unlimited", "null", "*"):
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def is_date_range_active(d_from_str: str, d_to_str: str) -> bool:
    df = parse_date_str(d_from_str)
    dt = parse_date_str(d_to_str)
    if df is None and dt is None:
        return True
    today = datetime.now().date()
    if df is not None and today < df:
        return False
    if dt is not None and today > dt:
        return False
    return True


def _file_sig(path: str):
    try:
        st = os.stat(path)
        return (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return None


class ModuleRegistry:
    """
    Реєстр незалежних підключуваних модулів (папка scripts/modules/).
    Автоматично знаходить нові модулі, підхоплює оновлення файлів модулів
    або бібліотек і не дозволяє збою одного модуля зламати програму.
    """
    def __init__(self, base_dir: str, extra_dirs=None):
        self.base_dir = os.path.abspath(base_dir)
        self._lock = threading.RLock()
        self._extra_dirs = [os.path.abspath(d) for d in (extra_dirs or [])]

        self._modules = {}        # mod_id -> entry dict
        self._errors = {}         # mod_id_or_file -> error info dict
        self._file_sigs = {}      # path -> (mtime_ns, size)
        self._last_scan_ts = 0.0
        self._generation = 0

        self.scan_and_load(force=False)

    def get_search_dirs(self) -> list:
        dirs = []
        primary = os.path.join(self.base_dir, "scripts", "modules")
        dirs.append(primary)
        for ed in self._extra_dirs:
            if ed not in dirs:
                dirs.append(ed)
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            bundled = os.path.join(sys._MEIPASS, "scripts", "modules")
            if bundled not in dirs:
                dirs.append(bundled)
        return [d for d in dirs if os.path.isdir(d)]

    def _discover_module_files(self) -> dict:
        """Повертає {stem: file_path}; зовнішній scripts/modules має пріоритет над _MEIPASS."""
        found = {}
        for folder in self.get_search_dirs():
            try:
                names = sorted(os.listdir(folder))
            except Exception:
                continue
            for fname in names:
                if fname.startswith("_") or fname.startswith("."):
                    continue
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(".py"):
                    stem = os.path.splitext(fname)[0]
                    if stem not in found:
                        found[stem] = fpath
                elif os.path.isdir(fpath):
                    init_py = os.path.join(fpath, "__init__.py")
                    if os.path.isfile(init_py) and fname not in found:
                        found[fname] = init_py
        return found

    def _load_one_file(self, stem: str, fpath: str, is_reload: bool = False):
        mod_full_name = f"scripts.modules.{stem}"
        try:
            importlib.invalidate_caches()
            spec = importlib.util.spec_from_file_location(mod_full_name, fpath)
            if spec is None or spec.loader is None:
                raise ImportError(f"Не вдалося створити import spec для {fpath}")
            py_mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_full_name] = py_mod
            # Читаємо й компілюємо напряму з вихідного файлу, щоб уникнути застарілого .pyc-кешу (1-сек квантування mtime)
            with open(fpath, "rb") as sf:
                src_bytes = sf.read()
            code_obj = compile(src_bytes, fpath, "exec")
            exec(code_obj, py_mod.__dict__)

            raw_spec = getattr(py_mod, "MODULE_SPEC", None)
            if not isinstance(raw_spec, dict) or not raw_spec.get("id"):
                return None, None  # Звичайний допоміжний файл без MODULE_SPEC пропускаємо без помилки

            mod_id = str(raw_spec["id"]).strip()
            norm_spec = copy.deepcopy(raw_spec)
            norm_spec.setdefault("version", "1.0.0")
            norm_spec.setdefault("order", 100)
            norm_spec.setdefault("kind", "ai")
            norm_spec.setdefault("name", mod_id.capitalize())
            norm_spec.setdefault("icon", "🧩")
            norm_spec.setdefault("short_badge", mod_id[:2].upper())
            norm_spec.setdefault("description", "")
            norm_spec.setdefault("provides", [])
            norm_spec.setdefault("library_deps", [])
            norm_spec.setdefault("settings_schema", [])
            norm_spec.setdefault("notices", [])
            norm_spec.setdefault("archive", {"has_archive": False})

            entry = {
                "id": mod_id,
                "stem": stem,
                "file": fpath,
                "sig": _file_sig(fpath),
                "py_mod": py_mod,
                "spec": norm_spec,
                "reloaded": is_reload,
            }
            return entry, None
        except Exception as e:
            err = {
                "id": stem,
                "file": fpath,
                "error": str(e),
                "traceback": traceback.format_exc(limit=4),
            }
            return None, err

    def scan_and_load(self, force: bool = False) -> dict:
        """
        Сканує папку модулів і підхоплює нові, оновлені або видалені модулі,
        а також оновлює динамічні можливості бібліотек у завантажених модулях.
        Повертає {'added': [...], 'updated': [...], 'removed': [...], 'errors': {...}, 'changed': bool}.
        """
        with self._lock:
            importlib.invalidate_caches()
            discovered = self._discover_module_files()
            added = []
            updated = []
            removed = []

            current_paths = set(discovered.values())
            for old_path in list(self._file_sigs.keys()):
                if old_path not in current_paths:
                    self._file_sigs.pop(old_path, None)

            # Визначаємо видалені модулі
            for mid in list(self._modules.keys()):
                m_entry = self._modules[mid]
                if m_entry["file"] not in current_paths:
                    self._modules.pop(mid, None)
                    removed.append(mid)

            # Видаляємо застарілі помилки, якщо файл зник
            for ek in list(self._errors.keys()):
                if self._errors[ek].get("file") not in current_paths:
                    self._errors.pop(ek, None)

            new_modules = {}
            for stem, fpath in discovered.items():
                sig = _file_sig(fpath)
                prev_entry = None
                for me in self._modules.values():
                    if me["file"] == fpath:
                        prev_entry = me
                        break

                if prev_entry is not None and not force and prev_entry["sig"] == sig:
                    # Файл не змінився — оновлюємо лише динамічні можливості бібліотек, якщо є hook
                    rf = prev_entry["spec"].get("refresh_capabilities")
                    if callable(rf):
                        try:
                            rf()
                        except Exception:
                            pass
                    new_modules[prev_entry["id"]] = prev_entry
                    self._file_sigs[fpath] = sig
                    continue

                is_upd = prev_entry is not None
                entry, err = self._load_one_file(stem, fpath, is_reload=is_upd)
                self._file_sigs[fpath] = sig
                if err is not None:
                    self._errors[stem] = err
                    if prev_entry is not None and prev_entry["id"] not in removed:
                        removed.append(prev_entry["id"])
                    print(f"[ModuleRegistry] Помилка завантаження модуля '{stem}' ({fpath}): {err['error']}")
                elif entry is not None:
                    self._errors.pop(stem, None)
                    self._errors.pop(entry["id"], None)
                    new_modules[entry["id"]] = entry
                    if is_upd:
                        updated.append(entry["id"])
                    else:
                        added.append(entry["id"])

            # Зливаємо без втрати вже завантажених, якщо був неперехоплений збій іншого
            self._modules = new_modules
            changed = bool(added or updated or removed or force)
            if changed:
                self._generation += 1
            self._last_scan_ts = datetime.now().timestamp()

            return {
                "added": added,
                "updated": updated,
                "removed": removed,
                "errors": copy.deepcopy(self._errors),
                "changed": changed,
                "generation": self._generation,
            }

    def check_for_updates(self) -> dict:
        """Швидка перевірка на появу/зміну/видалення файлів у scripts/modules/."""
        with self._lock:
            discovered = self._discover_module_files()
            paths_now = set(discovered.values())
            paths_known = set(self._file_sigs.keys())
            need_scan = (paths_now != paths_known)
            if not need_scan:
                for _, fpath in discovered.items():
                    if _file_sig(fpath) != self._file_sigs.get(fpath):
                        need_scan = True
                        break
        if need_scan:
            return self.scan_and_load(force=False)
        return {"added": [], "updated": [], "removed": [], "errors": copy.deepcopy(self._errors), "changed": False, "generation": self._generation}

    def refresh_libraries_capabilities(self):
        """Повідомляє всі активні модулі перечитати стан зовнішніх бібліотек (FFmpeg, ORT, fast_alpr тощо)."""
        importlib.invalidate_caches()
        with self._lock:
            entries = list(self._modules.values())
        for ent in entries:
            rf = ent["spec"].get("refresh_capabilities")
            if callable(rf):
                try:
                    rf()
                except Exception as e:
                    print(f"[ModuleRegistry] refresh_capabilities '{ent['id']}': {e}")

    def list_modules(self, kind: str = None) -> list:
        with self._lock:
            vals = list(self._modules.values())
        if kind is not None:
            vals = [v for v in vals if v["spec"].get("kind") == kind]
        vals.sort(key=lambda x: (int(x["spec"].get("order", 100)), str(x["id"])))
        return vals

    def get_module(self, mod_id: str):
        with self._lock:
            return self._modules.get(str(mod_id))

    def get_spec(self, mod_id: str):
        ent = self.get_module(mod_id)
        return ent["spec"] if ent else None

    def get_errors(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._errors)

    def get_generation(self) -> int:
        with self._lock:
            return self._generation

    def list_by_capability(self, cap_name: str) -> list:
        with self._lock:
            vals = [v for v in self._modules.values() if cap_name in (v["spec"].get("provides") or [])]
        vals.sort(key=lambda x: (int(x["spec"].get("order", 100)), str(x["id"])))
        return vals

    def get_default_module_config(self, mod_id: str) -> dict:
        spec = self.get_spec(mod_id)
        if not spec:
            return {}
        cfg = {"on": bool(spec.get("default_enabled", True))}
        for fld in spec.get("settings_schema") or []:
            k = fld.get("key")
            if k:
                cfg[k] = copy.deepcopy(fld.get("default"))
            k_to = fld.get("key_to")
            if k_to:
                cfg[k_to] = copy.deepcopy(fld.get("default_to", ""))
        return cfg

    def build_all_default_camera_modules(self) -> dict:
        out = {}
        for ent in self.list_modules():
            out[ent["id"]] = self.get_default_module_config(ent["id"])
        return out

    def get_all_help_entries(self) -> dict:
        help_map = {}
        for ent in self.list_modules():
            mid = ent["id"]
            on_desc = ent["spec"].get("on_help") or f"Увімкнути або вимкнути модуль '{ent['spec'].get('name', mid)}' для цієї камери."
            help_map[f"{mid}.on"] = on_desc
            for fld in ent["spec"].get("settings_schema") or []:
                k = fld.get("key")
                if k and fld.get("help"):
                    help_map[f"{mid}.{k}"] = fld["help"]
        return help_map

    def describe_modules_status(self) -> list:
        rows = []
        for ent in self.list_modules():
            spec = ent["spec"]
            status_fn = spec.get("get_status_summary")
            lib_status = "Готовий"
            if callable(status_fn):
                try:
                    lib_status = str(status_fn(self.base_dir))
                except Exception as e:
                    lib_status = f"Помилка перевірки: {e}"
            rows.append({
                "id": ent["id"],
                "name": spec.get("name", ent["id"]),
                "icon": spec.get("icon", "🧩"),
                "version": spec.get("version", "1.0.0"),
                "kind": spec.get("kind", "ai"),
                "file": ent["file"],
                "status": lib_status,
                "error": "",
            })
        for ek, eobj in self.get_errors().items():
            rows.append({
                "id": ek,
                "name": ek,
                "icon": "⚠",
                "version": "—",
                "kind": "error",
                "file": eobj.get("file", ""),
                "status": f"Помилка завантаження: {eobj.get('error', '')}",
                "error": eobj.get("traceback", ""),
            })
        return rows
