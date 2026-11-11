import os
import re
import sys
import json
import threading

DEFAULT_LANGUAGE = "en"

_CODE_SUFFIX_RE = re.compile(r"\[([^\[\]]+)\]\s*$")


def _default_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normalize_language_code(code: str) -> str:
    s = str(code or "").strip().replace("_", "-")
    if not s:
        return ""
    if "-" in s:
        parts = [p for p in s.split("-") if p]
        if len(parts) >= 2:
            return f"{parts[0].lower()}-{'-'.join(p.upper() if len(p) == 2 else p for p in parts[1:])}"
    return s.lower()


def format_language_label(meta: dict) -> str:
    code = str(meta.get("code") or DEFAULT_LANGUAGE)
    name = str(meta.get("name") or code).strip()
    native = str(meta.get("native_name") or name).strip()
    if native.lower() == name.lower():
        return f"{native} [{code}]"
    return f"{native} ({name}) [{code}]"


class LanguageManager:
    """
    Менеджер мультимовності додатку.
    Завантажує окремі JSON-файли локалізації з папки lang/ (<code >.json),
    забезпечує fallback на англійську ('en') та безпечну підстановку параметрів.
    """
    def __init__(self, base_dir: str = None, lang_code: str = None):
        self._lock = threading.RLock()
        self.base_dir = os.path.abspath(base_dir or _default_base_dir())
        self._languages = {}     # canonical_code -> {"meta": dict, "strings": dict, "path": str}
        self._norm_to_canon = {} # lowercase code -> canonical_code
        self._label_to_code = {} # display label -> canonical_code
        self._current_code = DEFAULT_LANGUAGE
        self._active_strings = {}
        self._en_strings = {}

        self.reload()
        self.set_language(lang_code or DEFAULT_LANGUAGE)

    def _candidate_lang_dirs(self) -> list:
        dirs = []
        primary = os.path.join(self.base_dir, "lang")
        dirs.append(primary)
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                alt = os.path.join(os.path.abspath(meipass), "lang")
                if os.path.normcase(alt) != os.path.normcase(primary):
                    dirs.append(alt)
        return dirs

    def get_lang_dir(self) -> str:
        return os.path.join(self.base_dir, "lang")

    def reload(self):
        loaded = {}
        norm_map = {}

        # Читаємо від менш пріоритетної директорії до основної, щоб файли у <base_dir>/lang/ перекривали _MEIPASS
        for lang_dir in reversed(self._candidate_lang_dirs()):
            if not os.path.isdir(lang_dir):
                continue
            try:
                fnames = sorted(os.listdir(lang_dir))
            except Exception:
                continue
            for fn in fnames:
                if not fn.lower().endswith(".json"):
                    continue
                fpath = os.path.join(lang_dir, fn)
                if not os.path.isfile(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8-sig") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"[i18n] Failed to load language file '{fpath}': {e}")
                    continue

                if not isinstance(data, dict):
                    continue
                raw_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
                raw_strings = data.get("strings") if isinstance(data.get("strings"), dict) else {}

                stem = os.path.splitext(fn)[0]
                code = str(raw_meta.get("code") or stem).strip()
                if not code:
                    continue

                try:
                    order = int(raw_meta.get("order", 999))
                except Exception:
                    order = 999

                name = str(raw_meta.get("name") or code).strip()
                native_name = str(raw_meta.get("native_name") or name).strip()
                direction = "rtl" if str(raw_meta.get("direction", "ltr")).strip().lower() == "rtl" else "ltr"

                meta = {
                    "code": code,
                    "name": name,
                    "native_name": native_name,
                    "direction": direction,
                    "order": order,
                }
                meta["label"] = format_language_label(meta)

                strings_clean = {}
                for k, v in raw_strings.items():
                    if k is not None and v is not None:
                        strings_clean[str(k)] = str(v)

                low_k = code.lower().replace("_", "-")
                prev_canon = norm_map.get(low_k)
                if prev_canon and prev_canon in loaded and prev_canon != code:
                    loaded.pop(prev_canon, None)

                loaded[code] = {
                    "meta": meta,
                    "strings": strings_clean,
                    "path": fpath,
                }
                norm_map[low_k] = code

        with self._lock:
            self._languages = loaded
            self._norm_to_canon = norm_map
            self._label_to_code = {
                ent["meta"]["label"]: code for code, ent in loaded.items()
            }
            en_canon = self._resolve_code_locked(DEFAULT_LANGUAGE)
            self._en_strings = dict((loaded.get(en_canon) or {}).get("strings") or {})

            cur_canon = self._resolve_code_locked(self._current_code)
            if cur_canon:
                self._current_code = cur_canon
                self._active_strings = dict((loaded.get(cur_canon) or {}).get("strings") or {})
            elif en_canon:
                self._current_code = en_canon
                self._active_strings = dict(self._en_strings)
            elif loaded:
                first_code = sorted(loaded.keys())[0]
                self._current_code = first_code
                self._active_strings = dict(loaded[first_code]["strings"])
            else:
                self._current_code = DEFAULT_LANGUAGE
                self._active_strings = {}

    def _resolve_code_locked(self, code: str) -> str:
        raw = str(code or "").strip()
        if not raw:
            return ""
        if raw in self._languages:
            return raw
        low = raw.lower().replace("_", "-")
        if low in self._norm_to_canon:
            return self._norm_to_canon[low]
        norm = normalize_language_code(raw)
        if norm in self._languages:
            return norm
        if norm.lower() in self._norm_to_canon:
            return self._norm_to_canon[norm.lower()]
        return ""

    def set_base_dir(self, base_dir: str, lang_code: str = None):
        target_dir = os.path.abspath(base_dir or _default_base_dir())
        with self._lock:
            need_reload = (os.path.normcase(target_dir) != os.path.normcase(self.base_dir)) or (not self._languages)
            self.base_dir = target_dir
        if need_reload:
            self.reload()
        if lang_code is not None:
            self.set_language(lang_code)

    def set_language(self, code: str) -> str:
        with self._lock:
            resolved = self._resolve_code_locked(code)
            if not resolved:
                resolved = self._resolve_code_locked(DEFAULT_LANGUAGE)
            if not resolved and self._languages:
                resolved = self.list_languages()[0]["code"]
            if resolved:
                self._current_code = resolved
                self._active_strings = dict(self._languages[resolved]["strings"])
            else:
                self._current_code = DEFAULT_LANGUAGE
                self._active_strings = {}
            return self._current_code

    def get_language(self) -> str:
        with self._lock:
            return self._current_code or DEFAULT_LANGUAGE

    def get_language_meta(self, code: str = None) -> dict:
        with self._lock:
            c = self._resolve_code_locked(code) if code is not None else self._current_code
            ent = self._languages.get(c)
            if ent:
                return dict(ent["meta"])
            return {
                "code": c or DEFAULT_LANGUAGE,
                "name": "English",
                "native_name": "English",
                "direction": "ltr",
                "order": 1,
                "label": "English [en]",
            }

    def get_language_label(self, code: str = None) -> str:
        return str(self.get_language_meta(code).get("label") or "English [en]")

    def list_languages(self) -> list:
        with self._lock:
            items = []
            for code, ent in self._languages.items():
                m = dict(ent["meta"])
                m["path"] = ent["path"]
                items.append(m)
            items.sort(key=lambda m: (int(m.get("order", 999)), str(m.get("name", "")).lower(), str(m.get("code", "")).lower()))
            return items

    def parse_language_code(self, label_or_code: str, fallback: str = DEFAULT_LANGUAGE) -> str:
        s = str(label_or_code or "").strip()
        if not s:
            return self._resolve_code_locked(fallback) or DEFAULT_LANGUAGE
        with self._lock:
            if s in self._label_to_code:
                return self._label_to_code[s]
            m = _CODE_SUFFIX_RE.search(s)
            if m:
                cand = self._resolve_code_locked(m.group(1))
                if cand:
                    return cand
            direct = self._resolve_code_locked(s)
            if direct:
                return direct
            fb = self._resolve_code_locked(fallback)
            return fb or (self._current_code or DEFAULT_LANGUAGE)

    def has_key(self, key: str) -> bool:
        with self._lock:
            return (key in self._active_strings) or (key in self._en_strings)

    def tr(self, key: str, default: str = None, **kwargs) -> str:
        k = str(key or "")
        with self._lock:
            active_map = self._active_strings
            en_map = self._en_strings

        raw = active_map.get(k)
        if raw is None or raw == "":
            raw = en_map.get(k)
        if raw is None or raw == "":
            raw = k if default is None else str(default)

        if not kwargs:
            return raw

        try:
            return raw.format(**kwargs)
        except Exception:
            fallback_tmpl = en_map.get(k)
            if not fallback_tmpl:
                fallback_tmpl = k if default is None else str(default)
            try:
                return str(fallback_tmpl).format(**kwargs)
            except Exception:
                return raw


_GLOBAL_LOCK = threading.RLock()
_GLOBAL_MANAGER = None


def get_language_manager() -> LanguageManager:
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = LanguageManager(base_dir=_default_base_dir(), lang_code=DEFAULT_LANGUAGE)
        return _GLOBAL_MANAGER


def init_i18n(base_dir: str = None, lang_code: str = None) -> LanguageManager:
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = LanguageManager(base_dir=base_dir or _default_base_dir(), lang_code=lang_code or DEFAULT_LANGUAGE)
        else:
            _GLOBAL_MANAGER.set_base_dir(base_dir or _default_base_dir(), lang_code=lang_code)
        return _GLOBAL_MANAGER


def get_language() -> str:
    return get_language_manager().get_language()


def set_language(code: str) -> str:
    return get_language_manager().set_language(code)


def get_language_meta(code: str = None) -> dict:
    return get_language_manager().get_language_meta(code)


def get_language_label(code: str = None) -> str:
    return get_language_manager().get_language_label(code)


def list_languages() -> list:
    return get_language_manager().list_languages()


def parse_language_code(label_or_code: str, fallback: str = DEFAULT_LANGUAGE) -> str:
    return get_language_manager().parse_language_code(label_or_code, fallback=fallback)


def reload_languages() -> list:
    mgr = get_language_manager()
    mgr.reload()
    return mgr.list_languages()


def get_text_direction(code: str = None) -> str:
    return str(get_language_meta(code).get("direction") or "ltr")


def tr(key: str, default: str = None, **kwargs) -> str:
    return get_language_manager().tr(key, default=default, **kwargs)
