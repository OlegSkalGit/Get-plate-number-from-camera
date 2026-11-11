import os
import copy
import threading

from scripts.core.hardware_info import prepare_cuda_dll_environment, check_cuda_libraries_status


def resolve_ort_providers(ai_backend: str, base_dir: str = None) -> list:
    backend = str(ai_backend or "cpu").strip().lower()
    if base_dir and backend in ("cuda", "auto"):
        try:
            prepare_cuda_dll_environment(base_dir, preload=False)
        except Exception:
            pass
    try:
        import onnxruntime as ort
        try:
            ort.set_default_logger_severity(3)
        except Exception:
            pass
        if base_dir and backend in ("cuda", "auto"):
            try:
                prepare_cuda_dll_environment(base_dir, preload=True)
            except Exception:
                pass
        avail = list(ort.get_available_providers())
    except Exception:
        avail = ["CPUExecutionProvider"]

    if base_dir and backend in ("cuda", "auto") and "CUDAExecutionProvider" in avail:
        try:
            if not check_cuda_libraries_status(base_dir, allow_network=False).get("ready"):
                avail = [p for p in avail if p != "CUDAExecutionProvider"]
        except Exception:
            pass

    def _pick(*wanted):
        res = [p for p in wanted if p in avail]
        if "CPUExecutionProvider" not in res:
            res.append("CPUExecutionProvider")
        return res

    if backend == "cuda":
        return _pick("CUDAExecutionProvider")
    if backend == "directml":
        return _pick("DmlExecutionProvider")
    if backend == "openvino":
        return _pick("OpenVINOExecutionProvider")
    if backend == "auto":
        return _pick("CUDAExecutionProvider", "DmlExecutionProvider", "OpenVINOExecutionProvider")
    return ["CPUExecutionProvider"]


class InferencePool:
    """
    Повністю модульний спільний пул нейромережевих сесій та спільних ресурсів.
    Не містить захардкоджених залежностей від конкретних модулів (plates/faces/people тощо) —
    усі ресурси створюються й оновлюються через хуки MODULE_SPEC у ModuleRegistry.
    """
    def __init__(self, base_dir: str, registry, app_context: dict = None, ai_backend: str = "cpu", ai_threads: int = 2):
        self.base_dir = os.path.abspath(base_dir)
        self.registry = registry
        self.app_context = dict(app_context or {})
        self.app_context["base_dir"] = self.base_dir

        self.ai_backend = str(ai_backend or "cpu").strip().lower()
        self.ai_threads = max(1, min(8, int(ai_threads or 2)))
        self.providers = resolve_ort_providers(self.ai_backend, base_dir=self.base_dir)

        self._lock = threading.RLock()
        # mod_id -> {"res": obj, "sig": file_sig, "ctx": dict}
        self._module_resources = {}

    def configure(self, ai_backend: str = None, ai_threads: int = None, app_context: dict = None):
        with self._lock:
            new_backend = str(ai_backend if ai_backend is not None else self.ai_backend).strip().lower()
            new_threads = max(1, min(8, int(ai_threads if ai_threads is not None else self.ai_threads)))
            new_ctx = dict(self.app_context)
            if app_context:
                new_ctx.update(app_context)
            new_ctx["base_dir"] = self.base_dir

            backend_changed = (new_backend != self.ai_backend) or (new_threads != self.ai_threads)
            ctx_changed = (new_ctx != self.app_context)

            old_providers = list(self.providers)
            self.ai_backend = new_backend
            self.ai_threads = new_threads
            self.app_context = new_ctx
            self.providers = resolve_ort_providers(new_backend, base_dir=self.base_dir)
            if list(self.providers) != old_providers:
                backend_changed = True

            if backend_changed:
                self._module_resources.clear()
            elif ctx_changed:
                for mid in list(self._module_resources.keys()):
                    ent = self.registry.get_module(mid)
                    on_ctx = ent["spec"].get("on_app_context_changed") if ent else None
                    if callable(on_ctx):
                        try:
                            cached = self._module_resources[mid]
                            cached["res"] = on_ctx(
                                cached["res"],
                                copy.deepcopy(self.app_context),
                                self.base_dir,
                                list(self.providers),
                                self.ai_threads,
                            )
                            cached["ctx"] = copy.deepcopy(self.app_context)
                        except Exception as e:
                            print(f"[InferencePool] Помилка on_app_context_changed для '{mid}': {e}")
                            self._module_resources.pop(mid, None)
                    else:
                        self._module_resources.pop(mid, None)

    def invalidate_module(self, mod_id: str):
        with self._lock:
            self._module_resources.pop(str(mod_id), None)

    def invalidate_all(self):
        with self._lock:
            self.providers = resolve_ort_providers(self.ai_backend, base_dir=self.base_dir)
            self._module_resources.clear()

    def get_module_resources(self, mod_id: str):
        """Повертає або ініціалізує спільні ресурси (напр. ONNX-сесії) для вказаного модуля."""
        with self._lock:
            ent = self.registry.get_module(mod_id)
            if not ent:
                self._module_resources.pop(mod_id, None)
                return None

            spec = ent["spec"]
            creator = spec.get("create_shared_resources")
            if not callable(creator):
                return None

            cached = self._module_resources.get(mod_id)
            if cached is not None:
                if cached.get("sig") == ent.get("sig"):
                    if cached.get("ctx") == self.app_context:
                        return cached["res"]
                    on_ctx = spec.get("on_app_context_changed")
                    if callable(on_ctx):
                        cached["res"] = on_ctx(
                            cached["res"],
                            copy.deepcopy(self.app_context),
                            self.base_dir,
                            list(self.providers),
                            self.ai_threads,
                        )
                        cached["ctx"] = copy.deepcopy(self.app_context)
                        return cached["res"]

            if self.ai_backend in ("cuda", "auto") or "CUDAExecutionProvider" in self.providers:
                try:
                    prepare_cuda_dll_environment(self.base_dir, preload=True)
                    self.providers = resolve_ort_providers(self.ai_backend, base_dir=self.base_dir)
                except Exception:
                    pass

            res = creator(
                self.base_dir,
                list(self.providers),
                self.ai_threads,
                copy.deepcopy(self.app_context),
            )
            self._module_resources[mod_id] = {
                "res": res,
                "sig": ent.get("sig"),
                "ctx": copy.deepcopy(self.app_context),
            }
            return res

    def get_capability_accessor(self, capability_name: str):
        """
        Знаходить перший активний модуль, що реалізує capability_name (напр. 'face_db'),
        і повертає UI/сервісний об'єкт доступу, не прив'язуючись до конкретного файлу модуля.
        """
        with self._lock:
            entries = self.registry.list_by_capability(capability_name)
            for ent in entries:
                getter = ent["spec"].get("get_db_accessor")
                if callable(getter):
                    cached = self._module_resources.get(ent["id"])
                    cached_res = cached["res"] if cached is not None else None
                    return getter(copy.deepcopy(self.app_context), cached_res)
        return None

    def get_active_summary(self) -> str:
        with self._lock:
            provs = []
            for entry in self._module_resources.values():
                res = entry.get("res")
                if isinstance(res, dict) and res.get("providers"):
                    provs = list(res["providers"])
                    break
            if not provs:
                provs = self.providers
            primary = provs[0].replace("ExecutionProvider", "") if provs else "CPU"
            return f"{self.ai_backend.upper()} ({primary})"
