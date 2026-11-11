import os
import sys
import time
import queue
import subprocess
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from scripts.ui.helpers import WindowGeometryTracker, ScrollableFrame

try:
    import cv2
except ImportError:
    cv2 = None


def _ask_text_modal(parent, title: str, prompt: str, initial_val: str = ""):
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.configure(bg="#181922")
    dlg.resizable(False, False)
    dlg.transient(parent)
    dlg.grab_set()

    res = {"value": None}
    body = tk.Frame(dlg, bg="#181922", padx=18, pady=16)
    body.pack(fill=tk.BOTH, expand=True)

    tk.Label(
        body, text=prompt, fg="#e0e4f0", bg="#181922",
        font=("Segoe UI", 9), justify=tk.LEFT, wraplength=380
    ).pack(anchor=tk.W, pady=(0, 8))

    var = tk.StringVar(value=initial_val)
    entry = tk.Entry(
        body, textvariable=var, width=40,
        bg="#262836", fg="#ffffff", insertbackground="white",
        relief=tk.FLAT, font=("Segoe UI", 10)
    )
    entry.pack(fill=tk.X, ipady=5, pady=(0, 14))
    entry.select_range(0, tk.END)
    entry.focus_set()

    btns = tk.Frame(body, bg="#181922")
    btns.pack(fill=tk.X)

    def _ok(*_):
        val = var.get().strip()
        if not val:
            return
        res["value"] = val
        dlg.destroy()

    def _cancel(*_):
        dlg.destroy()

    tk.Button(
        btns, text="Скасувати", command=_cancel,
        bg="#323544", fg="#d5dae8", relief=tk.FLAT,
        font=("Segoe UI", 9, "bold"), padx=14, pady=5, cursor="hand2"
    ).pack(side=tk.RIGHT, padx=(6, 0))
    tk.Button(
        btns, text="Зберегти", command=_ok,
        bg="#007acc", fg="#ffffff", relief=tk.FLAT,
        font=("Segoe UI", 9, "bold"), padx=16, pady=5, cursor="hand2"
    ).pack(side=tk.RIGHT)

    dlg.bind("<Return>", _ok)
    dlg.bind("<Escape>", _cancel)
    dlg.update_idletasks()
    try:
        px = parent.winfo_rootx() + (parent.winfo_width() - dlg.winfo_width()) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{max(0, px)}+{max(0, py)}")
    except Exception:
        pass
    dlg.wait_window()
    return res["value"]


class ReanalyzeCropDialog:
    """
    Діалог реаналізу помилкового кропу:
    показує кроп, запускає пошук по базі облич та кропах на диску (виключаючи поточного власника)
    і дозволяє обрати найбільш підходящу людину або створити нову з власним UUID.
    """
    def __init__(
        self,
        parent,
        db_accessor,
        event_info: dict,
        remaining_old_crops: list,
        person_crops_map: dict,
        default_min_sim: float = 0.45,
    ):
        self.parent = parent
        self.db = db_accessor
        self.ev = dict(event_info or {})
        self.remaining_old_crops = list(remaining_old_crops or [])
        self.person_crops_map = dict(person_crops_map or {})
        self.result = None

        self._thumbs_cache = []
        self._scan_seq = 0
        self._selected_target = None  # uid | "__NEW__"
        self._is_closing = False
        self._ui_queue = queue.SimpleQueue()
        self._ui_poll_after_id = None

        self.dlg = tk.Toplevel(parent)
        self.dlg.title("⚠ Не та людина — Реаналіз кропу обличчя")
        self.dlg.configure(bg="#14151e")
        self.dlg.geometry("680x580")
        self.dlg.minsize(560, 480)
        self.dlg.transient(parent)
        self.dlg.grab_set()
        self.dlg.protocol("WM_DELETE_WINDOW", self._close_dlg)

        self.thresh_var = tk.DoubleVar(value=round(float(default_min_sim or 0.45), 2))
        self.choice_var = tk.StringVar(value="__NEW__")

        self._build_ui()
        self._center_over_parent()
        self._drain_ui_queue()
        self._run_scan()
        self.dlg.wait_window()

    def _drain_ui_queue(self):
        if self._is_closing:
            return
        while True:
            try:
                cb, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cb(*args)
            except Exception:
                pass
        if not self._is_closing:
            try:
                if self.dlg.winfo_exists():
                    self._ui_poll_after_id = self.dlg.after(25, self._drain_ui_queue)
            except Exception:
                pass

    def _close_dlg(self):
        if self._is_closing:
            return
        self._is_closing = True
        if self._ui_poll_after_id is not None:
            try:
                self.dlg.after_cancel(self._ui_poll_after_id)
            except Exception:
                pass
            self._ui_poll_after_id = None
        try:
            self.dlg.destroy()
        except Exception:
            pass

    def _center_over_parent(self):
        try:
            self.dlg.update_idletasks()
            px = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - self.dlg.winfo_width()) // 2)
            py = self.parent.winfo_rooty() + max(0, (self.parent.winfo_height() - self.dlg.winfo_height()) // 2)
            self.dlg.geometry(f"+{px}+{py}")
        except Exception:
            pass

    def _build_ui(self):
        # Заголовок та інформація про проблемний кроп
        top = tk.Frame(self.dlg, bg="#1b1d29", padx=14, pady=12, highlightthickness=1, highlightbackground="#2b2e3d")
        top.pack(fill=tk.X, padx=12, pady=(12, 8))

        crop_lbl = tk.Label(top, bg="#101118", width=88, height=88, text="👤", fg="#70758a", font=("Segoe UI", 16))
        cp = self.ev.get("crop_path", "")
        if cp and os.path.exists(cp):
            try:
                with Image.open(cp) as pim:
                    pim.load()
                    pim = pim.resize((84, 84), Image.Resampling.LANCZOS)
                    tkimg = ImageTk.PhotoImage(pim)
                    self._thumbs_cache.append(tkimg)
                    crop_lbl.configure(image=tkimg, text="", width=84, height=84)
            except Exception:
                pass
        crop_lbl.pack(side=tk.LEFT, padx=(0, 12))

        info = tk.Frame(top, bg="#1b1d29")
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            info,
            text=f"Поточне помилкове призначення: {self.ev.get('person_name', '—')}",
            fg="#ffcc00", bg="#1b1d29", font=("Segoe UI", 10, "bold"), anchor=tk.W
        ).pack(fill=tk.X)
        tk.Label(
            info,
            text=f"UUID: {self.ev.get('person_uuid') or 'неприв’язано'}",
            fg="#8c92a4", bg="#1b1d29", font=("Consolas", 8), anchor=tk.W
        ).pack(fill=tk.X, pady=(2, 4))
        tk.Label(
            info,
            text=f"Камера: {self.ev.get('camera_name', '—')}  •  Час: {self.ev.get('day', '')} {self.ev.get('time', '')}",
            fg="#d5dae8", bg="#1b1d29", font=("Segoe UI", 9), anchor=tk.W
        ).pack(fill=tk.X)
        rem_cnt = len(self.remaining_old_crops)
        rem_txt = (
            f"Після вилучення решта кропів ({rem_cnt} шт.) буде перерахована для очищення еталону."
            if rem_cnt > 0 else
            "Це єдиний кроп поточної людини на диску."
        )
        tk.Label(
            info, text=rem_txt, fg="#75e08b" if rem_cnt > 0 else "#9ba3b8",
            bg="#1b1d29", font=("Segoe UI", 8), anchor=tk.W
        ).pack(fill=tk.X, pady=(4, 0))

        # Панель налаштування порогу та повторного сканування
        ctrl = tk.Frame(self.dlg, bg="#14151e", padx=12, pady=4)
        ctrl.pack(fill=tk.X)
        tk.Label(
            ctrl, text="Поріг збігу для пошуку:", fg="#d5dae8", bg="#14151e", font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)
        tk.Spinbox(
            ctrl, from_=0.20, to=0.90, increment=0.02, textvariable=self.thresh_var, width=6,
            bg="#262836", fg="#ffffff", buttonbackground="#323544", relief=tk.FLAT, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            ctrl, text="🔄 Реаналізувати базу кропів", command=self._run_scan,
            bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(4, 8))

        self.status_lbl = tk.Label(
            ctrl, text="Аналіз...", fg="#ffcc00", bg="#14151e", font=("Segoe UI", 8, "bold")
        )
        self.status_lbl.pack(side=tk.RIGHT)

        # Список кандидатів зі скролом
        self.list_scroll = ScrollableFrame(self.dlg, bg="#14151e")
        self.list_scroll.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Нижня панель дій
        bot = tk.Frame(self.dlg, bg="#1b1d29", padx=14, pady=10, highlightthickness=1, highlightbackground="#2b2e3d")
        bot.pack(fill=tk.X)
        self.btn_del_only = tk.Button(
            bot, text="🗑 Просто видалити кроп", command=self._delete_only,
            bg="#64262b", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=6, cursor="hand2"
        )
        self.btn_del_only.pack(side=tk.LEFT)
        tk.Button(
            bot, text="Скасувати", command=self._close_dlg,
            bg="#323544", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=14, pady=6, cursor="hand2"
        ).pack(side=tk.RIGHT, padx=(8, 0))
        self.btn_apply = tk.Button(
            bot, text="✅ Перепризначити кроп", command=self._apply,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=16, pady=6, cursor="hand2"
        )
        self.btn_apply.pack(side=tk.RIGHT)

    def _run_scan(self):
        self._scan_seq += 1
        seq = self._scan_seq
        for ch in list(self.list_scroll.inner.winfo_children()):
            ch.destroy()
        self.status_lbl.configure(text="⏳ Реаналіз ембедінгів...", fg="#ffcc00")
        self.btn_apply.configure(state=tk.DISABLED)

        old_uid = str(self.ev.get("person_uuid") or "")
        crop_path = str(self.ev.get("crop_path") or "")
        min_sim = float(self.thresh_var.get() or 0.45)

        def _worker():
            res = self.db.analyze_crop_candidates(
                crop_path=crop_path,
                exclude_uids=[old_uid] if old_uid else None,
                person_crops_map=self.person_crops_map,
                min_similarity=min_sim,
                max_candidates=10,
            )
            self._ui_queue.put((self._on_scan_done, (seq, res)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_scan_done(self, seq: int, res: dict):
        if seq != self._scan_seq or not self.dlg.winfo_exists():
            return
        for ch in list(self.list_scroll.inner.winfo_children()):
            ch.destroy()
        self._thumbs_cache.clear()

        if not res or not res.get("ok"):
            err = (res or {}).get("error") or "Не вдалося виконати аналіз."
            self.status_lbl.configure(text="❌ Помилка", fg="#ff6666")
            tk.Label(
                self.list_scroll.inner, text=err, fg="#ff7777", bg="#14151e",
                font=("Segoe UI", 9), wraplength=580, justify=tk.LEFT, pady=16
            ).pack(anchor=tk.W)
            return

        cands = list(res.get("candidates") or [])
        best = res.get("best_candidate")
        self.btn_apply.configure(state=tk.NORMAL)

        if best:
            self.choice_var.set(best["id"])
            self.status_lbl.configure(
                text=f"Знайдено збіг: {best['name']} ({best['similarity']*100:.1f}%)", fg="#75e08b"
            )
        else:
            self.choice_var.set("__NEW__")
            self.status_lbl.configure(
                text="Підходящих збігів ≥ порогу не знайдено — рекомендовано новий UUID", fg="#00e5ff"
            )

        # Опція створення нової людини з новим UUID
        new_card = tk.Frame(
            self.list_scroll.inner, bg="#1d2433", padx=10, pady=8,
            highlightthickness=1, highlightbackground="#00a8cc" if best is None else "#2b2e3d"
        )
        new_card.pack(fill=tk.X, pady=4)
        rb_new = tk.Radiobutton(
            new_card,
            text="➕ Створити нову окрему людину (новий унікальний UUID)",
            variable=self.choice_var,
            value="__NEW__",
            bg="#1d2433", fg="#00e5ff", selectcolor="#12131a",
            activebackground="#1d2433", activeforeground="#00e5ff",
            font=("Segoe UI", 9, "bold"), cursor="hand2"
        )
        rb_new.pack(anchor=tk.W)
        tk.Label(
            new_card,
            text="Використовуйте, якщо на кропі інша особа, якої ще немає в базі (або схожість занизька).",
            fg="#9ba3b8", bg="#1d2433", font=("Segoe UI", 8)
        ).pack(anchor=tk.W, padx=(22, 0))

        if not cands:
            tk.Label(
                self.list_scroll.inner,
                text="Інших відомих людей у базі поки немає.",
                fg="#8c92a4", bg="#14151e", font=("Segoe UI", 9), pady=12
            ).pack(anchor=tk.W)
            return

        tk.Label(
            self.list_scroll.inner,
            text=f"Кандидати з бази облич та збережених кропів (усього порівняно: {res.get('all_candidates_count', len(cands))}):",
            fg="#d5dae8", bg="#14151e", font=("Segoe UI", 9, "bold"), anchor=tk.W
        ).pack(fill=tk.X, pady=(8, 4))

        for cand in cands:
            is_pass = bool(cand.get("passes_threshold"))
            bg_c = "#1b2824" if is_pass else "#1b1d29"
            bd_c = "#38a169" if (best and cand["id"] == best["id"]) else "#2b2e3d"
            row = tk.Frame(self.list_scroll.inner, bg=bg_c, padx=10, pady=7, highlightthickness=1, highlightbackground=bd_c)
            row.pack(fill=tk.X, pady=3)

            t_lbl = tk.Label(row, bg="#101118", width=6, height=2, text="👤", fg="#70758a", font=("Segoe UI", 10))
            tp = cand.get("thumb_path") or cand.get("matched_crop") or ""
            if tp and os.path.exists(tp):
                try:
                    with Image.open(tp) as pim:
                        pim.load()
                        pim = pim.resize((44, 44), Image.Resampling.LANCZOS)
                        tkimg = ImageTk.PhotoImage(pim)
                        self._thumbs_cache.append(tkimg)
                        t_lbl.configure(image=tkimg, text="", width=44, height=44)
                except Exception:
                    pass
            t_lbl.pack(side=tk.LEFT, padx=(0, 10))

            sim_pct = cand["similarity"] * 100.0
            gal_pct = cand["gallery_sim"] * 100.0
            crop_pct_txt = f" | кроп на диску: {cand['crop_sim']*100:.1f}%" if cand.get("crop_sim") is not None else ""
            badge = " [ЗБІГ]" if is_pass else ""

            rb = tk.Radiobutton(
                row,
                text=f"{cand['name']}  —  {sim_pct:.1f}%{badge}",
                variable=self.choice_var,
                value=cand["id"],
                bg=bg_c, fg="#75e08b" if is_pass else "#ffffff", selectcolor="#12131a",
                activebackground=bg_c, activeforeground="#ffffff",
                font=("Segoe UI", 9, "bold"), cursor="hand2", anchor=tk.W
            )
            rb.pack(fill=tk.X, anchor=tk.W)
            tk.Label(
                row,
                text=f"UUID: {cand['id']}  •  еталон: {gal_pct:.1f}%{crop_pct_txt}",
                fg="#8c92a4", bg=bg_c, font=("Segoe UI", 8), anchor=tk.W
            ).pack(fill=tk.X, padx=(22, 0))

    def _apply(self):
        choice = self.choice_var.get()
        if not choice:
            return
        self.btn_apply.configure(state=tk.DISABLED)
        self.status_lbl.configure(text="⏳ Оновлення бази та перепризначення...", fg="#ffcc00")

        old_uid = str(self.ev.get("person_uuid") or "")
        crop_path = str(self.ev.get("crop_path") or "")
        min_sim = float(self.thresh_var.get() or 0.45)

        def _exec():
            res = self.db.reanalyze_wrong_crop(
                crop_path=crop_path,
                old_u_id=old_uid,
                remaining_old_crops=self.remaining_old_crops,
                target_uid=choice,
                person_crops_map=self.person_crops_map,
                min_similarity=min_sim,
                full_path=str(self.ev.get("full_path") or ""),
                day_str=str(self.ev.get("day") or ""),
                time_str=str(self.ev.get("time") or ""),
                timestamp=float(self.ev.get("timestamp") or 0.0),
                sharpness=self.ev.get("sharpness"),
                remove_empty_old=False,
            )
            self._ui_queue.put((self._on_apply_done, (res,)))

        threading.Thread(target=_exec, daemon=True).start()

    def _delete_only(self):
        if not messagebox.askyesno(
            "Видалення помилкового кропу",
            "Просто видалити цей помилковий кроп з диска та перерахувати еталон поточної людини по решті кропів?",
            parent=self.dlg,
        ):
            return
        self.btn_apply.configure(state=tk.DISABLED)
        self.btn_del_only.configure(state=tk.DISABLED)
        self.status_lbl.configure(text="⏳ Видалення кропу та перерахунок еталону...", fg="#ffcc00")

        def _exec_del():
            res = self.db.delete_crop_event(self.ev, self.remaining_old_crops)
            if isinstance(res, dict) and res.get("ok"):
                res["deleted"] = True
            self._ui_queue.put((self._on_apply_done, (res,)))

        threading.Thread(target=_exec_del, daemon=True).start()

    def _on_apply_done(self, res: dict):
        if self._is_closing or not self.dlg.winfo_exists():
            return
        if not res or not res.get("ok"):
            self.btn_apply.configure(state=tk.NORMAL)
            self.btn_del_only.configure(state=tk.NORMAL)
            self.status_lbl.configure(text="❌ Помилка", fg="#ff6666")
            messagebox.showerror(
                "Помилка операції",
                (res or {}).get("error") or "Не вдалося виконати дію з кропом.",
                parent=self.dlg,
            )
            return
        self.result = res
        self._close_dlg()


class PeopleWindow:
    """
    Незалежне вікно «Люди»:
    - список унікальних людей із бази облич (кожне нове обличчя має власний UUID);
    - усі кропи, повні скріни та відеофрагменти з обраною людиною по всіх камерах, відсортовані по часу;
    - перейменування, ігнорування, видалення окремих кропів або профілю;
    - позначка «Не та людина» з реаналізом бази кропів та пошуком підходящої людини або створенням нового UUID.
    """
    def __init__(self, parent_root, cfg_mgr, camera_manager, registry, on_close_cb=None):
        self.parent_root = parent_root
        self.cfg_mgr = cfg_mgr
        self.camera_manager = camera_manager
        self.registry = registry
        self.on_close_cb = on_close_cb

        self._is_closing = False
        self._load_seq = 0
        self._video_seq = 0
        self._notify_after_id = None
        self._ui_queue = queue.SimpleQueue()
        self._ui_poll_after_id = None

        self.win = tk.Toplevel(parent_root)
        self.win.title("👥 Люди — Унікальні особи, кропи, скріни та відеоархів")
        self.win.configure(bg="#12131a")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.geom_tracker = WindowGeometryTracker(
            self.win,
            get_geom_cb=lambda: self.cfg_mgr.get_window_geometry("people"),
            set_geom_cb=lambda g: self.cfg_mgr.set_window_geometry("people", g),
            min_w=860,
            min_h=540,
            default_geom={"left": 120, "top": 70, "width": 1180, "height": 760, "maximized": False},
        )

        # Дані поточного зрізу
        self._records = []              # список записів з faces.txt
        self._events_by_person = {}     # uid -> list[event_dict]
        self._unassigned_events = []    # list[event_dict]
        self._summary = {}              # uid -> stats
        self._selected_uid = None       # uid або "__UNASSIGNED__"
        self._matched_videos = []       # list[video_seg_dict] для обраної людини
        self._timeline_items = []       # відфільтровані та відсортовані елементи правої стрічки
        self._tree_item_map = {}        # iid -> timeline item dict
        self._current_item = None       # поточний обраний елемент таймлайну

        # UI-кеші зображень
        self._left_thumbs = []
        self._hdr_thumb_tk = None
        self._crop_preview_tk = None
        self._screen_preview_tk = None
        self._person_card_frames = {}

        # Відеоплеєр
        self._vid_cap = None
        self._vid_path = None
        self._vid_fps = 15.0
        self._vid_total_frames = 0
        self._vid_curr_frame = 0
        self._vid_playing = False
        self._vid_speed = 1
        self._vid_after_id = None
        self._vid_updating_slider = False
        self._vid_photo_tk = None

        # Змінні фільтрів
        self.search_var = tk.StringVar()
        self.people_sort_var = tk.StringVar(value="recent")
        self.kind_filter_var = tk.StringVar(value="all")
        self.sort_desc_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.search_var.trace_add("write", lambda *_: self._render_people_list())
        self._drain_ui_queue()
        self._init_refresh_after_id = self.win.after(60, self.refresh_all)

    def _drain_ui_queue(self):
        if self._is_closing:
            return
        while True:
            try:
                cb, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cb(*args)
            except Exception:
                pass
        if not self._is_closing:
            try:
                if self.win.winfo_exists():
                    self._ui_poll_after_id = self.win.after(20, self._drain_ui_queue)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Інтерфейс вікна
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Верхня панель інструментів
        top_bar = tk.Frame(self.win, bg="#191b26", padx=12, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
        top_bar.pack(fill=tk.X)

        tk.Label(
            top_bar, text="👥 Люди та історія появ", fg="#ffffff", bg="#191b26",
            font=("Segoe UI", 11, "bold")
        ).pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(top_bar, text="🔍", fg="#9ba3b8", bg="#191b26", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        tk.Entry(
            top_bar, textvariable=self.search_var, width=26,
            bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, ipady=4, padx=(4, 12))

        self.lbl_global_stats = tk.Label(
            top_bar, text="Людей: 0", fg="#00e5ff", bg="#222534",
            font=("Segoe UI", 8, "bold"), padx=10, pady=4
        )
        self.lbl_global_stats.pack(side=tk.LEFT, padx=(0, 10))

        tk.Button(
            top_bar, text="✖ Закрити", command=self.close,
            bg="#44252a", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=4, cursor="hand2"
        ).pack(side=tk.RIGHT)
        tk.Button(
            top_bar, text="↻ Оновити", command=self.refresh_all,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=4, cursor="hand2"
        ).pack(side=tk.RIGHT, padx=(0, 6))
        self.btn_reanalyze_all = tk.Button(
            top_bar, text="♻ Реаналіз бази з кропів", command=self._on_reanalyze_all_profiles,
            bg="#2b3145", fg="#75e08b", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=4, cursor="hand2"
        )
        self.btn_reanalyze_all.pack(side=tk.RIGHT, padx=(0, 6))

        # Основна горизонтальна область
        self.main_paned = tk.PanedWindow(
            self.win, orient=tk.HORIZONTAL, bg="#12131a", sashwidth=5, bd=0, relief=tk.FLAT
        )
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Ліва панель: список унікальних людей ---
        left_pane = tk.Frame(self.main_paned, bg="#171923", highlightthickness=1, highlightbackground="#292d3e")
        self.main_paned.add(left_pane, minsize=260, width=320)

        left_hdr = tk.Frame(left_pane, bg="#1e202c", padx=10, pady=7)
        left_hdr.pack(fill=tk.X)
        tk.Label(
            left_hdr, text="Унікальні люди", fg="#ffffff", bg="#1e202c", font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        sort_box = tk.Frame(left_hdr, bg="#1e202c")
        sort_box.pack(side=tk.RIGHT)
        self.btn_sort_people = tk.Button(
            sort_box, text="🕒 Нещодавні", command=self._cycle_people_sort,
            bg="#2b2f40", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=7, pady=2, cursor="hand2"
        )
        self.btn_sort_people.pack(side=tk.RIGHT)

        self.people_scroll = ScrollableFrame(left_pane, bg="#151720")
        self.people_scroll.pack(fill=tk.BOTH, expand=True)

        # --- Права панель: картка людини + таймлайн + перегляд ---
        right_pane = tk.Frame(self.main_paned, bg="#12131a")
        self.main_paned.add(right_pane, minsize=520)

        # Заголовок обраної людини
        self.person_hdr = tk.Frame(
            right_pane, bg="#191b26", padx=12, pady=10, highlightthickness=1, highlightbackground="#292d3e"
        )
        self.person_hdr.pack(fill=tk.X, pady=(0, 8))

        self.hdr_thumb_lbl = tk.Label(
            self.person_hdr, bg="#101118", width=8, height=4, text="👤", fg="#70758a", font=("Segoe UI", 16)
        )
        self.hdr_thumb_lbl.pack(side=tk.LEFT, padx=(0, 12))

        hdr_info = tk.Frame(self.person_hdr, bg="#191b26")
        hdr_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_person_name = tk.Label(
            hdr_info, text="Оберіть людину зі списку ліворуч",
            fg="#ffffff", bg="#191b26", font=("Segoe UI", 12, "bold"), anchor=tk.W
        )
        self.lbl_person_name.pack(fill=tk.X)

        uuid_row = tk.Frame(hdr_info, bg="#191b26")
        uuid_row.pack(fill=tk.X, pady=(2, 2))
        self.lbl_person_uuid = tk.Label(
            uuid_row, text="UUID: —", fg="#00e5ff", bg="#191b26", font=("Consolas", 8), anchor=tk.W
        )
        self.lbl_person_uuid.pack(side=tk.LEFT)
        self.btn_copy_uuid = tk.Button(
            uuid_row, text="📋 Копіювати UUID", command=self._copy_selected_uuid,
            bg="#262938", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 7, "bold"),
            padx=6, pady=1, cursor="hand2"
        )
        self.btn_copy_uuid.pack(side=tk.LEFT, padx=(8, 0))

        self.lbl_person_meta = tk.Label(
            hdr_info, text="", fg="#9ba3b8", bg="#191b26", font=("Segoe UI", 8), anchor=tk.W
        )
        self.lbl_person_meta.pack(fill=tk.X)

        hdr_actions = tk.Frame(self.person_hdr, bg="#191b26")
        hdr_actions.pack(side=tk.RIGHT)
        self.btn_rename = tk.Button(
            hdr_actions, text="✎ Дати ім'я", command=self._rename_selected_person,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=5, cursor="hand2"
        )
        self.btn_rename.pack(side=tk.LEFT, padx=3)
        self.btn_ignore = tk.Button(
            hdr_actions, text="🚫 Ігнор", command=self._toggle_ignore_selected,
            bg="#3a2f22", fg="#ffcc00", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=5, cursor="hand2"
        )
        self.btn_ignore.pack(side=tk.LEFT, padx=3)
        self.btn_del_person = tk.Button(
            hdr_actions, text="🗑 Видалити", command=self._delete_selected_person,
            bg="#44252a", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=5, cursor="hand2"
        )
        self.btn_del_person.pack(side=tk.LEFT, padx=3)

        # Панель фільтрів таймлайну
        flt_bar = tk.Frame(right_pane, bg="#171923", padx=10, pady=6, highlightthickness=1, highlightbackground="#292d3e")
        flt_bar.pack(fill=tk.X, pady=(0, 8))

        self._filter_buttons = {}
        for fkey, flbl in (("all", "Усі"), ("crop", "👤 Кропи"), ("screen", "🖼 Повні скріни"), ("video", "🎬 Відео")):
            btn = tk.Button(
                flt_bar, text=flbl, command=lambda k=fkey: self._set_kind_filter(k),
                bg="#252836", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                padx=10, pady=4, cursor="hand2"
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
            self._filter_buttons[fkey] = btn

        self.btn_time_order = tk.Button(
            flt_bar, text="▼ Спочатку новіші", command=self._toggle_time_order,
            bg="#2b2f40", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2"
        )
        self.btn_time_order.pack(side=tk.RIGHT)

        self.lbl_video_scan = tk.Label(
            flt_bar, text="", fg="#ffcc00", bg="#171923", font=("Segoe UI", 8, "bold"), padx=8
        )
        self.lbl_video_scan.pack(side=tk.RIGHT)

        # Внутрішній поділ правої області: [Хронологічний список] | [Перегляд + Плеєр + Дії]
        self.right_paned = tk.PanedWindow(
            right_pane, orient=tk.HORIZONTAL, bg="#12131a", sashwidth=5, bd=0, relief=tk.FLAT
        )
        self.right_paned.pack(fill=tk.BOTH, expand=True)

        # Таймлайн (список усіх кропів/скрінів/відео по часу)
        tl_box = tk.Frame(self.right_paned, bg="#171923", highlightthickness=1, highlightbackground="#292d3e")
        self.right_paned.add(tl_box, minsize=250, width=340)

        style = ttk.Style(self.win)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "PeopleTL.Treeview",
            background="#151720",
            foreground="#e2e6f0",
            fieldbackground="#151720",
            rowheight=26,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.map(
            "PeopleTL.Treeview",
            background=[("selected", "#0066ad")],
            foreground=[("selected", "#ffffff")],
        )

        self.tl_tree = ttk.Treeview(tl_box, style="PeopleTL.Treeview", show="tree", selectmode="browse")
        tl_scroll = ttk.Scrollbar(tl_box, orient=tk.VERTICAL, command=self.tl_tree.yview)
        self.tl_tree.configure(yscrollcommand=tl_scroll.set)
        tl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tl_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tl_tree.bind("<<TreeviewSelect>>", self._on_timeline_select)

        # Права область перегляду
        self.view_box = tk.Frame(self.right_paned, bg="#171923", highlightthickness=1, highlightbackground="#292d3e")
        self.right_paned.add(self.view_box, minsize=340)

        self.lbl_item_header = tk.Label(
            self.view_box, text="Оберіть подію у списку", fg="#d5dae8", bg="#1e202c",
            font=("Segoe UI", 9, "bold"), padx=10, pady=7, anchor=tk.W
        )
        self.lbl_item_header.pack(fill=tk.X)

        # Контейнер перегляду зображення або відеоплеєра
        self.media_stage = tk.Frame(self.view_box, bg="#101118")
        self.media_stage.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 1) Панель перегляду кропу / скріна
        self.img_panel = tk.Frame(self.media_stage, bg="#101118")
        self.img_row = tk.Frame(self.img_panel, bg="#101118")
        self.img_row.pack(fill=tk.BOTH, expand=True)

        self.crop_frame_box = tk.Frame(self.img_row, bg="#151720", padx=8, pady=8, highlightthickness=1, highlightbackground="#282b3a")
        tk.Label(
            self.crop_frame_box, text="Кроп обличчя", fg="#00e5ff", bg="#151720", font=("Segoe UI", 8, "bold")
        ).pack(pady=(0, 6))
        self.lbl_crop_preview = tk.Label(self.crop_frame_box, bg="#0d0e14", text="—", fg="#6e7385")
        self.lbl_crop_preview.pack(expand=True)

        self.screen_frame_box = tk.Frame(self.img_row, bg="#151720", padx=8, pady=8, highlightthickness=1, highlightbackground="#282b3a")
        tk.Label(
            self.screen_frame_box, text="Повний кадр камери", fg="#9ee8ff", bg="#151720", font=("Segoe UI", 8, "bold")
        ).pack(pady=(0, 6))
        self.lbl_screen_preview = tk.Label(self.screen_frame_box, bg="#0d0e14", text="Повний скрін відсутній", fg="#6e7385")
        self.lbl_screen_preview.pack(fill=tk.BOTH, expand=True)
        self.screen_frame_box.bind("<Configure>", lambda e: self._rerender_active_images())

        # 2) Панель відеоплеєра
        self.video_panel = tk.Frame(self.media_stage, bg="#101118")
        self.vid_canvas = tk.Canvas(self.video_panel, bg="#0a0b10", highlightthickness=0)
        self.vid_canvas.pack(fill=tk.BOTH, expand=True)
        self.vid_canvas.bind("<Configure>", lambda e: self._refresh_video_canvas_center())

        self.vid_hits_bar = tk.Frame(self.video_panel, bg="#181a24", padx=8, pady=4)
        self.vid_hits_bar.pack(fill=tk.X, pady=(4, 0))

        vid_ctrl = tk.Frame(self.video_panel, bg="#1b1d29", padx=8, pady=6)
        vid_ctrl.pack(fill=tk.X, pady=(4, 0))

        self.btn_vid_play = tk.Button(
            vid_ctrl, text="▶", command=self._toggle_vid_play,
            bg="#28a745", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, cursor="hand2"
        )
        self.btn_vid_play.pack(side=tk.LEFT, padx=(0, 6))

        self.vid_slider_var = tk.DoubleVar(value=0.0)
        self.vid_slider = tk.Scale(
            vid_ctrl, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.vid_slider_var,
            command=self._on_vid_slider_move, showvalue=False,
            bg="#1b1d29", fg="#ffffff", troughcolor="#2a2d3d", highlightthickness=0, sliderlength=14
        )
        self.vid_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.lbl_vid_time = tk.Label(
            vid_ctrl, text="00:00/00:00", fg="#d5dae8", bg="#1b1d29", font=("Consolas", 8), width=12
        )
        self.lbl_vid_time.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_vid_speed = tk.Button(
            vid_ctrl, text="1x", command=self._cycle_vid_speed,
            bg="#2b2f40", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            width=4, cursor="hand2"
        )
        self.btn_vid_speed.pack(side=tk.LEFT)

        # Нижня панель дій для обраного кропу/скріна/відео
        self.item_actions = tk.Frame(self.view_box, bg="#1b1d29", padx=10, pady=8, highlightthickness=1, highlightbackground="#292d3e")
        self.item_actions.pack(fill=tk.X)

        self.btn_wrong_person = tk.Button(
            self.item_actions, text="⚠ Не та людина (реаналізувати)", command=self._on_wrong_person_clicked,
            bg="#b45309", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=5, cursor="hand2"
        )
        self.btn_wrong_person.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_del_crop = tk.Button(
            self.item_actions, text="🗑 Видалити кроп", command=self._on_delete_crop_clicked,
            bg="#64262b", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=5, cursor="hand2"
        )
        self.btn_del_crop.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_jump_video = tk.Button(
            self.item_actions, text="🎬 Відео цього моменту", command=self._on_jump_to_matching_video,
            bg="#1f4e78", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=5, cursor="hand2"
        )
        self.btn_jump_video.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_open_folder = tk.Button(
            self.item_actions, text="📂 Папка", command=self._open_item_folder,
            bg="#2f3346", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=5, cursor="hand2"
        )
        self.btn_open_folder.pack(side=tk.RIGHT)

        self.btn_open_external = tk.Button(
            self.item_actions, text="↗ Відкрити файл", command=self._open_item_external,
            bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=5, cursor="hand2"
        )
        self.btn_open_external.pack(side=tk.RIGHT, padx=(0, 6))

        self._update_filter_buttons_style()
        self._update_person_header()
        self._update_item_action_buttons()

    # ------------------------------------------------------------------
    # Доступ до модулів без жорсткого імпорту
    # ------------------------------------------------------------------
    def _get_face_db(self):
        try:
            return self.camera_manager.get_capability_accessor("face_db")
        except Exception:
            return None

    def _get_cameras_info(self) -> list:
        cams = self.cfg_mgr.get_cameras() or []
        out = []
        for c in cams:
            uid = str(c.get("uuid") or "")
            if not uid:
                continue
            out.append({
                "uuid": uid,
                "name": str(c.get("name") or uid),
                "archive_dir": self.cfg_mgr.get_camera_archive_dir(uid),
            })
        return out

    def _get_default_min_similarity(self) -> float:
        cams = self.cfg_mgr.get_cameras() or []
        vals = []
        for c in cams:
            fcfg = (c.get("modules") or {}).get("faces") or {}
            if "min_similarity" in fcfg:
                try:
                    vals.append(float(fcfg["min_similarity"]))
                except Exception:
                    pass
        if vals:
            return max(0.20, min(0.90, sum(vals) / len(vals)))
        return 0.48

    def _find_matching_videos_for_person(self, person_events: list) -> list:
        if not person_events:
            return []
        try:
            video_mods = self.registry.list_by_capability("video_recorder")
        except Exception:
            video_mods = []

        list_fn = None
        for ent in video_mods:
            arch = (ent.get("spec") or {}).get("archive") or {}
            fn = arch.get("list_segments")
            if callable(fn):
                list_fn = fn
                break
        if not callable(list_fn):
            return []

        ts_by_cam = {}
        ev_by_cam = {}
        for ev in person_events:
            cuid = str(ev.get("camera_uuid") or "")
            ts_by_cam.setdefault(cuid, []).append(float(ev["timestamp"]))
            ev_by_cam.setdefault(cuid, []).append(ev)

        cams_by_uid = {c["uuid"]: c for c in self._get_cameras_info()}
        matched = []
        for cuid, ts_list in ts_by_cam.items():
            cam_info = cams_by_uid.get(cuid)
            if not cam_info:
                continue
            cam_arch = cam_info["archive_dir"]
            unit = self.camera_manager.get_unit(cuid)
            try:
                segs = list_fn(cam_arch, target_timestamps=ts_list, margin_sec=2.5)
            except Exception:
                segs = []

            for seg in (segs or []):
                fpath = seg["path"]
                is_busy = bool(unit and unit.is_archive_file_busy("video", fpath))
                hits = []
                for h_ts in (seg.get("hit_timestamps") or []):
                    rel_sec = max(0.0, float(h_ts) - float(seg["start_ts"]))
                    if seg.get("duration_sec"):
                        rel_sec = min(rel_sec, max(0.0, float(seg["duration_sec"]) - 0.5))
                    hits.append({
                        "event_ts": float(h_ts),
                        "seek_sec": rel_sec,
                        "time_str": datetime.fromtimestamp(float(h_ts)).strftime("%H:%M:%S"),
                    })
                if not hits:
                    continue
                matched.append({
                    "kind": "video",
                    "camera_uuid": cuid,
                    "camera_name": cam_info["name"],
                    "path": fpath,
                    "filename": seg["filename"],
                    "day": seg["day"],
                    "time": seg["time"],
                    "start_ts": float(seg["start_ts"]),
                    "end_ts": float(seg["end_ts"]),
                    "duration_sec": float(seg["duration_sec"]),
                    "size_mb": float(seg["size_mb"]),
                    "busy": is_busy,
                    "hits": hits,
                    "timestamp": hits[0]["event_ts"],
                    "latest_hit_ts": hits[-1]["event_ts"],
                    "seek_sec": hits[0]["seek_sec"],
                })

        matched.sort(key=lambda v: (v["timestamp"], v["filename"]))
        return matched

    # ------------------------------------------------------------------
    # Завантаження та оновлення даних
    # ------------------------------------------------------------------
    def refresh_all(self, select_uid: str = None):
        if getattr(self, "_init_refresh_after_id", None) is not None:
            try:
                self.win.after_cancel(self._init_refresh_after_id)
            except Exception:
                pass
            self._init_refresh_after_id = None
        if self._is_closing or not self.win.winfo_exists():
            return
        db = self._get_face_db()
        if db is None:
            self._records = []
            self._events_by_person = {}
            self._unassigned_events = []
            self._summary = {}
            self.lbl_global_stats.configure(text="Модуль облич ('face_db') недоступний")
            self._render_people_list()
            return

        if select_uid is not None:
            self._selected_uid = select_uid

        self._load_seq += 1
        seq = self._load_seq
        cams_info = self._get_cameras_info()

        def _worker():
            collected = db.collect_person_events(cams_info)
            self._ui_queue.put((self._on_collected_loaded, (seq, collected)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_collected_loaded(self, seq: int, collected: dict):
        if seq != self._load_seq or self._is_closing or not self.win.winfo_exists():
            return
        self._records = list(collected.get("records") or [])
        self._events_by_person = dict(collected.get("events_by_person") or {})
        self._unassigned_events = list(collected.get("unassigned_events") or [])
        self._summary = dict(collected.get("summary") or {})

        total_crops = sum(len(v) for v in self._events_by_person.values()) + len(self._unassigned_events)
        self.lbl_global_stats.configure(
            text=f"Унікальних людей: {len(self._records)}  |  Кропів: {total_crops}"
        )

        # Перевіряємо валідність вибраного UID
        valid_ids = {r["id"] for r in self._records}
        if self._selected_uid == "__UNASSIGNED__" and not self._unassigned_events:
            self._selected_uid = None
        elif self._selected_uid not in valid_ids and self._selected_uid != "__UNASSIGNED__":
            self._selected_uid = None

        self._render_people_list()
        if self._selected_uid is None and self._records:
            self.select_person(self._records[0]["id"])
        elif self._selected_uid is not None:
            self.select_person(self._selected_uid)
        else:
            self._matched_videos = []
            self._rebuild_timeline()

    def _cycle_people_sort(self):
        order = ["recent", "crops", "name"]
        labels = {"recent": "🕒 Нещодавні", "crops": "🔢 За кількістю", "name": "🔤 За ім'ям"}
        cur = self.people_sort_var.get()
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "recent"
        self.people_sort_var.set(nxt)
        self.btn_sort_people.configure(text=labels[nxt])
        self._render_people_list()

    def _render_people_list(self):
        for ch in list(self.people_scroll.inner.winfo_children()):
            ch.destroy()
        self._left_thumbs.clear()
        self._person_card_frames.clear()

        q = self.search_var.get().strip().upper()
        sort_mode = self.people_sort_var.get()

        recs = list(self._records)
        if q:
            recs = [r for r in recs if (q in r["name"].upper()) or (q in r["id"].upper())]

        def _sort_key(r):
            st = self._summary.get(r["id"]) or {}
            if sort_mode == "name":
                return (0, r["name"].upper(), r["id"])
            if sort_mode == "crops":
                return (0, -int(st.get("crop_count", 0)), -float(st.get("last_ts", 0.0)), r["name"].upper())
            return (0, -float(st.get("last_ts", 0.0)), -int(st.get("crop_count", 0)), r["name"].upper())

        recs.sort(key=_sort_key)

        if not recs and not self._unassigned_events:
            tk.Label(
                self.people_scroll.inner,
                text="База людей порожня\nабо нічого не знайдено.",
                fg="#8c92a4", bg="#151720", font=("Segoe UI", 9), pady=24
            ).pack()
            return

        for rec in recs:
            uid = rec["id"]
            st = self._summary.get(uid) or {}
            evs = self._events_by_person.get(uid) or []
            thumb_src = rec.get("thumb_path") or ""
            if (not thumb_src or not os.path.exists(thumb_src)) and evs:
                thumb_src = evs[-1].get("crop_path") or ""

            is_sel = (self._selected_uid == uid)
            bg_c = "#212738" if is_sel else "#1b1d29"
            bd_c = "#00e5ff" if is_sel else "#292d3e"
            card = tk.Frame(
                self.people_scroll.inner, bg=bg_c, padx=8, pady=7,
                highlightthickness=2 if is_sel else 1, highlightbackground=bd_c, cursor="hand2"
            )
            card.pack(fill=tk.X, padx=6, pady=3)
            self._person_card_frames[uid] = card

            t_lbl = tk.Label(card, bg="#101118", width=6, height=3, text="👤", fg="#70758a", font=("Segoe UI", 12))
            if thumb_src and os.path.exists(thumb_src):
                try:
                    with Image.open(thumb_src) as pim:
                        pim.load()
                        pim = pim.resize((48, 48), Image.Resampling.LANCZOS)
                        tkimg = ImageTk.PhotoImage(pim)
                        self._left_thumbs.append(tkimg)
                        t_lbl.configure(image=tkimg, text="", width=48, height=48)
                except Exception:
                    pass
            t_lbl.pack(side=tk.LEFT, padx=(0, 10))

            txt_box = tk.Frame(card, bg=bg_c)
            txt_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            ign = bool(rec.get("ignore", False))
            ign_str = " [ІГНОР]" if ign else ""
            l1 = tk.Label(
                txt_box,
                text=f"{rec['name']}{ign_str}",
                fg="#8c92a4" if ign else "#ffffff", bg=bg_c, font=("Segoe UI", 9, "bold"), anchor=tk.W
            )
            l1.pack(fill=tk.X)

            short_uid = uid if len(uid) <= 18 else f"{uid[:18]}..."
            l2 = tk.Label(
                txt_box, text=f"UUID: {short_uid}",
                fg="#00c4e0", bg=bg_c, font=("Consolas", 7), anchor=tk.W
            )
            l2.pack(fill=tk.X)

            c_cnt = int(st.get("crop_count", 0))
            last_ts = float(st.get("last_ts", 0.0))
            last_str = datetime.fromtimestamp(last_ts).strftime("%d.%m %H:%M") if last_ts > 0 else "—"
            l3 = tk.Label(
                txt_box, text=f"Кропів: {c_cnt}  •  Востаннє: {last_str}",
                fg="#9ba3b8", bg=bg_c, font=("Segoe UI", 8), anchor=tk.W
            )
            l3.pack(fill=tk.X)

            for w in (card, t_lbl, txt_box, l1, l2, l3):
                w.bind("<Button-1>", lambda e, target=uid: self.select_person(target))

        if self._unassigned_events and (not q or "НЕПРИВ" in q or "ВИДАЛЕН" in q):
            is_sel = (self._selected_uid == "__UNASSIGNED__")
            bg_c = "#2c221b" if is_sel else "#221b18"
            bd_c = "#ffcc00" if is_sel else "#4d3a27"
            ucard = tk.Frame(
                self.people_scroll.inner, bg=bg_c, padx=8, pady=8,
                highlightthickness=2 if is_sel else 1, highlightbackground=bd_c, cursor="hand2"
            )
            ucard.pack(fill=tk.X, padx=6, pady=(8, 4))
            self._person_card_frames["__UNASSIGNED__"] = ucard
            ul1 = tk.Label(
                ucard, text=f"⚠ Неприв'язані / видалені ({len(self._unassigned_events)})",
                fg="#ffcc00", bg=bg_c, font=("Segoe UI", 9, "bold"), anchor=tk.W
            )
            ul1.pack(fill=tk.X)
            ul2 = tk.Label(
                ucard, text="Старі кропи без запису у faces.txt — можна реаналізувати або видалити",
                fg="#b8a898", bg=bg_c, font=("Segoe UI", 7), anchor=tk.W
            )
            ul2.pack(fill=tk.X)
            for w in (ucard, ul1, ul2):
                w.bind("<Button-1>", lambda e: self.select_person("__UNASSIGNED__"))

    # ------------------------------------------------------------------
    # Вибір людини і формування хронологічної стрічки (кропи + скріни + відео)
    # ------------------------------------------------------------------
    def select_person(self, uid: str):
        self._selected_uid = uid
        self._stop_video_player()
        self._current_item = None
        self._render_people_list()
        self._update_person_header()

        evs = self._get_selected_events()
        self._matched_videos = []
        self._rebuild_timeline()

        # Фонове зіставлення відеофрагментів (щоб не блокувати UI при відкритті відео)
        self._video_seq += 1
        v_seq = self._video_seq
        if not evs:
            self.lbl_video_scan.configure(text="")
            return

        self.lbl_video_scan.configure(text="⏳ Пошук відео...", fg="#ffcc00")

        def _vid_worker():
            vids = self._find_matching_videos_for_person(evs)
            self._ui_queue.put((self._on_videos_matched, (v_seq, uid, vids)))

        threading.Thread(target=_vid_worker, daemon=True).start()

    def _on_videos_matched(self, v_seq: int, uid: str, vids: list):
        if v_seq != self._video_seq or uid != self._selected_uid or self._is_closing or not self.win.winfo_exists():
            return
        self._matched_videos = list(vids or [])
        self.lbl_video_scan.configure(text="")
        self._update_person_header()
        self._rebuild_timeline()

    def _get_selected_record(self):
        if self._selected_uid is None or self._selected_uid == "__UNASSIGNED__":
            return None
        for r in self._records:
            if r["id"] == self._selected_uid:
                return r
        return None

    def _get_selected_events(self) -> list:
        if self._selected_uid == "__UNASSIGNED__":
            return list(self._unassigned_events)
        if self._selected_uid:
            return list(self._events_by_person.get(self._selected_uid) or [])
        return []

    def _update_person_header(self):
        rec = self._get_selected_record()
        evs = self._get_selected_events()

        self._hdr_thumb_tk = None
        self.hdr_thumb_lbl.configure(image="", text="👤", width=8, height=4)

        if self._selected_uid == "__UNASSIGNED__":
            self.lbl_person_name.configure(text="⚠ Неприв'язані кропи", fg="#ffcc00")
            self.lbl_person_uuid.configure(text="UUID: (відсутній у faces.txt)")
            self.lbl_person_meta.configure(text=f"Кропів: {len(evs)}  •  Відеофрагментів: {len(self._matched_videos)}")
            self.btn_rename.configure(state=tk.DISABLED)
            self.btn_ignore.configure(state=tk.DISABLED)
            self.btn_del_person.configure(state=tk.DISABLED)
            self.btn_copy_uuid.configure(state=tk.DISABLED)
            return

        if rec is None:
            self.lbl_person_name.configure(text="Оберіть людину зі списку ліворуч", fg="#ffffff")
            self.lbl_person_uuid.configure(text="UUID: —")
            self.lbl_person_meta.configure(text="")
            for b in (self.btn_rename, self.btn_ignore, self.btn_del_person, self.btn_copy_uuid):
                b.configure(state=tk.DISABLED)
            return

        tp = rec.get("thumb_path") or ""
        if (not tp or not os.path.exists(tp)) and evs:
            tp = evs[-1].get("crop_path") or ""
        if tp and os.path.exists(tp):
            try:
                with Image.open(tp) as pim:
                    pim.load()
                    pim = pim.resize((68, 68), Image.Resampling.LANCZOS)
                    self._hdr_thumb_tk = ImageTk.PhotoImage(pim)
                    self.hdr_thumb_lbl.configure(image=self._hdr_thumb_tk, text="", width=68, height=68)
            except Exception:
                pass

        ign = bool(rec.get("ignore", False))
        self.lbl_person_name.configure(
            text=f"{rec['name']}{'  [ІГНОРУЄТЬСЯ]' if ign else ''}",
            fg="#ffcc00" if ign else "#ffffff"
        )
        self.lbl_person_uuid.configure(text=f"UUID: {rec['id']}")
        self.btn_copy_uuid.configure(state=tk.NORMAL)

        st = self._summary.get(rec["id"]) or {}
        cams = ", ".join(st.get("cameras") or []) or "—"
        c_cnt = int(st.get("crop_count", len(evs)))
        s_cnt = int(st.get("screen_count", 0))
        v_cnt = len(self._matched_videos)
        self.lbl_person_meta.configure(
            text=f"Ракурсів: {rec.get('gallery_count', 1)}/4  •  Кропів: {c_cnt}  •  Скрінів: {s_cnt}  •  Відео: {v_cnt}  •  Камери: {cams}"
        )

        self.btn_rename.configure(state=tk.NORMAL)
        self.btn_del_person.configure(state=tk.NORMAL)
        self.btn_ignore.configure(
            state=tk.NORMAL,
            text="✅ Зняти ігнор" if ign else "🚫 Ігнорувати",
            bg="#1f442c" if ign else "#3a2f22",
            fg="#75e08b" if ign else "#ffcc00",
        )

    def _set_kind_filter(self, kind_key: str):
        self.kind_filter_var.set(kind_key)
        self._update_filter_buttons_style()
        self._rebuild_timeline()

    def _toggle_time_order(self):
        desc = not bool(self.sort_desc_var.get())
        self.sort_desc_var.set(desc)
        self.btn_time_order.configure(text="▼ Спочатку новіші" if desc else "▲ Хронологічно ↑")
        self._rebuild_timeline()

    def _update_filter_buttons_style(self):
        cur = self.kind_filter_var.get()
        evs = self._get_selected_events()
        c_cnt = len(evs)
        s_cnt = sum(1 for e in evs if e.get("full_path"))
        v_cnt = len(self._matched_videos)
        tot_cnt = c_cnt + s_cnt + v_cnt

        labels = {
            "all": f"Усі ({tot_cnt})",
            "crop": f"👤 Кропи ({c_cnt})",
            "screen": f"🖼 Скріни ({s_cnt})",
            "video": f"🎬 Відео ({v_cnt})",
        }
        for k, btn in self._filter_buttons.items():
            active = (k == cur)
            btn.configure(
                text=labels[k],
                bg="#007acc" if active else "#252836",
                fg="#ffffff" if active else "#d5dae8",
            )

    def _rebuild_timeline(self):
        self._update_filter_buttons_style()
        prev_selected_key = self._current_item.get("item_key") if self._current_item else None

        for iid in self.tl_tree.get_children():
            self.tl_tree.delete(iid)
        self._tree_item_map.clear()
        self._timeline_items.clear()

        evs = self._get_selected_events()
        vids = list(self._matched_videos)
        fkind = self.kind_filter_var.get()
        desc = bool(self.sort_desc_var.get())

        raw_items = []
        if fkind in ("all", "crop"):
            for ev in evs:
                raw_items.append({
                    "kind": "crop",
                    "item_key": f"crop::{ev['crop_path']}",
                    "sort_ts": float(ev["timestamp"]),
                    "priority": 2,
                    "day": ev["day"],
                    "time": ev["time"],
                    "camera_name": ev["camera_name"],
                    "event": ev,
                })
        if fkind in ("all", "screen"):
            for ev in evs:
                if not ev.get("full_path"):
                    continue
                raw_items.append({
                    "kind": "screen",
                    "item_key": f"screen::{ev['full_path']}::{ev['crop_path']}",
                    "sort_ts": float(ev["timestamp"]),
                    "priority": 1,
                    "day": ev["day"],
                    "time": ev["time"],
                    "camera_name": ev["camera_name"],
                    "event": ev,
                })
        if fkind in ("all", "video"):
            for vid in vids:
                sort_t = float(vid["latest_hit_ts"] if desc else vid["timestamp"])
                raw_items.append({
                    "kind": "video",
                    "item_key": f"video::{vid['path']}",
                    "sort_ts": sort_t,
                    "priority": 0,
                    "day": vid["day"],
                    "time": vid["hits"][0]["time_str"] if vid.get("hits") else vid["time"],
                    "camera_name": vid["camera_name"],
                    "video": vid,
                })

        if desc:
            raw_items.sort(key=lambda x: (-x["sort_ts"], -x["priority"], x["item_key"]))
        else:
            raw_items.sort(key=lambda x: (x["sort_ts"], x["priority"], x["item_key"]))

        self._timeline_items = raw_items

        reselect_iid = None
        first_iid = None
        grouped = {}
        for item in raw_items:
            grouped.setdefault(item["day"], []).append(item)

        for day_str in sorted(grouped.keys(), reverse=desc):
            day_node = self.tl_tree.insert("", "end", text=f"📁 {day_str}", open=True)
            for item in grouped[day_str]:
                k = item["kind"]
                if k == "crop":
                    ev = item["event"]
                    tag_bit = f" [{ev['tag']}]" if ev.get("tag") else ""
                    txt = f"  👤 {item['time']}  {item['camera_name']}{tag_bit}"
                elif k == "screen":
                    txt = f"  🖼 {item['time']}  Повний кадр ({item['camera_name']})"
                else:
                    vid = item["video"]
                    hits_n = len(vid.get("hits") or [])
                    off_s = self._format_sec(vid.get("seek_sec", 0.0))
                    hits_bit = f" ({hits_n} появ)" if hits_n > 1 else ""
                    txt = f"  🎬 {item['time']}  Відео +{off_s}{hits_bit} [{item['camera_name']}]"

                iid = self.tl_tree.insert(day_node, "end", text=txt)
                self._tree_item_map[iid] = item
                if first_iid is None:
                    first_iid = iid
                if prev_selected_key and item["item_key"] == prev_selected_key:
                    reselect_iid = iid

        target_iid = reselect_iid or first_iid
        if target_iid is not None:
            self.tl_tree.selection_set(target_iid)
            self.tl_tree.focus(target_iid)
            self._show_timeline_item(self._tree_item_map[target_iid])
        else:
            self._clear_media_stage("Подій для відображення не знайдено")

    def _on_timeline_select(self, event=None):
        sel = self.tl_tree.selection()
        if not sel:
            return
        item = self._tree_item_map.get(sel[0])
        if item is not None:
            self._show_timeline_item(item)

    # ------------------------------------------------------------------
    # Відображення кропу, повного скріна або відеоплеєра
    # ------------------------------------------------------------------
    def _clear_media_stage(self, msg: str = "Оберіть подію зі списку"):
        self._stop_video_player()
        self._current_item = None
        self.lbl_item_header.configure(text=msg)
        self.video_panel.pack_forget()
        self.img_panel.pack_forget()
        self._crop_preview_tk = None
        self._screen_preview_tk = None
        self._update_item_action_buttons()

    def _show_timeline_item(self, item: dict):
        self._current_item = item
        kind = item["kind"]

        if kind == "video":
            vid = item["video"]
            self.img_panel.pack_forget()
            self.video_panel.pack(fill=tk.BOTH, expand=True)
            dur_s = self._format_sec(vid.get("duration_sec", 0.0))
            self.lbl_item_header.configure(
                text=f"🎬 Відео: {vid['day']} {vid['time']}  •  {vid['camera_name']}  •  {dur_s} ({vid['size_mb']:.1f} МБ)"
            )
            self._load_video_item(vid)
        else:
            self._stop_video_player()
            self.video_panel.pack_forget()
            self.img_panel.pack(fill=tk.BOTH, expand=True)
            ev = item["event"]
            sh_txt = f"  •  різкість {ev['sharpness']}" if ev.get("sharpness") else ""
            tag_txt = f" [{ev['tag']}]" if ev.get("tag") else ""
            k_lbl = "👤 Кроп обличчя" if kind == "crop" else "🖼 Повний кадр"
            self.lbl_item_header.configure(
                text=f"{k_lbl}: {ev['day']} {ev['time']}  •  {ev['camera_name']}{tag_txt}{sh_txt}"
            )
            if kind == "crop":
                self.crop_frame_box.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
                self.screen_frame_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            else:
                self.crop_frame_box.pack_forget()
                self.screen_frame_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self._rerender_active_images()

        self._update_item_action_buttons()

    def _rerender_active_images(self):
        item = self._current_item
        if not item or item.get("kind") == "video":
            return
        ev = item["event"]
        kind = item["kind"]

        if kind == "crop":
            cp = ev.get("crop_path") or ""
            self._crop_preview_tk = None
            if cp and os.path.exists(cp):
                try:
                    with Image.open(cp) as pim:
                        pim.load()
                        pim = pim.resize((176, 176), Image.Resampling.LANCZOS)
                        self._crop_preview_tk = ImageTk.PhotoImage(pim)
                        self.lbl_crop_preview.configure(image=self._crop_preview_tk, text="")
                except Exception:
                    self.lbl_crop_preview.configure(image="", text="[Помилка читання]")
            else:
                self.lbl_crop_preview.configure(image="", text="[Файл кропу відсутній]")

        fp = ev.get("full_path") or ""
        self._screen_preview_tk = None
        if fp and os.path.exists(fp):
            try:
                avail_w = max(220, self.screen_frame_box.winfo_width() - 24)
                avail_h = max(180, self.screen_frame_box.winfo_height() - 42)
                with Image.open(fp) as pim:
                    pim.load()
                    ow, oh = pim.size
                    if ow > 0 and oh > 0:
                        scale = min(avail_w / float(ow), avail_h / float(oh))
                        nw = max(1, int(round(ow * scale)))
                        nh = max(1, int(round(oh * scale)))
                        pim = pim.resize((nw, nh), Image.Resampling.LANCZOS)
                        self._screen_preview_tk = ImageTk.PhotoImage(pim)
                        self.lbl_screen_preview.configure(image=self._screen_preview_tk, text="")
            except Exception:
                self.lbl_screen_preview.configure(image="", text="[Помилка читання повного скріна]")
        else:
            self.lbl_screen_preview.configure(image="", text="[Повний скрін відсутній]")

    def _find_matching_video_for_event(self, ev: dict):
        if not ev:
            return None
        ev_ts = float(ev.get("timestamp") or 0.0)
        cam_uid = str(ev.get("camera_uuid") or "")
        for v in self._matched_videos:
            if v.get("camera_uuid") == cam_uid and (v["start_ts"] - 2.5) <= ev_ts <= (v["end_ts"] + 2.5):
                seek_sec = max(0.0, min(max(0.0, v["duration_sec"] - 0.5), ev_ts - v["start_ts"]))
                return v, seek_sec
        return None

    def _update_item_action_buttons(self):
        item = self._current_item
        if not item:
            for b in (
                self.btn_wrong_person, self.btn_del_crop,
                self.btn_jump_video, self.btn_open_folder, self.btn_open_external
            ):
                b.configure(state=tk.DISABLED)
            return

        self.btn_open_folder.configure(state=tk.NORMAL)
        self.btn_open_external.configure(state=tk.NORMAL)

        if item["kind"] == "video":
            self.btn_wrong_person.configure(state=tk.DISABLED)
            self.btn_del_crop.configure(state=tk.DISABLED)
            self.btn_jump_video.configure(state=tk.DISABLED)
        else:
            ev = item["event"]
            has_crop = bool(ev.get("crop_path") and os.path.exists(ev["crop_path"]))
            self.btn_wrong_person.configure(state=tk.NORMAL if has_crop else tk.DISABLED)
            self.btn_del_crop.configure(state=tk.NORMAL)
            has_vid = self._find_matching_video_for_event(ev) is not None
            self.btn_jump_video.configure(state=tk.NORMAL if has_vid else tk.DISABLED)

    # ------------------------------------------------------------------
    # Відеоплеєр для знайдених сегментів
    # ------------------------------------------------------------------
    @staticmethod
    def _format_sec(sec: float) -> str:
        s = max(0, int(round(sec)))
        if s >= 3600:
            h, rem = divmod(s, 3600)
            m, s2 = divmod(rem, 60)
            return f"{h:d}:{m:02d}:{s2:02d}"
        m, s2 = divmod(s, 60)
        return f"{m:02d}:{s2:02d}"

    def _stop_video_player(self):
        self._vid_playing = False
        if self._vid_after_id is not None:
            try:
                self.win.after_cancel(self._vid_after_id)
            except Exception:
                pass
            self._vid_after_id = None
        if self._vid_cap is not None:
            try:
                self._vid_cap.release()
            except Exception:
                pass
            self._vid_cap = None
        self._vid_path = None
        self._vid_total_frames = 0
        self._vid_curr_frame = 0
        self._vid_photo_tk = None
        try:
            self.btn_vid_play.configure(text="▶", bg="#28a745")
            self.lbl_vid_time.configure(text="00:00/00:00")
            self._vid_updating_slider = True
            self.vid_slider_var.set(0)
        except Exception:
            pass
        finally:
            self._vid_updating_slider = False

    def _set_video_placeholder(self, text: str, color: str = "#9ba3b8"):
        self.vid_canvas.delete("all")
        cw = max(120, self.vid_canvas.winfo_width())
        ch = max(120, self.vid_canvas.winfo_height())
        self.vid_canvas.create_text(
            cw // 2, ch // 2, text=text, fill=color, font=("Segoe UI", 10, "bold"), justify=tk.CENTER, tags="ph"
        )

    def _refresh_video_canvas_center(self):
        cw = max(64, self.vid_canvas.winfo_width())
        ch = max(64, self.vid_canvas.winfo_height())
        if self.vid_canvas.find_withtag("ph"):
            self.vid_canvas.coords("ph", cw // 2, ch // 2)
        if self.vid_canvas.find_withtag("vframe"):
            self.vid_canvas.coords("vframe", cw // 2, ch // 2)

    def _render_video_frame(self, bgr_img):
        if bgr_img is None or getattr(bgr_img, "size", 0) == 0:
            return
        try:
            cw = max(64, self.vid_canvas.winfo_width())
            ch = max(64, self.vid_canvas.winfo_height())
            oh, ow = bgr_img.shape[:2]
            if ow <= 0 or oh <= 0:
                return
            scale = min(cw / float(ow), ch / float(oh))
            dw = max(1, int(round(ow * scale)))
            dh = max(1, int(round(oh * scale)))
            if cv2 is not None:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(bgr_img, (dw, dh), interpolation=interp)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            else:
                rgb = bgr_img[:, :, ::-1]
                pil_img = Image.fromarray(rgb).resize((dw, dh), Image.Resampling.BILINEAR)
            self._vid_photo_tk = ImageTk.PhotoImage(pil_img)
            self.vid_canvas.delete("all")
            self.vid_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._vid_photo_tk, tags="vframe")
        except Exception:
            pass

    def _update_vid_time_ui(self):
        tot = max(1, self._vid_total_frames)
        cur = max(0, min(tot - 1, self._vid_curr_frame))
        fps = max(1.0, self._vid_fps)
        self.lbl_vid_time.configure(text=f"{self._format_sec(cur / fps)}/{self._format_sec(tot / fps)}")
        self._vid_updating_slider = True
        try:
            self.vid_slider.configure(to=max(1, tot - 1))
            self.vid_slider_var.set(cur)
        finally:
            self._vid_updating_slider = False

    def _load_video_item(self, vid: dict, override_seek_sec: float = None):
        self._stop_video_player()
        for ch in list(self.vid_hits_bar.winfo_children()):
            ch.destroy()

        tk.Label(
            self.vid_hits_bar, text="Моменти появи у відео:", fg="#9ba3b8", bg="#181a24", font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=(0, 6))
        for hit in (vid.get("hits") or []):
            s_sec = float(hit["seek_sec"])
            tk.Button(
                self.vid_hits_bar,
                text=f"▶ {hit['time_str']} (+{self._format_sec(s_sec)})",
                command=lambda sec=s_sec: self._seek_video_sec(sec),
                bg="#283548", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                padx=8, pady=2, cursor="hand2"
            ).pack(side=tk.LEFT, padx=2)

        fpath = str(vid.get("path") or "")
        if not fpath or not os.path.exists(fpath):
            self._set_video_placeholder("[Відеофайл відсутній]", color="#ff6666")
            return
        if vid.get("busy"):
            self._set_video_placeholder("[Відеофайл ще активно записується...]", color="#ffcc00")
            return
        if cv2 is None:
            self._set_video_placeholder(
                "[Для вбудованого плеєра потрібна бібліотека OpenCV (cv2)]\nСкористайтесь кнопкою «↗ Відкрити файл».",
                color="#ffcc00",
            )
            return

        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            self._set_video_placeholder("[Не вдалося відкрити відеофайл]", color="#ff6666")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        if fps <= 0 or fps > 120:
            fps = 15.0
        total_f = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))

        self._vid_cap = cap
        self._vid_path = fpath
        self._vid_fps = fps
        self._vid_total_frames = total_f

        seek_sec = float(override_seek_sec if override_seek_sec is not None else vid.get("seek_sec", 0.0))
        target_frame = max(0, min(total_f - 1, int(round(seek_sec * fps))))
        self._seek_video_frame(target_frame)

    def _seek_video_sec(self, sec: float):
        if self._vid_cap is None:
            return
        fidx = max(0, min(max(0, self._vid_total_frames - 1), int(round(float(sec) * max(1.0, self._vid_fps)))))
        self._seek_video_frame(fidx)

    def _seek_video_frame(self, frame_idx: int):
        if self._vid_cap is None or cv2 is None:
            return
        target = max(0, min(max(0, self._vid_total_frames - 1), int(frame_idx)))
        self._vid_cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self._vid_cap.read()
        if ret and frame is not None:
            self._vid_curr_frame = target
            self._render_video_frame(frame)
            self._update_vid_time_ui()

    def _on_vid_slider_move(self, val_str):
        if self._vid_updating_slider or self._vid_cap is None:
            return
        try:
            idx = int(float(val_str))
        except Exception:
            return
        if idx != self._vid_curr_frame:
            self._seek_video_frame(idx)

    def _toggle_vid_play(self):
        if self._vid_cap is None:
            return
        if self._vid_playing:
            self._vid_playing = False
            if self._vid_after_id is not None:
                try:
                    self.win.after_cancel(self._vid_after_id)
                except Exception:
                    pass
                self._vid_after_id = None
            self.btn_vid_play.configure(text="▶", bg="#28a745")
        else:
            if self._vid_curr_frame >= max(1, self._vid_total_frames - 1):
                self._seek_video_frame(0)
            self._vid_playing = True
            self.btn_vid_play.configure(text="⏸", bg="#d97706")
            self._schedule_vid_tick(16)

    def _cycle_vid_speed(self):
        speeds = [1, 2, 4]
        self._vid_speed = speeds[(speeds.index(self._vid_speed) + 1) % len(speeds)] if self._vid_speed in speeds else 1
        self.btn_vid_speed.configure(text=f"{self._vid_speed}x")

    def _schedule_vid_tick(self, delay_ms=None):
        if self._vid_after_id is not None:
            try:
                self.win.after_cancel(self._vid_after_id)
            except Exception:
                pass
            self._vid_after_id = None
        if not self._vid_playing or self._is_closing:
            return
        if delay_ms is None:
            rate = self._vid_fps * (2 if self._vid_speed >= 2 else 1)
            delay_ms = max(16, int(round(1000.0 / max(1.0, rate))))
        self._vid_after_id = self.win.after(delay_ms, self._vid_player_tick)

    def _vid_player_tick(self):
        self._vid_after_id = None
        if not self._vid_playing or self._vid_cap is None or self._is_closing:
            return
        t0 = time.perf_counter()
        step = 2 if self._vid_speed >= 4 else 1
        for _ in range(step - 1):
            if not self._vid_cap.grab():
                break
            self._vid_curr_frame += 1
        ret, frame = self._vid_cap.read()
        if not ret or frame is None:
            self._vid_playing = False
            self.btn_vid_play.configure(text="▶", bg="#28a745")
            return
        self._vid_curr_frame += 1
        self._render_video_frame(frame)
        self._update_vid_time_ui()

        rate = self._vid_fps * (2 if self._vid_speed >= 2 else 1)
        target_ms = 1000.0 / max(1.0, rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._schedule_vid_tick(max(12, int(round(target_ms - elapsed_ms))))

    # ------------------------------------------------------------------
    # Дії користувача: копіювання UUID, ім'я, ігнор, видалення, реаналіз
    # ------------------------------------------------------------------
    def _copy_selected_uuid(self):
        rec = self._get_selected_record()
        if not rec:
            return
        try:
            self.win.clipboard_clear()
            self.win.clipboard_append(rec["id"])
        except Exception:
            pass

    def _rename_selected_person(self):
        rec = self._get_selected_record()
        db = self._get_face_db()
        if not rec or db is None:
            return
        new_n = _ask_text_modal(
            self.win,
            "Дати ім'я людині",
            f"Введіть нове ім'я для людини\nUUID: {rec['id']}",
            rec["name"],
        )
        if new_n and db.rename_record(rec["id"], new_n, cameras_info=self._get_cameras_info()):
            self.refresh_all(select_uid=rec["id"])

    def _toggle_ignore_selected(self):
        rec = self._get_selected_record()
        db = self._get_face_db()
        if not rec or db is None:
            return
        if db.set_ignore(rec["id"], not bool(rec.get("ignore", False))):
            self.refresh_all(select_uid=rec["id"])

    def _delete_selected_person(self):
        rec = self._get_selected_record()
        db = self._get_face_db()
        if not rec or db is None:
            return
        evs = self._events_by_person.get(rec["id"]) or []
        ans = messagebox.askyesnocancel(
            "Видалення людини",
            f"Видалити '{rec['name']}' (UUID: {rec['id']})?\n\n"
            f"• Так — видалити профіль І всі його кропи з диска ({len(evs)} шт.).\n"
            f"• Ні — видалити лише еталон з бази faces.txt (кропи лишаться в 'Неприв’язаних').\n"
            f"• Скасувати — не видаляти.",
            parent=self.win,
        )
        if ans is None:
            return
        if ans:
            db.delete_person_completely(rec["id"], evs, delete_files=True)
        else:
            db.delete_record(rec["id"])
        self._selected_uid = None
        self.refresh_all()

    def _on_wrong_person_clicked(self):
        item = self._current_item
        db = self._get_face_db()
        if not item or item.get("kind") not in ("crop", "screen") or db is None:
            return
        ev = item["event"]
        crop_path = str(ev.get("crop_path") or "")
        if not crop_path or not os.path.exists(crop_path):
            messagebox.showwarning("Кроп відсутній", "Файл кропу обличчя не знайдено на диску.", parent=self.win)
            return

        old_uid = str(ev.get("person_uuid") or "")
        person_crops_map = {
            uid: [e["crop_path"] for e in evs if e.get("crop_path")]
            for uid, evs in self._events_by_person.items()
        }
        norm_cp = os.path.normpath(crop_path)
        remaining_old = [
            p for p in (person_crops_map.get(old_uid) or [])
            if os.path.normpath(p) != norm_cp
        ]

        dlg = ReanalyzeCropDialog(
            self.win,
            db_accessor=db,
            event_info=ev,
            remaining_old_crops=remaining_old,
            person_crops_map=person_crops_map,
            default_min_sim=self._get_default_min_similarity(),
        )
        if dlg.result and dlg.result.get("ok"):
            r = dlg.result
            if r.get("deleted"):
                self.refresh_all(select_uid=old_uid or self._selected_uid)
                return
            target_uid = r.get("new_uid")
            msg = (
                f"Створено нову людину: {r['new_name']}\nUUID: {target_uid}"
                if r.get("created_new")
                else f"Кроп перенесено до: {r['new_name']} (схожість {r['similarity']*100:.1f}%)\nUUID: {target_uid}"
            )
            messagebox.showinfo("Реаналіз виконано", msg, parent=self.win)
            # Якщо у попередньої людини залишилися події — можемо лишитися на ній або перейти на нового власника
            keep_uid = old_uid if (old_uid and r.get("old_remaining", 0) > 0) else target_uid
            self.refresh_all(select_uid=keep_uid)

    def _on_delete_crop_clicked(self):
        item = self._current_item
        db = self._get_face_db()
        if not item or item.get("kind") not in ("crop", "screen") or db is None:
            return
        ev = item["event"]
        if not messagebox.askyesno(
            "Видалення кропу",
            f"Видалити цей кроп ({ev['day']} {ev['time']}, {ev['camera_name']})?\n"
            f"Еталони людини будуть перераховані по решті кропів.",
            parent=self.win,
        ):
            return

        old_uid = str(ev.get("person_uuid") or "")
        norm_cp = os.path.normpath(ev.get("crop_path") or "")
        remaining_old = [
            e["crop_path"]
            for e in (self._events_by_person.get(old_uid) or [])
            if e.get("crop_path") and os.path.normpath(e["crop_path"]) != norm_cp
        ]
        res = db.delete_crop_event(ev, remaining_old)
        if not res or not res.get("ok"):
            messagebox.showerror(
                "Помилка видалення",
                (res or {}).get("error") or "Не вдалося видалити кроп.",
                parent=self.win,
            )
            return
        self.refresh_all(select_uid=self._selected_uid)

    def _on_reanalyze_all_profiles(self):
        db = self._get_face_db()
        if db is None:
            return
        self.btn_reanalyze_all.configure(state=tk.DISABLED, text="⏳ Реаналіз бази...")
        cams_info = self._get_cameras_info()

        def _worker():
            res = db.reanalyze_all_profiles_from_disk(cams_info)
            self._ui_queue.put((self._on_reanalyze_all_done, (res,)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_reanalyze_all_done(self, res: dict):
        if self._is_closing or not self.win.winfo_exists():
            return
        self.btn_reanalyze_all.configure(state=tk.NORMAL, text="♻ Реаналіз бази з кропів")
        if not res or not res.get("ok"):
            messagebox.showwarning(
                "Реаналіз бази",
                (res or {}).get("error") or "Не вдалося виконати реаналіз.",
                parent=self.win,
            )
            return
        messagebox.showinfo(
            "Реаналіз бази завершено",
            f"Оновлено профілів за реальними кропами на диску: {res.get('updated_profiles', 0)}\n"
            f"Пропущено (без кропів на диску): {res.get('skipped_empty', 0)}",
            parent=self.win,
        )
        self.refresh_all(select_uid=self._selected_uid)

    def _on_jump_to_matching_video(self):
        item = self._current_item
        if not item or item.get("kind") == "video":
            return
        match = self._find_matching_video_for_event(item["event"])
        if not match:
            messagebox.showinfo("Відео не знайдено", "Для цієї мітки часу немає збереженого відеофрагмента.", parent=self.win)
            return
        vid, seek_sec = match
        target_key = f"video::{vid['path']}"
        if self.kind_filter_var.get() not in ("all", "video"):
            self.kind_filter_var.set("all")
            self._update_filter_buttons_style()
            self._rebuild_timeline()

        for iid, t_item in self._tree_item_map.items():
            if t_item.get("item_key") == target_key:
                self.tl_tree.selection_set(iid)
                self.tl_tree.focus(iid)
                self.tl_tree.see(iid)
                self._show_timeline_item(t_item)
                self._seek_video_sec(seek_sec)
                return

    def _get_current_file_path(self) -> str:
        item = self._current_item
        if not item:
            return ""
        if item["kind"] == "video":
            return str((item.get("video") or {}).get("path") or "")
        ev = item.get("event") or {}
        if item["kind"] == "screen":
            return str(ev.get("full_path") or ev.get("crop_path") or "")
        return str(ev.get("crop_path") or "")

    def _open_item_external(self):
        fpath = self._get_current_file_path()
        if not fpath or not os.path.exists(fpath):
            return
        try:
            if sys.platform == "win32":
                os.startfile(fpath)
            else:
                subprocess.Popen(["xdg-open", fpath])
        except Exception:
            pass

    def _open_item_folder(self):
        fpath = self._get_current_file_path()
        if not fpath:
            return
        folder = os.path.dirname(fpath)
        if not os.path.exists(folder):
            return
        try:
            if sys.platform == "win32" and os.path.exists(fpath):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(fpath)])
            elif sys.platform == "win32":
                os.startfile(folder)
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Події від головної системи й життєвий цикл
    # ------------------------------------------------------------------
    def notify_saved(self, mod_name: str):
        if self._is_closing or str(mod_name) != "faces":
            return
        if self._notify_after_id is not None:
            try:
                self.win.after_cancel(self._notify_after_id)
            except Exception:
                pass
        self._notify_after_id = self.win.after(700, lambda: self.refresh_all(select_uid=self._selected_uid))

    def on_modules_reloaded(self):
        if not self._is_closing:
            self.refresh_all(select_uid=self._selected_uid)

    def focus(self):
        try:
            if self.win.state() == "iconic":
                self.win.state("normal")
            self.win.deiconify()
            self.win.lift()
            self.win.focus_force()
        except Exception:
            pass

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        for attr in ("_init_refresh_after_id", "_notify_after_id", "_ui_poll_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.win.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._stop_video_player()
        try:
            self.geom_tracker.close()
        except Exception:
            pass
        try:
            self.win.destroy()
        except Exception:
            pass
        if self.on_close_cb:
            try:
                self.on_close_cb()
            except Exception:
                pass
