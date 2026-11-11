import os
import re
import sys
import time
import queue
import subprocess
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from scripts.core.i18n import tr
from scripts.ui.helpers import (
    WindowGeometryTracker,
    ScrollableFrame,
    ensure_clam_theme,
    update_scale_range_and_value,
    select_best_video_segment_for_event,
    open_video_capture_for_playback,
    seek_video_capture,
)

try:
    import cv2
except ImportError:
    cv2 = None

_CYR_TO_LAT_UI = str.maketrans("АВЕІКМНОРСТХУЁЫЭЯИЙ", "ABEIKMHOPCTXYEYERII")
_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


def _normalize_plate_input(text: str) -> str:
    raw = str(text or "").strip().upper().translate(_CYR_TO_LAT_UI)
    return re.sub(r"[^A-Z0-9]", "", raw)


def _render_contained_pil(pim: Image.Image, box_w: int, box_h: int, bg_rgb=(16, 17, 24)) -> Image.Image:
    bw = max(8, int(box_w))
    bh = max(8, int(box_h))
    src = pim.convert("RGB")
    ow, oh = src.size
    if ow <= 0 or oh <= 0:
        return Image.new("RGB", (bw, bh), bg_rgb)
    scale = min(bw / float(ow), bh / float(oh))
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))
    resized = src.resize((nw, nh), _LANCZOS)
    canvas = Image.new("RGB", (bw, bh), bg_rgb)
    canvas.paste(resized, ((bw - nw) // 2, (bh - nh) // 2))
    return canvas


def _compute_image_sharpness(path: str):
    if not path or not os.path.exists(path) or cv2 is None:
        return None
    try:
        import numpy as np
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return None
        return round(float(cv2.Laplacian(img, cv2.CV_64F).var()), 1)
    except Exception:
        return None


def _status_visual(status: str):
    st = str(status or "normal").strip().lower()
    if st == "whitelist":
        return {
            "key": "whitelist",
            "short": tr("ui.plates.status.wl_short", " [WL]"),
            "badge": tr("ui.plates.status.wl_badge", "★ WHITELIST"),
            "fg": "#75e08b",
            "badge_bg": "#1a3826",
        }
    if st == "blacklist":
        return {
            "key": "blacklist",
            "short": tr("ui.plates.status.bl_short", " [BL]"),
            "badge": tr("ui.plates.status.bl_badge", "🚨 BLACKLIST"),
            "fg": "#ff6b6b",
            "badge_bg": "#441f24",
        }
    if st == "ignore":
        return {
            "key": "ignore",
            "short": tr("ui.plates.status.ign_short", " [IGN]"),
            "badge": tr("ui.plates.status.ign_badge", "🚫 IGNORED"),
            "fg": "#9ba3b8",
            "badge_bg": "#2c2f3e",
        }
    return {
        "key": "normal",
        "short": "",
        "badge": tr("ui.plates.status.normal_badge", "● NORMAL"),
        "fg": "#00e5ff",
        "badge_bg": "#1e2736",
    }


class EditPlateMetaDialog:
    """
    Діалог редагування або додавання номера в базі:
    номер (при додаванні), підпис/власник, статус (Звичайний / Білий / Чорний / Ігнор) та примітка.
    """

    def __init__(
        self,
        parent,
        plate_text: str = "",
        initial_label: str = "",
        initial_status: str = "normal",
        initial_notes: str = "",
        allow_plate_edit: bool = False,
    ):
        self.parent = parent
        self.result = None
        self.allow_plate_edit = bool(allow_plate_edit)

        self.dlg = tk.Toplevel(parent)
        self.dlg.title(
            tr("ui.plates.meta_dlg.add_title", "➕ Add License Plate")
            if self.allow_plate_edit
            else tr("ui.plates.meta_dlg.edit_title", "✎ Plate Note & Status — {plate}", plate=plate_text)
        )
        self.dlg.configure(bg="#181922")
        self.dlg.resizable(False, False)
        self.dlg.transient(parent)
        self.dlg.grab_set()

        self.plate_var = tk.StringVar(value=str(plate_text or "").upper())
        self.label_var = tk.StringVar(value=str(initial_label or ""))
        st = str(initial_status or "normal").strip().lower()
        if st not in ("normal", "whitelist", "blacklist", "ignore"):
            st = "normal"
        self.status_var = tk.StringVar(value=st)
        self.notes_var = tk.StringVar(value=str(initial_notes or ""))

        self._build_ui()
        self._center_over_parent()
        self.dlg.wait_window()

    def _center_over_parent(self):
        try:
            self.dlg.update_idletasks()
            px = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - self.dlg.winfo_width()) // 2)
            py = self.parent.winfo_rooty() + max(0, (self.parent.winfo_height() - self.dlg.winfo_height()) // 2)
            self.dlg.geometry(f"+{px}+{py}")
        except Exception:
            pass

    def _build_ui(self):
        body = tk.Frame(self.dlg, bg="#181922", padx=18, pady=16)
        body.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            body,
            text=tr("ui.plates.meta_dlg.plate_lbl", "License plate number (letters/digits):"),
            fg="#e0e4f0", bg="#181922", font=("Segoe UI", 9, "bold"), anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 4))

        plate_entry = tk.Entry(
            body,
            textvariable=self.plate_var,
            width=38,
            bg="#262836",
            fg="#00e5ff",
            disabledbackground="#1f212d",
            disabledforeground="#00e5ff",
            insertbackground="white",
            relief=tk.FLAT,
            font=("Consolas", 12, "bold"),
        )
        if not self.allow_plate_edit:
            plate_entry.configure(state="disabled")
        plate_entry.pack(fill=tk.X, ipady=5, pady=(0, 10))

        tk.Label(
            body,
            text=tr("ui.plates.meta_dlg.label_lbl", "Owner / Vehicle label (optional):"),
            fg="#e0e4f0", bg="#181922", font=("Segoe UI", 9), anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 4))

        lbl_entry = tk.Entry(
            body, textvariable=self.label_var, width=42,
            bg="#262836", fg="#ffffff", insertbackground="white",
            relief=tk.FLAT, font=("Segoe UI", 10),
        )
        lbl_entry.pack(fill=tk.X, ipady=5, pady=(0, 10))

        tk.Label(
            body,
            text=tr("ui.plates.meta_dlg.status_lbl", "List / Reaction status:"),
            fg="#e0e4f0", bg="#181922", font=("Segoe UI", 9), anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 4))

        st_frame = tk.Frame(body, bg="#1d1f2b", padx=10, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
        st_frame.pack(fill=tk.X, pady=(0, 10))

        opts = [
            ("normal", tr("ui.plates.meta_dlg.st_normal", "● Normal (standard detection and archive)"), "#e0e4f0"),
            ("whitelist", tr("ui.plates.meta_dlg.st_whitelist", "★ Whitelist (allowed / VIP / resident)"), "#75e08b"),
            ("blacklist", tr("ui.plates.meta_dlg.st_blacklist", "🚨 Blacklist (watch list / alert)"), "#ff6b6b"),
            ("ignore", tr("ui.plates.meta_dlg.st_ignore", "🚫 Ignore (skip saving new snapshots on disk)"), "#9ba3b8"),
        ]
        for key, text, col in opts:
            tk.Radiobutton(
                st_frame,
                text=text,
                variable=self.status_var,
                value=key,
                bg="#1d1f2b",
                fg=col,
                selectcolor="#262836",
                activebackground="#1d1f2b",
                activeforeground=col,
                font=("Segoe UI", 9, "bold" if key in ("whitelist", "blacklist") else "normal"),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=2)

        tk.Label(
            body,
            text=tr("ui.plates.meta_dlg.notes_lbl", "Notes (optional):"),
            fg="#e0e4f0", bg="#181922", font=("Segoe UI", 9), anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 4))

        notes_entry = tk.Entry(
            body, textvariable=self.notes_var, width=42,
            bg="#262836", fg="#ffffff", insertbackground="white",
            relief=tk.FLAT, font=("Segoe UI", 9),
        )
        notes_entry.pack(fill=tk.X, ipady=4, pady=(0, 14))

        btns = tk.Frame(body, bg="#181922")
        btns.pack(fill=tk.X)

        def _ok(*_):
            norm = _normalize_plate_input(self.plate_var.get())
            if len(norm) < 2:
                messagebox.showwarning(
                    tr("ui.plates.meta_dlg.invalid_title", "Invalid plate"),
                    tr("ui.plates.meta_dlg.invalid_msg", "Enter at least 2 alphanumeric characters."),
                    parent=self.dlg,
                )
                return
            self.result = {
                "plate": norm,
                "label": self.label_var.get().strip(),
                "status": self.status_var.get(),
                "notes": self.notes_var.get().strip(),
            }
            self.dlg.destroy()

        def _cancel(*_):
            self.dlg.destroy()

        tk.Button(
            btns, text=tr("ui.common.cancel", "Cancel"), command=_cancel,
            bg="#323544", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=14, pady=5, cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(6, 0))
        tk.Button(
            btns, text=tr("ui.common.save", "Save"), command=_ok,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=16, pady=5, cursor="hand2",
        ).pack(side=tk.RIGHT)

        self.dlg.bind("<Return>", _ok)
        self.dlg.bind("<Escape>", _cancel)
        (plate_entry if self.allow_plate_edit else lbl_entry).focus_set()


class CorrectPlateDialog:
    """
    Діалог виправлення помилки OCR і повторного розпізнавання кропу номера:
    показує зображення пластини, запускає OCR повторно, шукає схожі відомі номери
    в базі/архіві й дозволяє виправити один знімок або об'єднати всі знімки номера.
    """

    def __init__(
        self,
        parent,
        db_accessor,
        event_info: dict,
        known_plates: list,
        total_events_for_old: int = 1,
    ):
        self.parent = parent
        self.db = db_accessor
        self.ev = dict(event_info or {})
        self.known_plates = list(known_plates or [])
        self.total_events_for_old = max(1, int(total_events_for_old or 1))
        self.result = None

        self._thumbs_cache = []
        self._scan_seq = 0
        self._is_closing = False
        self._ui_queue = queue.SimpleQueue()
        self._ui_poll_after_id = None
        self._ocr_conf_hint = None
        self._ocr_pred_plate = ""

        self.dlg = tk.Toplevel(parent)
        self.dlg.title(tr("ui.plate_fix.window_title", "✎ Fix Plate / Re-OCR"))
        self.dlg.configure(bg="#14151e")
        self.dlg.geometry("700x600")
        self.dlg.minsize(580, 500)
        self.dlg.transient(parent)
        self.dlg.grab_set()
        self.dlg.protocol("WM_DELETE_WINDOW", self._close_dlg)

        old_plate = str(self.ev.get("plate") or "")
        self.new_plate_var = tk.StringVar(value=old_plate)
        self.cand_choice_var = tk.StringVar(value="__MANUAL__")
        self.merge_all_var = tk.BooleanVar(value=False)

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
        top = tk.Frame(self.dlg, bg="#1b1d29", padx=14, pady=12, highlightthickness=1, highlightbackground="#2b2e3d")
        top.pack(fill=tk.X, padx=12, pady=(12, 8))

        crop_lbl = tk.Label(top, bg="#101118", text="🚘", fg="#70758a", font=("Segoe UI", 16), width=28, height=4)
        cp = str(self.ev.get("crop_path") or "")
        if cp and os.path.exists(cp):
            try:
                with Image.open(cp) as pim:
                    pim.load()
                    contained = _render_contained_pil(pim, 220, 74)
                    tkimg = ImageTk.PhotoImage(contained)
                    self._thumbs_cache.append(tkimg)
                    crop_lbl.configure(image=tkimg, text="", width=220, height=74)
            except Exception:
                pass
        crop_lbl.pack(side=tk.LEFT, padx=(0, 14))

        info = tk.Frame(top, bg="#1b1d29")
        info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        conf_txt = f"{self.ev['conf']}%" if self.ev.get("conf") is not None else "—"
        tk.Label(
            info,
            text=tr("ui.plate_fix.curr_plate", "Current plate: {plate}  (OCR: {conf})", plate=self.ev.get("plate", "—"), conf=conf_txt),
            fg="#ffcc00", bg="#1b1d29", font=("Segoe UI", 10, "bold"), anchor=tk.W,
        ).pack(fill=tk.X)
        tk.Label(
            info,
            text=tr(
                "ui.plate_fix.cam_time_fmt",
                "Camera: {cam}  •  Time: {day} {time}",
                cam=self.ev.get("camera_name", "—"),
                day=self.ev.get("day", ""),
                time=self.ev.get("time", ""),
            ),
            fg="#d5dae8", bg="#1b1d29", font=("Segoe UI", 9), anchor=tk.W,
        ).pack(fill=tk.X, pady=(4, 2))
        tk.Label(
            info,
            text=tr("ui.plate_fix.file_fmt", "File: {fname}", fname=self.ev.get("filename", "")),
            fg="#8c92a4", bg="#1b1d29", font=("Consolas", 8), anchor=tk.W,
        ).pack(fill=tk.X)

        ctrl = tk.Frame(self.dlg, bg="#14151e", padx=12, pady=4)
        ctrl.pack(fill=tk.X)
        tk.Button(
            ctrl, text=tr("ui.plate_fix.btn_reocr", "🔄 Re-run OCR on crop"), command=self._run_scan,
            bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2",
        ).pack(side=tk.LEFT)
        self.lbl_ocr_status = tk.Label(
            ctrl, text=tr("ui.plate_fix.status.analyzing", "Running OCR..."),
            fg="#ffcc00", bg="#14151e", font=("Segoe UI", 9, "bold"), anchor=tk.W,
        )
        self.lbl_ocr_status.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)

        # Поле ручного вводу / результату OCR
        edit_box = tk.Frame(self.dlg, bg="#191b26", padx=12, pady=10, highlightthickness=1, highlightbackground="#292d3e")
        edit_box.pack(fill=tk.X, padx=12, pady=6)

        row1 = tk.Frame(edit_box, bg="#191b26")
        row1.pack(fill=tk.X)
        tk.Label(
            row1, text=tr("ui.plate_fix.target_lbl", "Corrected plate number:"),
            fg="#e0e4f0", bg="#191b26", font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 10))
        self.plate_entry = tk.Entry(
            row1, textvariable=self.new_plate_var, width=22,
            bg="#262836", fg="#00ff88", insertbackground="white",
            relief=tk.FLAT, font=("Consolas", 13, "bold"),
        )
        self.plate_entry.pack(side=tk.LEFT, ipady=4)

        self.btn_use_ocr = tk.Button(
            row1, text=tr("ui.plate_fix.btn_use_ocr", "⬇ Use OCR suggestion"),
            command=self._use_ocr_suggestion,
            bg="#254c5a", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=4, cursor="hand2", state=tk.DISABLED,
        )
        self.btn_use_ocr.pack(side=tk.LEFT, padx=(10, 0))

        if self.total_events_for_old > 1:
            tk.Checkbutton(
                edit_box,
                text=tr(
                    "ui.plate_fix.merge_all_chk",
                    "Apply correction to all {n} snapshots of '{old}' (merge plate history)",
                    n=self.total_events_for_old,
                    old=self.ev.get("plate", ""),
                ),
                variable=self.merge_all_var,
                bg="#191b26",
                fg="#ffcc00",
                selectcolor="#262836",
                activebackground="#191b26",
                activeforeground="#ffcc00",
                font=("Segoe UI", 8, "bold"),
                anchor=tk.W,
            ).pack(fill=tk.X, pady=(8, 0))

        tk.Label(
            self.dlg,
            text=tr("ui.plate_fix.similar_hdr", "Similar known plates in archive / database (1-click select):"),
            fg="#9ba3b8", bg="#14151e", font=("Segoe UI", 9, "bold"), anchor=tk.W,
        ).pack(fill=tk.X, padx=12, pady=(6, 4))

        self.cand_scroll = ScrollableFrame(self.dlg, bg="#181923")
        self.cand_scroll.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        btns = tk.Frame(self.dlg, bg="#191b26", padx=12, pady=10, highlightthickness=1, highlightbackground="#292d3e")
        btns.pack(fill=tk.X)

        self.btn_apply = tk.Button(
            btns, text=tr("ui.plate_fix.btn_apply", "✅ Save plate fix"),
            command=self._apply,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=14, pady=6, cursor="hand2",
        )
        self.btn_apply.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btns, text=tr("ui.plate_fix.btn_del_only", "🗑 Just delete snapshot"),
            command=self._delete_only,
            bg="#64262b", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=6, cursor="hand2",
        ).pack(side=tk.LEFT)

        tk.Button(
            btns, text=tr("ui.common.cancel", "Cancel"),
            command=self._close_dlg,
            bg="#323544", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=14, pady=6, cursor="hand2",
        ).pack(side=tk.RIGHT)

        self.dlg.bind("<Escape>", lambda e: self._close_dlg())
        self.plate_entry.bind("<Return>", lambda e: self._apply())
        self.plate_entry.select_range(0, tk.END)
        self.plate_entry.focus_set()

    def _use_ocr_suggestion(self):
        if self._ocr_pred_plate:
            self.new_plate_var.set(self._ocr_pred_plate)
            self.cand_choice_var.set(f"SIM::{self._ocr_pred_plate}")

    def _run_scan(self):
        self._scan_seq += 1
        seq = self._scan_seq
        self.lbl_ocr_status.configure(text=tr("ui.plate_fix.status.running", "⏳ Running OCR & fuzzy search..."), fg="#ffcc00")
        cp = str(self.ev.get("crop_path") or "")
        cur_plate = str(self.ev.get("plate") or "")
        known = list(self.known_plates)

        def _worker():
            res = self.db.reanalyze_crop_ocr(cp, current_plate=cur_plate, known_plates=known, max_diff=2)
            self._ui_queue.put((self._on_scan_done, (seq, res)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_scan_done(self, seq: int, res: dict):
        if seq != self._scan_seq or self._is_closing or not self.dlg.winfo_exists():
            return
        for ch in list(self.cand_scroll.inner.winfo_children()):
            ch.destroy()

        if not res or not res.get("ok"):
            self.lbl_ocr_status.configure(
                text=(res or {}).get("error") or tr("ui.plate_fix.status.err", "❌ Failed to run OCR."),
                fg="#ff6666",
            )
            return

        ocr_p = str(res.get("ocr_plate") or "")
        ocr_c = int(res.get("ocr_conf") or 0)
        sharp = res.get("sharpness")
        self._ocr_pred_plate = ocr_p
        self._ocr_conf_hint = ocr_c if ocr_p else None

        if ocr_p:
            sh_bit = f"  •  {tr('ui.plate_fix.sharp_fmt', 'sharpness {v}', v=sharp)}" if sharp is not None else ""
            self.lbl_ocr_status.configure(
                text=tr("ui.plate_fix.status.ocr_found", "OCR: {plate} ({pct}%){sh}", plate=ocr_p, pct=ocr_c, sh=sh_bit),
                fg="#75e08b" if ocr_p != _normalize_plate_input(self.ev.get("plate", "")) else "#00e5ff",
            )
            self.btn_use_ocr.configure(state=tk.NORMAL)
        elif not res.get("ocr_available", True):
            self.lbl_ocr_status.configure(
                text=tr("ui.plate_fix.status.ocr_unavail", "ℹ OCR model unavailable — edit plate manually or pick a similar plate below."),
                fg="#9ba3b8",
            )
        else:
            self.lbl_ocr_status.configure(
                text=tr("ui.plate_fix.status.ocr_empty", "ℹ OCR returned no confident reading — edit manually or pick below."),
                fg="#9ba3b8",
            )

        similar = list(res.get("similar") or [])
        if not similar:
            tk.Label(
                self.cand_scroll.inner,
                text=tr("ui.plate_fix.no_similar", "No similar plates (distance ≤ 2) found in archive or database."),
                fg="#8c92a4", bg="#181923", font=("Segoe UI", 9), pady=12,
            ).pack()
            return

        for cand in similar:
            plate = cand["plate"]
            dist = cand["distance"]
            lbl = cand.get("label") or ""
            st_vis = _status_visual(cand.get("status"))
            row = tk.Frame(
                self.cand_scroll.inner,
                bg="#1e212e",
                padx=10,
                pady=7,
                highlightthickness=1,
                highlightbackground="#2b2f40",
            )
            row.pack(fill=tk.X, padx=6, pady=3)
            val = f"SIM::{plate}"

            def _pick(p=plate, v=val):
                self.cand_choice_var.set(v)
                self.new_plate_var.set(p)

            rb = tk.Radiobutton(
                row,
                variable=self.cand_choice_var,
                value=val,
                command=_pick,
                bg="#1e212e",
                activebackground="#1e212e",
                selectcolor="#262836",
                fg="#ffffff",
            )
            rb.pack(side=tk.LEFT)

            title_txt = f"{plate}{st_vis['short']}"
            if lbl:
                title_txt += f"  —  {lbl}"
            l1 = tk.Label(row, text=title_txt, fg=st_vis["fg"], bg="#1e212e", font=("Consolas", 10, "bold"), anchor=tk.W)
            l1.pack(side=tk.LEFT, fill=tk.X, expand=True)

            dist_lbl = tk.Label(
                row,
                text=tr("ui.plate_fix.diff_fmt", "diff {d}", d=dist),
                fg="#9ba3b8",
                bg="#1e212e",
                font=("Segoe UI", 8, "bold"),
            )
            dist_lbl.pack(side=tk.RIGHT)
            for w in (row, l1, dist_lbl):
                w.bind("<Button-1>", lambda e, fn=_pick: fn())

    def _apply(self):
        norm = _normalize_plate_input(self.new_plate_var.get())
        if len(norm) < 2:
            messagebox.showwarning(
                tr("ui.plates.meta_dlg.invalid_title", "Invalid plate"),
                tr("ui.plates.meta_dlg.invalid_msg", "Enter at least 2 alphanumeric characters."),
                parent=self.dlg,
            )
            return

        new_conf = None
        if self._ocr_pred_plate and norm == self._ocr_pred_plate and self._ocr_conf_hint is not None:
            new_conf = self._ocr_conf_hint

        self.result = {
            "action": "merge_all" if (self.total_events_for_old > 1 and self.merge_all_var.get()) else "fix_one",
            "new_plate": norm,
            "new_conf": new_conf,
        }
        self._close_dlg()

    def _delete_only(self):
        if not messagebox.askyesno(
            tr("ui.plate_fix.del_title", "Delete snapshot"),
            tr("ui.plate_fix.del_confirm", "Delete this plate snapshot from disk permanently?"),
            parent=self.dlg,
        ):
            return
        self.result = {"action": "delete_one"}
        self._close_dlg()


class PlatesWindow:
    """
    Незалежне глобальне вікно «Номери»:
    - агрегує всі розпізнані номерні знаки та їхні знімки з усіх камер;
    - об'єднує номери із загальною базою нотаток та списків (Whitelist / Blacklist / Ignore);
    - шукає відповідні відеосегменти across all cameras і відтворює відео з переходом до появи авто;
    - підтримує швидке виправлення OCR / Re-OCR, перейменування/злиття і збереження скролу при живому поповненні.
    """

    def __init__(self, parent, cfg_mgr, camera_manager, registry, on_close_cb=None):
        self.parent = parent
        self.cfg_mgr = cfg_mgr
        self.camera_manager = camera_manager
        self.registry = registry
        self.on_close_cb = on_close_cb

        self.win = tk.Toplevel(parent)
        self.win.title(tr("ui.plates.window_title", "🚘 Plates — License plate history, snapshots, statuses and video archive"))
        self.win.configure(bg="#12131a")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self._is_closing = False
        self._ui_queue = queue.SimpleQueue()
        self._ui_poll_after_id = None
        self._init_refresh_after_id = None
        self._notify_after_id = None
        self._pending_flush_after_id = None
        self._load_seq = 0
        self._video_seq = 0
        self._load_in_flight = False
        self._pending_live_refresh = False
        self._active_modal_count = 0
        self._vid_slider_dragging = False
        self._tl_tree_syncing = False

        self.geom_tracker = WindowGeometryTracker(
            self.win,
            get_geom_cb=lambda: self.cfg_mgr.get_window_geometry("plates"),
            set_geom_cb=lambda g: self.cfg_mgr.set_window_geometry("plates", g),
            min_w=900,
            min_h=560,
            default_geom={"left": 140, "top": 80, "width": 1180, "height": 760, "maximized": False},
        )

        # Дані
        self._records = []          # list[dict] унікальних номерів
        self._events_by_plate = {}  # plate -> list[event_dict]
        self._summary = {}          # plate -> summary stats
        self._selected_plate = None
        self._displayed_plate = None
        self._matched_videos = []
        self._timeline_items = []
        self._tree_item_map = {}
        self._current_item = None
        self._rendered_selected_events_sig = None
        self._rendered_videos_sig = None
        self._rendered_tl_sig = None
        self._rendered_hits_sig = None
        self._last_active_img_sig = None
        self._global_stats_mode = "init"
        self._video_scanning = False
        self._media_empty_key = ("ui.plates.timeline.select_event", "Select an event from the list")
        self._crop_ph_key = ("ui.plates.ph.dash", "—")
        self._screen_ph_key = ("ui.plates.ph.screen_missing_plain", "Full screenshot missing")
        self._vid_ph_spec = None
        self._vid_hits_lbl = None

        # UI кеші
        self._left_thumbs = []
        self._thumb_cache = {}
        self._sharpness_cache = {}
        self._hdr_thumb_tk = None
        self._hdr_thumb_sig = None
        self._crop_preview_tk = None
        self._screen_preview_tk = None
        self._plate_card_frames = {}
        self._plate_cards = {}
        self._rendered_plates_order = []
        self._rendered_card_keys = []
        self._empty_list_lbl = None

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
        self._vid_last_frame_bgr = None
        self._vid_last_render_size = None

        # Фільтри
        self.search_var = tk.StringVar()
        self.status_filter_var = tk.StringVar(value="all")
        self.plates_sort_var = tk.StringVar(value="recent")
        self.kind_filter_var = tk.StringVar(value="all")
        self.sort_desc_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.search_var.trace_add("write", lambda *_: self._render_plates_list(keep_existing_order=False))
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
        top_bar = tk.Frame(self.win, bg="#191b26", padx=12, pady=8, highlightthickness=1, highlightbackground="#2b2e3d")
        top_bar.pack(fill=tk.X)

        self.lbl_top_title = tk.Label(
            top_bar, text=tr("ui.plates.top_title", "🚘 License Plates & History"),
            fg="#ffffff", bg="#191b26", font=("Segoe UI", 11, "bold"),
        )
        self.lbl_top_title.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(top_bar, text="🔍", fg="#9ba3b8", bg="#191b26", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        tk.Entry(
            top_bar, textvariable=self.search_var, width=24,
            bg="#252836", fg="#ffffff", insertbackground="white", relief=tk.FLAT, font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, ipady=4, padx=(4, 8))

        self.btn_status_filter = tk.Button(
            top_bar, text=tr("ui.plates.st_filter.all", "Filter: All"), command=self._cycle_status_filter,
            bg="#2b2f40", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=9, pady=4, cursor="hand2",
        )
        self.btn_status_filter.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_global_stats = tk.Label(
            top_bar, text=tr("ui.plates.stats_init", "Plates: {n}", n=0),
            fg="#00e5ff", bg="#222534", font=("Segoe UI", 8, "bold"), padx=10, pady=4,
        )
        self.lbl_global_stats.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_close_win = tk.Button(
            top_bar, text=tr("ui.plates.btn_close", "✖ Close"), command=self.close,
            bg="#44252a", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=4, cursor="hand2",
        )
        self.btn_close_win.pack(side=tk.RIGHT)

        self.btn_refresh = tk.Button(
            top_bar, text=tr("ui.plates.btn_refresh", "↻ Refresh"), command=self.refresh_all,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=12, pady=4, cursor="hand2",
        )
        self.btn_refresh.pack(side=tk.RIGHT, padx=(0, 6))

        self.btn_add_plate = tk.Button(
            top_bar, text=tr("ui.plates.btn_add_plate", "➕ Add plate"), command=self._add_new_plate_clicked,
            bg="#234733", fg="#75e08b", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=4, cursor="hand2",
        )
        self.btn_add_plate.pack(side=tk.RIGHT, padx=(0, 6))

        self.lbl_operator_banner = tk.Label(
            self.win,
            text=tr(
                "ui.plates.operator_banner",
                "👤 Operator mode: read-only view enabled. Adding, editing notes/statuses, OCR corrections, and disk deletion are blocked.",
            ),
            fg="#ffcc00",
            bg="#2c2416",
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=6,
            anchor=tk.W,
            justify=tk.LEFT,
            highlightthickness=1,
            highlightbackground="#66521e",
        )

        self.main_paned = tk.PanedWindow(
            self.win, orient=tk.HORIZONTAL, bg="#12131a", sashwidth=5, bd=0, relief=tk.FLAT
        )
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Ліва панель: список унікальних номерів ---
        left_pane = tk.Frame(self.main_paned, bg="#171923", highlightthickness=1, highlightbackground="#292d3e")
        self.main_paned.add(left_pane, minsize=270, width=340)

        left_hdr = tk.Frame(left_pane, bg="#1e202c", padx=10, pady=7)
        left_hdr.pack(fill=tk.X)
        self.lbl_left_title = tk.Label(
            left_hdr, text=tr("ui.plates.left_title", "Unique plates"),
            fg="#ffffff", bg="#1e202c", font=("Segoe UI", 9, "bold"),
        )
        self.lbl_left_title.pack(side=tk.LEFT)

        sort_box = tk.Frame(left_hdr, bg="#1e202c")
        sort_box.pack(side=tk.RIGHT)
        self.btn_sort_plates = tk.Button(
            sort_box, text=tr("ui.plates.sort.recent", "🕒 Recent"), command=self._cycle_plates_sort,
            bg="#2b2f40", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=7, pady=2, cursor="hand2",
        )
        self.btn_sort_plates.pack(side=tk.RIGHT)

        self.plates_scroll = ScrollableFrame(left_pane, bg="#151720")
        self.plates_scroll.pack(fill=tk.BOTH, expand=True)

        # --- Права панель: заголовок номера + таймлайн + перегляд ---
        right_pane = tk.Frame(self.main_paned, bg="#12131a")
        self.main_paned.add(right_pane, minsize=520)

        self.plate_hdr = tk.Frame(
            right_pane, bg="#191b26", padx=12, pady=10, highlightthickness=1, highlightbackground="#292d3e"
        )
        self.plate_hdr.pack(fill=tk.X, pady=(0, 8))

        self.hdr_thumb_lbl = tk.Label(
            self.plate_hdr, bg="#101118", width=16, height=3, text="🚘", fg="#70758a", font=("Segoe UI", 14)
        )
        self.hdr_thumb_lbl.pack(side=tk.LEFT, padx=(0, 12))

        hdr_info = tk.Frame(self.plate_hdr, bg="#191b26")
        hdr_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        title_row = tk.Frame(hdr_info, bg="#191b26")
        title_row.pack(fill=tk.X)

        self.lbl_plate_title = tk.Label(
            title_row, text=tr("ui.plates.select_plate", "Select a license plate from the list"),
            fg="#ffffff", bg="#191b26", font=("Consolas", 14, "bold"), anchor=tk.W,
        )
        self.lbl_plate_title.pack(side=tk.LEFT)

        self.lbl_status_badge = tk.Label(
            title_row, text="", fg="#00e5ff", bg="#1e2736",
            font=("Segoe UI", 8, "bold"), padx=7, pady=2,
        )

        self.lbl_plate_note = tk.Label(
            hdr_info, text="", fg="#00e5ff", bg="#191b26", font=("Segoe UI", 9, "bold"), anchor=tk.W
        )
        self.lbl_plate_note.pack(fill=tk.X, pady=(2, 1))

        self.lbl_plate_meta = tk.Label(
            hdr_info, text="", fg="#9ba3b8", bg="#191b26", font=("Segoe UI", 8), anchor=tk.W
        )
        self.lbl_plate_meta.pack(fill=tk.X)

        hdr_actions = tk.Frame(self.plate_hdr, bg="#191b26")
        hdr_actions.pack(side=tk.RIGHT)

        self.btn_edit_meta = tk.Button(
            hdr_actions, text=tr("ui.plates.btn_edit_meta", "✎ Note / Status"), command=self._edit_selected_meta,
            bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=9, pady=5, cursor="hand2",
        )
        self.btn_edit_meta.pack(side=tk.LEFT, padx=2)

        self.btn_rename_plate = tk.Button(
            hdr_actions, text=tr("ui.plates.btn_rename_plate", "🔄 Rename / Merge"), command=self._rename_selected_plate,
            bg="#2b354a", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=9, pady=5, cursor="hand2",
        )
        self.btn_rename_plate.pack(side=tk.LEFT, padx=2)

        self.btn_copy_plate = tk.Button(
            hdr_actions, text=tr("ui.plates.btn_copy", "📋 Copy"), command=self._copy_selected_plate,
            bg="#262938", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=5, cursor="hand2",
        )
        self.btn_copy_plate.pack(side=tk.LEFT, padx=2)

        self.btn_del_plate = tk.Button(
            hdr_actions, text=tr("ui.plates.btn_delete", "🗑 Delete"), command=self._delete_selected_plate,
            bg="#44252a", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=9, pady=5, cursor="hand2",
        )
        self.btn_del_plate.pack(side=tk.LEFT, padx=(2, 0))

        # Фільтри таймлайну
        flt_bar = tk.Frame(right_pane, bg="#171923", padx=10, pady=6, highlightthickness=1, highlightbackground="#292d3e")
        flt_bar.pack(fill=tk.X, pady=(0, 8))

        self._filter_buttons = {}
        for fkey, flbl in (
            ("all", tr("ui.plates.filter.all_short", "All")),
            ("crop", tr("ui.plates.filter.crop_short", "🚘 Plates")),
            ("screen", tr("ui.plates.filter.screen_short", "🖼 Full frames")),
            ("video", tr("ui.plates.filter.video_short", "🎬 Videos")),
        ):
            btn = tk.Button(
                flt_bar, text=flbl, command=lambda k=fkey: self._set_kind_filter(k),
                bg="#252836", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                padx=10, pady=4, cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
            self._filter_buttons[fkey] = btn

        self.btn_time_order = tk.Button(
            flt_bar, text=tr("ui.plates.order_newest", "▼ Newest first"), command=self._toggle_time_order,
            bg="#2b2f40", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2",
        )
        self.btn_time_order.pack(side=tk.RIGHT)

        self.lbl_video_scan = tk.Label(
            flt_bar, text="", fg="#ffcc00", bg="#171923", font=("Segoe UI", 8, "bold"), padx=8
        )
        self.lbl_video_scan.pack(side=tk.RIGHT)

        self.right_paned = tk.PanedWindow(
            right_pane, orient=tk.HORIZONTAL, bg="#12131a", sashwidth=5, bd=0, relief=tk.FLAT
        )
        self.right_paned.pack(fill=tk.BOTH, expand=True)

        # Таймлайн
        tl_box = tk.Frame(self.right_paned, bg="#171923", highlightthickness=1, highlightbackground="#292d3e")
        self.right_paned.add(tl_box, minsize=240, width=330)

        style = ensure_clam_theme(self.win)
        style.configure(
            "PlatesTL.Treeview",
            background="#151720",
            foreground="#e2e6f0",
            fieldbackground="#151720",
            rowheight=26,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.map(
            "PlatesTL.Treeview",
            background=[("selected", "#0066ad")],
            foreground=[("selected", "#ffffff")],
        )

        self.tl_tree = ttk.Treeview(tl_box, style="PlatesTL.Treeview", show="tree", selectmode="browse")
        tl_scroll = ttk.Scrollbar(tl_box, orient=tk.VERTICAL, command=self.tl_tree.yview)
        self.tl_tree.configure(yscrollcommand=tl_scroll.set)
        tl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tl_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tl_tree.bind("<<TreeviewSelect>>", self._on_timeline_select)

        # Права область перегляду
        self.view_box = tk.Frame(self.right_paned, bg="#171923", highlightthickness=1, highlightbackground="#292d3e")
        self.right_paned.add(self.view_box, minsize=340)

        self.lbl_item_header = tk.Label(
            self.view_box, text=tr("ui.plates.timeline.select_event", "Select an event from the list"),
            fg="#d5dae8", bg="#1e202c", font=("Segoe UI", 9, "bold"), padx=10, pady=7, anchor=tk.W,
        )
        self.lbl_item_header.pack(fill=tk.X)

        self.media_stage = tk.Frame(self.view_box, bg="#101118")
        self.media_stage.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 1) Перегляд кропу номера (зверху) та повного кадру (знизу)
        self.img_panel = tk.Frame(self.media_stage, bg="#101118")
        self.img_row = tk.Frame(self.img_panel, bg="#101118")
        self.img_row.pack(fill=tk.BOTH, expand=True)

        self.crop_frame_box = tk.Frame(
            self.img_row, bg="#151720", padx=8, pady=6, highlightthickness=1, highlightbackground="#282b3a"
        )
        self.lbl_crop_box_title = tk.Label(
            self.crop_frame_box, text=tr("ui.plates.crop_box_title", "License plate snapshot"),
            fg="#00e5ff", bg="#151720", font=("Segoe UI", 8, "bold"),
        )
        self.lbl_crop_box_title.pack(pady=(0, 4))
        self.lbl_crop_preview = tk.Label(
            self.crop_frame_box, bg="#0d0e14", text=tr("ui.plates.ph.dash", "—"), fg="#6e7385"
        )
        self.lbl_crop_preview.pack()

        self.screen_frame_box = tk.Frame(
            self.img_row, bg="#151720", padx=8, pady=6, highlightthickness=1, highlightbackground="#282b3a"
        )
        self.lbl_screen_box_title = tk.Label(
            self.screen_frame_box, text=tr("ui.plates.screen_box_title", "Full camera frame"),
            fg="#9ee8ff", bg="#151720", font=("Segoe UI", 8, "bold"),
        )
        self.lbl_screen_box_title.pack(pady=(0, 4))
        self.lbl_screen_preview = tk.Label(
            self.screen_frame_box, bg="#0d0e14",
            text=tr("ui.plates.ph.screen_missing_plain", "Full screenshot missing"), fg="#6e7385",
        )
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
            width=3, cursor="hand2",
        )
        self.btn_vid_play.pack(side=tk.LEFT, padx=(0, 6))

        self.vid_slider_var = tk.DoubleVar(value=0.0)
        self.vid_slider = tk.Scale(
            vid_ctrl, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.vid_slider_var,
            command=self._on_vid_slider_move, showvalue=False,
            bg="#1b1d29", fg="#ffffff", troughcolor="#2a2d3d", highlightthickness=0, sliderlength=14,
        )
        self.vid_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.vid_slider.bind("<ButtonPress-1>", self._on_vid_slider_press, add="+")
        self.vid_slider.bind("<ButtonRelease-1>", self._on_vid_slider_release, add="+")

        self.lbl_vid_time = tk.Label(
            vid_ctrl, text="00:00/00:00", fg="#d5dae8", bg="#1b1d29", font=("Consolas", 8), width=12
        )
        self.lbl_vid_time.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_vid_speed = tk.Button(
            vid_ctrl, text="1x", command=self._cycle_vid_speed,
            bg="#2b2f40", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            width=4, cursor="hand2",
        )
        self.btn_vid_speed.pack(side=tk.LEFT)

        # Нижня панель дій для обраного елемента
        self.item_actions = tk.Frame(
            self.view_box, bg="#1b1d29", padx=10, pady=8, highlightthickness=1, highlightbackground="#292d3e"
        )
        self.item_actions.pack(fill=tk.X)

        self.btn_fix_plate = tk.Button(
            self.item_actions, text=tr("ui.plates.btn_fix_plate", "✎ Fix plate / Re-OCR"),
            command=self._on_fix_plate_clicked,
            bg="#b45309", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=10, pady=5, cursor="hand2",
        )
        self.btn_fix_plate.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_del_crop = tk.Button(
            self.item_actions, text=tr("ui.plates.btn_del_crop", "🗑 Delete snapshot"),
            command=self._on_delete_crop_clicked,
            bg="#64262b", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=9, pady=5, cursor="hand2",
        )
        self.btn_del_crop.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_jump_video = tk.Button(
            self.item_actions, text=tr("ui.plates.btn_jump_video", "🎬 Video of this moment"),
            command=self._on_jump_to_matching_video,
            bg="#1f4e78", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=9, pady=5, cursor="hand2",
        )
        self.btn_jump_video.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_debug_log = tk.Button(
            self.item_actions, text=tr("ui.plates.btn_debug", "📄 Debug"),
            command=self._on_view_debug_clicked,
            bg="#2b3042", fg="#ffcc00", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=5, cursor="hand2",
        )
        self.btn_debug_log.pack(side=tk.LEFT)

        self.btn_open_folder = tk.Button(
            self.item_actions, text=tr("ui.plates.btn_open_folder", "📂 Folder"),
            command=self._open_item_folder,
            bg="#2f3346", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=5, cursor="hand2",
        )
        self.btn_open_folder.pack(side=tk.RIGHT)

        self.btn_open_external = tk.Button(
            self.item_actions, text=tr("ui.plates.btn_open_file", "↗ Open file"),
            command=self._open_item_external,
            bg="#2f3346", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
            padx=8, pady=5, cursor="hand2",
        )
        self.btn_open_external.pack(side=tk.RIGHT, padx=(0, 6))

        self._update_filter_buttons_style()
        self._update_plate_header()
        self._update_item_action_buttons()
        self._sync_operator_mode_ui()

    def _sync_global_stats_label(self):
        if not hasattr(self, "lbl_global_stats"):
            return
        if self._global_stats_mode == "no_db":
            self.lbl_global_stats.configure(
                text=tr("ui.plates.err_no_plate_db", "Plates module ('plate_db') unavailable")
            )
        elif self._global_stats_mode == "loaded":
            total_events = sum(len(v) for v in self._events_by_plate.values())
            self.lbl_global_stats.configure(
                text=tr("ui.plates.stats_fmt", "Unique plates: {plates}  |  Detections: {events}", plates=len(self._records), events=total_events)
            )
        else:
            self.lbl_global_stats.configure(text=tr("ui.plates.stats_init", "Plates: {n}", n=len(self._records)))

    def _sync_status_filter_btn_text(self):
        cur = self.status_filter_var.get()
        labels = {
            "all": tr("ui.plates.st_filter.all", "Filter: All"),
            "whitelist": tr("ui.plates.st_filter.whitelist", "Filter: ★ Whitelist"),
            "blacklist": tr("ui.plates.st_filter.blacklist", "Filter: 🚨 Blacklist"),
            "noted": tr("ui.plates.st_filter.noted", "Filter: 🏷 Annotated"),
            "ignore": tr("ui.plates.st_filter.ignore", "Filter: 🚫 Ignored"),
        }
        fg_map = {
            "all": "#d5dae8",
            "whitelist": "#75e08b",
            "blacklist": "#ff6b6b",
            "noted": "#00e5ff",
            "ignore": "#9ba3b8",
        }
        self.btn_status_filter.configure(
            text=labels.get(cur, labels["all"]),
            fg=fg_map.get(cur, "#d5dae8"),
        )

    def _sync_plates_sort_btn_text(self):
        cur = self.plates_sort_var.get()
        labels = {
            "recent": tr("ui.plates.sort.recent", "🕒 Recent"),
            "events": tr("ui.plates.sort.events", "🔢 By count"),
            "plate": tr("ui.plates.sort.plate", "🔤 By plate"),
            "conf": tr("ui.plates.sort.conf", "🎯 By OCR %"),
        }
        self.btn_sort_plates.configure(text=labels.get(cur, labels["recent"]))

    def _sync_time_order_btn_text(self):
        desc = bool(self.sort_desc_var.get())
        self.btn_time_order.configure(
            text=tr("ui.plates.order_newest", "▼ Newest first") if desc else tr("ui.plates.order_oldest", "▲ Chronological ↑")
        )

    def _format_item_header(self, item: dict) -> str:
        kind = item["kind"]
        if kind == "video":
            vid = item["video"]
            dur_s = self._format_sec(vid.get("duration_sec", 0.0))
            return tr(
                "ui.plates.item_hdr.video",
                "🎬 Video: {day} {time}  •  {cam}  •  {dur} ({mb:.1f} MB)",
                day=vid["day"],
                time=vid["time"],
                cam=vid["camera_name"],
                dur=dur_s,
                mb=vid["size_mb"],
            )
        ev = item["event"]
        conf_bit = f" ({ev['conf']}%)" if ev.get("conf") is not None else ""
        sh = self._get_event_sharpness(ev)
        sh_txt = tr("ui.plates.item_hdr.sharp_bit", "  •  sharpness {val}", val=sh) if sh is not None else ""
        k_lbl = (
            tr("ui.plates.item_hdr.crop_kind", "🚘 Plate snapshot")
            if kind == "crop" else
            tr("ui.plates.item_hdr.screen_kind", "🖼 Full frame")
        )
        return f"{k_lbl} {ev['plate']}{conf_bit}: {ev['day']} {ev['time']}  •  {ev['camera_name']}{sh_txt}"

    def apply_language(self):
        if self._is_closing:
            return
        try:
            self.win.title(tr("ui.plates.window_title", "🚘 Plates — License plate history, snapshots, statuses and video archive"))
            self.lbl_top_title.configure(text=tr("ui.plates.top_title", "🚘 License Plates & History"))
            self._sync_status_filter_btn_text()
            self._sync_global_stats_label()
            self.btn_close_win.configure(text=tr("ui.plates.btn_close", "✖ Close"))
            self.btn_refresh.configure(text=tr("ui.plates.btn_refresh", "↻ Refresh"))
            self.btn_add_plate.configure(text=tr("ui.plates.btn_add_plate", "➕ Add plate"))
            self.lbl_operator_banner.configure(
                text=tr(
                    "ui.plates.operator_banner",
                    "👤 Operator mode: read-only view enabled. Adding, editing notes/statuses, OCR corrections, and disk deletion are blocked.",
                )
            )
            self.lbl_left_title.configure(text=tr("ui.plates.left_title", "Unique plates"))
            self._sync_plates_sort_btn_text()
            self._sync_time_order_btn_text()
            self.btn_edit_meta.configure(text=tr("ui.plates.btn_edit_meta", "✎ Note / Status"))
            self.btn_rename_plate.configure(text=tr("ui.plates.btn_rename_plate", "🔄 Rename / Merge"))
            self.btn_copy_plate.configure(text=tr("ui.plates.btn_copy", "📋 Copy"))
            self.btn_del_plate.configure(text=tr("ui.plates.btn_delete", "🗑 Delete"))
            if self._video_scanning:
                self.lbl_video_scan.configure(text=tr("ui.plates.scanning_video", "⏳ Searching videos..."))
            self.lbl_crop_box_title.configure(text=tr("ui.plates.crop_box_title", "License plate snapshot"))
            self.lbl_screen_box_title.configure(text=tr("ui.plates.screen_box_title", "Full camera frame"))
            self.btn_fix_plate.configure(text=tr("ui.plates.btn_fix_plate", "✎ Fix plate / Re-OCR"))
            self.btn_del_crop.configure(text=tr("ui.plates.btn_del_crop", "🗑 Delete snapshot"))
            self.btn_jump_video.configure(text=tr("ui.plates.btn_jump_video", "🎬 Video of this moment"))
            self.btn_debug_log.configure(text=tr("ui.plates.btn_debug", "📄 Debug"))
            self.btn_open_folder.configure(text=tr("ui.plates.btn_open_folder", "📂 Folder"))
            self.btn_open_external.configure(text=tr("ui.plates.btn_open_file", "↗ Open file"))

            self._render_plates_list(keep_existing_order=True)
            self._update_plate_header()
            self._rebuild_timeline(reload_media=False, preserve_scroll=True)

            if self._current_item is not None:
                self.lbl_item_header.configure(text=self._format_item_header(self._current_item))
                if self._current_item.get("kind") == "video":
                    if self._vid_hits_lbl is not None and self._vid_hits_lbl.winfo_exists():
                        self._vid_hits_lbl.configure(text=tr("ui.plates.vid_hits_label", "Appearances in video:"))
                    if self._vid_ph_spec is not None and self.vid_canvas.find_withtag("ph"):
                        pk, pdflt, pcol = self._vid_ph_spec
                        self._set_video_placeholder(tr(pk, pdflt), color=pcol)
                else:
                    if self._crop_preview_tk is None:
                        ck, cd = self._crop_ph_key
                        self.lbl_crop_preview.configure(text=tr(ck, cd))
                    if self._screen_preview_tk is None:
                        sk, sd = self._screen_ph_key
                        self.lbl_screen_preview.configure(text=tr(sk, sd))
            elif self._media_empty_key is not None:
                mk, md = self._media_empty_key
                self.lbl_item_header.configure(text=tr(mk, md))
        except Exception:
            pass

    def _is_operator_mode(self) -> bool:
        try:
            return bool(self.cfg_mgr and self.cfg_mgr.is_operator_mode())
        except Exception:
            return False

    def _sync_operator_mode_ui(self):
        is_op = self._is_operator_mode()
        try:
            if is_op:
                if str(self.lbl_operator_banner.winfo_manager()) != "pack":
                    self.lbl_operator_banner.pack(fill=tk.X, padx=8, pady=(6, 0), before=self.main_paned)
                for b in (
                    self.btn_add_plate,
                    self.btn_edit_meta,
                    self.btn_rename_plate,
                    self.btn_del_plate,
                    self.btn_fix_plate,
                    self.btn_del_crop,
                ):
                    b.configure(state=tk.DISABLED)
            else:
                if str(self.lbl_operator_banner.winfo_manager()) == "pack":
                    self.lbl_operator_banner.pack_forget()
                self.btn_add_plate.configure(state=tk.NORMAL)
        except Exception:
            pass

    def _guard_operator_write(self, action_label: str) -> bool:
        if not self._is_operator_mode():
            return False
        self._enter_modal()
        try:
            messagebox.showinfo(
                tr("ui.op.title", "Operator Mode"),
                tr(
                    "ui.plates.op_blocked_msg",
                    "Action '{action}' is unavailable in Operator mode (disk writes and changes are disabled).\nSwitch the system to Server mode in Settings to edit the license plate database.",
                    action=action_label,
                ),
                parent=self.win,
            )
        finally:
            self._exit_modal(schedule_flush=True)
        return True

    # ------------------------------------------------------------------
    # Доступ до модулів без жорсткого імпорту
    # ------------------------------------------------------------------
    def _get_plate_db(self):
        try:
            return self.camera_manager.get_capability_accessor("plate_db")
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

    def _find_matching_videos_for_plate(self, plate_events: list) -> list:
        if not plate_events:
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
        for ev in plate_events:
            cuid = str(ev.get("camera_uuid") or "")
            ts_by_cam.setdefault(cuid, []).append(float(ev["timestamp"]))

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
                seg_fps = float(seg.get("fps") or 0.0)
                end_pad = min(0.25, 0.5 / seg_fps) if seg_fps > 0.1 else 0.05
                hits = []
                for h_ts in (seg.get("hit_timestamps") or []):
                    rel_sec = max(0.0, float(h_ts) - float(seg["start_ts"]))
                    if seg.get("duration_sec"):
                        rel_sec = min(rel_sec, max(0.0, float(seg["duration_sec"]) - end_pad))
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
                    "fps": seg_fps,
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
    # Кешування, підписи та оновлення даних без мерехтіння
    # ------------------------------------------------------------------
    @staticmethod
    def _file_sig(path: str):
        if not path:
            return None
        try:
            st = os.stat(path)
            return (os.path.normpath(path), int(st.st_mtime_ns), int(st.st_size))
        except Exception:
            return None

    def _get_cached_thumb(self, path: str, size_wh: tuple):
        sig = self._file_sig(path)
        if sig is None:
            return None, None
        key = (sig[0], sig[1], sig[2], int(size_wh[0]), int(size_wh[1]))
        cached = self._thumb_cache.get(key)
        if cached is not None:
            return cached, sig
        try:
            with Image.open(path) as pim:
                pim.load()
                contained = _render_contained_pil(pim, int(size_wh[0]), int(size_wh[1]))
                tkimg = ImageTk.PhotoImage(contained)
            self._thumb_cache[key] = tkimg
            while len(self._thumb_cache) > 600:
                try:
                    self._thumb_cache.pop(next(iter(self._thumb_cache)))
                except Exception:
                    break
            return tkimg, sig
        except Exception:
            return None, None

    def _get_event_sharpness(self, ev: dict):
        if not ev:
            return None
        if ev.get("sharpness") is not None:
            return ev.get("sharpness")
        cp = str(ev.get("crop_path") or "")
        sig = self._file_sig(cp)
        if sig is None:
            return None
        if sig in self._sharpness_cache:
            return self._sharpness_cache[sig]
        sh = _compute_image_sharpness(cp)
        self._sharpness_cache[sig] = sh
        if len(self._sharpness_cache) > 400:
            try:
                self._sharpness_cache.pop(next(iter(self._sharpness_cache)))
            except Exception:
                pass
        return sh

    @staticmethod
    def _compute_events_sig(evs: list):
        return tuple(
            (
                str(e.get("crop_path") or ""),
                str(e.get("full_path") or ""),
                str(e.get("plate") or ""),
                e.get("conf"),
                float(e.get("timestamp") or 0.0),
            )
            for e in (evs or [])
        )

    @staticmethod
    def _compute_videos_sig(vids: list):
        return tuple(
            (
                str(v.get("path") or ""),
                bool(v.get("busy")),
                float(v.get("latest_hit_ts") or 0.0),
                tuple((float(h.get("event_ts") or 0.0), float(h.get("seek_sec") or 0.0)) for h in (v.get("hits") or [])),
            )
            for v in (vids or [])
        )

    def _is_live_refresh_blocked(self) -> bool:
        if self._is_closing or self._active_modal_count > 0 or self._vid_slider_dragging:
            return True
        try:
            if self.win.winfo_exists():
                grab_w = self.win.grab_current()
                if grab_w is not None and grab_w.winfo_toplevel() is not self.win:
                    return True
        except Exception:
            pass
        return False

    def _enter_modal(self):
        self._active_modal_count += 1

    def _exit_modal(self, schedule_flush: bool = True):
        self._active_modal_count = max(0, self._active_modal_count - 1)
        if schedule_flush and self._pending_live_refresh and not self._is_closing:
            self._schedule_pending_live_refresh(delay_ms=80)

    def _schedule_pending_live_refresh(self, delay_ms: int = 120):
        if self._is_closing:
            return
        if self._pending_flush_after_id is not None:
            try:
                self.win.after_cancel(self._pending_flush_after_id)
            except Exception:
                pass
            self._pending_flush_after_id = None
        try:
            if self.win.winfo_exists():
                self._pending_flush_after_id = self.win.after(int(delay_ms), self._flush_pending_live_refresh)
        except Exception:
            pass

    def _flush_pending_live_refresh(self):
        self._pending_flush_after_id = None
        if self._is_closing or not self._pending_live_refresh:
            return
        if self._is_live_refresh_blocked() or self._load_in_flight:
            return
        self._pending_live_refresh = False
        self.refresh_all(select_plate=self._selected_plate, live_update=True, preserve_active_item=True)

    def refresh_all(self, select_plate: str = None, *, live_update: bool = False, preserve_active_item: bool = True):
        for attr in ("_init_refresh_after_id", "_pending_flush_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.win.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._is_closing or not self.win.winfo_exists():
            return
        if live_update and self._load_in_flight:
            self._pending_live_refresh = True
            return
        if not live_update:
            self._pending_live_refresh = False

        db = self._get_plate_db()
        if db is None:
            self._records = []
            self._events_by_plate = {}
            self._summary = {}
            self._selected_plate = None
            self._displayed_plate = None
            self._global_stats_mode = "no_db"
            self._sync_global_stats_label()
            self._render_plates_list(keep_existing_order=False)
            self._update_plate_header()
            self._matched_videos = []
            self._rendered_videos_sig = None
            self._rendered_selected_events_sig = None
            self._rebuild_timeline(reload_media=False, preserve_scroll=False)
            return

        if select_plate is not None:
            self._selected_plate = select_plate

        self._load_seq += 1
        seq = self._load_seq
        self._load_in_flight = True
        cams_info = self._get_cameras_info()
        is_live = bool(live_update)
        keep_item = bool(preserve_active_item)

        def _worker():
            collected = None
            try:
                collected = db.collect_plate_events(cams_info)
            except Exception:
                collected = None
            self._ui_queue.put((self._on_collected_loaded, (seq, collected, is_live, keep_item)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_collected_loaded(
        self,
        seq: int,
        collected: dict,
        is_live_update: bool = False,
        preserve_active_item: bool = True,
    ):
        if seq == self._load_seq:
            self._load_in_flight = False
        if seq != self._load_seq or self._is_closing or not self.win.winfo_exists():
            return
        if collected is None:
            return
        if is_live_update and self._is_live_refresh_blocked():
            self._pending_live_refresh = True
            return

        self._records = list(collected.get("plates") or [])
        self._events_by_plate = dict(collected.get("events_by_plate") or {})
        self._summary = dict(collected.get("summary") or {})

        self._global_stats_mode = "loaded"
        self._sync_global_stats_label()

        valid_plates = {r["plate"] for r in self._records}
        if self._selected_plate not in valid_plates:
            self._selected_plate = None
        if self._selected_plate is None and self._records:
            ordered = self._get_sorted_filtered_records(keep_existing_order=False)
            self._selected_plate = ordered[0]["plate"] if ordered else self._records[0]["plate"]

        self._render_plates_list(keep_existing_order=bool(is_live_update and preserve_active_item))

        if self._selected_plate is None:
            self._displayed_plate = None
            self._stop_video_player()
            self._current_item = None
            self._rendered_selected_events_sig = None
            self._rendered_videos_sig = None
            self._matched_videos = []
            self._update_plate_header()
            self._rebuild_timeline(reload_media=False, preserve_scroll=False)
        elif self._selected_plate != self._displayed_plate or not preserve_active_item:
            self.select_plate(self._selected_plate, force=True, render_list=False)
        else:
            self._refresh_selected_plate_inplace(is_live_update=is_live_update)

        self._sync_operator_mode_ui()
        if self._pending_live_refresh and not self._is_live_refresh_blocked():
            self._schedule_pending_live_refresh(delay_ms=150)

    def _cycle_status_filter(self):
        order = ["all", "whitelist", "blacklist", "noted", "ignore"]
        cur = self.status_filter_var.get()
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "all"
        self.status_filter_var.set(nxt)
        self._sync_status_filter_btn_text()
        self._render_plates_list(keep_existing_order=False)

    def _cycle_plates_sort(self):
        order = ["recent", "events", "plate", "conf"]
        cur = self.plates_sort_var.get()
        nxt = order[(order.index(cur) + 1) % len(order)] if cur in order else "recent"
        self.plates_sort_var.set(nxt)
        self._sync_plates_sort_btn_text()
        self._render_plates_list(keep_existing_order=False)

    def _get_sorted_filtered_records(self, keep_existing_order: bool = False) -> list:
        q_raw = self.search_var.get().strip().upper()
        q_norm = _normalize_plate_input(q_raw)
        st_flt = self.status_filter_var.get()
        sort_mode = self.plates_sort_var.get()

        recs = []
        for r in self._records:
            st = str(r.get("status") or "normal")
            if st_flt == "whitelist" and st != "whitelist":
                continue
            elif st_flt == "blacklist" and st != "blacklist":
                continue
            elif st_flt == "ignore" and st != "ignore":
                continue
            elif st_flt == "noted" and not (r.get("label") or r.get("notes") or st != "normal"):
                continue

            if q_raw:
                p_txt = r["plate"].upper()
                lbl_txt = str(r.get("label") or "").upper()
                notes_txt = str(r.get("notes") or "").upper()
                cams_txt = " ".join((self._summary.get(r["plate"]) or {}).get("cameras") or []).upper()
                matched = (
                    (q_raw in p_txt)
                    or (q_norm and q_norm in p_txt)
                    or (q_raw in lbl_txt)
                    or (q_raw in notes_txt)
                    or (q_raw in cams_txt)
                )
                if not matched:
                    continue
            recs.append(r)

        def _sort_key(r):
            s = self._summary.get(r["plate"]) or {}
            if sort_mode == "plate":
                return (0, r["plate"])
            if sort_mode == "events":
                return (0, -int(s.get("event_count", 0)), -float(s.get("last_ts", 0.0)), r["plate"])
            if sort_mode == "conf":
                bc = int(s.get("best_conf") if s.get("best_conf") is not None else -1)
                return (0, -bc, -float(s.get("last_ts", 0.0)), r["plate"])
            return (0, -float(s.get("last_ts", 0.0)), -int(s.get("event_count", 0)), r["plate"])

        if keep_existing_order and self._rendered_plates_order:
            pos_map = {p: idx for idx, p in enumerate(self._rendered_plates_order)}
            existing_recs = [r for r in recs if r["plate"] in pos_map]
            existing_recs.sort(key=lambda r: pos_map[r["plate"]])
            new_recs = [r for r in recs if r["plate"] not in pos_map]
            new_recs.sort(key=_sort_key)
            return existing_recs + new_recs

        recs.sort(key=_sort_key)
        return recs

    def _build_plate_card_specs(self, keep_existing_order: bool = False):
        recs = self._get_sorted_filtered_records(keep_existing_order=keep_existing_order)
        specs = []
        for rec in recs:
            plate = rec["plate"]
            s = self._summary.get(plate) or {}
            evs = self._events_by_plate.get(plate) or []
            thumb_src = rec.get("thumb_path") or ""
            thumb_sig = self._file_sig(thumb_src)
            if thumb_sig is None and evs:
                thumb_src = evs[-1].get("crop_path") or ""
                thumb_sig = self._file_sig(thumb_src)

            is_sel = (self._selected_plate == plate)
            st_vis = _status_visual(rec.get("status"))
            title_txt = f"{plate}{st_vis['short']}"
            fg_title = st_vis["fg"] if st_vis["key"] != "normal" else "#ffffff"

            lbl = str(rec.get("label") or "").strip()
            best_c = s.get("best_conf")
            cams_list = s.get("cameras") or []
            if lbl:
                sub_txt = lbl
                sub_fg = "#00c4e0"
            else:
                conf_part = f"OCR {best_c}%" if best_c is not None else tr("ui.plates.manual_entry", "Manual DB entry")
                cam_part = ", ".join(cams_list[:2]) + ("..." if len(cams_list) > 2 else "")
                sub_txt = f"{conf_part}  •  {cam_part}" if cam_part else conf_part
                sub_fg = "#9ba3b8"

            e_cnt = int(s.get("event_count", 0))
            last_ts = float(s.get("last_ts", 0.0))
            last_str = datetime.fromtimestamp(last_ts).strftime("%d.%m %H:%M") if last_ts > 0 else "—"
            meta_txt = tr("ui.plates.card_meta_fmt", "Snapshots: {events}  •  Last: {last}", events=e_cnt, last=last_str)

            bg_c = "#212738" if is_sel else "#1b1d29"
            if is_sel:
                bd_c = "#00e5ff"
            elif st_vis["key"] == "blacklist":
                bd_c = "#6e2e36"
            elif st_vis["key"] == "whitelist":
                bd_c = "#255738"
            else:
                bd_c = "#292d3e"
            bd_w = 2 if is_sel else 1

            specs.append({
                "key": plate,
                "thumb_src": thumb_src,
                "thumb_sig": thumb_sig,
                "title_txt": title_txt,
                "fg_title": fg_title,
                "sub_txt": sub_txt,
                "sub_fg": sub_fg,
                "meta_txt": meta_txt,
                "bg_c": bg_c,
                "bd_c": bd_c,
                "bd_w": bd_w,
                "state": (title_txt, fg_title, sub_txt, sub_fg, meta_txt, thumb_sig, is_sel, bd_c),
            })
        return specs

    def _create_plate_card(self, sp: dict) -> dict:
        key = sp["key"]
        bg_c = sp["bg_c"]
        card = tk.Frame(
            self.plates_scroll.inner, bg=bg_c, padx=8, pady=7,
            highlightthickness=sp["bd_w"], highlightbackground=sp["bd_c"], cursor="hand2",
        )
        t_lbl = tk.Label(card, bg="#101118", width=11, height=2, text="🚘", fg="#70758a", font=("Segoe UI", 11))
        tkimg, thumb_sig = self._get_cached_thumb(sp["thumb_src"], (88, 32))
        if tkimg is not None:
            t_lbl.configure(image=tkimg, text="", width=88, height=32)
        t_lbl.pack(side=tk.LEFT, padx=(0, 10))

        txt_box = tk.Frame(card, bg=bg_c)
        txt_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        l1 = tk.Label(
            txt_box, text=sp["title_txt"],
            fg=sp["fg_title"], bg=bg_c, font=("Consolas", 10, "bold"), anchor=tk.W,
        )
        l1.pack(fill=tk.X)
        l2 = tk.Label(
            txt_box, text=sp["sub_txt"],
            fg=sp["sub_fg"], bg=bg_c, font=("Segoe UI", 8), anchor=tk.W,
        )
        l2.pack(fill=tk.X)
        l3 = tk.Label(
            txt_box, text=sp["meta_txt"],
            fg="#8c92a4", bg=bg_c, font=("Segoe UI", 8), anchor=tk.W,
        )
        l3.pack(fill=tk.X)

        for w in (card, t_lbl, txt_box, l1, l2, l3):
            w.bind("<Button-1>", lambda e, target=key: self.select_plate(target))
        self._plate_card_frames[key] = card
        return {
            "frame": card,
            "t_lbl": t_lbl,
            "txt_box": txt_box,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "thumb_sig": thumb_sig,
            "tkimg": tkimg,
            "state": sp["state"],
        }

    def _update_plate_card_inplace(self, cinfo: dict, sp: dict):
        if cinfo["state"] == sp["state"]:
            return
        prev_state = cinfo["state"]
        cinfo["state"] = sp["state"]
        bg_c = sp["bg_c"]

        if prev_state[6] != sp["state"][6] or prev_state[7] != sp["state"][7]:
            cinfo["frame"].configure(bg=bg_c, highlightthickness=sp["bd_w"], highlightbackground=sp["bd_c"])
            cinfo["txt_box"].configure(bg=bg_c)
            cinfo["l1"].configure(bg=bg_c)
            cinfo["l2"].configure(bg=bg_c)
            cinfo["l3"].configure(bg=bg_c)

        if prev_state[0] != sp["state"][0] or prev_state[1] != sp["state"][1]:
            cinfo["l1"].configure(text=sp["title_txt"], fg=sp["fg_title"])
        if prev_state[2] != sp["state"][2] or prev_state[3] != sp["state"][3]:
            cinfo["l2"].configure(text=sp["sub_txt"], fg=sp["sub_fg"])
        if prev_state[4] != sp["state"][4]:
            cinfo["l3"].configure(text=sp["meta_txt"])
        if prev_state[5] != sp["state"][5]:
            tkimg, thumb_sig = self._get_cached_thumb(sp["thumb_src"], (88, 32))
            cinfo["tkimg"] = tkimg
            cinfo["thumb_sig"] = thumb_sig
            if tkimg is not None:
                cinfo["t_lbl"].configure(image=tkimg, text="", width=88, height=32)
            else:
                cinfo["t_lbl"].configure(image="", text="🚘", width=11, height=2)

    def _destroy_all_plate_cards(self):
        for ch in list(self.plates_scroll.inner.winfo_children()):
            ch.destroy()
        self._left_thumbs.clear()
        self._plate_card_frames.clear()
        self._plate_cards.clear()
        self._empty_list_lbl = None
        self._rendered_plates_order = []
        self._rendered_card_keys = []

    def _render_plates_list(self, keep_existing_order: bool = False):
        specs = self._build_plate_card_specs(keep_existing_order=keep_existing_order)
        desired_keys = [sp["key"] for sp in specs]
        old_keys = list(self._rendered_card_keys)

        if not specs:
            if (
                self._empty_list_lbl is not None
                and self._empty_list_lbl.winfo_exists()
                and not self._plate_cards
            ):
                self._empty_list_lbl.configure(
                    text=tr("ui.plates.list_empty", "License plate archive is empty\nor nothing matched your search.")
                )
                return
            self._destroy_all_plate_cards()
            self._empty_list_lbl = tk.Label(
                self.plates_scroll.inner,
                text=tr("ui.plates.list_empty", "License plate archive is empty\nor nothing matched your search."),
                fg="#8c92a4", bg="#151720", font=("Segoe UI", 9), pady=24,
            )
            self._empty_list_lbl.pack()
            return

        if self._empty_list_lbl is not None:
            try:
                self._empty_list_lbl.destroy()
            except Exception:
                pass
            self._empty_list_lbl = None

        all_cards_alive = (
            bool(old_keys)
            and all(
                k in self._plate_cards and self._plate_cards[k]["frame"].winfo_exists()
                for k in old_keys
            )
        )

        # Швидкий шлях 1: порядок і набір не змінились — оновлюємо in-place
        if desired_keys == old_keys and all_cards_alive:
            for sp in specs:
                self._update_plate_card_inplace(self._plate_cards[sp["key"]], sp)
            self._left_thumbs = [c["tkimg"] for c in self._plate_cards.values() if c.get("tkimg") is not None]
            return

        # Швидкий шлях 2: нові номери дописано в кінець у режимі живого оновлення
        is_prefix_append = (
            keep_existing_order
            and all_cards_alive
            and bool(old_keys)
            and len(desired_keys) >= len(old_keys)
            and desired_keys[:len(old_keys)] == old_keys
        )
        if is_prefix_append:
            def _append_update():
                existing_set = set(old_keys)
                for sp in specs:
                    k = sp["key"]
                    if k in existing_set:
                        self._update_plate_card_inplace(self._plate_cards[k], sp)
                    else:
                        cinfo = self._create_plate_card(sp)
                        self._plate_cards[k] = cinfo
                        cinfo["frame"].pack(fill=tk.X, padx=6, pady=3)
                self._rendered_plates_order = desired_keys
                self._rendered_card_keys = desired_keys
                self._left_thumbs = [c["tkimg"] for c in self._plate_cards.values() if c.get("tkimg") is not None]

            self.plates_scroll.preserve_scroll_during(_append_update)
            return

        # Шлях 3: повна перебудова зі збереженням скролу
        def _full_rebuild():
            self._destroy_all_plate_cards()
            for sp in specs:
                cinfo = self._create_plate_card(sp)
                self._plate_cards[sp["key"]] = cinfo
                cinfo["frame"].pack(fill=tk.X, padx=6, pady=3)
            self._rendered_plates_order = desired_keys
            self._rendered_card_keys = desired_keys
            self._left_thumbs = [c["tkimg"] for c in self._plate_cards.values() if c.get("tkimg") is not None]

        self.plates_scroll.preserve_scroll_during(_full_rebuild)

    # ------------------------------------------------------------------
    # Вибір номера та формування хронологічної стрічки
    # ------------------------------------------------------------------
    def _start_video_scan_for_selected(self, plate: str, evs: list, *, is_live_update: bool = False):
        self._video_seq += 1
        v_seq = self._video_seq
        if not evs:
            self._video_scanning = False
            self._matched_videos = []
            self._rendered_videos_sig = ()
            self.lbl_video_scan.configure(text="")
            if is_live_update:
                self._update_plate_header()
                self._rebuild_timeline(reload_media=False, preserve_scroll=True)
            return

        self._video_scanning = True
        self.lbl_video_scan.configure(text=tr("ui.plates.scanning_video", "⏳ Searching videos..."), fg="#ffcc00")

        def _vid_worker():
            vids = self._find_matching_videos_for_plate(evs)
            self._ui_queue.put((self._on_videos_matched, (v_seq, plate, vids, bool(is_live_update))))

        threading.Thread(target=_vid_worker, daemon=True).start()

    def _refresh_selected_plate_inplace(self, is_live_update: bool = True):
        plate = self._selected_plate
        evs = self._get_selected_events()
        new_evs_sig = self._compute_events_sig(evs)
        evs_changed = (new_evs_sig != self._rendered_selected_events_sig)

        self._update_plate_header()
        self._update_filter_buttons_style()

        if evs_changed or not is_live_update:
            self._rendered_selected_events_sig = new_evs_sig
            self._rebuild_timeline(reload_media=False, preserve_scroll=True)
            self._start_video_scan_for_selected(plate, evs, is_live_update=True)

    def select_plate(self, plate: str, force: bool = False, render_list: bool = True):
        if not force and plate == self._displayed_plate and self._selected_plate == plate:
            return
        self._selected_plate = plate
        self._displayed_plate = plate
        self._stop_video_player()
        self._current_item = None
        self._rendered_selected_events_sig = None
        self._rendered_videos_sig = None
        self._rendered_tl_sig = None
        self._rendered_hits_sig = None
        self._last_active_img_sig = None

        if render_list:
            self._render_plates_list(keep_existing_order=True)
        self._update_plate_header()

        evs = self._get_selected_events()
        self._rendered_selected_events_sig = self._compute_events_sig(evs)
        self._matched_videos = []
        self._rebuild_timeline(reload_media=True, preserve_scroll=False)
        self._start_video_scan_for_selected(plate, evs, is_live_update=False)

    def _on_videos_matched(self, v_seq: int, plate: str, vids: list, is_live_update: bool = False):
        if v_seq != self._video_seq or plate != self._selected_plate or self._is_closing or not self.win.winfo_exists():
            return
        self._video_scanning = False
        self.lbl_video_scan.configure(text="")
        if is_live_update and self._is_live_refresh_blocked():
            self._pending_live_refresh = True
            return

        vids_list = list(vids or [])
        new_vids_sig = self._compute_videos_sig(vids_list)
        if new_vids_sig == self._rendered_videos_sig:
            return

        self._matched_videos = vids_list
        self._rendered_videos_sig = new_vids_sig
        self._update_plate_header()
        self._rebuild_timeline(reload_media=False, preserve_scroll=True)

    def _get_selected_record(self):
        if not self._selected_plate:
            return None
        for r in self._records:
            if r["plate"] == self._selected_plate:
                return r
        return None

    def _get_selected_events(self) -> list:
        if self._selected_plate:
            return list(self._events_by_plate.get(self._selected_plate) or [])
        return []

    def _set_header_placeholder(self):
        if self._hdr_thumb_sig is None and self._hdr_thumb_tk is None:
            return
        self._hdr_thumb_tk = None
        self._hdr_thumb_sig = None
        self.hdr_thumb_lbl.configure(image="", text="🚘", width=16, height=3)

    def _update_plate_header(self):
        rec = self._get_selected_record()
        evs = self._get_selected_events()

        if rec is None:
            self._set_header_placeholder()
            self.lbl_plate_title.configure(
                text=tr("ui.plates.select_plate", "Select a license plate from the list"),
                fg="#ffffff",
            )
            self.lbl_status_badge.pack_forget()
            self.lbl_plate_note.configure(text="")
            self.lbl_plate_meta.configure(text="")
            for b in (self.btn_edit_meta, self.btn_rename_plate, self.btn_copy_plate, self.btn_del_plate):
                b.configure(state=tk.DISABLED)
            self._sync_operator_mode_ui()
            return

        tp = rec.get("thumb_path") or ""
        new_sig = self._file_sig(tp)
        if new_sig is None and evs:
            tp = evs[-1].get("crop_path") or ""
            new_sig = self._file_sig(tp)

        if new_sig != self._hdr_thumb_sig or (new_sig is not None and self._hdr_thumb_tk is None):
            if new_sig is not None:
                tkimg, resolved_sig = self._get_cached_thumb(tp, (140, 48))
                if tkimg is not None:
                    self._hdr_thumb_tk = tkimg
                    self._hdr_thumb_sig = resolved_sig
                    self.hdr_thumb_lbl.configure(image=tkimg, text="", width=140, height=48)
                else:
                    self._set_header_placeholder()
            else:
                self._set_header_placeholder()

        st_vis = _status_visual(rec.get("status"))
        self.lbl_plate_title.configure(text=rec["plate"], fg=st_vis["fg"] if st_vis["key"] != "normal" else "#ffffff")
        self.lbl_status_badge.configure(text=st_vis["badge"], fg=st_vis["fg"], bg=st_vis["badge_bg"])
        if str(self.lbl_status_badge.winfo_manager()) != "pack":
            self.lbl_status_badge.pack(side=tk.LEFT, padx=(10, 0))

        lbl = str(rec.get("label") or "").strip()
        notes = str(rec.get("notes") or "").strip()
        if lbl and notes:
            note_disp = f"{lbl}  •  {notes}"
        elif lbl or notes:
            note_disp = lbl or notes
        else:
            note_disp = tr("ui.plates.no_note_hint", "No owner / note assigned")
        self.lbl_plate_note.configure(text=note_disp, fg="#00e5ff" if (lbl or notes) else "#6e7385")

        st = self._summary.get(rec["plate"]) or {}
        cams = ", ".join(st.get("cameras") or []) or "—"
        e_cnt = int(st.get("event_count", len(evs)))
        s_cnt = int(st.get("screen_count", 0))
        v_cnt = len(self._matched_videos)
        bc = st.get("best_conf")
        ac = st.get("avg_conf")
        ocr_str = f"{bc}% ({tr('ui.plates.avg_lbl', 'avg')} {ac}%)" if bc is not None and ac is not None else "—"
        self.lbl_plate_meta.configure(
            text=tr(
                "ui.plates.plate_meta_fmt",
                "Snapshots: {events}  •  Screenshots: {screens}  •  Videos: {videos}  •  OCR: {ocr}  •  Cameras: {cams}",
                events=e_cnt,
                screens=s_cnt,
                videos=v_cnt,
                ocr=ocr_str,
                cams=cams,
            )
        )

        self.btn_edit_meta.configure(state=tk.NORMAL)
        self.btn_rename_plate.configure(state=tk.NORMAL)
        self.btn_copy_plate.configure(state=tk.NORMAL)
        self.btn_del_plate.configure(state=tk.NORMAL)
        self._sync_operator_mode_ui()

    def _set_kind_filter(self, kind_key: str):
        self.kind_filter_var.set(kind_key)
        self._update_filter_buttons_style()
        self._rebuild_timeline(reload_media=False, preserve_scroll=False)

    def _toggle_time_order(self):
        desc = not bool(self.sort_desc_var.get())
        self.sort_desc_var.set(desc)
        self._sync_time_order_btn_text()
        self._rebuild_timeline(reload_media=False, preserve_scroll=False)

    def _update_filter_buttons_style(self):
        cur = self.kind_filter_var.get()
        evs = self._get_selected_events()
        c_cnt = len(evs)
        s_cnt = sum(1 for e in evs if e.get("full_path"))
        v_cnt = len(self._matched_videos)
        tot_cnt = c_cnt + s_cnt + v_cnt

        labels = {
            "all": tr("ui.plates.filter.all", "All ({n})", n=tot_cnt),
            "crop": tr("ui.plates.filter.crop", "🚘 Plates ({n})", n=c_cnt),
            "screen": tr("ui.plates.filter.screen", "🖼 Screenshots ({n})", n=s_cnt),
            "video": tr("ui.plates.filter.video", "🎬 Videos ({n})", n=v_cnt),
        }
        for k, btn in self._filter_buttons.items():
            active = (k == cur)
            btn.configure(
                text=labels[k],
                bg="#007acc" if active else "#252836",
                fg="#ffffff" if active else "#d5dae8",
            )

    def _rebuild_timeline(self, reload_media: bool = True, preserve_scroll: bool = True):
        self._update_filter_buttons_style()
        prev_selected_key = self._current_item.get("item_key") if self._current_item else None

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

        grouped = {}
        for item in raw_items:
            k = item["kind"]
            if k == "crop":
                ev = item["event"]
                conf_bit = f" [{ev['conf']}%]" if ev.get("conf") is not None else ""
                item["_row_text"] = f"  🚘 {item['time']}  {item['camera_name']}{conf_bit}"
            elif k == "screen":
                item["_row_text"] = tr(
                    "ui.plates.tl.screen_row",
                    "  🖼 {time}  Full frame ({cam})",
                    time=item["time"],
                    cam=item["camera_name"],
                )
            else:
                vid = item["video"]
                hits_n = len(vid.get("hits") or [])
                off_s = self._format_sec(vid.get("seek_sec", 0.0))
                hits_bit = tr("ui.plates.tl.hits_bit", " ({n} appearances)", n=hits_n) if hits_n > 1 else ""
                item["_row_text"] = tr(
                    "ui.plates.tl.video_row",
                    "  🎬 {time}  Video +{offset}{hits_bit} [{cam}]",
                    time=item["time"],
                    offset=off_s,
                    hits_bit=hits_bit,
                    cam=item["camera_name"],
                )
            grouped.setdefault(item["day"], []).append(item)

        ordered_days = sorted(grouped.keys(), reverse=desc)
        new_tl_sig = (
            fkind,
            desc,
            tuple(
                (day_str, tuple((it["item_key"], it["_row_text"]) for it in grouped[day_str]))
                for day_str in ordered_days
            ),
        )

        if not reload_media and self._rendered_tl_sig == new_tl_sig and self.tl_tree.get_children():
            self._timeline_items = raw_items
            by_key = {it["item_key"]: it for it in raw_items}
            for iid, old_it in list(self._tree_item_map.items()):
                fresh = by_key.get(old_it["item_key"])
                if fresh is not None:
                    self._tree_item_map[iid] = fresh
            if prev_selected_key and prev_selected_key in by_key:
                self._current_item = by_key[prev_selected_key]
                self.lbl_item_header.configure(text=self._format_item_header(self._current_item))
                if self._current_item.get("kind") == "video":
                    self._refresh_video_hits_bar_if_needed(self._current_item["video"])
                self._update_item_action_buttons()
            return

        prev_day_nodes = self.tl_tree.get_children()
        had_days = bool(prev_day_nodes)
        open_days = set()
        known_days = set()
        for d_id in prev_day_nodes:
            d_txt = self.tl_tree.item(d_id, "text")
            if d_txt.startswith("📁 "):
                d_name = d_txt[2:].strip()
                known_days.add(d_name)
                if self.tl_tree.item(d_id, "open"):
                    open_days.add(d_name)

        prev_yview_top = 0.0
        top_anchor = None
        if preserve_scroll and had_days:
            try:
                prev_yview_top = float(self.tl_tree.yview()[0])
                if prev_yview_top > 0.001:
                    top_iid = self.tl_tree.identify_row(4)
                    if top_iid:
                        if top_iid in self._tree_item_map:
                            top_anchor = ("item", self._tree_item_map[top_iid]["item_key"])
                        else:
                            t_txt = self.tl_tree.item(top_iid, "text")
                            if t_txt.startswith("📁 "):
                                top_anchor = ("day", t_txt[2:].strip())
            except Exception:
                pass

        self._tl_tree_syncing = True
        try:
            if prev_day_nodes:
                self.tl_tree.delete(*prev_day_nodes)
            self._tree_item_map.clear()
            self._timeline_items = raw_items
            self._rendered_tl_sig = new_tl_sig

            reselect_iid = None
            reselect_day_iid = None
            first_iid = None

            for day_str in ordered_days:
                is_open = (day_str in open_days) if (had_days and day_str in known_days) else True
                day_node = self.tl_tree.insert("", "end", text=f"📁 {day_str}", open=is_open)
                for item in grouped[day_str]:
                    iid = self.tl_tree.insert(day_node, "end", text=item["_row_text"])
                    self._tree_item_map[iid] = item
                    if first_iid is None:
                        first_iid = iid
                    if prev_selected_key and item["item_key"] == prev_selected_key:
                        reselect_iid = iid
                        reselect_day_iid = day_node

            target_iid = reselect_iid or first_iid
            if target_iid is not None:
                if reselect_day_iid is not None:
                    self.tl_tree.item(reselect_day_iid, open=True)
                self.tl_tree.selection_set(target_iid)
                self.tl_tree.focus(target_iid)
                target_item = self._tree_item_map[target_iid]
                same_item_kept = bool(prev_selected_key and reselect_iid is not None)
                if reload_media or not same_item_kept:
                    self._show_timeline_item(target_item)
                else:
                    self._current_item = target_item
                    self.lbl_item_header.configure(text=self._format_item_header(target_item))
                    if target_item.get("kind") == "video":
                        vid = target_item["video"]
                        if self._vid_cap is None and not vid.get("busy"):
                            self._show_timeline_item(target_item)
                        else:
                            self._refresh_video_hits_bar_if_needed(vid)
                            self._update_item_action_buttons()
                    else:
                        self._update_item_action_buttons()

                if preserve_scroll and had_days:
                    if prev_yview_top > 0.001:
                        visible_rows = 0
                        anchor_idx = None
                        for d_node in self.tl_tree.get_children():
                            d_row = visible_rows
                            visible_rows += 1
                            dt = self.tl_tree.item(d_node, "text")
                            dn = dt[2:].strip() if dt.startswith("📁 ") else ""
                            if top_anchor == ("day", dn):
                                anchor_idx = d_row
                            if self.tl_tree.item(d_node, "open"):
                                for ch_id in self.tl_tree.get_children(d_node):
                                    c_row = visible_rows
                                    visible_rows += 1
                                    ch_it = self._tree_item_map.get(ch_id)
                                    if ch_it and top_anchor == ("item", ch_it["item_key"]):
                                        anchor_idx = c_row
                        self.tl_tree.update_idletasks()
                        if anchor_idx is not None and visible_rows > 0:
                            self.tl_tree.yview_moveto(max(0.0, min(1.0, anchor_idx / float(visible_rows))))
                        else:
                            self.tl_tree.yview_moveto(prev_yview_top)
                elif not preserve_scroll:
                    self.tl_tree.see(target_iid)
            else:
                self._clear_media_stage_by_key("ui.plates.timeline.empty", "No events to display")

            self.tl_tree.update_idletasks()
        finally:
            self._tl_tree_syncing = False

    def _on_timeline_select(self, event=None):
        if self._tl_tree_syncing:
            return
        sel = self.tl_tree.selection()
        if not sel:
            return
        item = self._tree_item_map.get(sel[0])
        if item is None:
            return
        if event is not None and self._current_item is not None and item.get("item_key") == self._current_item.get("item_key"):
            return
        self._show_timeline_item(item)

    # ------------------------------------------------------------------
    # Перегляд кропу, повного скріна або відео
    # ------------------------------------------------------------------
    def _clear_media_stage_by_key(self, key: str, default_text: str):
        self._stop_video_player()
        self._current_item = None
        self._last_active_img_sig = None
        self._rendered_hits_sig = None
        self._media_empty_key = (key, default_text)
        self.lbl_item_header.configure(text=tr(key, default_text))
        self.video_panel.pack_forget()
        self.img_panel.pack_forget()
        self._crop_preview_tk = None
        self._screen_preview_tk = None
        self._update_item_action_buttons()

    def _show_timeline_item(self, item: dict, override_seek_sec: float = None):
        self._current_item = item
        self._media_empty_key = None
        kind = item["kind"]

        if kind == "video":
            vid = item["video"]
            self._last_active_img_sig = None
            self.img_panel.pack_forget()
            self.video_panel.pack(fill=tk.BOTH, expand=True)
            self.lbl_item_header.configure(text=self._format_item_header(item))
            try:
                self.video_panel.update_idletasks()
            except Exception:
                pass
            self._load_video_item(vid, override_seek_sec=override_seek_sec)
        else:
            self._stop_video_player()
            self._last_active_img_sig = None
            self.video_panel.pack_forget()
            self.img_panel.pack(fill=tk.BOTH, expand=True)
            self.lbl_item_header.configure(text=self._format_item_header(item))
            if kind == "crop":
                self.crop_frame_box.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
                self.screen_frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            else:
                self.crop_frame_box.pack_forget()
                self.screen_frame_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self._rerender_active_images()

        self._update_item_action_buttons()

    def _rerender_active_images(self):
        item = self._current_item
        if not item or item.get("kind") == "video":
            return
        ev = item["event"]
        kind = item["kind"]
        cp = ev.get("crop_path") or ""
        fp = ev.get("full_path") or ""
        avail_w = max(220, self.screen_frame_box.winfo_width() - 24)
        avail_h = max(160, self.screen_frame_box.winfo_height() - 38)
        crop_max_w = max(240, min(460, avail_w - 16))
        img_sig = (
            item.get("item_key"),
            kind,
            self._file_sig(cp) if kind == "crop" else None,
            self._file_sig(fp),
            avail_w,
            avail_h,
        )
        if img_sig == self._last_active_img_sig:
            return
        self._last_active_img_sig = img_sig

        if kind == "crop":
            self._crop_preview_tk = None
            if cp and os.path.exists(cp):
                try:
                    with Image.open(cp) as pim:
                        pim.load()
                        contained = _render_contained_pil(pim, crop_max_w, 92)
                        self._crop_preview_tk = ImageTk.PhotoImage(contained)
                        self.lbl_crop_preview.configure(image=self._crop_preview_tk, text="")
                except Exception:
                    self._crop_ph_key = ("ui.plates.ph.read_err", "[Read error]")
                    self.lbl_crop_preview.configure(image="", text=tr(*self._crop_ph_key))
            else:
                self._crop_ph_key = ("ui.plates.ph.crop_missing", "[Snapshot missing]")
                self.lbl_crop_preview.configure(image="", text=tr(*self._crop_ph_key))

        self._screen_preview_tk = None
        if fp and os.path.exists(fp):
            try:
                with Image.open(fp) as pim:
                    pim.load()
                    ow, oh = pim.size
                    if ow > 0 and oh > 0:
                        scale = min(avail_w / float(ow), avail_h / float(oh))
                        nw = max(1, int(round(ow * scale)))
                        nh = max(1, int(round(oh * scale)))
                        pim = pim.resize((nw, nh), _LANCZOS)
                        self._screen_preview_tk = ImageTk.PhotoImage(pim)
                        self.lbl_screen_preview.configure(image=self._screen_preview_tk, text="")
            except Exception:
                self._screen_ph_key = ("ui.plates.ph.screen_read_err", "[Error reading full screenshot]")
                self.lbl_screen_preview.configure(image="", text=tr(*self._screen_ph_key))
        else:
            self._screen_ph_key = ("ui.plates.ph.screen_missing", "[Full screenshot missing]")
            self.lbl_screen_preview.configure(image="", text=tr(*self._screen_ph_key))

    def _find_matching_video_for_event(self, ev: dict):
        return select_best_video_segment_for_event(self._matched_videos, ev, margin_sec=2.5)

    def _update_item_action_buttons(self):
        item = self._current_item
        if not item:
            for b in (
                self.btn_fix_plate, self.btn_del_crop, self.btn_jump_video,
                self.btn_debug_log, self.btn_open_folder, self.btn_open_external,
            ):
                b.configure(state=tk.DISABLED)
            self._sync_operator_mode_ui()
            return

        self.btn_open_folder.configure(state=tk.NORMAL)
        self.btn_open_external.configure(state=tk.NORMAL)

        if item["kind"] == "video":
            self.btn_fix_plate.configure(state=tk.DISABLED)
            self.btn_del_crop.configure(state=tk.DISABLED)
            self.btn_jump_video.configure(state=tk.DISABLED)
            self.btn_debug_log.configure(state=tk.DISABLED)
        else:
            ev = item["event"]
            has_crop = bool(ev.get("crop_path") and os.path.exists(ev["crop_path"]))
            has_dbg = bool(ev.get("debug_path") and os.path.exists(ev["debug_path"]))
            self.btn_fix_plate.configure(state=tk.NORMAL if has_crop else tk.DISABLED)
            self.btn_del_crop.configure(state=tk.NORMAL)
            has_vid = self._find_matching_video_for_event(ev) is not None
            self.btn_jump_video.configure(state=tk.NORMAL if has_vid else tk.DISABLED)
            self.btn_debug_log.configure(state=tk.NORMAL if has_dbg else tk.DISABLED)
        self._sync_operator_mode_ui()

    # ------------------------------------------------------------------
    # Відеоплеєр
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
        self._rendered_hits_sig = None
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
        self._vid_last_frame_bgr = None
        self._vid_last_render_size = None
        try:
            self.btn_vid_play.configure(text="▶", bg="#28a745")
            self.lbl_vid_time.configure(text="00:00/00:00")
            self._vid_updating_slider = True
            update_scale_range_and_value(self.vid_slider, self.vid_slider_var, 0, 1)
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
        if self._vid_last_frame_bgr is not None:
            prev_sz = self._vid_last_render_size
            if prev_sz is None or abs(cw - prev_sz[0]) > 2 or abs(ch - prev_sz[1]) > 2:
                self._render_video_frame(self._vid_last_frame_bgr)
                return
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
                pil_img = Image.fromarray(rgb).resize((dw, dh), _BILINEAR)
            self._vid_last_frame_bgr = bgr_img
            self._vid_last_render_size = (cw, ch)
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
            update_scale_range_and_value(self.vid_slider, self.vid_slider_var, cur, max(1, tot - 1))
        finally:
            self._vid_updating_slider = False

    def _refresh_video_hits_bar_if_needed(self, vid: dict):
        hits = list((vid or {}).get("hits") or [])
        hits_sig = tuple(
            (float(h.get("event_ts") or 0.0), float(h.get("seek_sec") or 0.0), str(h.get("time_str") or ""))
            for h in hits
        )
        if (
            self._rendered_hits_sig == hits_sig
            and self._vid_hits_lbl is not None
            and self._vid_hits_lbl.winfo_exists()
        ):
            return
        self._rendered_hits_sig = hits_sig
        for ch in list(self.vid_hits_bar.winfo_children()):
            ch.destroy()

        self._vid_hits_lbl = tk.Label(
            self.vid_hits_bar,
            text=tr("ui.plates.vid_hits_label", "Appearances in video:"),
            fg="#9ba3b8",
            bg="#181a24",
            font=("Segoe UI", 8),
        )
        self._vid_hits_lbl.pack(side=tk.LEFT, padx=(0, 6))
        for hit in hits:
            s_sec = float(hit["seek_sec"])
            tk.Button(
                self.vid_hits_bar,
                text=f"▶ {hit['time_str']} (+{self._format_sec(s_sec)})",
                command=lambda sec=s_sec: self._seek_video_sec(sec),
                bg="#283548",
                fg="#00e5ff",
                relief=tk.FLAT,
                font=("Segoe UI", 8, "bold"),
                padx=8,
                pady=2,
                cursor="hand2",
            ).pack(side=tk.LEFT, padx=2)

    def _load_video_item(self, vid: dict, override_seek_sec: float = None):
        self._stop_video_player()
        self._vid_ph_spec = None
        self._refresh_video_hits_bar_if_needed(vid)

        fpath = str(vid.get("path") or "")
        if not fpath or not os.path.exists(fpath):
            self._vid_ph_spec = ("ui.plates.vid_ph.missing", "[Video file missing]", "#ff6666")
            self._set_video_placeholder(tr(*self._vid_ph_spec[:2]), color="#ff6666")
            return
        if vid.get("busy"):
            self._vid_ph_spec = ("ui.plates.vid_ph.recording", "[Video file is still recording...]", "#ffcc00")
            self._set_video_placeholder(tr(*self._vid_ph_spec[:2]), color="#ffcc00")
            return
        if cv2 is None:
            self._vid_ph_spec = (
                "ui.plates.vid_ph.no_cv2",
                "[OpenCV (cv2) is required for the built-in player]\nUse the \"↗ Open file\" button.",
                "#ffcc00",
            )
            self._set_video_placeholder(tr(*self._vid_ph_spec[:2]), color="#ffcc00")
            return

        cap, fps, total_f = open_video_capture_for_playback(cv2, fpath, vid_meta=vid)
        if cap is None:
            self._vid_ph_spec = ("ui.plates.vid_ph.open_err", "[Failed to open video file]", "#ff6666")
            self._set_video_placeholder(tr(*self._vid_ph_spec[:2]), color="#ff6666")
            return

        self._vid_cap = cap
        self._vid_path = fpath
        self._vid_fps = fps
        self._vid_total_frames = total_f

        seek_sec = float(override_seek_sec if override_seek_sec is not None else vid.get("seek_sec", 0.0))
        self._seek_video_sec(seek_sec)

    def _seek_video_sec(self, sec: float):
        if self._vid_cap is None or cv2 is None:
            return
        ret, frame, landed_idx, updated_tot = seek_video_capture(
            self._vid_cap,
            cv2,
            self._vid_fps,
            self._vid_total_frames,
            target_sec=max(0.0, float(sec)),
        )
        if ret and frame is not None:
            self._vid_total_frames = updated_tot
            self._vid_curr_frame = landed_idx
            self._render_video_frame(frame)
            self._update_vid_time_ui()

    def _seek_video_frame(self, frame_idx: int):
        if self._vid_cap is None or cv2 is None:
            return
        ret, frame, landed_idx, updated_tot = seek_video_capture(
            self._vid_cap,
            cv2,
            self._vid_fps,
            self._vid_total_frames,
            target_frame=int(frame_idx),
        )
        if ret and frame is not None:
            self._vid_total_frames = updated_tot
            self._vid_curr_frame = landed_idx
            self._render_video_frame(frame)
            self._update_vid_time_ui()

    def _on_vid_slider_press(self, event=None):
        self._vid_slider_dragging = True

    def _on_vid_slider_release(self, event=None):
        self._vid_slider_dragging = False
        if self._pending_live_refresh and not self._is_live_refresh_blocked():
            self._schedule_pending_live_refresh(delay_ms=100)

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
        pos_f = float(self._vid_cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0) if cv2 is not None else 0.0
        if pos_f >= 1.0:
            self._vid_curr_frame = max(0, int(round(pos_f)) - 1)
        else:
            self._vid_curr_frame += 1
        if self._vid_curr_frame >= self._vid_total_frames:
            self._vid_total_frames = self._vid_curr_frame + 1
        self._render_video_frame(frame)
        self._update_vid_time_ui()

        rate = self._vid_fps * (2 if self._vid_speed >= 2 else 1)
        target_ms = 1000.0 / max(1.0, rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._schedule_vid_tick(max(12, int(round(target_ms - elapsed_ms))))

    # ------------------------------------------------------------------
    # Дії користувача
    # ------------------------------------------------------------------
    def _copy_selected_plate(self):
        rec = self._get_selected_record()
        if not rec:
            return
        try:
            self.win.clipboard_clear()
            self.win.clipboard_append(rec["plate"])
        except Exception:
            pass

    def _add_new_plate_clicked(self):
        if self._guard_operator_write(tr("ui.plates.op_act.add", "Add plate record")):
            return
        db = self._get_plate_db()
        if db is None:
            return
        did_refresh = False
        self._enter_modal()
        try:
            dlg = EditPlateMetaDialog(
                self.win,
                plate_text="",
                initial_label="",
                initial_status="whitelist",
                initial_notes="",
                allow_plate_edit=True,
            )
            if dlg.result:
                res = db.add_plate_record(
                    dlg.result["plate"],
                    label=dlg.result["label"],
                    status=dlg.result["status"],
                    notes=dlg.result["notes"],
                )
                if res and res.get("ok"):
                    did_refresh = True
                    self.refresh_all(select_plate=res["plate"], preserve_active_item=False)
                elif res:
                    messagebox.showwarning(
                        tr("ui.plates.err_title", "Error"),
                        res.get("error") or tr("ui.plates.err_save", "Failed to save plate record."),
                        parent=self.win,
                    )
        finally:
            self._exit_modal(schedule_flush=not did_refresh)

    def _edit_selected_meta(self):
        if self._guard_operator_write(tr("ui.plates.op_act.edit_meta", "Edit plate note / status")):
            return
        rec = self._get_selected_record()
        db = self._get_plate_db()
        if not rec or db is None:
            return
        did_refresh = False
        self._enter_modal()
        try:
            dlg = EditPlateMetaDialog(
                self.win,
                plate_text=rec["plate"],
                initial_label=rec.get("label", ""),
                initial_status=rec.get("status", "normal"),
                initial_notes=rec.get("notes", ""),
                allow_plate_edit=False,
            )
            if dlg.result:
                res = db.set_plate_metadata(
                    rec["plate"],
                    label=dlg.result["label"],
                    status=dlg.result["status"],
                    notes=dlg.result["notes"],
                )
                if res and res.get("ok"):
                    did_refresh = True
                    self.refresh_all(select_plate=rec["plate"])
        finally:
            self._exit_modal(schedule_flush=not did_refresh)

    def _rename_selected_plate(self):
        if self._guard_operator_write(tr("ui.plates.op_act.rename_plate", "Rename / Merge plate")):
            return
        rec = self._get_selected_record()
        db = self._get_plate_db()
        if not rec or db is None:
            return
        old_p = rec["plate"]
        evs = self._events_by_plate.get(old_p) or []
        did_refresh = False
        self._enter_modal()
        try:
            dlg = tk.Toplevel(self.win)
            dlg.title(tr("ui.plates.rename_dlg.title", "🔄 Rename / Merge License Plate"))
            dlg.configure(bg="#181922")
            dlg.resizable(False, False)
            dlg.transient(self.win)
            dlg.grab_set()

            res_holder = {"value": None}
            body = tk.Frame(dlg, bg="#181922", padx=18, pady=16)
            body.pack(fill=tk.BOTH, expand=True)
            tk.Label(
                body,
                text=tr(
                    "ui.plates.rename_dlg.prompt",
                    "Enter the corrected license plate for '{old}'.\nAll snapshots ({n}) and note/status will be moved/merged into it:",
                    old=old_p,
                    n=len(evs),
                ),
                fg="#e0e4f0", bg="#181922", font=("Segoe UI", 9), justify=tk.LEFT, wraplength=400,
            ).pack(anchor=tk.W, pady=(0, 8))

            var = tk.StringVar(value=old_p)
            ent = tk.Entry(
                body, textvariable=var, width=28,
                bg="#262836", fg="#00ff88", insertbackground="white",
                relief=tk.FLAT, font=("Consolas", 12, "bold"),
            )
            ent.pack(fill=tk.X, ipady=5, pady=(0, 14))
            ent.select_range(0, tk.END)
            ent.focus_set()

            btns = tk.Frame(body, bg="#181922")
            btns.pack(fill=tk.X)

            def _ok(*_):
                norm = _normalize_plate_input(var.get())
                if len(norm) < 2:
                    return
                res_holder["value"] = norm
                dlg.destroy()

            def _cancel(*_):
                dlg.destroy()

            tk.Button(
                btns, text=tr("ui.common.cancel", "Cancel"), command=_cancel,
                bg="#323544", fg="#d5dae8", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
                padx=14, pady=5, cursor="hand2",
            ).pack(side=tk.RIGHT, padx=(6, 0))
            tk.Button(
                btns, text=tr("ui.common.save", "Save"), command=_ok,
                bg="#007acc", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
                padx=16, pady=5, cursor="hand2",
            ).pack(side=tk.RIGHT)

            dlg.bind("<Return>", _ok)
            dlg.bind("<Escape>", _cancel)
            dlg.update_idletasks()
            try:
                px = self.win.winfo_rootx() + (self.win.winfo_width() - dlg.winfo_width()) // 2
                py = self.win.winfo_rooty() + (self.win.winfo_height() - dlg.winfo_height()) // 2
                dlg.geometry(f"+{max(0, px)}+{max(0, py)}")
            except Exception:
                pass
            dlg.wait_window()

            new_p = res_holder["value"]
            if new_p and new_p != old_p:
                r = db.merge_or_rename_plate(old_p, new_p, evs)
                if r and r.get("ok"):
                    did_refresh = True
                    self.refresh_all(select_plate=r["new_plate"], preserve_active_item=False)
                elif r:
                    messagebox.showerror(
                        tr("ui.plates.err_title", "Error"),
                        r.get("error") or tr("ui.plates.err_rename", "Failed to rename plate."),
                        parent=self.win,
                    )
        finally:
            self._exit_modal(schedule_flush=not did_refresh)

    def _delete_selected_plate(self):
        if self._guard_operator_write(tr("ui.plates.op_act.delete_plate", "Delete plate")):
            return
        rec = self._get_selected_record()
        db = self._get_plate_db()
        if not rec or db is None:
            return
        plate = rec["plate"]
        evs = self._events_by_plate.get(plate) or []
        did_refresh = False
        self._enter_modal()
        try:
            if evs:
                ans = messagebox.askyesnocancel(
                    tr("ui.plates.delete_plate.title", "Delete license plate"),
                    tr(
                        "ui.plates.delete_plate.confirm",
                        "Delete '{plate}'?\n\n• Yes — delete all snapshots from disk ({n}) AND database note/status.\n• No — clear only note/status from plates.json (snapshots remain in archive).\n• Cancel — do not delete.",
                        plate=plate,
                        n=len(evs),
                    ),
                    parent=self.win,
                )
            else:
                ok_del = messagebox.askyesno(
                    tr("ui.plates.delete_plate.title", "Delete license plate"),
                    tr("ui.plates.delete_plate.confirm_meta_only", "Delete database record for '{plate}'?", plate=plate),
                    parent=self.win,
                )
                ans = False if ok_del else None

            if ans is None:
                return
            db.delete_plate_completely(plate, evs, delete_files=bool(ans))
            if ans or not evs:
                self._selected_plate = None
                self._displayed_plate = None
                self._current_item = None
            did_refresh = True
            self.refresh_all(preserve_active_item=not bool(ans))
        finally:
            self._exit_modal(schedule_flush=not did_refresh)

    def _on_fix_plate_clicked(self):
        if self._guard_operator_write(tr("ui.plates.op_act.fix_plate", "Fix plate / Re-OCR")):
            return
        item = self._current_item
        db = self._get_plate_db()
        if not item or item.get("kind") not in ("crop", "screen") or db is None:
            return
        ev = item["event"]
        crop_path = str(ev.get("crop_path") or "")
        did_refresh = False
        self._enter_modal()
        try:
            if not crop_path or not os.path.exists(crop_path):
                messagebox.showwarning(
                    tr("ui.plates.crop_missing.title", "Snapshot missing"),
                    tr("ui.plates.crop_missing.msg", "Plate snapshot file was not found on disk."),
                    parent=self.win,
                )
                return

            old_plate = str(ev.get("plate") or "")
            old_evs = list(self._events_by_plate.get(old_plate) or [])
            known_list = [r["plate"] for r in self._records]

            dlg = CorrectPlateDialog(
                self.win,
                db_accessor=db,
                event_info=ev,
                known_plates=known_list,
                total_events_for_old=len(old_evs),
            )
            if not dlg.result:
                return
            act = dlg.result.get("action")
            if act == "delete_one":
                res = db.delete_crop_event(ev)
                if res and res.get("ok"):
                    did_refresh = True
                    self.refresh_all(select_plate=old_plate, preserve_active_item=False)
                return

            new_plate = dlg.result["new_plate"]
            if act == "merge_all":
                res = db.merge_or_rename_plate(old_plate, new_plate, old_evs)
                if res and res.get("ok"):
                    did_refresh = True
                    self.refresh_all(select_plate=res["new_plate"], preserve_active_item=False)
                elif res:
                    messagebox.showerror(
                        tr("ui.plates.err_title", "Error"),
                        res.get("error") or tr("ui.plates.err_fix", "Failed to apply fix."),
                        parent=self.win,
                    )
            else:
                res = db.correct_event_plate(ev, new_plate, new_conf=dlg.result.get("new_conf"))
                if res and res.get("ok"):
                    keep_target = old_plate if len(old_evs) > 1 and old_plate == self._selected_plate and new_plate == old_plate else new_plate
                    did_refresh = True
                    self.refresh_all(select_plate=keep_target, preserve_active_item=False)
                elif res:
                    messagebox.showerror(
                        tr("ui.plates.err_title", "Error"),
                        res.get("error") or tr("ui.plates.err_fix", "Failed to apply fix."),
                        parent=self.win,
                    )
        finally:
            self._exit_modal(schedule_flush=not did_refresh)

    def _on_delete_crop_clicked(self):
        if self._guard_operator_write(tr("ui.plates.op_act.delete_crop", "Delete plate snapshot")):
            return
        item = self._current_item
        db = self._get_plate_db()
        if not item or item.get("kind") not in ("crop", "screen") or db is None:
            return
        ev = item["event"]
        did_refresh = False
        self._enter_modal()
        try:
            if not messagebox.askyesno(
                tr("ui.plates.del_crop.title", "Delete snapshot"),
                tr(
                    "ui.plates.del_crop.confirm",
                    "Delete snapshot of '{plate}' ({day} {time}, {cam}) from disk?",
                    plate=ev.get("plate", ""),
                    day=ev["day"],
                    time=ev["time"],
                    cam=ev["camera_name"],
                ),
                parent=self.win,
            ):
                return

            res = db.delete_crop_event(ev)
            if not res or not res.get("ok"):
                messagebox.showerror(
                    tr("ui.plates.del_crop.err_title", "Delete error"),
                    (res or {}).get("error") or tr("ui.plates.del_crop.err_msg", "Failed to delete snapshot."),
                    parent=self.win,
                )
                return
            did_refresh = True
            self.refresh_all(select_plate=self._selected_plate, preserve_active_item=False)
        finally:
            self._exit_modal(schedule_flush=not did_refresh)

    def _on_view_debug_clicked(self):
        item = self._current_item
        if not item or item.get("kind") == "video":
            return
        dbg_path = str((item.get("event") or {}).get("debug_path") or "")
        if not dbg_path or not os.path.exists(dbg_path):
            return
        try:
            with open(dbg_path, "r", encoding="utf-8") as df:
                content = df.read()
        except Exception:
            return

        dlg = tk.Toplevel(self.win)
        dlg.title(os.path.basename(dbg_path))
        dlg.configure(bg="#14151e")
        dlg.geometry("680x440")
        dlg.transient(self.win)
        txt = tk.Text(
            dlg, bg="#191b26", fg="#d5dae8", font=("Consolas", 9),
            relief=tk.FLAT, wrap=tk.NONE, padx=10, pady=10,
        )
        vs = ttk.Scrollbar(dlg, orient=tk.VERTICAL, command=txt.yview)
        txt.configure(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        txt.insert("1.0", content)
        txt.configure(state=tk.DISABLED)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def _on_jump_to_matching_video(self):
        item = self._current_item
        if not item or item.get("kind") == "video":
            return
        match = self._find_matching_video_for_event(item["event"])
        if not match:
            self._enter_modal()
            try:
                messagebox.showinfo(
                    tr("ui.plates.no_video.title", "Video not found"),
                    tr("ui.plates.no_video.msg", "No saved video segment matches this timestamp."),
                    parent=self.win,
                )
            finally:
                self._exit_modal(schedule_flush=True)
            return
        vid, seek_sec = match
        target_key = f"video::{vid['path']}"
        if self.kind_filter_var.get() not in ("all", "video"):
            self.kind_filter_var.set("all")
            self._update_filter_buttons_style()
            self._rebuild_timeline(reload_media=False, preserve_scroll=True)

        self._tl_tree_syncing = True
        try:
            for iid, t_item in self._tree_item_map.items():
                if t_item.get("item_key") == target_key:
                    parent_iid = self.tl_tree.parent(iid)
                    if parent_iid:
                        self.tl_tree.item(parent_iid, open=True)
                    self.tl_tree.selection_set(iid)
                    self.tl_tree.focus(iid)
                    self.tl_tree.see(iid)
                    self._show_timeline_item(t_item, override_seek_sec=seek_sec)
                    return
        finally:
            self._tl_tree_syncing = False

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
    # Події від головної системи та життєвий цикл
    # ------------------------------------------------------------------
    def notify_saved(self, mod_name: str):
        if self._is_closing or str(mod_name) != "plates":
            return
        if self._is_live_refresh_blocked():
            self._pending_live_refresh = True
            return
        if self._notify_after_id is None:
            try:
                if self.win.winfo_exists():
                    self._notify_after_id = self.win.after(900, self._on_notify_timer)
            except Exception:
                pass

    def _on_notify_timer(self):
        self._notify_after_id = None
        if self._is_closing or not self.win.winfo_exists():
            return
        if self._is_live_refresh_blocked() or self._load_in_flight:
            self._pending_live_refresh = True
            return
        self.refresh_all(select_plate=self._selected_plate, live_update=True, preserve_active_item=True)

    def on_modules_reloaded(self):
        if not self._is_closing:
            self._sync_operator_mode_ui()
            self.refresh_all(select_plate=self._selected_plate, live_update=True, preserve_active_item=True)

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
        for attr in ("_init_refresh_after_id", "_notify_after_id", "_pending_flush_after_id", "_ui_poll_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.win.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._stop_video_player()
        self._thumb_cache.clear()
        self._sharpness_cache.clear()
        self._left_thumbs.clear()
        self._plate_cards.clear()
        self._plate_card_frames.clear()
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
