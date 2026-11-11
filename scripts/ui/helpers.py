import sys
import tkinter as tk
from tkinter import ttk

from scripts.core.i18n import tr
from tkinter import font as tkfont

EVT_CONFIGURE = "<Configure>"
EVT_ENTER = "<Enter>"
EVT_LEAVE = "<Leave>"
EVT_MOTION = "<Motion>"
EVT_MOUSEWHEEL = "<MouseWheel>"
EVT_BTN4 = "<Button-4>"
EVT_BTN5 = "<Button-5>"


def get_virtual_screen_bounds(widget=None):
    """Повертає (x, y, w, h) віртуального робочого столу через усі підключені монітори."""
    if sys.platform == "win32":
        try:
            import ctypes
            u32 = ctypes.windll.user32
            vx = int(u32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
            vy = int(u32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
            vw = int(u32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
            vh = int(u32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
            if vw > 0 and vh > 0:
                return vx, vy, vw, vh
        except Exception:
            pass
    if widget is not None:
        try:
            return 0, 0, max(640, widget.winfo_screenwidth()), max(480, widget.winfo_screenheight())
        except Exception:
            pass
    return 0, 0, 1280, 720


def clamp_window_geometry(left, top, width, height, min_w=320, min_h=240, widget=None):
    """Перевіряє та повертає видиму на віртуальному екрані геометрію (left, top, width, height)."""
    vx, vy, vw, vh = get_virtual_screen_bounds(widget)
    w = max(int(min_w), min(int(vw), int(width)))
    h = max(int(min_h), min(int(vh), int(height)))
    x = int(left)
    y = int(top)

    min_x = vx - max(0, w - 100)
    max_x = vx + vw - 100
    min_y = vy
    max_y = vy + vh - 60

    if x < min_x or x > max_x:
        x = max(vx, min(x, vx + vw - w))
    if y < min_y or y > max_y:
        y = max(vy, min(y, vy + vh - h))
    return x, y, w, h


def get_current_monitor_geometry(widget):
    """Визначає (x, y, width, height) саме того монітора, на якому знаходиться вікно/віджет."""
    try:
        widget.update_idletasks()
    except Exception:
        pass

    if sys.platform == "win32" and widget is not None:
        try:
            import ctypes
            from ctypes import wintypes

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG),
                ]

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            user32 = ctypes.windll.user32
            hmon = 0
            try:
                rx = int(widget.winfo_rootx())
                ry = int(widget.winfo_rooty())
                rw = max(1, int(widget.winfo_width()))
                rh = max(1, int(widget.winfo_height()))
                if rw > 1 and rh > 1:
                    rc_target = RECT(rx, ry, rx + rw, ry + rh)
                    hmon = user32.MonitorFromRect(ctypes.byref(rc_target), 2)
            except Exception:
                hmon = 0

            if not hmon:
                hwnd = widget.winfo_id()
                hmon = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST

            if hmon:
                mi = MONITORINFO()
                mi.cbSize = ctypes.sizeof(MONITORINFO)
                if user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                    rc = mi.rcMonitor
                    mw = rc.right - rc.left
                    mh = rc.bottom - rc.top
                    if mw > 0 and mh > 0:
                        return rc.left, rc.top, mw, mh
        except Exception:
            pass

    if widget is not None:
        try:
            return (
                int(widget.winfo_rootx()),
                int(widget.winfo_rooty()),
                max(640, int(widget.winfo_width())),
                max(480, int(widget.winfo_height())),
            )
        except Exception:
            pass
    return 0, 0, 1280, 720


def format_geometry_str(left, top, width, height) -> str:
    x_s = f"+{int(left)}" if int(left) >= 0 else f"{int(left)}"
    y_s = f"+{int(top)}" if int(top) >= 0 else f"{int(top)}"
    return f"{int(width)}x{int(height)}{x_s}{y_s}"


class WindowGeometryTracker:
    """
    Автоматично відновлює та зберігає координати, розмір і стан максимізації вікна
    (у тому числі при переміщенні на інші монітори), без виведення сирих координат у UI налаштувань.
    """
    def __init__(self, window, get_geom_cb, set_geom_cb, min_w=480, min_h=320, default_geom=None):
        self.win = window
        self.get_geom_cb = get_geom_cb
        self.set_geom_cb = set_geom_cb
        self.min_w = min_w
        self.min_h = min_h
        self._save_after_id = None
        self._restore_after_id = None
        self._restoring = True
        self._closed = False

        base = dict(default_geom or {"left": 80, "top": 60, "width": 1160, "height": 720, "maximized": False})
        loaded = get_geom_cb() or {}
        self._normal_geom = {
            "left": int(loaded.get("left", base["left"])),
            "top": int(loaded.get("top", base["top"])),
            "width": max(min_w, int(loaded.get("width", base["width"]))),
            "height": max(min_h, int(loaded.get("height", base["height"]))),
        }
        self._maximized = bool(loaded.get("maximized", base.get("maximized", False)))

        self.restore()
        self.win.bind(EVT_CONFIGURE, self._on_configure, add="+")

    def restore(self):
        self._restoring = True
        x, y, w, h = clamp_window_geometry(
            self._normal_geom["left"],
            self._normal_geom["top"],
            self._normal_geom["width"],
            self._normal_geom["height"],
            min_w=self.min_w,
            min_h=self.min_h,
            widget=self.win,
        )
        self._normal_geom = {"left": x, "top": y, "width": w, "height": h}
        try:
            self.win.minsize(self.min_w, self.min_h)
            self.win.geometry(format_geometry_str(x, y, w, h))
            self.win.update_idletasks()
        except Exception:
            pass

        def _finish_restore():
            self._restore_after_id = None
            if self._closed:
                return
            if self._maximized and sys.platform == "win32":
                try:
                    self.win.state("zoomed")
                except Exception:
                    pass
            elif self._maximized:
                try:
                    self.win.attributes("-zoomed", True)
                except Exception:
                    pass
            try:
                self._restore_after_id = self.win.after(150, self._end_restore)
            except Exception:
                self._restoring = False

        self._restore_after_id = self.win.after(20, _finish_restore)

    def _end_restore(self):
        self._restoring = False

    def _is_zoomed(self) -> bool:
        try:
            if str(self.win.state()).lower() == "zoomed":
                return True
        except Exception:
            pass
        try:
            if bool(self.win.attributes("-zoomed")):
                return True
        except Exception:
            pass
        return False

    def _on_configure(self, event=None):
        if self._restoring or self._closed:
            return
        if event is not None and event.widget is not self.win:
            return
        try:
            st = str(self.win.state()).lower()
            if st == "iconic" or st == "withdrawn":
                return
        except Exception:
            pass

        zoomed = self._is_zoomed()
        self._maximized = zoomed
        if not zoomed:
            try:
                x = int(self.win.winfo_x())
                y = int(self.win.winfo_y())
                w = int(self.win.winfo_width())
                h = int(self.win.winfo_height())
                if w >= self.min_w and h >= self.min_h and x > -25000 and y > -25000:
                    self._normal_geom = {"left": x, "top": y, "width": w, "height": h}
            except Exception:
                pass

        if self._save_after_id is not None:
            try:
                self.win.after_cancel(self._save_after_id)
            except Exception:
                pass
        self._save_after_id = self.win.after(350, self.flush)

    def flush(self):
        self._save_after_id = None
        if self._closed:
            return
        payload = {
            "left": int(self._normal_geom["left"]),
            "top": int(self._normal_geom["top"]),
            "width": int(self._normal_geom["width"]),
            "height": int(self._normal_geom["height"]),
            "maximized": bool(self._maximized),
        }
        try:
            self.set_geom_cb(payload)
        except Exception:
            pass

    def close(self):
        for attr in ("_restore_after_id", "_save_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.win.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        self.flush()
        self._closed = True


class ScrollableFrame(tk.Frame):
    """Контейнер із вертикальною прокруткою та автоматичною підгонкою ширини контенту."""
    def __init__(self, parent, bg="#181820", **kwargs):
        super().__init__(parent, bg=bg, **kwargs)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)

        self._win_id = self.canvas.create_window((0, 0), window=self.inner, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner.bind(EVT_CONFIGURE, self._on_inner_configure)
        self.canvas.bind(EVT_CONFIGURE, self._on_canvas_configure)

        self._wheel_bound = False
        self._wheel_hook_toplevel = None
        self._suspend_scrollregion = False
        self.canvas.bind(EVT_ENTER, self._on_enter)
        self.canvas.bind(EVT_LEAVE, self._on_leave)
        self.inner.bind(EVT_ENTER, self._on_enter)
        self.inner.bind(EVT_LEAVE, self._on_leave)

    def _on_inner_configure(self, event=None):
        if self._suspend_scrollregion:
            return
        try:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception:
            pass

    def get_scroll_y_px(self) -> float:
        try:
            bbox = self.canvas.bbox("all")
            if not bbox:
                return 0.0
            total_h = float(bbox[3] - bbox[1])
            if total_h <= 1.0:
                return 0.0
            top_frac, _ = self.canvas.yview()
            return max(0.0, float(top_frac) * total_h)
        except Exception:
            return 0.0

    def set_scroll_y_px(self, y_px: float):
        try:
            for _ in range(2):
                self.canvas.update_idletasks()
            bbox = self.canvas.bbox("all")
            if not bbox:
                return
            self.canvas.configure(scrollregion=bbox)
            total_h = float(bbox[3] - bbox[1])
            ch = float(self.canvas.winfo_height())
            max_top = max(0.0, total_h - ch) if ch > 10.0 else max(0.0, total_h)
            clamped = max(0.0, min(float(y_px), max_top))
            if total_h > 0:
                self.canvas.yview_moveto(clamped / total_h)
        except Exception:
            pass

    def preserve_scroll_during(self, fn):
        y_px = self.get_scroll_y_px()
        prev = self._suspend_scrollregion
        self._suspend_scrollregion = True
        try:
            return fn()
        finally:
            self._suspend_scrollregion = prev
            if not prev:
                self.set_scroll_y_px(y_px)

    def _on_canvas_configure(self, event=None):
        try:
            cw = int(event.width) if event else int(self.canvas.winfo_width())
            if cw > 20:
                self.canvas.itemconfigure(self._win_id, width=cw)
        except Exception:
            pass

    def _on_enter(self, event=None):
        self._wheel_bound = True
        try:
            top = self.winfo_toplevel()
            if self._wheel_hook_toplevel is not top:
                self._wheel_hook_toplevel = top
                top.bind(EVT_MOUSEWHEEL, self._on_mousewheel, add="+")
                top.bind(EVT_BTN4, self._on_mousewheel, add="+")
                top.bind(EVT_BTN5, self._on_mousewheel, add="+")
        except Exception:
            pass

    def _on_leave(self, event=None):
        self._wheel_bound = False

    def _on_mousewheel(self, event):
        if not self._wheel_bound:
            return
        try:
            px, py = self.winfo_pointerxy()
            rx = self.canvas.winfo_rootx()
            ry = self.canvas.winfo_rooty()
            rw = self.canvas.winfo_width()
            rh = self.canvas.winfo_height()
            if not (rx <= px <= rx + rw and ry <= py <= ry + rh):
                self._wheel_bound = False
                return
            hovered = self.winfo_containing(px, py)
            if hovered is not None and isinstance(hovered, ttk.Treeview):
                return
            bbox = self.canvas.bbox("all")
            if not bbox or (bbox[3] - bbox[1]) <= rh:
                return
            if getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            elif getattr(event, "delta", 0):
                delta = int(-1 * (event.delta / 120))
            else:
                delta = 0
            if delta != 0:
                self.canvas.yview_scroll(delta * 3, "units")
                return "break"
        except Exception:
            pass


_ACTIVE_HELP_POPOVER = None


def close_active_help_popover():
    global _ACTIVE_HELP_POPOVER
    if _ACTIVE_HELP_POPOVER is not None:
        try:
            _ACTIVE_HELP_POPOVER.destroy()
        except Exception:
            pass
        _ACTIVE_HELP_POPOVER = None


class HelpButton(tk.Label):
    """
    Кнопка '?' біля кожного параметра у налаштуваннях.
    Наведення показує коротку підказку, клік відкриває повний детальний опис параметра.
    """
    def __init__(self, parent, help_text: str, title: str = None, bg="#181820"):
        super().__init__(
            parent,
            text="?",
            fg="#00e5ff",
            bg="#262838",
            font=("Segoe UI", 8, "bold"),
            width=2,
            relief=tk.FLAT,
            cursor="hand2",
            padx=2,
            pady=0,
        )
        self.help_text = str(help_text or "").strip()
        self.help_title = str(title if title is not None else tr("help.dialog_title", "Help"))
        self._hover_after = None
        self._tooltip_win = None

        self.bind(EVT_ENTER, self._on_enter)
        self.bind(EVT_LEAVE, self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _on_enter(self, event=None):
        self.configure(bg="#007acc", fg="#ffffff")
        self._cancel_hover()
        self._hover_after = self.after(400, lambda: self._show_popup(is_pinned=False))

    def _on_leave(self, event=None):
        self.configure(bg="#262838", fg="#00e5ff")
        self._cancel_hover()
        self._hide_tooltip()

    def _on_click(self, event=None):
        self._cancel_hover()
        self._hide_tooltip()
        self._show_popup(is_pinned=True)
        return "break"

    def _cancel_hover(self):
        if self._hover_after is not None:
            try:
                self.after_cancel(self._hover_after)
            except Exception:
                pass
            self._hover_after = None

    def _hide_tooltip(self):
        if self._tooltip_win is not None:
            try:
                self._tooltip_win.destroy()
            except Exception:
                pass
            self._tooltip_win = None

    def _show_popup(self, is_pinned: bool):
        global _ACTIVE_HELP_POPOVER
        if is_pinned:
            was_same = getattr(_ACTIVE_HELP_POPOVER, "_owner_btn", None) is self
            close_active_help_popover()
            if was_same:
                return

        if not self.help_text:
            return

        try:
            top = self.winfo_toplevel()
            pop = tk.Toplevel(top)
            pop.overrideredirect(True)
            pop.configure(bg="#00e5ff")
            pop._owner_btn = self
        except Exception:
            return

        inner = tk.Frame(pop, bg="#1d202b", padx=12, pady=10)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        if is_pinned:
            hdr = tk.Frame(inner, bg="#1d202b")
            hdr.pack(fill=tk.X, pady=(0, 6))
            tk.Label(
                hdr, text=f"ℹ {self.help_title}", fg="#00e5ff", bg="#1d202b",
                font=("Segoe UI", 9, "bold"), anchor=tk.W
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(
                hdr, text="✕", command=close_active_help_popover,
                bg="#2d3142", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"),
                padx=5, cursor="hand2"
            ).pack(side=tk.RIGHT)

        tk.Label(
            inner,
            text=self.help_text,
            fg="#e8ebf5",
            bg="#1d202b",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
            wraplength=420,
            anchor=tk.W,
        ).pack(fill=tk.BOTH, expand=True)

        try:
            pop.update_idletasks()
            pw = pop.winfo_reqwidth()
            ph = pop.winfo_reqheight()
            vx, vy, vw, vh = get_virtual_screen_bounds(self)
            bx = self.winfo_rootx() + self.winfo_width() + 6
            by = self.winfo_rooty() - 4
            if bx + pw > vx + vw - 10:
                bx = max(vx + 8, self.winfo_rootx() - pw - 6)
            if by + ph > vy + vh - 10:
                by = max(vy + 8, vy + vh - ph - 10)
            pop.geometry(f"+{bx}+{by}")
        except Exception:
            pass

        if is_pinned:
            _ACTIVE_HELP_POPOVER = pop
            pop.bind("<FocusOut>", lambda e: close_active_help_popover())
            pop.bind("<Escape>", lambda e: close_active_help_popover())
            try:
                pop.focus_set()
            except Exception:
                pass
        else:
            self._tooltip_win = pop


def compute_auto_grid(num_items: int, area_w: int, area_h: int) -> tuple:
    """Автоматично підбирає (rows, cols) під кількість камер і пропорції монітора."""
    n = max(1, int(num_items))
    w = max(160, int(area_w))
    h = max(120, int(area_h))
    aspect = w / float(h)

    if n == 1:
        return 1, 1
    if n == 2:
        return (1, 2) if aspect >= 1.15 else (2, 1)
    if n == 3:
        return (1, 3) if aspect >= 2.2 else (2, 2)
    if n == 4:
        return (2, 2)
    if n <= 6:
        return (2, 3) if aspect >= 1.25 else (3, 2)
    if n <= 8:
        return (2, 4) if aspect >= 1.85 else (3, 3)
    if n <= 9:
        return (3, 3)
    if n <= 12:
        return (3, 4) if aspect >= 1.25 else (4, 3)
    if n <= 16:
        return (4, 4)

    import math
    cols = max(1, int(math.ceil(math.sqrt(n * aspect * (9.0 / 16.0)))))
    rows = max(1, int(math.ceil(n / float(cols))))
    return rows, cols


def _get_cached_tk_font(widget, font_spec=None):
    """Повертає або кешує tkfont.Font на кореневому вікні (безпечно для кількох Tk() у тестах)."""
    try:
        top = widget.winfo_toplevel()
    except Exception:
        top = widget
    cache = getattr(top, "_agy_ellipsis_fonts", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(top, "_agy_ellipsis_fonts", cache)
        except Exception:
            pass

    try:
        resolved = font_spec or widget.cget("font")
    except Exception:
        resolved = ("Segoe UI", 8)
    key = tuple(resolved) if isinstance(resolved, (list, tuple)) else str(resolved)
    cached = cache.get(key)
    if cached is not None:
        return cached

    try:
        if isinstance(resolved, (list, tuple)):
            family = str(resolved[0]) if len(resolved) > 0 else "Segoe UI"
            size = int(resolved[1]) if len(resolved) > 1 else 9
            mods = " ".join(str(x) for x in resolved[2:]).lower()
            weight = "bold" if "bold" in mods else "normal"
            slant = "italic" if "italic" in mods else "roman"
            fobj = tkfont.Font(root=top, family=family, size=size, weight=weight, slant=slant)
        else:
            fobj = tkfont.Font(root=top, font=resolved)
        cache[key] = fobj
        return fobj
    except Exception:
        return None


def fit_text_ellipsis(widget, text: str, max_px: int, font_spec=None) -> str:
    """Обрізає однорядковий текст із «…» точно під доступну ширину в пікселях."""
    s = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    if not s:
        return ""
    limit = int(max_px or 0)
    if limit <= 0:
        return s

    fobj = _get_cached_tk_font(widget, font_spec=font_spec)
    if fobj is None:
        approx = max(3, limit // 6)
        return s if len(s) <= approx else (s[: max(1, approx - 1)].rstrip() + "…")

    try:
        if fobj.measure(s) <= limit:
            return s
        ell = "…"
        if fobj.measure(ell) >= limit:
            return ell
        lo, hi = 1, len(s)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid].rstrip() + ell
            if fobj.measure(cand) <= limit:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return s[:best].rstrip() + ell
    except Exception:
        return s


def _refresh_single_line_ellipsis(lbl: tk.Label, event_width: int = None):
    if lbl is None:
        return
    try:
        if not lbl.winfo_exists():
            return
    except Exception:
        return

    full_raw = str(getattr(lbl, "_ellipsis_full_text", "") or "")
    full_norm = " ".join(full_raw.replace("\r", " ").replace("\n", " ").split())
    pad = int(getattr(lbl, "_ellipsis_pad", 4) or 0)
    min_px = int(getattr(lbl, "_ellipsis_min_px", 40) or 40)
    init_px = int(getattr(lbl, "_ellipsis_initial_px", 160) or 160)

    w = 0
    if event_width is not None and int(event_width) > 10:
        w = int(event_width)
    else:
        try:
            cur_w = int(lbl.winfo_width() or 0)
            if cur_w > 10:
                w = cur_w
            elif lbl.master is not None:
                mw = int(lbl.master.winfo_width() or 0)
                if mw > 20:
                    w = mw - 6
        except Exception:
            w = 0

    max_px = max(min_px, (w if w > 10 else init_px) - pad)
    disp = fit_text_ellipsis(lbl, full_norm, max_px)
    lbl._ellipsis_truncated = bool(full_norm and disp != full_norm)

    sig = (full_norm, disp)
    if getattr(lbl, "_ellipsis_last_sig", None) == sig:
        return
    lbl._ellipsis_last_sig = sig
    try:
        lbl.configure(text=disp)
    except Exception:
        pass


def set_single_line_ellipsis_text(lbl: tk.Label, full_text: str):
    """Оновлює повний текст однорядкового заголовка та одразу підганяє обрізання «…»."""
    if lbl is None:
        return
    lbl._ellipsis_full_text = str(full_text or "")
    lbl._ellipsis_last_sig = None
    _refresh_single_line_ellipsis(lbl)


def bind_single_line_ellipsis(
    lbl: tk.Label,
    full_text: str = None,
    pad: int = 4,
    min_px: int = 40,
    initial_px: int = 160,
):
    """Налаштовує Label як суворо 1-рядковий з автоматичним обрізанням «…» при зміні ширини."""
    if full_text is None:
        try:
            full_text = str(lbl.cget("text") or "")
        except Exception:
            full_text = ""

    lbl._ellipsis_full_text = str(full_text)
    lbl._ellipsis_pad = int(pad)
    lbl._ellipsis_min_px = int(min_px)
    lbl._ellipsis_initial_px = int(initial_px)
    lbl._ellipsis_truncated = False
    lbl._ellipsis_last_sig = None

    try:
        lbl.configure(wraplength=0)
    except Exception:
        pass

    _refresh_single_line_ellipsis(lbl)

    def _on_cfg(event):
        ew = getattr(event, "width", None)
        _refresh_single_line_ellipsis(lbl, event_width=ew)

    lbl.bind(EVT_CONFIGURE, _on_cfg, add="+")
    return lbl


def bind_hover_tooltip(
    widget,
    text_or_fn,
    only_when_truncated: bool = False,
    truncate_sources=None,
    delay_ms: int = 350,
):
    """Показує легку підказку при наведенні курсора (опційно — лише коли текст обрізано «…»)."""
    if widget is None:
        return

    state = {"after_id": None, "win": None}

    def _cancel_timer():
        aid = state.get("after_id")
        if aid is not None:
            try:
                widget.after_cancel(aid)
            except Exception:
                pass
            state["after_id"] = None

    def _hide(event=None):
        _cancel_timer()
        pop = state.get("win")
        state["win"] = None
        if pop is not None:
            try:
                pop.destroy()
            except Exception:
                pass

    def _show():
        state["after_id"] = None
        try:
            if not widget.winfo_exists():
                return
        except Exception:
            return

        if only_when_truncated:
            srcs = tuple(truncate_sources) if truncate_sources is not None else (widget,)
            if not any(bool(getattr(s, "_ellipsis_truncated", False)) for s in srcs if s is not None):
                return

        try:
            txt = text_or_fn() if callable(text_or_fn) else str(text_or_fn or "")
        except Exception:
            txt = ""
        txt = str(txt or "").strip()
        if not txt:
            return

        _hide()
        try:
            top = widget.winfo_toplevel()
            pop = tk.Toplevel(top)
            pop.overrideredirect(True)
            pop.configure(bg="#00e5ff")
        except Exception:
            return

        inner = tk.Frame(pop, bg="#1d202b", padx=10, pady=6)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        tk.Label(
            inner,
            text=txt,
            fg="#e8ebf5",
            bg="#1d202b",
            font=("Segoe UI", 8),
            justify=tk.LEFT,
            wraplength=360,
            anchor=tk.W,
        ).pack(fill=tk.BOTH, expand=True)

        try:
            pop.update_idletasks()
            pw = pop.winfo_reqwidth()
            ph = pop.winfo_reqheight()
            vx, vy, vw, vh = get_virtual_screen_bounds(widget)
            try:
                px, py = widget.winfo_pointerxy()
                bx = int(px) + 12
                by = int(py) + 16
            except Exception:
                bx = int(widget.winfo_rootx()) + 8
                by = int(widget.winfo_rooty()) + int(widget.winfo_height()) + 4
            if bx + pw > vx + vw - 8:
                bx = max(vx + 8, vx + vw - pw - 8)
            if by + ph > vy + vh - 8:
                by = max(vy + 8, int(widget.winfo_rooty()) - ph - 4)
            pop.geometry(f"+{bx}+{by}")
        except Exception:
            pass
        state["win"] = pop

    def _on_enter(event=None):
        _cancel_timer()
        try:
            state["after_id"] = widget.after(int(delay_ms), _show)
        except Exception:
            pass

    try:
        widget.bind(EVT_ENTER, _on_enter, add="+")
        widget.bind(EVT_LEAVE, _hide, add="+")
        widget.bind("<Button-1>", _hide, add="+")
        widget.bind("<Destroy>", _hide, add="+")
    except Exception:
        pass
