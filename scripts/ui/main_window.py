import os
import sys
import time
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from scripts.core.i18n import (
    init_i18n,
    tr,
    set_language,
    get_language,
)
from scripts.core.module_registry import ModuleRegistry
from scripts.core.config_manager import ConfigManager
from scripts.core.io_pool import flush_io
from scripts.core.hardware_info import (
    HardwareMonitor,
    prepare_cuda_dll_environment,
    prepare_ort_backend_environment,
    check_cuda_libraries_status,
    download_cuda_libraries,
    is_onnxruntime_gpu_installed,
    has_nvidia_gpu,
)
from scripts.engine.inference_pool import InferencePool
from scripts.engine.camera_manager import CameraManager
from scripts.ui.helpers import (
    WindowGeometryTracker,
    apply_borderless_window_style,
    compute_auto_grid,
    get_virtual_screen_bounds,
)
from scripts.ui.detail_window import CameraDetailWindow
from scripts.ui.settings_window import SettingsWindow
from scripts.ui.people_window import PeopleWindow
from scripts.ui.plates_window import PlatesWindow

# Базова папка додатку (підтримка PyInstaller EXE та структури scripts/ui/main_window.py)
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import winreg
except ImportError:
    winreg = None

EVT_MOTION = "<Motion>"
EVT_CONFIGURE = "<Configure>"
EVT_ADMIN_EXIT_UP = "<Control-Alt-Shift-KeyPress-Q>"
EVT_ADMIN_EXIT_LOW = "<Control-Alt-Shift-KeyPress-q>"


def _is_win_modifiers_ctrl_alt_shift() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        u32 = ctypes.windll.user32
        ctrl = (u32.GetAsyncKeyState(0x11) & 0x8000) != 0
        alt = (u32.GetAsyncKeyState(0x12) & 0x8000) != 0
        shift = (u32.GetAsyncKeyState(0x10) & 0x8000) != 0
        return bool(ctrl and alt and shift)
    except Exception:
        return False


def _is_win_foreground_admin_exit_pressed() -> bool:
    """Резервна перевірка Ctrl+Alt+Shift+Q через Win32 API для активного вікна нашого процесу."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        hwnd = u32.GetForegroundWindow()
        if not hwnd:
            return False
        pid = wintypes.DWORD(0)
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) != int(os.getpid()):
            return False
        q_down = (u32.GetAsyncKeyState(0x51) & 0x8000) != 0
        if not q_down:
            return False
        return _is_win_modifiers_ctrl_alt_shift()
    except Exception:
        return False


def is_admin_exit_keypress(event) -> bool:
    """Перевіряє Ctrl+Alt+Shift+Q незалежно від розкладки клавіатури, мови та регістру/CapsLock."""
    if event is None:
        return False
    try:
        state = int(getattr(event, "state", 0) or 0)
    except Exception:
        state = 0

    tk_ctrl = bool(state & 0x0004)
    tk_shift = bool(state & 0x0001)
    tk_alt = bool(state & (0x0008 | 0x0010 | 0x20000))
    mods_ok = (tk_ctrl and tk_shift and tk_alt) or _is_win_modifiers_ctrl_alt_shift()
    if not mods_ok:
        return False

    try:
        kc = int(getattr(event, "keycode", 0) or 0)
    except Exception:
        kc = 0
    ks = str(getattr(event, "keysym", "") or "").strip().lower()
    ch = str(getattr(event, "char", "") or "")

    # На Windows VK_Q == 81 для будь-якої мови та регістру; на X11/macOS перевіряємо також keysym і char
    if sys.platform == "win32" and kc == 81:
        return True
    if ks in ("q", "cyrillic_shorti", "hebrew_qoph", "arabic_dad"):
        return True
    if ch in ("q", "Q", "й", "Й", "\x11"):
        return True
    return False

LOCKDOWN_KEYS = [
    "<Alt-F4>",
    "<Alt-KeyPress-F4>",
    "<Control-w>",
    "<Control-W>",
    "<Control-q>",
    "<Control-Q>",
    "<F11>",
]


class GridWindowBase:
    """Shared tile layout, border highlight, and frame-rendering logic for Monitor grid windows."""

    def _grid_win_widget(self):
        return getattr(self, "win", None) or getattr(self, "root", None)

    def _clear_tiles(self):
        for slot in self._tiles.values():
            for key in ("canvas", "hl_top", "hl_bottom", "hl_left", "hl_right"):
                w = slot.get(key)
                if w is not None:
                    try:
                        w.destroy()
                    except Exception:
                        pass
        self._tiles.clear()
        for es in self._empty_slots:
            try:
                es.destroy()
            except Exception:
                pass
        self._empty_slots.clear()
        if self._grid_surface is not None:
            try:
                self._grid_surface.destroy()
            except Exception:
                pass
            self._grid_surface = None
        self._last_grid_dims_key = None

    def _apply_tile_highlight(self, t: dict, state: str, bcol: str):
        cx, cy, cw, ch = t["cell_rect"]
        bw = t["bw"]
        strips = (
            ("hl_top", cx - bw, cy - bw, cw + 2 * bw, bw),
            ("hl_bottom", cx - bw, cy + ch, cw + 2 * bw, bw),
            ("hl_left", cx - bw, cy, bw, ch),
            ("hl_right", cx + cw, cy, bw, ch),
        )
        if state == "normal":
            for key, *_ in strips:
                try:
                    t[key].place_forget()
                except Exception:
                    pass
        else:
            for key, sx, sy, sw, sh in strips:
                try:
                    t[key].configure(bg=bcol)
                    t[key].place(x=sx, y=sy, width=sw, height=sh)
                except Exception:
                    pass

    def _restack_grid_highlights(self):
        for target_state in ("offline", "active"):
            for t in self._tiles.values():
                if t.get("last_hl_state") == target_state:
                    for key in ("hl_top", "hl_bottom", "hl_left", "hl_right"):
                        try:
                            t[key].lift()
                        except Exception:
                            pass
        if self._panel_visible:
            try:
                self.ctrl_frame.lift()
            except Exception:
                pass

    def _get_grid_cfg(self) -> dict:
        return (self._fast_cfg().get("grid") or {})

    def _is_operator_mode(self) -> bool:
        try:
            gen = self._fast_cfg().get("general") or {}
            return str(gen.get("operation_mode", "server")).strip().lower() == "operator"
        except Exception:
            return False

    def _can_show_top_bar(self) -> bool:
        if self._is_operator_mode():
            if getattr(self, "btn_settings", None) is not None:
                return True
            return int(getattr(self, "_total_pages", 1) or 1) > 1
        return True

    def _sync_top_bar_for_mode(self):
        is_op = self._is_operator_mode()
        has_pages = int(getattr(self, "_total_pages", 1) or 1) > 1
        has_settings = getattr(self, "btn_settings", None) is not None
        lb = getattr(self, "_left_box", None)
        rb = getattr(self, "_right_box", None)
        op_hidden_left = getattr(self, "_op_hidden_left", None)

        if is_op:
            if has_settings and op_hidden_left:
                for w in op_hidden_left:
                    if w is not None and w.winfo_manager():
                        w.pack_forget()
                if lb is not None and not lb.winfo_manager():
                    if self.page_box.winfo_manager():
                        lb.pack(side=tk.LEFT, before=self.page_box)
                    else:
                        lb.pack(side=tk.LEFT)
                if not self.btn_settings.winfo_manager():
                    self.btn_settings.pack(side=tk.LEFT, padx=(0, 8))
            else:
                if lb is not None and lb.winfo_manager():
                    lb.pack_forget()

            if rb is not None and rb.winfo_manager():
                rb.pack_forget()

            if has_pages:
                if not self.page_box.winfo_manager():
                    self.page_box.pack(side=tk.LEFT, padx=14 if has_settings else 6)
            else:
                if self.page_box.winfo_manager():
                    self.page_box.pack_forget()
                if (not has_settings) and (getattr(self, "_panel_visible", False) or self.ctrl_frame.winfo_manager()):
                    try:
                        self.ctrl_frame.place_forget()
                    except Exception:
                        pass
                    self._panel_visible = False
        else:
            if rb is not None and not rb.winfo_manager():
                rb.pack(side=tk.RIGHT)
            if has_settings and op_hidden_left:
                if lb is not None and not lb.winfo_manager():
                    if self.page_box.winfo_manager():
                        lb.pack(side=tk.LEFT, before=self.page_box)
                    elif rb is not None and rb.winfo_manager():
                        lb.pack(side=tk.LEFT, before=rb)
                    else:
                        lb.pack(side=tk.LEFT)
                # Ensure exact original left_box child order
                lbl_brand, lbl_cam, lbl_mode, btn_ppl, btn_plt = op_hidden_left
                if not lbl_brand.winfo_manager():
                    lbl_brand.pack(side=tk.LEFT, padx=(0, 10), before=self.btn_settings)
                if not lbl_cam.winfo_manager():
                    lbl_cam.pack(side=tk.LEFT, padx=(0, 8), before=self.btn_settings)
                if not lbl_mode.winfo_manager():
                    lbl_mode.pack(side=tk.LEFT, padx=(0, 10), before=self.btn_settings)
                if not btn_ppl.winfo_manager():
                    btn_ppl.pack(side=tk.LEFT, padx=(0, 8))
                if not btn_plt.winfo_manager():
                    btn_plt.pack(side=tk.LEFT, padx=(0, 8))
            elif lb is not None and not lb.winfo_manager():
                if self.page_box.winfo_manager():
                    lb.pack(side=tk.LEFT, before=self.page_box)
                elif rb is not None and rb.winfo_manager():
                    lb.pack(side=tk.LEFT, before=rb)
                else:
                    lb.pack(side=tk.LEFT)

            if has_pages:
                if not self.page_box.winfo_manager():
                    if rb is not None and rb.winfo_manager():
                        self.page_box.pack(side=tk.LEFT, padx=14, before=rb)
                    else:
                        self.page_box.pack(side=tk.LEFT, padx=14)
            elif self.page_box.winfo_manager():
                self.page_box.pack_forget()

    def _update_tile_borders(self):
        grid_cfg = self._get_grid_cfg()
        col_normal = grid_cfg.get("border_color", "#00e5ff")
        col_active = grid_cfg.get("active_border_color", "#00ff88")
        col_offline = grid_cfg.get("offline_border_color", "#ff4444")

        if self._grid_surface is not None and self._grid_surface.winfo_exists():
            try:
                if str(self._grid_surface.cget("bg")) != col_normal:
                    self._grid_surface.configure(bg=col_normal)
            except Exception:
                pass

        changed = False
        for uid, t in self._tiles.items():
            unit = self.camera_manager.get_unit(uid)
            if uid in self.detail_windows:
                hl_state, bcol = "active", col_active
            elif unit is not None:
                snap = unit.get_grid_snapshot() or {}
                if snap.get("sub_status") in ("offline", "reconnecting") and snap.get("main_status") != "active":
                    hl_state, bcol = "offline", col_offline
                else:
                    hl_state, bcol = "normal", col_normal
            else:
                hl_state, bcol = "offline", col_offline

            bkey = (hl_state, bcol, t.get("bw", 2))
            if bkey != t["last_border"]:
                t["last_border"] = bkey
                t["last_hl_state"] = hl_state
                self._apply_tile_highlight(t, hl_state, bcol)
                changed = True

        if changed:
            self._restack_grid_highlights()

    def _relayout_grid_with_units(self, units: list, force: bool = False):
        self._relayout_after_id = None
        if self._is_closing or getattr(self, "_in_relayout", False):
            return
        self._in_relayout = True
        try:
            self._relayout_grid_with_units_inner(units, force=force)
        finally:
            self._in_relayout = False

    def _relayout_grid_with_units_inner(self, units: list, force: bool = False):
        w_obj = self._grid_win_widget()
        if w_obj is None:
            return
        area_w = self.grid_area.winfo_width()
        area_h = self.grid_area.winfo_height()

        snap = self._fast_cfg()
        grid_cfg = self._get_grid_cfg()
        total_cams = len(units)
        self._last_total_cams = total_cams
        self.lbl_cam_count.configure(text=tr("ui.main.cam_count", "Cameras: {n}", n=total_cams))

        mode = str(grid_cfg.get("mode", "auto")).lower()
        if mode == "manual":
            rows = max(1, min(8, int(grid_cfg.get("rows", 2))))
            cols = max(1, min(8, int(grid_cfg.get("cols", 2))))
            per_page = rows * cols
            self._total_pages = max(1, (total_cams + per_page - 1) // per_page) if total_cams > 0 else 1
            if self._current_page >= self._total_pages:
                self._current_page = max(0, self._total_pages - 1)
            start_i = self._current_page * per_page
            visible_units = units[start_i:start_i + per_page]
        else:
            eff_w = area_w if area_w >= 80 else max(640, w_obj.winfo_width())
            eff_h = area_h if area_h >= 80 else max(400, w_obj.winfo_height())
            rows, cols = compute_auto_grid(max(1, total_cams), eff_w, eff_h)
            self._total_pages = 1
            self._current_page = 0
            visible_units = units

        self._visible_grid_uuids = {u.uuid for u in visible_units}
        self._notify_visible_changed()

        if self._total_pages > 1:
            self.lbl_page.configure(
                text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=self._current_page + 1, total=self._total_pages)
            )
        self._sync_top_bar_for_mode()

        if area_w < 80 or area_h < 80:
            return

        bw = max(1, min(10, int(grid_cfg.get("border_width", 2))))
        gap = max(0, min(20, int(grid_cfg.get("tile_gap", 4))))
        col_normal = grid_cfg.get("border_color", "#00e5ff")

        layout_key = (
            area_w, area_h, rows, cols, bw, gap,
            tuple(u.uuid for u in visible_units),
            bool(grid_cfg.get("show_labels", True)),
            bool(grid_cfg.get("show_status", True)),
            bool(grid_cfg.get("show_fps", True)),
            col_normal,
            grid_cfg.get("active_border_color", "#00ff88"),
            grid_cfg.get("offline_border_color", "#ff4444"),
        )
        if not force and layout_key == self._last_layout_key and (bool(self._tiles) or not visible_units):
            return
        self._last_layout_key = layout_key

        if not visible_units:
            self._clear_tiles()
            empty_txt = tr(
                "ui.main.empty_cams",
                "No enabled cameras.\nOpen ⚙ Settings above to add or enable cameras.",
            )
            if not hasattr(self, "_empty_lbl") or not self._empty_lbl.winfo_exists():
                self._empty_lbl = tk.Label(
                    self.grid_area,
                    text=empty_txt,
                    fg="#9ba3b8", bg="#0b0c10", font=("Segoe UI", 13)
                )
            else:
                self._empty_lbl.configure(text=empty_txt)
            self._empty_lbl.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            return
        elif hasattr(self, "_empty_lbl") and self._empty_lbl.winfo_exists():
            self._empty_lbl.place_forget()

        outer_pad = gap
        avail_w = max(cols * 32, area_w - 2 * outer_pad - bw * (cols + 1))
        avail_h = max(rows * 24, area_h - 2 * outer_pad - bw * (rows + 1))
        base_w, rem_w = divmod(avail_w, cols)
        base_h, rem_h = divmod(avail_h, rows)
        col_widths = [base_w + (1 if c < rem_w else 0) for c in range(cols)]
        row_heights = [base_h + (1 if r < rem_h else 0) for r in range(rows)]

        col_xs = []
        x_cur = bw
        for c in range(cols):
            col_xs.append(x_cur)
            x_cur += col_widths[c] + bw

        row_ys = []
        y_cur = bw
        for r in range(rows):
            row_ys.append(y_cur)
            y_cur += row_heights[r] + bw

        total_grid_w = x_cur
        total_grid_h = y_cur
        origin_x = max(0, (area_w - total_grid_w) // 2)
        origin_y = max(0, (area_h - total_grid_h) // 2)

        needed_uids = tuple(u.uuid for u in visible_units)
        grid_dims_key = (rows, cols, needed_uids)
        if grid_dims_key != self._last_grid_dims_key or self._grid_surface is None or not self._grid_surface.winfo_exists():
            self._clear_tiles()
            self._last_grid_dims_key = grid_dims_key
            self._grid_surface = tk.Frame(self.grid_area, bg=col_normal, highlightthickness=0, bd=0)

            for u in visible_units:
                canvas = tk.Canvas(self._grid_surface, bg="#08090d", highlightthickness=0, bd=0, cursor="hand2")
                img_id = canvas.create_image(1, 1, anchor=tk.CENTER, tags="vid_img")
                ph_id = canvas.create_text(
                    1, 1, text=tr("ui.main.tile.connecting", "Connecting..."),
                    fill="#ffcc00", font=("Segoe UI", 10, "bold"), tags="placeholder"
                )

                lbl_bg_id = canvas.create_rectangle(0, 0, 0, 0, fill="#10131e", outline="#282d3f", width=1, state=tk.HIDDEN, tags="ov_lbl")
                lbl_txt_id = canvas.create_text(10, 10, text="", fill="#ffffff", font=("Segoe UI", 9, "bold"), anchor=tk.NW, state=tk.HIDDEN, tags="ov_lbl")

                st_bg_id = canvas.create_rectangle(0, 0, 0, 0, fill="#10131e", outline="#282d3f", width=1, state=tk.HIDDEN, tags="ov_st")
                st_txt_id = canvas.create_text(10, 10, text="", fill="#75e08b", font=("Segoe UI", 8, "bold"), anchor=tk.NE, state=tk.HIDDEN, tags="ov_st")

                warn_bg_id = canvas.create_rectangle(0, 0, 0, 0, fill="#351418", outline="#ff5555", width=1, state=tk.HIDDEN, tags="ov_warn")
                warn_txt_id = canvas.create_text(10, 10, text="", fill="#ff9999", font=("Segoe UI", 8, "bold"), anchor=tk.SW, state=tk.HIDDEN, tags="ov_warn")

                canvas.bind("<Button-1>", lambda e, uid=u.uuid: self.open_camera_detail(uid))

                hl_top = tk.Frame(self._grid_surface, bg=col_normal, highlightthickness=0, bd=0)
                hl_bottom = tk.Frame(self._grid_surface, bg=col_normal, highlightthickness=0, bd=0)
                hl_left = tk.Frame(self._grid_surface, bg=col_normal, highlightthickness=0, bd=0)
                hl_right = tk.Frame(self._grid_surface, bg=col_normal, highlightthickness=0, bd=0)

                self._tiles[u.uuid] = {
                    "canvas": canvas,
                    "hl_top": hl_top,
                    "hl_bottom": hl_bottom,
                    "hl_left": hl_left,
                    "hl_right": hl_right,
                    "img_id": img_id,
                    "ph_id": ph_id,
                    "lbl_bg_id": lbl_bg_id,
                    "lbl_txt_id": lbl_txt_id,
                    "st_bg_id": st_bg_id,
                    "st_txt_id": st_txt_id,
                    "warn_bg_id": warn_bg_id,
                    "warn_txt_id": warn_txt_id,
                    "photo_tk": None,
                    "last_seq": -1,
                    "inner_w": 0,
                    "inner_h": 0,
                    "cell_rect": (bw, bw, 32, 24),
                    "bw": bw,
                    "last_border": "",
                    "last_hl_state": "",
                    "last_lbl_text": None,
                    "last_st_key": None,
                    "last_warn_text": None,
                }

            total_slots = rows * cols
            for _ in range(total_slots - len(visible_units)):
                es = tk.Frame(self._grid_surface, bg="#08090d", highlightthickness=0, bd=0)
                self._empty_slots.append(es)
        else:
            try:
                if str(self._grid_surface.cget("bg")) != col_normal:
                    self._grid_surface.configure(bg=col_normal)
            except Exception:
                pass

        self._grid_surface.place(x=origin_x, y=origin_y, width=total_grid_w, height=total_grid_h)

        for slot_i, u in enumerate(visible_units):
            r = slot_i // cols
            c = slot_i % cols
            cx = col_xs[c]
            cy = row_ys[r]
            cw = col_widths[c]
            ch = row_heights[r]
            t = self._tiles[u.uuid]

            cv = t["canvas"]
            cv.place(x=cx, y=cy, width=cw, height=ch)
            t["inner_w"] = cw
            t["inner_h"] = ch
            t["cell_rect"] = (cx, cy, cw, ch)
            t["bw"] = bw
            t["last_border"] = ""
            t["last_lbl_text"] = None
            t["last_st_key"] = None
            t["last_warn_text"] = None

            cv.coords(t["img_id"], cw // 2, ch // 2)
            cv.coords(t["ph_id"], cw // 2, ch // 2)

            u.set_grid_target_size(cw, ch)

        for empty_idx, es in enumerate(self._empty_slots):
            slot_i = len(visible_units) + empty_idx
            r = slot_i // cols
            c = slot_i % cols
            es.place(x=col_xs[c], y=row_ys[r], width=col_widths[c], height=row_heights[r])

        self._update_tile_borders()
        if self._panel_visible:
            self.ctrl_frame.lift()

    def _render_tiles_with_units(self, units: list):
        if self._is_closing:
            return 83

        w_obj = self._grid_win_widget()
        if w_obj is None:
            return 250
        try:
            st = str(w_obj.state()).lower()
            if st == "iconic":
                if not self._window_minimized:
                    self._window_minimized = True
                    self._notify_visible_changed()
                return 250
            elif self._window_minimized and st in ("normal", "zoomed"):
                self._window_minimized = False
                self._notify_visible_changed()
        except Exception:
            pass
        if self._window_minimized:
            return 250

        if not self._tiles and units:
            self._relayout_grid(force=False)

        snap = self._fast_cfg()
        grid_cfg = self._get_grid_cfg()
        hw_cfg = snap.get("hardware") or {}
        grid_fps = max(2.0, min(30.0, float(hw_cfg.get("display_grid_fps", 12.0))))
        interval_ms = max(25, int(round(1000.0 / grid_fps)))

        show_labels = bool(grid_cfg.get("show_labels", True))
        show_status = bool(grid_cfg.get("show_status", True))
        show_fps = bool(grid_cfg.get("show_fps", True))
        col_normal = grid_cfg.get("border_color", "#00e5ff")
        col_active = grid_cfg.get("active_border_color", "#00ff88")
        col_offline = grid_cfg.get("offline_border_color", "#ff4444")

        ai_entries = self.registry.list_modules(kind="ai")
        border_changed = False

        for uid, t in list(self._tiles.items()):
            unit = self.camera_manager.get_unit(uid)
            if unit is None:
                continue
            gs = unit.get_grid_snapshot()
            cv = t["canvas"]
            iw = max(32, t["inner_w"])
            ih = max(24, t["inner_h"])

            if uid in self.detail_windows:
                hl_state, bcol = "active", col_active
            elif gs.get("sub_status") in ("offline", "reconnecting") and gs.get("main_status") != "active":
                hl_state, bcol = "offline", col_offline
            else:
                hl_state, bcol = "normal", col_normal
            bkey = (hl_state, bcol, t.get("bw", 2))
            if bkey != t["last_border"]:
                t["last_border"] = bkey
                t["last_hl_state"] = hl_state
                self._apply_tile_highlight(t, hl_state, bcol)
                border_changed = True

            is_cam_enabled = bool(gs.get("enabled", True))
            rgb = gs.get("rgb") if is_cam_enabled else None
            seq = int(gs.get("seq", 0))
            layer_dirty = False
            if rgb is not None:
                if seq != t["last_seq"] or t["photo_tk"] is None:
                    t["last_seq"] = seq
                    t["last_ph_key"] = None
                    try:
                        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
                        t["photo_tk"] = tk_img
                        cv.itemconfigure(t["img_id"], image=tk_img, state=tk.NORMAL)
                        cv.coords(t["img_id"], iw // 2, ih // 2)
                        cv.itemconfigure(t["ph_id"], state=tk.HIDDEN)
                        layer_dirty = True
                    except Exception:
                        pass
            else:
                if not is_cam_enabled:
                    sub_s = "stopped"
                    p_txt, p_col = tr("ui.main.tile.stopped", "Stopped"), "#9ba3b8"
                else:
                    st_map = {
                        "offline": (tr("ui.main.tile.offline", "No signal (offline)"), "#ff6666"),
                        "reconnecting": (tr("ui.main.tile.reconnecting", "Reconnecting sub-stream..."), "#ffcc00"),
                        "connecting": (tr("ui.main.tile.connecting_sub", "Connecting sub-stream..."), "#ffcc00"),
                        "fallback_main": (tr("ui.main.tile.waiting_main", "Waiting for main stream..."), "#ffcc00"),
                    }
                    sub_s = gs.get("sub_status", "")
                    p_txt, p_col = st_map.get(
                        sub_s,
                        (tr("ui.main.tile.starting", "Starting..."), "#9ba3b8"),
                    )
                ph_key = (sub_s, p_txt, p_col, iw, ih)
                if t.get("last_ph_key") != ph_key:
                    t["last_ph_key"] = ph_key
                    cv.itemconfigure(t["img_id"], state=tk.HIDDEN)
                    cv.itemconfigure(t["ph_id"], text=p_txt, fill=p_col, state=tk.NORMAL)
                    cv.coords(t["ph_id"], iw // 2, ih // 2)
                    layer_dirty = True

            if show_labels:
                cam_nm = str(gs.get("name") or "").strip() or tr("camera.default_single", "Camera")
                name_txt = f"📹 {cam_nm}"
                if uid in self.detail_windows:
                    name_txt += "  " + tr("ui.main.tile.open_tag", "[OPEN]")
                if t["last_lbl_text"] != name_txt:
                    t["last_lbl_text"] = name_txt
                    cv.itemconfigure(t["lbl_txt_id"], text=name_txt, state=tk.NORMAL)
                    cv.coords(t["lbl_txt_id"], 10, 8)
                    bbox = cv.bbox(t["lbl_txt_id"])
                    if bbox:
                        cv.coords(t["lbl_bg_id"], bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3)
                        cv.itemconfigure(t["lbl_bg_id"], state=tk.NORMAL)
                    layer_dirty = True
            else:
                if t["last_lbl_text"] != "__HIDDEN__":
                    t["last_lbl_text"] = "__HIDDEN__"
                    cv.itemconfigure(t["lbl_txt_id"], text="", state=tk.HIDDEN)
                    cv.itemconfigure(t["lbl_bg_id"], state=tk.HIDDEN)

            if show_status or show_fps:
                parts = []
                rec_on = bool(gs.get("recording"))
                op_mode = str(gs.get("operation_mode") or "server")
                if show_status:
                    if rec_on:
                        enc = gs.get("encoder") or "rec"
                        parts.append(f"● REC ({enc})")
                    elif op_mode == "operator":
                        parts.append(tr("ui.main.tile.no_rec_badge", "👤 NO RECORDING"))
                    m_on = gs.get("modules_on") or {}
                    m_tags = []
                    for ent in ai_entries:
                        mid = ent["id"]
                        if m_on.get(mid):
                            raw_badge = ent["spec"].get("short_badge")
                            if raw_badge != "":
                                default_badge = raw_badge if raw_badge is not None else mid[:2].upper()
                                badge = tr(f"mod.{mid}.short_badge", str(default_badge))
                                if badge:
                                    m_tags.append(str(badge))
                    if m_tags:
                        parts.append(" ".join(m_tags))
                if show_fps and gs.get("sub_status") == "active":
                    parts.append(f"{float(gs.get('fps', 0.0)):.1f} FPS")

                st_txt = "  |  ".join(parts)
                st_col = "#75e08b" if rec_on else ("#ffcc00" if op_mode == "operator" else "#9ee8ff")
                st_key = (st_txt, st_col, iw)
                if st_txt:
                    if t["last_st_key"] != st_key:
                        t["last_st_key"] = st_key
                        cv.itemconfigure(t["st_txt_id"], text=st_txt, fill=st_col, state=tk.NORMAL)
                        cv.coords(t["st_txt_id"], iw - 10, 8)
                        bbox = cv.bbox(t["st_txt_id"])
                        if bbox:
                            cv.coords(t["st_bg_id"], bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3)
                            cv.itemconfigure(t["st_bg_id"], state=tk.NORMAL)
                        layer_dirty = True
                else:
                    if t["last_st_key"] != ("__HIDDEN__",):
                        t["last_st_key"] = ("__HIDDEN__",)
                        cv.itemconfigure(t["st_txt_id"], text="", state=tk.HIDDEN)
                        cv.itemconfigure(t["st_bg_id"], state=tk.HIDDEN)
            else:
                if t["last_st_key"] != ("__HIDDEN__",):
                    t["last_st_key"] = ("__HIDDEN__",)
                    cv.itemconfigure(t["st_txt_id"], text="", state=tk.HIDDEN)
                    cv.itemconfigure(t["st_bg_id"], state=tk.HIDDEN)

            warn_txt = ""
            if rgb is not None and gs.get("sub_status") in ("offline", "reconnecting"):
                warn_txt = tr("ui.main.tile.stream_lost", "⚠ Stream lost (reconnecting...)")
            if t["last_warn_text"] != (warn_txt, ih):
                t["last_warn_text"] = (warn_txt, ih)
                if warn_txt:
                    cv.itemconfigure(t["warn_txt_id"], text=warn_txt, state=tk.NORMAL)
                    cv.coords(t["warn_txt_id"], 10, ih - 8)
                    bbox = cv.bbox(t["warn_txt_id"])
                    if bbox:
                        cv.coords(t["warn_bg_id"], bbox[0] - 5, bbox[1] - 3, bbox[2] + 5, bbox[3] + 3)
                        cv.itemconfigure(t["warn_bg_id"], state=tk.NORMAL)
                    layer_dirty = True
                else:
                    cv.itemconfigure(t["warn_txt_id"], text="", state=tk.HIDDEN)
                    cv.itemconfigure(t["warn_bg_id"], state=tk.HIDDEN)

            if layer_dirty:
                cv.tag_raise("ov_lbl")
                cv.tag_raise("ov_st")
                cv.tag_raise("ov_warn")

        if border_changed:
            self._restack_grid_highlights()

        return interval_ms


class SecondaryMonitorWindow(GridWindowBase):
    """Independent camera grid window for 2nd+ Monitors."""

    def __init__(self, parent_grid, monitor_id: str, monitor_name: str):
        self.parent_grid = parent_grid
        self.monitor_id = str(monitor_id)
        self.monitor_name = str(monitor_name or "Monitor")
        self.camera_manager = parent_grid.camera_manager
        self.registry = parent_grid.registry
        self.detail_windows = parent_grid.detail_windows
        self._is_closing = False

        self.win = tk.Toplevel(parent_grid.root)
        self.win.configure(bg="#0b0c10")
        self.win.title(f"{self.monitor_name} — {tr('ui.main.window_title', 'ANPR Multi-Cam NVR — Camera Grid')}")

        self._current_page = 0
        self._total_pages = 1
        self._tiles = {}
        self._grid_surface = None
        self._empty_slots = []
        self._last_grid_dims_key = None
        self._last_layout_key = None
        self._relayout_after_id = None
        self._hide_bar_after_id = None
        self._panel_visible = True
        self._last_total_cams = 0
        self._window_minimized = False
        self._visible_grid_uuids = set()

        self.grid_area = tk.Frame(self.win, bg="#0b0c10")
        self.grid_area.pack(fill=tk.BOTH, expand=True)

        self.ctrl_frame = tk.Frame(
            self.win, bg="#171923", padx=12, pady=7,
            highlightthickness=1, highlightbackground="#292d3e"
        )
        left_box = tk.Frame(self.ctrl_frame, bg="#171923")
        left_box.pack(side=tk.LEFT)
        self._left_box = left_box

        self.lbl_brand = tk.Label(
            left_box, text=f"🖥 {self.monitor_name}", fg="#ffffff", bg="#171923",
            font=("Segoe UI", 11, "bold")
        )
        self.lbl_brand.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_cam_count = tk.Label(
            left_box, text=tr("ui.main.cam_count", "Cameras: {n}", n=0), fg="#9ba3b8", bg="#171923",
            font=("Segoe UI", 9)
        )
        self.lbl_cam_count.pack(side=tk.LEFT, padx=(0, 10))

        self.page_box = tk.Frame(self.ctrl_frame, bg="#171923")
        self.btn_prev_page = tk.Button(
            self.page_box, text="◀", command=lambda: self._change_page(-1),
            bg="#2b2f40", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, pady=2, cursor="hand2"
        )
        self.btn_prev_page.pack(side=tk.LEFT, padx=(0, 4))

        self.lbl_page = tk.Label(
            self.page_box, text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=1, total=1), fg="#d5dae8", bg="#171923",
            font=("Segoe UI", 9, "bold"), padx=6
        )
        self.lbl_page.pack(side=tk.LEFT)

        self.btn_next_page = tk.Button(
            self.page_box, text="▶", command=lambda: self._change_page(1),
            bg="#2b2f40", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, pady=2, cursor="hand2"
        )
        self.btn_next_page.pack(side=tk.LEFT, padx=(4, 0))

        self._top_drag_widgets = (self.ctrl_frame, left_box, self.lbl_brand, self.lbl_cam_count, self.page_box, self.lbl_page)
        self._sync_top_bar_for_mode()
        if self._can_show_top_bar():
            self._show_top_bar()
        else:
            self._panel_visible = False

        self._borderless_ctl = apply_borderless_window_style(self.win)
        if self._borderless_ctl is not None:
            self._borderless_ctl.bind_drag_widgets(
                self._top_drag_widgets,
                on_interaction=self._on_top_bar_interact,
            )

        self.geom_tracker = WindowGeometryTracker(
            self.win,
            get_geom_cb=lambda mid=self.monitor_id: self.parent_grid.cfg_mgr.get_monitor_window_geometry(mid),
            set_geom_cb=lambda g, mid=self.monitor_id: self.parent_grid.cfg_mgr.set_monitor_window_geometry(mid, g),
            min_w=480,
            min_h=320,
            default_geom={"left": 120, "top": 80, "width": 1160, "height": 700, "maximized": False},
        )

        self.win.bind(EVT_MOTION, self._on_mouse_motion, add="+")
        self.win.bind(EVT_CONFIGURE, self._on_configure, add="+")
        self.win.bind("<Unmap>", self._on_unmap, add="+")
        self.win.bind("<Map>", self._on_map, add="+")
        self.win.protocol("WM_DELETE_WINDOW", self._on_user_close)
        self._schedule_hide_bar(5000)

    def _fast_cfg(self) -> dict:
        return self.parent_grid._fast_cfg()

    def _get_grid_cfg(self) -> dict:
        snap = self._fast_cfg()
        for m in (snap.get("camera_monitors") or []):
            if isinstance(m, dict) and str(m.get("id")) == self.monitor_id:
                if isinstance(m.get("grid"), dict):
                    return m["grid"]
                break
        return super()._get_grid_cfg()

    def open_camera_detail(self, cam_uuid: str):
        self.parent_grid.open_camera_detail(cam_uuid)

    def _notify_visible_changed(self):
        self.parent_grid._sync_visible_grid_uuids()

    def get_visible_uuids(self) -> set:
        if self._is_closing or self._window_minimized:
            return set()
        return set(self._visible_grid_uuids)

    def set_monitor_name(self, name: str):
        self.monitor_name = str(name or "Monitor")
        try:
            self.lbl_brand.configure(text=f"🖥 {self.monitor_name}")
            self.win.title(f"{self.monitor_name} — {tr('ui.main.window_title', 'ANPR Multi-Cam NVR — Camera Grid')}")
        except Exception:
            pass

    def apply_language(self):
        self.set_monitor_name(self.monitor_name)
        self.lbl_cam_count.configure(text=tr("ui.main.cam_count", "Cameras: {n}", n=self._last_total_cams))
        self.lbl_page.configure(text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=self._current_page + 1, total=self._total_pages))
        self._last_layout_key = None
        self._relayout_grid(force=True)

    def _show_top_bar(self):
        if self._is_closing or not self._can_show_top_bar():
            return
        self._sync_top_bar_for_mode()
        if not self._panel_visible or not self.ctrl_frame.winfo_manager():
            self.ctrl_frame.place(x=0, y=0, relwidth=1.0)
            self.ctrl_frame.lift()
            self._panel_visible = True

    def _hide_top_bar(self):
        self._hide_bar_after_id = None
        if self._is_closing:
            return
        try:
            px, py = self.win.winfo_pointerxy()
            rx = self.win.winfo_rootx()
            ry = self.win.winfo_rooty()
            rw = self.win.winfo_width()
            if rx <= px <= rx + rw and ry <= py <= ry + 64:
                self._schedule_hide_bar(2000)
                return
        except Exception:
            pass
        if self._panel_visible:
            self.ctrl_frame.place_forget()
            self._panel_visible = False

    def _schedule_hide_bar(self, delay_ms: int = 3200):
        if self._hide_bar_after_id is not None:
            try:
                self.win.after_cancel(self._hide_bar_after_id)
            except Exception:
                pass
            self._hide_bar_after_id = None
        if not self._is_closing:
            self._hide_bar_after_id = self.win.after(delay_ms, self._hide_top_bar)

    def _on_top_bar_interact(self):
        self._show_top_bar()
        if self._can_show_top_bar():
            self._schedule_hide_bar(3800)

    def _on_mouse_motion(self, event=None):
        if self._is_closing or not self._can_show_top_bar():
            return
        try:
            py = self.win.winfo_pointery() - self.win.winfo_rooty()
        except Exception:
            return
        if py <= 54:
            self._show_top_bar()
            self._schedule_hide_bar(3200)
        elif self._panel_visible and self._hide_bar_after_id is None:
            self._schedule_hide_bar(2500)

    def _on_unmap(self, event=None):
        if self._is_closing or (event is not None and event.widget is not self.win):
            return
        try:
            if str(self.win.state()).lower() not in ("iconic", "withdrawn"):
                return
        except Exception:
            pass
        if not self._window_minimized:
            self._window_minimized = True
            self._notify_visible_changed()

    def _on_map(self, event=None):
        if self._is_closing or (event is not None and event.widget is not self.win):
            return
        if self._window_minimized:
            self._window_minimized = False
            self._notify_visible_changed()
            try:
                self.render_tiles_once()
            except Exception:
                pass

    def _on_configure(self, event=None):
        if self._is_closing:
            return
        if event is not None and event.widget not in (self.win, self.grid_area):
            return
        if self._relayout_after_id is not None:
            try:
                self.win.after_cancel(self._relayout_after_id)
            except Exception:
                pass
        self._relayout_after_id = self.win.after(50, lambda: self._relayout_grid(force=False))

    def _change_page(self, delta: int):
        new_p = max(0, min(max(0, self._total_pages - 1), self._current_page + delta))
        if new_p != self._current_page:
            self._current_page = new_p
            self._relayout_grid(force=True)

    def _relayout_grid(self, force: bool = False):
        units = self.parent_grid._get_units_for_monitor(self.monitor_id)
        self._relayout_grid_with_units(units, force=force)

    def render_tiles_once(self):
        units = self.parent_grid._get_units_for_monitor(self.monitor_id)
        return self._render_tiles_with_units(units)

    def _on_user_close(self):
        try:
            snap = self._fast_cfg()
            if bool((snap.get("general") or {}).get("kiosk_mode", False)):
                return
        except Exception:
            pass
        try:
            self.win.iconify()
        except Exception:
            pass

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        for attr in ("_relayout_after_id", "_hide_bar_after_id"):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.win.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)
        try:
            self.geom_tracker.close()
        except Exception:
            pass
        self._clear_tiles()
        try:
            self.win.destroy()
        except Exception:
            pass


class MainGridWindow(GridWindowBase):
    """
    Головне вікно системи: адаптивна або ручна сітка перегляду стиснутих потоків (sub_source)
    усіх камер. Лівий клік по плитці відкриває незалежне вікно повного перегляду (main_source + архів)
    конкретної камери. Фоновий запис та AI-аналіз усіх камер працюють безперервно незалежно від вікон,
    а незалежні модулі автоматично підхоплюються через ModuleRegistry.
    """
    def __init__(self, root: tk.Tk, base_dir: str = None, registry=None, cfg_mgr=None):
        self.root = root
        self._manage_os_startup = (base_dir is None)
        self.base_dir = os.path.abspath(base_dir or BASE_DIR)
        if self._manage_os_startup:
            try:
                os.chdir(self.base_dir)
            except Exception:
                pass

        self.root.configure(bg="#0b0c10")

        self._is_closing = False
        self._startup_ts = time.time()

        # 1. Автоматичне виявлення незалежних модулів (scripts/modules/), конфіг і локалізація
        try:
            prepare_ort_backend_environment(self.base_dir)
            prepare_cuda_dll_environment(self.base_dir, preload=False)
        except Exception:
            pass
        self.registry = registry if registry is not None else ModuleRegistry(self.base_dir)
        self.cfg_mgr = cfg_mgr if cfg_mgr is not None else ConfigManager(self.base_dir, registry=self.registry)
        if self.cfg_mgr.is_locked():
            raise PermissionError("Configuration is locked. Enter access password before starting MainGridWindow.")
        self.cfg_mgr.sync_with_registry(self.registry)
        init_i18n(self.base_dir, lang_code=self.cfg_mgr.get_language())
        self.root.title(tr("ui.main.window_title", "ANPR Multi-Cam NVR — Camera Grid"))

        snap = self.cfg_mgr.get_snapshot()
        gen_cfg = snap.get("general") or {}
        op_mode = str(gen_cfg.get("operation_mode", "server")).strip().lower()
        if op_mode not in ("server", "operator"):
            op_mode = "server"
        allow_write = (op_mode == "server")
        hw_cfg = snap.get("hardware") or {}

        # 2. Телеметрія заліза, спільний пул ресурсів та менеджер камер
        self.hw_monitor = HardwareMonitor(self.base_dir)
        get_plates_fn = getattr(self.cfg_mgr, "get_plates_db_path", None)
        if callable(get_plates_fn):
            plates_db_path = get_plates_fn(create=allow_write)
        else:
            plates_db_path = os.path.join(self.cfg_mgr.get_archive_root(), "plates_db", "plates.json")
        self.inference_pool = InferencePool(
            self.base_dir,
            self.registry,
            app_context={
                "operation_mode": op_mode,
                "allow_disk_write": allow_write,
                "faces_db_path": self.cfg_mgr.get_faces_db_path(create=allow_write),
                "plates_db_path": plates_db_path,
            },
            ai_backend=hw_cfg.get("ai_backend", "cpu"),
            ai_threads=int(hw_cfg.get("ai_threads", 2)),
        )
        self.camera_manager = CameraManager(
            self.cfg_mgr,
            self.inference_pool,
            self.registry,
            on_saved_event=self._on_camera_saved_cb,
        )

        # Відкриті дочірні вікна
        self.settings_win = None
        self.people_win = None
        self.plates_win = None
        self.detail_windows = {}  # cam_uuid -> CameraDetailWindow
        self.monitor_windows = {}  # monitor_id -> SecondaryMonitorWindow

        # Стан сітки та пагінації
        self._current_page = 0
        self._total_pages = 1
        self._tiles = {}          # uuid -> dict widget refs & cache
        self._grid_surface = None
        self._empty_slots = []
        self._last_grid_dims_key = None
        self._last_layout_key = None
        self._relayout_after_id = None
        self._render_after_id = None
        self._hw_after_id = None
        self._hide_bar_after_id = None
        self._module_poll_after_id = None
        self._start_after_id = None
        self._ui_poll_after_id = None
        self._cuda_note_clear_after_id = None
        self._panel_visible = True
        self._last_total_cams = 0
        self._window_minimized = False
        self._visible_grid_uuids = set()

        # Потокобезпечна черга UI-викликів із фонових потоків
        self._ui_queue = queue.SimpleQueue()
        self._cuda_ensure_lock = threading.Lock()
        self._cuda_downloading = False
        self._cuda_status_note = ""
        self._cuda_note_color = "#ffcc00"

        # Кеш фонового збору телеметрії (щоб не блокувати UI-потік)
        self._cached_hw_metrics = None
        self._hw_poll_running = True

        # Синхронізація автозапуску та режиму кіоску
        self._sync_windows_startup(bool(gen_cfg.get("autostart_os_and_play", False)))
        self._apply_kiosk_mode(bool(gen_cfg.get("kiosk_mode", False)))
        self.root.bind_all(EVT_ADMIN_EXIT_UP, self._admin_exit)
        self.root.bind_all(EVT_ADMIN_EXIT_LOW, self._admin_exit)
        self.root.bind_all("<KeyPress>", self._on_global_keypress, add="+")

        self._build_ui()

        # Прибираємо стандартний заголовок Windows лише для головного вікна (зі збереженням Taskbar і rcWork)
        self._borderless_ctl = apply_borderless_window_style(self.root)
        if self._borderless_ctl is not None:
            self._borderless_ctl.bind_drag_widgets(
                getattr(self, "_top_drag_widgets", ()),
                on_interaction=self._on_top_bar_interact,
            )

        # Відновлення й автозбереження позиції вікна на будь-якому моніторі
        self.geom_tracker = WindowGeometryTracker(
            self.root,
            get_geom_cb=lambda: self.cfg_mgr.get_window_geometry("grid"),
            set_geom_cb=lambda g: self.cfg_mgr.set_window_geometry("grid", g),
            min_w=640,
            min_h=400,
            default_geom={"left": 40, "top": 30, "width": 1280, "height": 760, "maximized": False},
        )

        self.root.bind(EVT_MOTION, self._on_mouse_motion, add="+")
        self.root.bind(EVT_CONFIGURE, self._on_root_configure, add="+")
        self.root.bind("<Unmap>", self._on_root_unmap, add="+")
        self.root.bind("<Map>", self._on_root_map, add="+")

        self._drain_ui_queue()
        self._start_hw_poll_thread()
        self._schedule_hide_bar(5000)

        self._start_after_id = self.root.after(120, self._start_system)

    # ------------------------------------------------------------------
    # Ініціалізація UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_area = tk.Frame(self.root, bg="#0b0c10")
        self.grid_area.pack(fill=tk.BOTH, expand=True)

        # Автоприховувана верхня панель (поверх сітки без зміщення її висоти)
        self.ctrl_frame = tk.Frame(
            self.root, bg="#171923", padx=12, pady=7,
            highlightthickness=1, highlightbackground="#292d3e"
        )

        left_box = tk.Frame(self.ctrl_frame, bg="#171923")
        left_box.pack(side=tk.LEFT)
        self._left_box = left_box

        lbl_brand = tk.Label(
            left_box, text="📹 ANPR Multi-Cam NVR", fg="#ffffff", bg="#171923",
            font=("Segoe UI", 10, "bold")
        )
        lbl_brand.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_cam_count = tk.Label(
            left_box, text=tr("ui.main.cam_count", "Cameras: {n}", n=0), fg="#00e5ff", bg="#222534",
            font=("Segoe UI", 8, "bold"), padx=8, pady=3
        )
        self.lbl_cam_count.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_mode_badge = tk.Label(
            left_box, text="", fg="#75e08b", bg="#1b2e24",
            font=("Segoe UI", 8, "bold"), padx=8, pady=3
        )
        self.lbl_mode_badge.pack(side=tk.LEFT, padx=(0, 10))
        self._refresh_mode_badge()

        self.btn_settings = tk.Button(
            left_box, text=tr("ui.main.btn_settings", "⚙ Settings"), command=self.open_settings,
            bg="#007acc", fg="#ffffff", activebackground="#0066ad", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_settings.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_people = tk.Button(
            left_box, text=tr("ui.main.btn_people", "👥 People"), command=self.open_people,
            bg="#254c5a", fg="#ffffff", activebackground="#2f6375", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_people.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_plates = tk.Button(
            left_box, text=tr("ui.main.btn_plates", "🚘 Plates"), command=self.open_plates,
            bg="#27493d", fg="#ffffff", activebackground="#336151", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_plates.pack(side=tk.LEFT, padx=(0, 8))
        self._op_hidden_left = (lbl_brand, self.lbl_cam_count, self.lbl_mode_badge, self.btn_people, self.btn_plates)

        # Блок пагінації (видимий лише коли в ручній сітці > 1 сторінки)
        self.page_box = tk.Frame(self.ctrl_frame, bg="#171923")
        self.btn_prev_page = tk.Button(
            self.page_box, text="◀", command=lambda: self._change_page(-1),
            bg="#2b2f40", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, pady=2, cursor="hand2"
        )
        self.btn_prev_page.pack(side=tk.LEFT, padx=(0, 4))

        self.lbl_page = tk.Label(
            self.page_box, text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=1, total=1), fg="#d5dae8", bg="#171923",
            font=("Segoe UI", 9, "bold"), padx=6
        )
        self.lbl_page.pack(side=tk.LEFT)

        self.btn_next_page = tk.Button(
            self.page_box, text="▶", command=lambda: self._change_page(1),
            bg="#2b2f40", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, pady=2, cursor="hand2"
        )
        self.btn_next_page.pack(side=tk.LEFT, padx=(4, 0))

        # Правий блок: телеметрія заліза + вихід
        right_box = tk.Frame(self.ctrl_frame, bg="#171923")
        right_box.pack(side=tk.RIGHT)
        self._right_box = right_box

        self.lbl_hw_badge = tk.Label(
            right_box, text=tr("ui.main.hw_badge_init", "CPU: --% | RAM: -- | AI: CPU"),
            fg="#9ee8ff", bg="#222534", font=("Segoe UI", 8, "bold"),
            padx=10, pady=4
        )
        self.lbl_hw_badge.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_exit = tk.Button(
            right_box, text=tr("ui.main.btn_exit", "🚪 Exit"), command=self.on_close,
            bg="#64262b", fg="#ffffff", activebackground="#7e2f36", activeforeground="#ffffff",
            relief=tk.FLAT, font=("Segoe UI", 9, "bold"), padx=12, pady=4, cursor="hand2"
        )
        self.btn_exit.pack(side=tk.LEFT)
        self._apply_exit_btn_text()

        self._top_drag_widgets = (
            self.ctrl_frame,
            left_box,
            lbl_brand,
            self.lbl_cam_count,
            self.lbl_mode_badge,
            self.page_box,
            self.lbl_page,
            right_box,
            self.lbl_hw_badge,
        )

        self._sync_top_bar_for_mode()
        if self._can_show_top_bar():
            self._show_top_bar()
        else:
            self._panel_visible = False

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
                if self.root.winfo_exists():
                    self._ui_poll_after_id = self.root.after(40, self._drain_ui_queue)
            except Exception:
                pass

    def _start_system(self):
        self._start_after_id = None
        if self._is_closing:
            return
        self.camera_manager.sync_from_config()
        self._sync_monitor_windows()
        self._relayout_grid(force=True)
        self.camera_manager.start_all()
        self._render_grid_tick()
        self._refresh_hw_badge_tick()
        self._module_poll_after_id = self.root.after(3000, self._poll_modules_tick)
        self._maybe_ensure_cuda_async()
        if getattr(self.cfg_mgr, "is_first_run", False):
            self.cfg_mgr.is_first_run = False
            self.root.after(200, lambda: self.open_settings(initial_node="cameras_root"))

    def _set_cuda_note(self, text: str, color: str = "#ffcc00", clear_after_ms: int = 0):
        if self._is_closing:
            return
        if self._cuda_note_clear_after_id is not None:
            try:
                self.root.after_cancel(self._cuda_note_clear_after_id)
            except Exception:
                pass
            self._cuda_note_clear_after_id = None
        self._cuda_status_note = str(text or "")
        self._cuda_note_color = str(color or "#ffcc00")
        if self._cuda_status_note:
            try:
                self.lbl_hw_badge.configure(text=self._cuda_status_note, fg=self._cuda_note_color)
            except Exception:
                pass
            if clear_after_ms > 0:
                self._cuda_note_clear_after_id = self.root.after(int(clear_after_ms), self._clear_cuda_note)
        else:
            try:
                self.lbl_hw_badge.configure(fg="#9ee8ff")
            except Exception:
                pass

    def _clear_cuda_note(self):
        self._cuda_note_clear_after_id = None
        self._cuda_status_note = ""
        self._cuda_note_color = "#9ee8ff"
        if not self._is_closing:
            try:
                self.lbl_hw_badge.configure(fg="#9ee8ff")
            except Exception:
                pass

    def _maybe_ensure_cuda_async(self):
        """
        Автоматично перевіряє наявність CUDA / cuDNN бібліотек при вибраному GPU-режимі ('cuda' або 'auto' з onnxruntime-gpu).
        Якщо потрібних DLL немає у системі та в <BASE_DIR>/cuda/, фоново завантажує їх у локальну папку cuda/.
        """
        if self._is_closing:
            return
        hw_cfg = self.cfg_mgr.get_snapshot().get("hardware") or {}
        backend = str(hw_cfg.get("ai_backend", "cpu")).strip().lower()
        gpu_ort = is_onnxruntime_gpu_installed()
        if not (backend == "cuda" or (backend == "auto" and gpu_ort)):
            return

        with self._cuda_ensure_lock:
            if self._cuda_downloading:
                return
            self._cuda_downloading = True

        def _worker():
            try:
                if self._is_closing or not has_nvidia_gpu():
                    return
                st = check_cuda_libraries_status(self.base_dir, allow_network=False)
                if st.get("ready"):
                    prepare_cuda_dll_environment(self.base_dir, preload=True)
                    return

                self._ui_queue.put((self._set_cuda_note, (tr("ui.main.cuda_dl_start", "⬇ Checking and downloading CUDA to cuda/..."), "#ffcc00", 0)))
                last_msg_ts = [0.0]

                def _prog(msg):
                    if self._is_closing:
                        return
                    now_t = time.time()
                    if now_t - last_msg_ts[0] < 0.35:
                        return
                    last_msg_ts[0] = now_t
                    short_m = f"⬇ CUDA: {msg[:60]}"
                    self._ui_queue.put((self._set_cuda_note, (short_m, "#ffcc00", 0)))

                res = download_cuda_libraries(
                    self.base_dir,
                    progress_cb=_prog,
                    include_optional=True,
                    force_into_local=False,
                )
                if self._is_closing:
                    return

                prepare_cuda_dll_environment(self.base_dir, preload=True)
                caps = self.hw_monitor.probe_acceleration_capabilities(force=True)
                cuda_ok = bool(((caps.get("ai_backends") or {}).get("cuda") or {}).get("supported"))
                gpu_pkg_now = is_onnxruntime_gpu_installed()

                def _on_done():
                    if self._is_closing:
                        return
                    if res.get("ok"):
                        try:
                            ai_ids = [ent["id"] for ent in self.registry.list_modules(kind="ai")]
                            self.inference_pool.invalidate_all()
                            self.camera_manager.sync_from_config()
                            for u in list(self.camera_manager.units.values()):
                                u.force_reload_modules(ai_ids)
                        except Exception as e:
                            print(f"[CUDA hot-reload error]: {e}")

                        if self.settings_win is not None:
                            try:
                                if self.settings_win.win.winfo_exists():
                                    self.settings_win.on_cuda_libraries_updated(caps=caps)
                            except Exception:
                                pass

                        if cuda_ok:
                            self._set_cuda_note(tr("ui.main.cuda_ok_active", "✔ CUDA libraries downloaded to cuda/ — GPU activated!"), "#75e08b", 8000)
                        elif not gpu_pkg_now:
                            self._set_cuda_note(tr("ui.main.cuda_ok_need_pkg", "✔ CUDA DLLs downloaded to cuda/ (onnxruntime-gpu required)"), "#ffcc00", 12000)
                        else:
                            self._set_cuda_note(tr("ui.main.cuda_ok_restart", "✔ CUDA DLLs in cuda/ — restart to activate GPU"), "#ffcc00", 12000)
                    else:
                        if self.settings_win is not None:
                            try:
                                if self.settings_win.win.winfo_exists():
                                    self.settings_win.on_cuda_libraries_updated(caps=caps)
                            except Exception:
                                pass
                        self._set_cuda_note(tr("ui.main.cuda_dl_fail", "⚠ Failed to download CUDA DLLs (CPU fallback active)"), "#ff8888", 10000)

                self._ui_queue.put((_on_done, ()))
            except Exception as e:
                print(f"[CUDA auto-ensure error]: {e}")
            finally:
                with self._cuda_ensure_lock:
                    self._cuda_downloading = False

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Автоматичне виявлення нових/оновлених модулів і бібліотек
    # ------------------------------------------------------------------
    def refresh_modules_and_libraries(self, force: bool = True) -> dict:
        if self._is_closing:
            return {"added": [], "updated": [], "removed": [], "errors": {}, "changed": False}
        res = self.registry.scan_and_load(force=force)
        if force:
            try:
                self.hw_monitor.refresh_all(force=True)
            except Exception:
                pass
        return self._apply_registry_update(res, force_reload_all=force)

    def _apply_registry_update(self, res: dict, force_reload_all: bool = False) -> dict:
        if self._is_closing:
            return res
        changed = bool(res and res.get("changed"))
        if changed or force_reload_all:
            try:
                self.registry.refresh_libraries_capabilities()
            except Exception:
                pass
            self.cfg_mgr.sync_with_registry(self.registry)
            self._refresh_mode_badge()

            added = list(res.get("added") or [])
            updated = list(res.get("updated") or [])
            removed = list(res.get("removed") or [])
            try:
                self.camera_manager.reload_from_registry(
                    changed_mod_ids=added + updated,
                    removed_mod_ids=removed,
                    force_all=force_reload_all,
                )
            except Exception as e:
                print(f"[MainGridWindow reload_from_registry]: {e}")

            if self.settings_win is not None:
                try:
                    if self.settings_win.win.winfo_exists():
                        self.settings_win.on_modules_reloaded()
                except Exception:
                    pass

            if self.people_win is not None:
                try:
                    if self.people_win.win.winfo_exists():
                        self.people_win.on_modules_reloaded()
                except Exception:
                    pass

            if self.plates_win is not None:
                try:
                    if self.plates_win.win.winfo_exists():
                        self.plates_win.on_modules_reloaded()
                except Exception:
                    pass

            for dwin in list(self.detail_windows.values()):
                try:
                    if dwin.win.winfo_exists():
                        dwin.on_modules_reloaded()
                except Exception:
                    pass

            self._last_layout_key = None
            self._relayout_grid(force=True)
        return res

    def _poll_modules_tick(self):
        self._module_poll_after_id = None
        if self._is_closing:
            return
        try:
            res = self.registry.check_for_updates()
            if res and res.get("changed"):
                self._apply_registry_update(res, force_reload_all=False)
        except Exception as e:
            print(f"[ModulePoll error]: {e}")
        if not self._is_closing:
            self._module_poll_after_id = self.root.after(3000, self._poll_modules_tick)

    # ------------------------------------------------------------------
    # Реєстр автозапуску Windows, перезапуск і режим кіоску
    # ------------------------------------------------------------------
    def _sync_windows_startup(self, enable: bool):
        if not self._manage_os_startup or winreg is None or sys.platform != "win32":
            return
        app_name = "ANPR_Video_Monitor"
        run_key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        if getattr(sys, "frozen", False):
            launch_cmd = f'"{os.path.abspath(sys.executable)}"'
        else:
            py_exe = sys.executable
            pyw = os.path.join(os.path.dirname(py_exe), "pythonw.exe")
            if os.path.exists(pyw):
                py_exe = pyw
            entry_py = os.path.abspath(os.path.join(self.base_dir, "app.py"))
            launch_cmd = f'"{py_exe}" "{entry_py}"'

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key_path, 0, winreg.KEY_SET_VALUE) as key:
                if enable:
                    winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, launch_cmd)
                else:
                    try:
                        winreg.DeleteValue(key, app_name)
                    except FileNotFoundError:
                        pass
        except Exception:
            pass

    def restart_application(self):
        if self._is_closing:
            return
        if getattr(sys, "frozen", False):
            cmd = [os.path.abspath(sys.executable)]
        else:
            py_exe = sys.executable
            pyw = os.path.join(os.path.dirname(py_exe), "pythonw.exe")
            if os.path.basename(py_exe).lower() == "pythonw.exe" and os.path.exists(pyw):
                py_exe = pyw
            cmd = [py_exe, os.path.abspath(os.path.join(self.base_dir, "app.py"))]

        self.on_close(exit_app=False)
        try:
            subprocess.Popen(cmd, cwd=self.base_dir, close_fds=True)
        except Exception as e:
            print(f"[Restart error]: {e}")
        sys.exit(0)

    def _apply_exit_btn_text(self):
        if hasattr(self, "btn_exit"):
            if getattr(self, "_kiosk_mode", False):
                self.btn_exit.configure(
                    text=tr("ui.main.btn_kiosk", "🔒 Kiosk (Ctrl+Alt+Shift+Q)"),
                    bg="#2e3142",
                    fg="#9ba3b8",
                    command=self._kiosk_hint,
                )
            else:
                self.btn_exit.configure(
                    text=tr("ui.main.btn_exit", "🚪 Exit"),
                    bg="#64262b",
                    fg="#ffffff",
                    command=self.on_close,
                )

    def _apply_kiosk_mode(self, kiosk_on: bool):
        self._kiosk_mode = bool(kiosk_on)
        if self._kiosk_mode:
            self.root.protocol("WM_DELETE_WINDOW", self._prevent_close)
            for key in LOCKDOWN_KEYS:
                self.root.bind(key, lambda e: "break")
        else:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            for key in LOCKDOWN_KEYS:
                try:
                    self.root.unbind(key)
                except Exception:
                    pass

        self._apply_exit_btn_text()

    def _prevent_close(self):
        pass

    def _kiosk_hint(self):
        messagebox.showinfo(
            tr("ui.main.kiosk_hint_title", "Kiosk Mode"),
            tr(
                "ui.main.kiosk_hint_msg",
                "Exit lock is enabled.\nTo exit the application, press Ctrl+Alt+Shift+Q or disable Kiosk Mode in Settings.",
            ),
            parent=self.root,
        )

    def _on_global_keypress(self, event=None):
        if self._is_closing:
            return
        if is_admin_exit_keypress(event):
            self._admin_exit(event)
            return "break"

    def _admin_exit(self, event=None):
        self.on_close()

    # ------------------------------------------------------------------
    # Автоприховувана верхня панель
    # ------------------------------------------------------------------
    def apply_language(self):
        if self._is_closing:
            return
        try:
            self.root.title(tr("ui.main.window_title", "ANPR Multi-Cam NVR — Camera Grid"))
            self.lbl_cam_count.configure(text=tr("ui.main.cam_count", "Cameras: {n}", n=getattr(self, "_last_total_cams", 0)))
            self._refresh_mode_badge()
            self.btn_settings.configure(text=tr("ui.main.btn_settings", "⚙ Settings"))
            self.btn_people.configure(text=tr("ui.main.btn_people", "👥 People"))
            self.btn_plates.configure(text=tr("ui.main.btn_plates", "🚘 Plates"))
            self._apply_exit_btn_text()
            self.lbl_page.configure(
                text=tr("ui.main.page_fmt", "Page {cur} / {total}", cur=self._current_page + 1, total=self._total_pages)
            )
            if hasattr(self, "_empty_lbl") and self._empty_lbl.winfo_exists():
                self._empty_lbl.configure(
                    text=tr(
                        "ui.main.empty_cams",
                        "No enabled cameras.\nOpen ⚙ Settings above to add or enable cameras.",
                    )
                )
            self._render_hw_badge_once()
            for t in self._tiles.values():
                t["last_lbl_text"] = None
                t["last_st_key"] = None
                t["last_warn_text"] = None
            self._last_layout_key = None
            self._relayout_grid(force=True)
            self._render_tiles_once()
        except Exception:
            pass

    def change_language(self, lang_code: str, source: str = "settings") -> str:
        saved_code = self.cfg_mgr.set_language(lang_code)
        eff_code = set_language(saved_code)
        try:
            self.hw_monitor.relocalize_cached_caps()
        except Exception:
            pass

        self.apply_language()

        if self.settings_win is not None:
            try:
                if self.settings_win.win.winfo_exists() and not getattr(self.settings_win, "_is_closing", False):
                    if source != "settings":
                        self.settings_win.on_external_language_changed(eff_code)
                    else:
                        self.settings_win.apply_language()
            except Exception:
                pass

        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.apply_language()
            except Exception:
                pass

        if self.plates_win is not None:
            try:
                if self.plates_win.win.winfo_exists():
                    self.plates_win.apply_language()
            except Exception:
                pass

        for dwin in list(self.detail_windows.values()):
            try:
                if dwin.win.winfo_exists():
                    dwin.apply_language()
            except Exception:
                pass

        for mwin in list(self.monitor_windows.values()):
            try:
                if mwin.win.winfo_exists():
                    mwin.apply_language()
            except Exception:
                pass

        return eff_code

    def _show_top_bar(self):
        if self._is_closing or not self._can_show_top_bar():
            return
        self._sync_top_bar_for_mode()
        if not self._panel_visible or not self.ctrl_frame.winfo_manager():
            self.ctrl_frame.place(x=0, y=0, relwidth=1.0)
            self.ctrl_frame.lift()
            self._panel_visible = True

    def _hide_top_bar(self):
        self._hide_bar_after_id = None
        if self._is_closing:
            return
        try:
            px, py = self.root.winfo_pointerxy()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            bh = max(44, self.ctrl_frame.winfo_height())
            if rx <= px <= (rx + self.root.winfo_width()) and ry <= py <= (ry + bh + 4):
                self._schedule_hide_bar(2500)
                return
        except Exception:
            pass
        if self._panel_visible:
            self.ctrl_frame.place_forget()
            self._panel_visible = False

    def _schedule_hide_bar(self, ms=5000):
        if self._hide_bar_after_id is not None:
            try:
                self.root.after_cancel(self._hide_bar_after_id)
            except Exception:
                pass
        self._hide_bar_after_id = self.root.after(ms, self._hide_top_bar)

    def _on_top_bar_interact(self):
        if self._is_closing:
            return
        self._show_top_bar()
        if self._can_show_top_bar():
            self._schedule_hide_bar(5000)

    def _on_mouse_motion(self, event=None):
        if self._is_closing or not self._can_show_top_bar():
            return
        try:
            ry = self.root.winfo_rooty()
            px_y = event.y_root if event is not None else self.root.winfo_pointery()
            local_y = px_y - ry
            if local_y <= 34:
                self._show_top_bar()
                self._schedule_hide_bar(5000)
            elif self._panel_visible:
                self._schedule_hide_bar(5000)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Керування незалежними вікнами (Налаштування + Детальний перегляд камер)
    # ------------------------------------------------------------------
    def open_settings(self, initial_node: str = None):
        if self._is_closing:
            return
        if self.settings_win is not None:
            try:
                if self.settings_win.win.winfo_exists():
                    self.settings_win.focus()
                    return
            except Exception:
                pass
            self.settings_win = None

        if not self._prompt_settings_password():
            return

        self.settings_win = SettingsWindow(
            self.root,
            cfg_mgr=self.cfg_mgr,
            hw_monitor=self.hw_monitor,
            camera_manager=self.camera_manager,
            registry=self.registry,
            on_apply_cb=self.on_settings_applied,
            on_close_cb=self._on_settings_closed,
            on_refresh_modules_cb=self.refresh_modules_and_libraries,
            on_restart_app_cb=self.restart_application,
        )
        if initial_node and hasattr(self.settings_win, "_select_nav_node"):
            try:
                self.settings_win._select_nav_node(initial_node)
            except Exception:
                pass

    def _prompt_settings_password(self) -> bool:
        """Запитує пароль доступу перед відкриттям налаштувань, якщо пароль задано."""
        try:
            if not self.cfg_mgr.has_password():
                return True
        except Exception:
            return True

        result = {"ok": False}
        dlg = tk.Toplevel(self.root)
        dlg.title(tr("ui.main.btn_settings", "Settings"))
        dlg.configure(bg="#181818")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        w, h = 380, 185
        try:
            self.root.update_idletasks()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            px = rx + max(0, (rw - w) // 2)
            py = ry + max(0, (rh - h) // 2)
        except Exception:
            px = max(0, (dlg.winfo_screenwidth() - w) // 2)
            py = max(0, (dlg.winfo_screenheight() - h) // 2)
        dlg.geometry(f"{w}x{h}+{px}+{py}")

        frm = tk.Frame(dlg, bg="#181818", padx=18, pady=16)
        frm.pack(fill="both", expand=True)

        tk.Label(
            frm,
            text=tr("ui.auth.pwd_label", "Password:"),
            bg="#181818",
            fg="#F3F3F3",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        pwd_var = tk.StringVar()
        err_var = tk.StringVar()

        entry_row = tk.Frame(frm, bg="#181818")
        entry_row.pack(fill="x")

        ent = tk.Entry(
            entry_row,
            textvariable=pwd_var,
            show="●",
            bg="#242424",
            fg="#F3F3F3",
            insertbackground="#F3F3F3",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#3A3A3A",
            highlightcolor="#0078D4",
            font=("Segoe UI", 10),
        )
        ent.pack(side="left", fill="x", expand=True, ipady=5)

        show_state = {"visible": False}

        def _toggle_show():
            show_state["visible"] = not show_state["visible"]
            ent.configure(show="" if show_state["visible"] else "●")

        tk.Button(
            entry_row,
            text="👁",
            command=_toggle_show,
            bg="#2A2A2A",
            fg="#D8D8D8",
            activebackground="#3A3A3A",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            padx=8,
            cursor="hand2",
        ).pack(side="left", padx=(6, 0), ipady=3)

        tk.Label(
            frm,
            textvariable=err_var,
            bg="#181818",
            fg="#F48771",
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(fill="x", pady=(6, 8))

        btn_row = tk.Frame(frm, bg="#181818")
        btn_row.pack(fill="x")

        def _submit(_evt=None):
            pwd = pwd_var.get()
            if not pwd:
                err_var.set(tr("ui.auth.err_empty", "Please enter the password."))
                ent.focus_set()
                return
            try:
                if self.cfg_mgr.verify_password(pwd):
                    result["ok"] = True
                    dlg.destroy()
                    return
            except Exception:
                pass
            err_var.set(tr("ui.auth.err_wrong", "Incorrect password. Try again."))
            ent.selection_range(0, "end")
            ent.focus_set()

        def _cancel(_evt=None):
            result["ok"] = False
            dlg.destroy()

        tk.Button(
            btn_row,
            text=tr("ui.common.cancel", "Cancel"),
            command=_cancel,
            bg="#2D2D2D",
            fg="#E5E5E5",
            activebackground="#3D3D3D",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            padx=14,
            pady=5,
            cursor="hand2",
        ).pack(side="right")

        tk.Button(
            btn_row,
            text="OK",
            command=_submit,
            bg="#0078D4",
            fg="#FFFFFF",
            activebackground="#1084DD",
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            padx=18,
            pady=5,
            cursor="hand2",
        ).pack(side="right", padx=(0, 8))

        dlg.bind("<Return>", _submit)
        dlg.bind("<Escape>", _cancel)
        dlg.protocol("WM_DELETE_WINDOW", _cancel)

        ent.focus_set()
        self.root.wait_window(dlg)
        return bool(result["ok"])

    def _on_settings_closed(self):
        self.settings_win = None

    def open_people(self):
        if self._is_closing:
            return
        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.focus()
                    return
            except Exception:
                pass
            self.people_win = None

        self.people_win = PeopleWindow(
            self.root,
            cfg_mgr=self.cfg_mgr,
            camera_manager=self.camera_manager,
            registry=self.registry,
            on_close_cb=self._on_people_closed,
        )

    def _on_people_closed(self):
        self.people_win = None

    def open_plates(self):
        if self._is_closing:
            return
        if self.plates_win is not None:
            try:
                if self.plates_win.win.winfo_exists():
                    self.plates_win.focus()
                    return
            except Exception:
                pass
            self.plates_win = None

        self.plates_win = PlatesWindow(
            self.root,
            cfg_mgr=self.cfg_mgr,
            camera_manager=self.camera_manager,
            registry=self.registry,
            on_close_cb=self._on_plates_closed,
        )

    def _on_plates_closed(self):
        self.plates_win = None

    def _refresh_mode_badge(self):
        if not hasattr(self, "lbl_mode_badge"):
            return
        try:
            if self.cfg_mgr.is_operator_mode():
                self.lbl_mode_badge.configure(
                    text=tr("ui.mode.operator_badge", "👤 Operator (no recording)"),
                    fg="#ffcc00",
                    bg="#332917",
                )
            else:
                self.lbl_mode_badge.configure(
                    text=tr("ui.mode.server_badge", "🖥 Server"),
                    fg="#75e08b",
                    bg="#1b2e24",
                )
        except Exception:
            pass

    def on_settings_applied(self):
        if self._is_closing:
            return
        snap = self.cfg_mgr.get_snapshot()
        gen_cfg = snap.get("general") or {}
        hw_cfg = snap.get("hardware") or {}

        new_lang = self.cfg_mgr.get_language()
        if new_lang != get_language():
            self.change_language(new_lang, source="settings")

        self._refresh_mode_badge()
        self._sync_windows_startup(bool(gen_cfg.get("autostart_os_and_play", False)))
        self._apply_kiosk_mode(bool(gen_cfg.get("kiosk_mode", False)))
        self.camera_manager.sync_from_config()

        cams_by_uid = {c["uuid"]: c for c in (snap.get("cameras") or [])}
        for uid in list(self.detail_windows.keys()):
            dwin = self.detail_windows[uid]
            cam_cfg = cams_by_uid.get(uid)
            unit = self.camera_manager.get_unit(uid)
            if cam_cfg is None or not cam_cfg.get("enabled", True) or unit is None:
                try:
                    dwin.close()
                except Exception:
                    self.detail_windows.pop(uid, None)
            else:
                try:
                    dwin.update_camera_config(cam_cfg, hw_cfg, unit)
                except Exception:
                    pass

        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.on_modules_reloaded()
            except Exception:
                pass

        if self.plates_win is not None:
            try:
                if self.plates_win.win.winfo_exists():
                    self.plates_win.on_modules_reloaded()
            except Exception:
                pass

        self._last_layout_key = None
        self._sync_monitor_windows()
        self._relayout_grid(force=True)
        self._render_tiles_once()
        self._maybe_ensure_cuda_async()

    def open_camera_detail(self, cam_uuid: str):
        if self._is_closing or not cam_uuid:
            return
        existing = self.detail_windows.get(cam_uuid)
        if existing is not None:
            try:
                if existing.win.winfo_exists():
                    existing.focus()
                    return
            except Exception:
                pass
            self.detail_windows.pop(cam_uuid, None)

        unit = self.camera_manager.get_unit(cam_uuid)
        if unit is None:
            return

        dwin = CameraDetailWindow(
            self.root,
            cam_uuid=cam_uuid,
            camera_unit=unit,
            cfg_mgr=self.cfg_mgr,
            registry=self.registry,
            on_close_cb=self._on_detail_closed,
        )
        self.detail_windows[cam_uuid] = dwin
        self._update_all_tile_borders()

    def _update_all_tile_borders(self):
        self._update_tile_borders()
        for mwin in list(self.monitor_windows.values()):
            try:
                if mwin.win.winfo_exists() and not mwin._is_closing:
                    mwin._update_tile_borders()
            except Exception:
                pass

    def _on_detail_closed(self, cam_uuid: str):
        self.detail_windows.pop(cam_uuid, None)
        if not self._is_closing:
            self._update_all_tile_borders()

    def _on_camera_saved_cb(self, cam_uuid, mod_name, day_str, time_str, label, fpath):
        if self._is_closing:
            return
        try:
            self._ui_queue.put((self._dispatch_saved_to_ui, (cam_uuid, mod_name)))
        except Exception:
            pass

    def _dispatch_saved_to_ui(self, cam_uuid: str, mod_name: str):
        if self._is_closing:
            return
        dwin = self.detail_windows.get(cam_uuid)
        if dwin is not None:
            try:
                dwin.notify_saved(mod_name)
            except Exception:
                pass
        if self.people_win is not None:
            try:
                if self.people_win.win.winfo_exists():
                    self.people_win.notify_saved(mod_name)
            except Exception:
                pass
        if self.plates_win is not None:
            try:
                if self.plates_win.win.winfo_exists():
                    self.plates_win.notify_saved(mod_name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Побудова, адаптація і рендер сітки камер (тільки sub_source)
    # ------------------------------------------------------------------
    def _fast_cfg(self) -> dict:
        if hasattr(self.cfg_mgr, "get_ui_fast_config"):
            return self.cfg_mgr.get_ui_fast_config()
        return self.cfg_mgr.get_snapshot()

    def _get_monitors(self) -> list:
        snap = self._fast_cfg()
        mons = sorted(
            [m for m in (snap.get("camera_monitors") or []) if isinstance(m, dict) and m.get("id")],
            key=lambda m: int(m.get("tree_order", 0) or 0),
        )
        return mons

    def _get_grid_cfg(self) -> dict:
        mons = self._get_monitors()
        if mons and isinstance(mons[0].get("grid"), dict):
            return mons[0]["grid"]
        return super()._get_grid_cfg()

    def _get_units_for_monitor(self, monitor_id: str) -> list:
        snap = self._fast_cfg()
        mons = self._get_monitors()
        default_mid = str(mons[0]["id"]) if mons else ""
        valid_mids = {str(m["id"]) for m in mons}
        cam_mid_map = {}
        for c in (snap.get("cameras") or []):
            if not isinstance(c, dict):
                continue
            uid = str(c.get("uuid") or "")
            mid = str(c.get("monitor_id") or "")
            if mid not in valid_mids:
                mid = default_mid
            cam_mid_map[uid] = mid
        units = self.camera_manager.get_ordered_units()
        if not mons:
            return units
        target = str(monitor_id or default_mid)
        return [u for u in units if cam_mid_map.get(u.uuid, default_mid) == target]

    def _get_primary_monitor_units(self) -> list:
        mons = self._get_monitors()
        if not mons:
            return self.camera_manager.get_ordered_units()
        return self._get_units_for_monitor(str(mons[0]["id"]))

    def _sync_monitor_windows(self):
        if self._is_closing:
            return
        mons = self._get_monitors()
        secondary = mons[1:] if len(mons) > 1 else []
        want_ids = {str(m["id"]): m for m in secondary}

        for mid in list(self.monitor_windows.keys()):
            if mid not in want_ids:
                mwin = self.monitor_windows.pop(mid)
                try:
                    mwin.close()
                except Exception:
                    pass

        for mid, mon in want_ids.items():
            mname = str(mon.get("name") or "Monitor")
            existing = self.monitor_windows.get(mid)
            if existing is not None:
                try:
                    if existing.win.winfo_exists() and not existing._is_closing:
                        existing.set_monitor_name(mname)
                        existing._last_layout_key = None
                        existing._relayout_grid(force=True)
                        existing.render_tiles_once()
                        continue
                except Exception:
                    pass
                self.monitor_windows.pop(mid, None)

            mwin = SecondaryMonitorWindow(self, mid, mname)
            self.monitor_windows[mid] = mwin
            try:
                mwin._relayout_grid(force=True)
                mwin.render_tiles_once()
            except Exception:
                pass

        self._sync_visible_grid_uuids()

    def _notify_visible_changed(self):
        self._sync_visible_grid_uuids()

    def _sync_visible_grid_uuids(self, _unused=None):
        if hasattr(self.camera_manager, "set_visible_grid_uuids"):
            all_vis = set()
            if not self._window_minimized:
                all_vis.update(self._visible_grid_uuids)
            for mwin in list(self.monitor_windows.values()):
                try:
                    if mwin.win.winfo_exists() and not getattr(mwin, "_window_minimized", False):
                        all_vis.update(mwin.get_visible_uuids())
                except Exception:
                    pass
            try:
                self.camera_manager.set_visible_grid_uuids(all_vis)
            except Exception:
                pass

    def _on_root_unmap(self, event=None):
        if self._is_closing or (event is not None and event.widget is not self.root):
            return
        try:
            st = str(self.root.state()).lower()
            if st not in ("iconic", "withdrawn"):
                return
        except Exception:
            pass
        if not self._window_minimized:
            self._window_minimized = True
            self._sync_visible_grid_uuids(set())

    def _on_root_map(self, event=None):
        if self._is_closing or (event is not None and event.widget is not self.root):
            return
        if self._window_minimized:
            self._window_minimized = False
            self._sync_visible_grid_uuids(self._visible_grid_uuids)
            try:
                self._render_tiles_once()
            except Exception:
                pass

    def _on_root_configure(self, event=None):
        if self._is_closing:
            return
        if event is not None and event.widget not in (self.root, self.grid_area):
            return
        if self._relayout_after_id is not None:
            try:
                self.root.after_cancel(self._relayout_after_id)
            except Exception:
                pass
        self._relayout_after_id = self.root.after(50, lambda: self._relayout_grid(force=False))

    def _change_page(self, delta: int):
        new_p = max(0, min(max(0, self._total_pages - 1), self._current_page + delta))
        if new_p != self._current_page:
            self._current_page = new_p
            self._relayout_grid(force=True)

    def _relayout_grid(self, force: bool = False):
        units = self._get_primary_monitor_units()
        self._relayout_grid_with_units(units, force=force)

    def _render_tiles_once(self):
        units = self._get_primary_monitor_units()
        return self._render_tiles_with_units(units)

    def _render_grid_tick(self):
        if self._is_closing:
            return
        if _is_win_foreground_admin_exit_pressed():
            self._admin_exit()
            return
        t0 = time.perf_counter()
        interval_ms = self._render_tiles_once()
        for mwin in list(self.monitor_windows.values()):
            try:
                if mwin.win.winfo_exists() and not mwin._is_closing:
                    mwin.render_tiles_once()
            except Exception:
                pass
        dt_ms = int((time.perf_counter() - t0) * 1000.0)
        delay_ms = max(25, int(interval_ms) - dt_ms)
        if not self._is_closing:
            self._render_after_id = self.root.after(delay_ms, self._render_grid_tick)

    # ------------------------------------------------------------------
    # Телеметрія апаратних ресурсів у фоні (без блокування UI)
    # ------------------------------------------------------------------
    def _start_hw_poll_thread(self):
        def _loop():
            while self._hw_poll_running and not self._is_closing:
                if (not self._window_minimized) or (self._cached_hw_metrics is None):
                    try:
                        self._cached_hw_metrics = self.hw_monitor.get_live_metrics(include_disks=False)
                    except TypeError:
                        try:
                            self._cached_hw_metrics = self.hw_monitor.get_live_metrics()
                        except Exception:
                            pass
                    except Exception:
                        pass
                for _ in range(15):
                    if not self._hw_poll_running or self._is_closing:
                        break
                    time.sleep(0.1)

        threading.Thread(target=_loop, daemon=True).start()

    def _render_hw_badge_once(self):
        if self._is_closing or not hasattr(self, "lbl_hw_badge"):
            return
        if self._cuda_status_note:
            try:
                self.lbl_hw_badge.configure(text=self._cuda_status_note, fg=self._cuda_note_color)
            except Exception:
                pass
            return
        m = self._cached_hw_metrics
        if m is None:
            try:
                self.lbl_hw_badge.configure(
                    text=tr("ui.main.hw_badge_init", "CPU: --% | RAM: -- | AI: CPU"),
                    fg="#9ee8ff",
                )
            except Exception:
                pass
            return
        try:
            cpu_pct = float(m["cpu"]["usage_pct"])
            ram_used = float(m["ram"]["used_gb"])
            ram_tot = float(m["ram"]["total_gb"])
            ai_sum = self.inference_pool.get_active_summary()
            gpu_parts = []
            gb_unit = tr("ui.unit.gb", "GB")
            for g in (m.get("gpus") or []):
                if g.get("live_metrics") and (g.get("gpu_util_pct") is not None or g.get("vram_used_mb") is not None):
                    if g.get("gpu_util_pct") is not None:
                        gpu_parts.append(f"GPU: {float(g['gpu_util_pct']):.0f}%")
                    v_used = g.get("vram_used_mb")
                    v_tot = g.get("vram_total_mb")
                    if v_used is not None and v_tot is not None and float(v_tot) > 0:
                        gpu_parts.append(f"VRAM: {float(v_used) / 1024.0:.1f}/{float(v_tot) / 1024.0:.1f} {gb_unit}")
                    elif v_used is not None:
                        gpu_parts.append(f"VRAM: {float(v_used) / 1024.0:.1f} {gb_unit}")
                    break
            gpu_bit = (" | " + " | ".join(gpu_parts)) if gpu_parts else ""
            self.lbl_hw_badge.configure(
                text=tr(
                    "ui.main.hw_badge_fmt",
                    "CPU: {cpu:.0f}%{gpu_bit} | RAM: {ram_used:.1f}/{ram_tot:.1f} GB | AI: {ai}",
                    cpu=cpu_pct,
                    gpu_bit=gpu_bit,
                    ram_used=ram_used,
                    ram_tot=ram_tot,
                    ai=ai_sum,
                ),
                fg="#9ee8ff",
            )
        except Exception:
            pass

    def _refresh_hw_badge_tick(self):
        if self._is_closing:
            return
        self._render_hw_badge_once()
        if not self._is_closing:
            self._hw_after_id = self.root.after(1500, self._refresh_hw_badge_tick)

    # ------------------------------------------------------------------
    # Завершення роботи
    # ------------------------------------------------------------------
    def on_close(self, exit_app: bool = True):
        if self._is_closing:
            return
        self._is_closing = True
        self._hw_poll_running = False

        for attr in (
            "_start_after_id",
            "_render_after_id",
            "_hw_after_id",
            "_hide_bar_after_id",
            "_relayout_after_id",
            "_module_poll_after_id",
            "_ui_poll_after_id",
            "_cuda_note_clear_after_id",
        ):
            aid = getattr(self, attr, None)
            if aid is not None:
                try:
                    self.root.after_cancel(aid)
                except Exception:
                    pass
                setattr(self, attr, None)

        if self.settings_win is not None:
            try:
                self.settings_win.close()
            except Exception:
                pass
            self.settings_win = None

        if self.people_win is not None:
            try:
                self.people_win.close()
            except Exception:
                pass
            self.people_win = None

        if self.plates_win is not None:
            try:
                self.plates_win.close()
            except Exception:
                pass
            self.plates_win = None

        for dwin in list(self.detail_windows.values()):
            try:
                dwin.close()
            except Exception:
                pass
        self.detail_windows.clear()

        for mwin in list(self.monitor_windows.values()):
            try:
                mwin.close()
            except Exception:
                pass
        self.monitor_windows.clear()

        try:
            self.geom_tracker.close()
        except Exception:
            pass

        try:
            self.camera_manager.stop_all()
        except Exception as e:
            print(f"[Shutdown CameraManager error]: {e}")

        try:
            flush_io(timeout=3.0)
        except Exception:
            pass

        try:
            borderless = getattr(self, "_borderless_ctl", None)
            if borderless is not None:
                borderless.detach()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

        if exit_app:
            sys.exit(0)


def _get_pointer_monitor_work_area(widget) -> tuple[int, int, int, int]:
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

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

            u32 = ctypes.windll.user32
            pt = POINT()
            u32.GetCursorPos(ctypes.byref(pt))
            hmon = u32.MonitorFromPoint(pt, 2)  # MONITOR_DEFAULTTONEAREST
            if hmon:
                mi = MONITORINFO()
                mi.cbSize = ctypes.sizeof(MONITORINFO)
                if u32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                    rc = mi.rcWork
                    mw = rc.right - rc.left
                    mh = rc.bottom - rc.top
                    if mw > 0 and mh > 0:
                        return int(rc.left), int(rc.top), int(mw), int(mh)
        except Exception:
            pass
    return get_virtual_screen_bounds(widget)


def prompt_startup_login(root: tk.Tk, cfg_mgr: ConfigManager) -> bool:
    if not cfg_mgr.is_locked():
        return True

    result = {"unlocked": False, "checking": False}

    dlg = tk.Toplevel(root)
    try:
        dlg.withdraw()
    except Exception:
        pass
    dlg.title(tr("ui.auth.login_title", "ANPR Multi-Cam NVR — Sign In"))
    dlg.configure(bg="#141620")
    dlg.resizable(False, False)

    card = tk.Frame(
        dlg,
        bg="#1c1f2b",
        padx=24,
        pady=20,
        highlightthickness=1,
        highlightbackground="#2b3042",
    )
    card.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

    tk.Label(
        card,
        text="🔒 ANPR Multi-Cam NVR",
        fg="#00e5ff",
        bg="#1c1f2b",
        font=("Segoe UI", 13, "bold"),
        anchor=tk.W,
    ).pack(fill=tk.X)

    tk.Label(
        card,
        text=tr(
            "ui.auth.login_prompt",
            "Enter your access password to decrypt video sources and start monitoring.",
        ),
        fg="#c8d0e4",
        bg="#1c1f2b",
        font=("Segoe UI", 9),
        justify=tk.LEFT,
        wraplength=360,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(6, 14))

    tk.Label(
        card,
        text=tr("ui.auth.pwd_label", "Password:"),
        fg="#d8deef",
        bg="#1c1f2b",
        font=("Segoe UI", 9, "bold"),
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(0, 4))

    entry_row = tk.Frame(card, bg="#1c1f2b")
    entry_row.pack(fill=tk.X)

    pwd_var = tk.StringVar(value="")
    entry = tk.Entry(
        entry_row,
        textvariable=pwd_var,
        show="•",
        bg="#252836",
        fg="#ffffff",
        insertbackground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 11),
        width=28,
    )
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6, padx=(0, 6))

    show_flag = [False]

    def _toggle_show():
        show_flag[0] = not show_flag[0]
        entry.configure(show="" if show_flag[0] else "•")
        btn_eye.configure(fg="#00e5ff" if show_flag[0] else "#c8d0e4")
        entry.focus_set()

    btn_eye = tk.Button(
        entry_row,
        text="👁",
        command=_toggle_show,
        bg="#2a2e3f",
        fg="#c8d0e4",
        activebackground="#353b50",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 10),
        width=3,
        pady=3,
        cursor="hand2",
    )
    btn_eye.pack(side=tk.LEFT)

    err_var = tk.StringVar(value="")
    tk.Label(
        card,
        textvariable=err_var,
        fg="#ff6b6b",
        bg="#1c1f2b",
        font=("Segoe UI", 9, "bold"),
        wraplength=360,
        justify=tk.LEFT,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(8, 6))

    btn_row = tk.Frame(card, bg="#1c1f2b")
    btn_row.pack(fill=tk.X, pady=(6, 0))

    def _cancel():
        if result["checking"]:
            return
        result["unlocked"] = False
        try:
            dlg.destroy()
        except Exception:
            pass

    def _try_unlock(event=None):
        if result["checking"]:
            return "break"
        pw = pwd_var.get()
        if not pw:
            err_var.set(tr("ui.auth.err_empty", "Please enter the password."))
            entry.focus_set()
            return "break"

        result["checking"] = True
        btn_unlock.configure(state=tk.DISABLED)
        err_var.set("")
        try:
            dlg.update_idletasks()
        except Exception:
            pass

        try:
            ok, err_code = cfg_mgr.unlock_with_error(pw)
        except Exception:
            ok, err_code = False, "corrupt_data"
        finally:
            result["checking"] = False

        if ok:
            result["unlocked"] = True
            try:
                dlg.destroy()
            except Exception:
                pass
            return "break"

        try:
            btn_unlock.configure(state=tk.NORMAL)
            if err_code == "corrupt_data":
                err_var.set(
                    tr(
                        "ui.auth.err_corrupt",
                        "Failed to decrypt saved camera source links in config.json.",
                    )
                )
            else:
                err_var.set(tr("ui.auth.err_wrong", "Incorrect password. Try again."))
            entry.selection_range(0, tk.END)
            entry.focus_set()
        except Exception:
            pass
        return "break"

    btn_exit = tk.Button(
        btn_row,
        text=tr("ui.auth.btn_exit", "🚪 Exit"),
        command=_cancel,
        bg="#353849",
        fg="#ffffff",
        activebackground="#44485d",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 9, "bold"),
        padx=14,
        pady=6,
        cursor="hand2",
    )
    btn_exit.pack(side=tk.RIGHT, padx=(8, 0))

    btn_unlock = tk.Button(
        btn_row,
        text=tr("ui.auth.btn_unlock", "🔓 Unlock"),
        command=_try_unlock,
        bg="#007acc",
        fg="#ffffff",
        activebackground="#0066ad",
        activeforeground="#ffffff",
        relief=tk.FLAT,
        font=("Segoe UI", 9, "bold"),
        padx=16,
        pady=6,
        cursor="hand2",
    )
    btn_unlock.pack(side=tk.RIGHT)

    dlg.protocol("WM_DELETE_WINDOW", _cancel)
    dlg.bind("<Return>", _try_unlock)
    dlg.bind("<Escape>", lambda e: _cancel())
    entry.bind("<Return>", _try_unlock)

    try:
        dlg.update_idletasks()
        dw = max(410, int(dlg.winfo_reqwidth()))
        dh = max(230, int(dlg.winfo_reqheight()))
        mx, my, mw, mh = _get_pointer_monitor_work_area(dlg)
        x = mx + max(0, (mw - dw) // 2)
        y = my + max(0, (mh - dh) // 2)
        dlg.geometry(f"{dw}x{dh}+{x}+{y}")
        dlg.deiconify()
        dlg.lift()
        dlg.focus_force()
        entry.focus_set()
        dlg.grab_set()
    except Exception:
        pass

    root._startup_login_dlg = dlg
    try:
        root.wait_window(dlg)
    finally:
        try:
            delattr(root, "_startup_login_dlg")
        except Exception:
            pass
    return bool(result["unlocked"])


def main(base_dir: str = None):
    target_dir = os.path.abspath(base_dir or BASE_DIR)
    if base_dir is None and not getattr(sys, "frozen", False):
        try:
            os.chdir(target_dir)
        except Exception:
            pass
    try:
        prepare_ort_backend_environment(target_dir)
        prepare_cuda_dll_environment(target_dir, preload=False)
    except Exception:
        pass

    registry = ModuleRegistry(target_dir)
    cfg_mgr = ConfigManager(target_dir, registry=registry)
    init_i18n(target_dir, lang_code=cfg_mgr.get_language())

    root = tk.Tk()
    if cfg_mgr.is_locked():
        try:
            root.withdraw()
        except Exception:
            pass
        unlocked = prompt_startup_login(root, cfg_mgr)
        if not unlocked or cfg_mgr.is_locked():
            try:
                root.destroy()
            except Exception:
                pass
            return
        try:
            root.deiconify()
        except Exception:
            pass

    cfg_mgr.sync_with_registry(registry)
    MainGridWindow(root, base_dir=base_dir, registry=registry, cfg_mgr=cfg_mgr)
    root.mainloop()


if __name__ == "__main__":
    main()
