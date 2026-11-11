import os
import time
import copy
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

try:
    import cv2
except ImportError:
    cv2 = None

from scripts.core.i18n import tr
from scripts.core.module_registry import is_date_range_active
from scripts.ui.helpers import (
    WindowGeometryTracker,
    clamp_window_geometry,
    get_current_monitor_geometry,
    ensure_clam_theme,
    update_scale_range_and_value,
    open_video_capture_for_playback,
    seek_video_capture,
)

EVT_MOTION = "<Motion>"
EVT_TREE_SELECT = "<<TreeviewSelect>>"
EVT_DOUBLE_CLICK = "<Double-1>"
EVT_BTN_PRESS = "<ButtonPress-1>"
EVT_B1_MOTION = "<B1-Motion>"
EVT_BTN_RELEASE = "<ButtonRelease-1>"
EVT_RIGHT_CLICK = "<Button-3>"
EVT_MOUSEWHEEL = "<MouseWheel>"
EVT_BTN4 = "<Button-4>"
EVT_BTN5 = "<Button-5>"
EVT_ENTER = "<Enter>"
EVT_LEAVE = "<Leave>"


class CameraDetailWindow:
    """
    Незалежне вікно повного перегляду однієї камери + перегляд її архівів
    (кнопки, парсери й оверлеї формуються динамічно через ModuleRegistry) та плаваючий ROI-кропер.
    Не керує фоновим записом: камера працює постійно у CameraManager.
    """
    def __init__(self, master, cam_uuid: str, camera_unit, cfg_mgr, registry, on_close_cb=None):
        self.master = master
        self.cam_uuid = str(cam_uuid)
        self.camera_unit = camera_unit
        self.cfg_mgr = cfg_mgr
        self.registry = registry
        self.on_close_cb = on_close_cb

        self.cam_cfg = cfg_mgr.get_camera(self.cam_uuid) or (camera_unit.cam_cfg if camera_unit else {})
        snap = cfg_mgr.get_snapshot()
        self.hw_cfg = snap.get("hardware") or {}
        self.archive_dir = cfg_mgr.get_camera_archive_dir(self.cam_uuid)

        self.win = tk.Toplevel(master)
        self.win.title(tr("ui.detail.window_title", "{name} — Detail View", name=self._get_cam_display_name()))
        self.win.configure(bg="#0d0d11")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self._is_closing = False
        self._render_after_id = None
        self._refresh_debounce_id = None
        self._hide_timer = None
        self.panel_visible = True
        self._wheel_zone = None

        self._last_drawn_seq = -1
        self._last_draw_ts = 0.0
        self._last_detail_snap = None
        self._last_main_offline = False
        self.img_main_tk = None
        self.preview_photo_tk = None
        self.screen_photo_tk = None
        self.roi_photo_tk = None

        # Інтерактивне кадрування (зум) наживо лівою клавішею миші на main_video_canvas
        self._live_crop_norm = None
        self._live_last_raw_frame = None
        self._live_last_overlay = {}
        self._live_last_now = 0.0
        self._main_img_id = None
        self._main_ph_id = None
        self._main_disp_w = 1
        self._main_disp_h = 1
        self._main_img_off_x = 0
        self._main_img_off_y = 0
        self._live_drag_start_x = None
        self._live_drag_start_y = None
        self._live_drag_moved = False
        self._live_configure_after_id = None

        # Динамічні кнопки архіву з ModuleRegistry: mod_id -> tk.Button
        self._archive_buttons = {}

        # Параметри стану архівного бічного сайдбару
        self.sidebar_visible = False
        self.sidebar_mode = self._pick_default_sidebar_mode()
        self.tree_item_map = {}
        self.raw_records = []
        self.sort_mode = 0
        self._arch_tree_syncing = False

        self.current_screen_img_cv = None
        self.current_preview_fpath = None
        self.current_item_preview_pil = None
        self.current_item_base_name = ""
        self.current_item_disp_h = 40
        self._item_placeholder_key = ("ui.detail.ph.select_record", "[Select a record from list]")
        self._screen_placeholder_spec = ("ui.detail.ph.select_record", "[Select a record from list]", "#707080")
        self._screen_placeholder_info = (tr("ui.detail.ph.select_record", "[Select a record from list]"), "#707080")
        self._preview_revision = 0
        self.last_roi_crop_bgr = None
        self.current_roi_coords = None
        self.canvas_scale_x = 1.0
        self.canvas_scale_y = 1.0
        self.canvas_disp_w = 340
        self.canvas_disp_h = 190
        self.canvas_slot_w = 340
        self.canvas_slot_h = 190
        self.canvas_img_offset_x = 1
        self.canvas_img_offset_y = 1
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_moved = False
        self.side_panel_width = 360
        self._last_main_body_size = None
        self._last_sidebar_layout_key = None
        self._in_sidebar_relayout = False
        self.screen_canvas_img_id = None

        # Вбудований плеєр архівного відео
        self.arch_cap = None
        self.arch_fpath = None
        self.arch_playing = False
        self.arch_fps = 15.0
        self.arch_total_frames = 0
        self.arch_curr_frame_idx = 0
        self.arch_speed = 1
        self.arch_after_id = None
        self._arch_click_after_id = None
        self._ignore_release_tag = None
        self._ignore_release_ts = 0.0
        self.arch_updating_slider = False

        # Геометрія плаваючого ROI вікна з config.json
        roi_g = cfg_mgr.get_camera_window_geometry(self.cam_uuid, "roi_window") or {}
        self.roi_user_top = int(roi_g.get("top", 40))
        self.roi_user_left = int(roi_g.get("left", 40))
        self.roi_user_width = max(260, int(roi_g.get("width", 520)))
        self.roi_user_height = max(180, int(roi_g.get("height", 360)))

        self.roi_visible = False
        self.roi_crop_mode = False
        self.roi_maximized = False
        self.roi_scale_x = 1.0
        self.roi_scale_y = 1.0
        self.roi_img_offset_x = 0
        self.roi_img_offset_y = 0
        self.roi_disp_w = 1
        self.roi_disp_h = 1
        self.roi_canvas_img_id = None
        self.roi_canvas_drag_start_x = None
        self.roi_canvas_drag_start_y = None
        self.roi_canvas_drag_moved = False
        self.roi_drag_moved_win = False

        self._build_ui()

        self.geom_tracker = WindowGeometryTracker(
            self.win,
            get_geom_cb=lambda: self.cfg_mgr.get_camera_window_geometry(self.cam_uuid, "window"),
            set_geom_cb=lambda g: self.cfg_mgr.set_camera_window_geometry(self.cam_uuid, "window", g),
            min_w=640,
            min_h=420,
            default_geom={"left": 80, "top": 60, "width": 1160, "height": 720, "maximized": False},
        )

        # Повідомляємо CameraUnit, що основний потік потрібен для живого перегляду
        if self.camera_unit is not None:
            self.camera_unit.add_detail_subscriber()

        self.win.bind(EVT_MOTION, self._on_mouse_motion)
        self.win.bind(EVT_MOUSEWHEEL, self._route_mousewheel)
        self.win.bind(EVT_BTN4, self._route_mousewheel)
        self.win.bind(EVT_BTN5, self._route_mousewheel)

        self._sync_module_buttons()
        self._schedule_hide(5000)
        self._render_tick()

    # ------------------------------------------------------------------
    # Метадані архівних модулів з ModuleRegistry
    # ------------------------------------------------------------------
    def _get_archive_entries(self) -> list:
        if self.registry is None:
            return []
        return [
            ent for ent in self.registry.list_modules()
            if (ent["spec"].get("archive") or {}).get("has_archive")
        ]

    def _get_archive_spec(self, mod_id: str) -> dict:
        if self.registry is None or not mod_id:
            return {}
        spec = self.registry.get_spec(mod_id)
        if not spec:
            return {}
        return spec.get("archive") or {}

    def _get_current_record_type(self, mode: str = None) -> str:
        arch = self._get_archive_spec(mode or self.sidebar_mode)
        return str(arch.get("record_type") or "single_image")

    def _get_cam_display_name(self) -> str:
        return str((self.cam_cfg or {}).get("name") or "").strip() or tr("camera.default_single", "Camera")

    def _resolve_archive_text(self, arch: dict, key: str, mod_id: str, fallback: str = "") -> str:
        val = (arch or {}).get(key)
        if callable(val):
            mod_cfg = ((self.cam_cfg or {}).get("modules") or {}).get(mod_id) or {}
            try:
                return str(val(mod_cfg))
            except Exception:
                return str(fallback)
        if val:
            return tr(f"mod.{mod_id}.archive.{key}", str(val))
        return str(fallback)

    def _pick_default_sidebar_mode(self) -> str:
        mods = (self.cam_cfg or {}).get("modules") or {}
        entries = self._get_archive_entries()
        for ent in entries:
            mid = ent["id"]
            if (mods.get(mid) or {}).get("on", False):
                return mid
        return entries[0]["id"] if entries else ""

    def _has_archive_files(self, mod_name: str) -> bool:
        d = os.path.join(self.archive_dir, mod_name)
        if not os.path.isdir(d):
            return False
        for _, _, files in os.walk(d):
            if files:
                return True
        return False

    def focus(self):
        try:
            if str(self.win.state()) == "iconic":
                self.win.deiconify()
            self.win.lift()
            self.win.focus_force()
        except Exception:
            pass

    def update_camera_config(self, cam_cfg: dict, hw_cfg: dict, camera_unit=None):
        self.cam_cfg = dict(cam_cfg or {})
        self.hw_cfg = dict(hw_cfg or {})
        if camera_unit is not None and camera_unit is not self.camera_unit:
            if self.camera_unit is not None:
                self.camera_unit.remove_detail_subscriber()
            self.camera_unit = camera_unit
            self.camera_unit.add_detail_subscriber()

        self.archive_dir = self.cfg_mgr.get_camera_archive_dir(self.cam_uuid)
        try:
            name = self._get_cam_display_name()
            self.win.title(tr("ui.detail.window_title", "{name} — Detail View", name=name))
            self.cam_title_lbl.configure(text=f"📹 {name}")
            self._sync_module_buttons()
            if self.sidebar_visible and self.sidebar_mode:
                arch = self._get_archive_spec(self.sidebar_mode)
                self.side_header_lbl.configure(
                    text=self._resolve_archive_text(
                        arch, "sidebar_title", self.sidebar_mode,
                        tr("ui.detail.archive_title_fmt", "Archive: {mod}", mod=self.sidebar_mode),
                    )
                )
        except Exception:
            pass

    def apply_language(self):
        if self._is_closing:
            return
        try:
            name = self._get_cam_display_name()
            self.win.title(tr("ui.detail.window_title", "{name} — Detail View", name=name))
            self.cam_title_lbl.configure(text=f"📹 {name}")
            self._sync_module_buttons()
            self._sync_sort_button_text()
            self.btn_expand_preview.configure(text=tr("ui.detail.btn_expand_preview", "❐ Window"))

            if self.sidebar_mode:
                arch = self._get_archive_spec(self.sidebar_mode)
                self.side_header_lbl.configure(
                    text=self._resolve_archive_text(
                        arch, "sidebar_title", self.sidebar_mode,
                        tr("ui.detail.archive_title_fmt", "Archive: {mod}", mod=self.sidebar_mode),
                    )
                )
                self._update_preview_layout(self.sidebar_mode)
            else:
                self.side_header_lbl.configure(text=tr("ui.detail.archive_default", "Archive"))

            if self.current_item_preview_pil is not None:
                self._render_item_preview_image()
            else:
                self._clear_item_preview_by_key(*self._item_placeholder_key)

            if self.current_screen_img_cv is None:
                k, dflt, col = self._screen_placeholder_spec
                self._set_screen_canvas_placeholder(tr(k, dflt), color=col)

            self._update_status_bar_from_snapshot(self._last_detail_snap)
            if self.img_main_tk is None:
                self._update_main_video_placeholder()
            elif self._live_crop_norm is not None:
                self._draw_live_crop_badge()

            self._update_roi_toolbar_labels()
            self._render_current_roi_view()
        except Exception:
            pass

    def on_modules_reloaded(self):
        """Оновлює кнопки архіву, бічний сайдбар та оверлеї після гарячого перезавантаження модулів."""
        if self._is_closing:
            return
        try:
            fresh_cfg = self.cfg_mgr.get_camera(self.cam_uuid)
            if fresh_cfg is not None:
                self.cam_cfg = fresh_cfg

            arch_ids = {e["id"] for e in self._get_archive_entries()}
            if self.sidebar_visible and self.sidebar_mode not in arch_ids:
                self._stop_archive_player()
                self._hide_roi_panel(clear_coords=True)
                self.side_panel.pack_forget()
                self.video_container.pack_forget()
                self.video_container.pack(fill=tk.BOTH, expand=True)
                self.sidebar_visible = False
                self.sidebar_mode = self._pick_default_sidebar_mode()

            self._sync_module_buttons()
            if self.sidebar_visible and self.sidebar_mode:
                arch = self._get_archive_spec(self.sidebar_mode)
                self.side_header_lbl.configure(
                    text=self._resolve_archive_text(
                        arch, "sidebar_title", self.sidebar_mode,
                        tr("ui.detail.archive_title_fmt", "Archive: {mod}", mod=self.sidebar_mode),
                    )
                )
                self._update_preview_layout(self.sidebar_mode)
                self.refresh_saved_list()
                self._relayout_sidebar_previews(force=True)
        except Exception as e:
            print(f"[DetailWindow on_modules_reloaded]: {e}")

    def notify_saved(self, mod_name: str):
        if self._is_closing:
            return
        if self.sidebar_visible and self.sidebar_mode == mod_name:
            if self._refresh_debounce_id is None:
                self._refresh_debounce_id = self.win.after(150, self._run_debounced_refresh)

    def _run_debounced_refresh(self):
        self._refresh_debounce_id = None
        if not self._is_closing and self.sidebar_visible:
            self.refresh_saved_list()

    def _build_ui(self):
        self.ctrl_frame = tk.Frame(self.win, bg="#1e1e24", padx=10, pady=8)
        self.ctrl_frame.pack(fill=tk.X, side=tk.TOP)

        cam_name = self._get_cam_display_name()
        self.cam_title_lbl = tk.Label(
            self.ctrl_frame,
            text=f"📹 {cam_name}",
            fg="#ffffff",
            bg="#1e1e24",
            font=("Segoe UI", 10, "bold"),
        )
        self.cam_title_lbl.pack(side=tk.LEFT, padx=(2, 10))

        # Статус та показник запису праворуч (кнопки архівів пакуються перед self.status_lbl динамічно)
        self.btn_fullscreen = tk.Button(
            self.ctrl_frame, text="⛶",
            command=self._toggle_fullscreen_window,
            bg="#303240", fg="#00e5ff", font=("Segoe UI", 9, "bold"),
            width=3, relief=tk.FLAT, cursor="hand2"
        )
        self.btn_fullscreen.pack(side=tk.RIGHT, padx=(6, 2))

        self.status_lbl = tk.Label(
            self.ctrl_frame,
            text=tr("ui.detail.status_connecting", "Connecting..."),
            fg="#ffcc00",
            bg="#1e1e24",
            font=("Segoe UI", 9, "bold"),
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=8)

        # Основна область: сайдбар зліва + живе відео праворуч
        self.main_body = tk.Frame(self.win, bg="#0d0d11")
        self.main_body.pack(fill=tk.BOTH, expand=True)
        self.main_body.bind("<Configure>", self._on_main_body_configure)

        self.side_panel = tk.Frame(self.main_body, bg="#1e1e24", width=self.side_panel_width)
        self.side_panel.pack_propagate(False)
        self.side_panel.bind("<Configure>", self._on_side_panel_configure)

        self.side_header = tk.Frame(self.side_panel, bg="#282832", padx=10, pady=6)
        self.side_header.pack(fill=tk.X, side=tk.TOP)
        self.side_header_lbl = tk.Label(
            self.side_header, text=tr("ui.detail.archive_default", "Archive"), fg="#00e5ff", bg="#282832",
            font=("Segoe UI", 10, "bold"), anchor=tk.W, width=1
        )
        self.side_header_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            self.side_header, text="↻", command=self.refresh_saved_list,
            bg="#3a3a48", fg="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            width=3, cursor="hand2"
        ).pack(side=tk.RIGHT)

        self.search_frame = tk.Frame(self.side_panel, bg="#22222c", padx=6, pady=6)
        self.search_frame.pack(fill=tk.X, side=tk.TOP)

        tk.Label(self.search_frame, text="🔍", fg="#888899", bg="#22222c", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(2, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self._apply_search_and_sort())
        self.search_entry = tk.Entry(
            self.search_frame, textvariable=self.search_var,
            bg="#14141a", fg="#ffffff", insertbackground="white", bd=1, relief=tk.FLAT, font=("Segoe UI", 9)
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2), ipady=2)

        tk.Button(
            self.search_frame, text="✕", command=lambda: self.search_var.set(""),
            bg="#2d2d38", fg="#ff7777", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), width=2, cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 4))

        self.btn_sort = tk.Button(
            self.search_frame, text=tr("ui.detail.sort_time_desc", "▼ Time"), command=self._toggle_sort,
            bg="#333342", fg="#00e5ff", relief=tk.FLAT, font=("Segoe UI", 8, "bold"), padx=5, cursor="hand2"
        )
        self.btn_sort.pack(side=tk.RIGHT)

        self.preview_container = tk.Frame(self.side_panel, bg="#14141a", pady=4, padx=8)
        self.preview_container.pack(fill=tk.X, side=tk.BOTTOM)

        self.preview_info_lbl = tk.Label(
            self.preview_container, text=tr("ui.detail.item_preview_default", "Snapshot:"), fg="#9e9ea8", bg="#14141a",
            font=("Segoe UI", 9), anchor=tk.W, width=1
        )
        self.preview_info_lbl.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        self.preview_label = tk.Label(
            self.preview_container, bg="#1a1a22", text=tr("ui.detail.ph.select_record", "[Select a record from list]"),
            fg="#707080", font=("Segoe UI", 9), pady=8, relief=tk.SOLID, borderwidth=1
        )
        self.preview_label.pack(side=tk.TOP, fill=tk.X)
        self.preview_label.bind(EVT_DOUBLE_CLICK, lambda e: self._open_roi_viewer(crop_mode=False))

        self.screen_header_frame = tk.Frame(self.preview_container, bg="#14141a")
        self.screen_header_frame.pack(side=tk.TOP, fill=tk.X, pady=(4, 2))

        self.btn_expand_preview = tk.Button(
            self.screen_header_frame, text=tr("ui.detail.btn_expand_preview", "❐ Window"),
            command=lambda: self._open_roi_viewer(crop_mode=False),
            bg="#2d2d3a", fg="#00e5ff", font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT, padx=6, pady=1, cursor="hand2"
        )
        self.btn_expand_preview.pack(side=tk.RIGHT)

        self.screen_info_lbl = tk.Label(
            self.screen_header_frame, text=tr("ui.detail.screen_header_photo", "Screenshot (crop / 2x click):"),
            fg="#9e9ea8", bg="#14141a", font=("Segoe UI", 9), anchor=tk.W, width=1
        )
        self.screen_info_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.screen_canvas = tk.Canvas(
            self.preview_container, bg="#1a1a22", width=340, height=190,
            highlightthickness=1, highlightbackground="#333340", cursor="cross"
        )
        self.screen_canvas.pack(side=tk.TOP, pady=(0, 2))

        self.tree_frame = tk.Frame(self.side_panel, bg="#16161d", padx=6, pady=4)
        self.tree_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(self.tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        style = ensure_clam_theme(self.win)
        style.configure("ProDark.Treeview", background="#16161d", foreground="#ffffff", fieldbackground="#16161d", font=("Segoe UI", 10), rowheight=26, borderwidth=0)
        style.map("ProDark.Treeview", background=[("selected", "#007acc")], foreground=[("selected", "#ffffff")])

        self.items_tree = ttk.Treeview(
            self.tree_frame, style="ProDark.Treeview", show="tree",
            selectmode="browse", yscrollcommand=scrollbar.set
        )
        self.items_tree.column("#0", width=330, stretch=True)
        self.items_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.items_tree.yview)

        self.items_tree.tag_configure("day_tag", font=("Segoe UI", 10, "bold"), foreground="#00e5ff")
        self.items_tree.tag_configure("item_tag", font=("Consolas", 10, "bold"), foreground="#75e08b")
        self.items_tree.bind(EVT_TREE_SELECT, self._on_item_selected)
        self.items_tree.bind(EVT_DOUBLE_CLICK, self._on_item_double_clicked)

        for w in (self.items_tree, scrollbar, self.tree_frame):
            w.bind(EVT_ENTER, lambda e: self._set_wheel_zone("tree"))
            w.bind(EVT_LEAVE, lambda e: self._clear_wheel_zone("tree"))

        self.screen_canvas.bind(EVT_BTN_PRESS, self._on_screen_canvas_press)
        self.screen_canvas.bind(EVT_B1_MOTION, self._on_screen_canvas_motion)
        self.screen_canvas.bind(EVT_BTN_RELEASE, self._on_screen_canvas_release)
        self.screen_canvas.bind(EVT_DOUBLE_CLICK, self._on_screen_canvas_double_click)
        self.screen_canvas.bind(EVT_RIGHT_CLICK, self._on_roi_right_click)

        # Плеєр архівного відео
        self.video_controls_frame = tk.Frame(self.preview_container, bg="#1b1b26", padx=4, pady=4)
        self.btn_arch_play = tk.Button(
            self.video_controls_frame, text="▶", command=self._toggle_arch_play,
            bg="#28a745", fg="white", font=("Segoe UI", 9, "bold"), width=3, relief=tk.FLAT, cursor="hand2"
        )
        self.btn_arch_play.pack(side=tk.LEFT, padx=(0, 4))

        self.arch_slider_var = tk.IntVar(value=0)
        self.arch_slider = tk.Scale(
            self.video_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.arch_slider_var, showvalue=False, bg="#1b1b26", fg="#00e5ff",
            troughcolor="#2e2e3d", highlightthickness=0, bd=0, sliderlength=14,
            command=self._on_arch_slider_move
        )
        self.arch_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.lbl_arch_time = tk.Label(
            self.video_controls_frame, text="00:00/00:00",
            fg="#c8c8d8", bg="#1b1b26", font=("Consolas", 8)
        )
        self.lbl_arch_time.pack(side=tk.LEFT, padx=(4, 4))

        self.btn_arch_speed = tk.Button(
            self.video_controls_frame, text="1x", command=self._cycle_arch_speed,
            bg="#333344", fg="#00e5ff", font=("Segoe UI", 8, "bold"), width=3, relief=tk.FLAT, cursor="hand2"
        )
        self.btn_arch_speed.pack(side=tk.RIGHT)

        for pw in (self.preview_container, self.preview_label, self.screen_canvas):
            pw.bind(EVT_ENTER, lambda e: self._set_wheel_zone("preview"))
            pw.bind(EVT_LEAVE, lambda e: self._clear_wheel_zone("preview"))

        # Живий екран (Canvas для інтерактивного виділення/кадрування ЛКМ та скидання ПКМ)
        self.video_container = tk.Frame(self.main_body, bg="#0d0d11")
        self.video_container.pack(fill=tk.BOTH, expand=True)
        self.main_video_canvas = tk.Canvas(
            self.video_container, bg="#0d0d11", highlightthickness=0, bd=0, cursor="cross"
        )
        self.main_video_canvas.pack(fill=tk.BOTH, expand=True)
        self._main_ph_id = self.main_video_canvas.create_text(
            320, 180,
            text=tr("ui.detail.waiting_main", "Waiting for main stream..."),
            fill="#8c92a4",
            font=("Segoe UI", 12, "bold"),
            justify=tk.CENTER,
            tags="live_ph",
        )

        self.main_video_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self.main_video_canvas.bind(EVT_BTN_PRESS, self._on_main_canvas_press)
        self.main_video_canvas.bind(EVT_B1_MOTION, self._on_main_canvas_motion)
        self.main_video_canvas.bind(EVT_BTN_RELEASE, self._on_main_canvas_release)
        self.main_video_canvas.bind(EVT_RIGHT_CLICK, self._on_main_canvas_right_click)
        self.video_container.bind(EVT_BTN_PRESS, self._on_video_area_click)

        # Плаваюче вікно детального ROI/кадру
        self.roi_panel = tk.Toplevel(self.win, bg="#00e5ff")
        self.roi_panel.overrideredirect(True)
        self.roi_panel.transient(self.win)
        self.roi_panel.protocol("WM_DELETE_WINDOW", self._hide_roi_panel)
        self.roi_panel.bind("<Escape>", self._on_roi_escape)
        self.roi_panel.bind("<Configure>", self._on_roi_panel_configure)
        self.roi_panel.withdraw()

        self.roi_inner = tk.Frame(self.roi_panel, bg="#16161e")
        self.roi_inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.roi_header = tk.Frame(self.roi_inner, bg="#20202c", padx=6, pady=3, cursor="fleur")
        self.roi_header.pack(fill=tk.X, side=tk.TOP)

        self.roi_btn_close = tk.Button(
            self.roi_header, text="✕", command=self._hide_roi_panel,
            bg="#333342", fg="#ff5555", font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, padx=5, cursor="hand2"
        )
        self.roi_btn_close.pack(side=tk.RIGHT)

        self.roi_btn_max = tk.Button(
            self.roi_header, text="⛶", command=self._toggle_roi_maximize,
            bg="#333342", fg="#00e5ff", font=("Segoe UI", 8, "bold"), bd=0, relief=tk.FLAT, padx=5, cursor="hand2"
        )
        self.roi_btn_max.pack(side=tk.RIGHT, padx=(0, 3))

        self.roi_title_lbl = tk.Label(
            self.roi_header, text=tr("ui.detail.roi.view_title", "Preview"), fg="#00e5ff", bg="#20202c",
            font=("Segoe UI", 9, "bold"), anchor=tk.W, width=1, cursor="fleur"
        )
        self.roi_title_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.roi_toolbar = tk.Frame(self.roi_inner, bg="#1b1b26", padx=6, pady=3, cursor="fleur")
        self.roi_toolbar.pack(fill=tk.X, side=tk.TOP)

        self.roi_btn_prev = tk.Button(
            self.roi_toolbar, text=tr("ui.detail.roi.prev_short", "◀ Back"), command=lambda: self._navigate_items(-1),
            bg="#2e2e3c", fg="#ffffff", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2, cursor="hand2"
        )
        self.roi_btn_prev.pack(side=tk.LEFT, padx=(0, 4))

        self.roi_btn_next = tk.Button(
            self.roi_toolbar, text=tr("ui.detail.roi.next_short", "Next ▶"), command=lambda: self._navigate_items(1),
            bg="#2e2e3c", fg="#ffffff", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2, cursor="hand2"
        )
        self.roi_btn_next.pack(side=tk.LEFT, padx=(0, 6))

        self.roi_btn_crop_mode = tk.Button(
            self.roi_toolbar, text=tr("ui.detail.roi.frame_short", "🖼 Frame"), command=self._toggle_roi_crop_mode,
            bg="#2d3b4e", fg="#00e5ff", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2, cursor="hand2"
        )
        self.roi_btn_clear_crop = tk.Button(
            self.roi_toolbar, text=tr("ui.detail.roi.reset_short", "✕ Reset"), command=self._clear_roi_crop,
            bg="#44333a", fg="#ffaaaa", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2, cursor="hand2"
        )

        for w in (self.roi_header, self.roi_title_lbl, self.roi_toolbar):
            w.bind(EVT_BTN_PRESS, self._on_roi_drag_start)
            w.bind(EVT_B1_MOTION, self._on_roi_drag_motion)
            w.bind(EVT_BTN_RELEASE, self._on_roi_drag_release)
        for w in (self.roi_header, self.roi_title_lbl):
            w.bind(EVT_DOUBLE_CLICK, self._on_roi_header_double_click)

        self.roi_controls_frame = tk.Frame(self.roi_inner, bg="#1b1b24", padx=6, pady=4)
        self.roi_btn_play = tk.Button(
            self.roi_controls_frame, text="▶", command=self._toggle_arch_play,
            bg="#28a745", fg="white", font=("Segoe UI", 9, "bold"), width=3, relief=tk.FLAT, cursor="hand2"
        )
        self.roi_btn_play.pack(side=tk.LEFT, padx=(0, 6))

        self.roi_slider_var = tk.IntVar(value=0)
        self.roi_slider = tk.Scale(
            self.roi_controls_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.roi_slider_var, showvalue=False, bg="#1b1b24", fg="#00e5ff",
            troughcolor="#2e2e3d", highlightthickness=0, bd=0, sliderlength=14,
            command=self._on_arch_slider_move
        )
        self.roi_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self.roi_time_lbl = tk.Label(
            self.roi_controls_frame, text="00:00/00:00", fg="#e0e0f0", bg="#1b1b24", font=("Consolas", 9, "bold")
        )
        self.roi_time_lbl.pack(side=tk.LEFT, padx=6)

        self.roi_btn_speed = tk.Button(
            self.roi_controls_frame, text="1x", command=self._cycle_arch_speed,
            bg="#333344", fg="#00e5ff", font=("Segoe UI", 8, "bold"), width=3, relief=tk.FLAT, cursor="hand2"
        )
        self.roi_btn_speed.pack(side=tk.RIGHT, padx=(0, 14))

        self.roi_canvas = tk.Canvas(self.roi_inner, bg="#0d0d11", highlightthickness=0, cursor="cross")
        self.roi_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.roi_canvas.bind("<Configure>", self._on_roi_canvas_configure)
        self.roi_canvas.bind(EVT_BTN_PRESS, self._on_roi_canvas_press)
        self.roi_canvas.bind(EVT_B1_MOTION, self._on_roi_canvas_motion)
        self.roi_canvas.bind(EVT_BTN_RELEASE, self._on_roi_canvas_release)
        self.roi_canvas.bind(EVT_DOUBLE_CLICK, self._on_roi_canvas_double_click)
        self.roi_canvas.bind(EVT_RIGHT_CLICK, self._on_roi_right_click)
        for mw_w in (self.roi_panel, self.roi_canvas):
            mw_w.bind(EVT_MOUSEWHEEL, self._on_preview_mousewheel)
            mw_w.bind(EVT_BTN4, self._on_preview_mousewheel)
            mw_w.bind(EVT_BTN5, self._on_preview_mousewheel)

        self.roi_resize_grip = tk.Label(
            self.roi_inner, text="◢", fg="#00e5ff", bg="#16161e", font=("Segoe UI", 10, "bold"), cursor="sizing"
        )
        self.roi_resize_grip.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-1, y=-1)
        self.roi_resize_grip.bind(EVT_BTN_PRESS, self._on_roi_resize_start)
        self.roi_resize_grip.bind(EVT_B1_MOTION, self._on_roi_resize_motion)
        self.roi_resize_grip.bind(EVT_BTN_RELEASE, self._on_roi_resize_release)

    def _is_operator_mode(self) -> bool:
        try:
            return bool(self.cfg_mgr and self.cfg_mgr.is_operator_mode())
        except Exception:
            return False

    def _sync_module_buttons(self):
        mods = (self.cam_cfg or {}).get("modules") or {}
        entries = self._get_archive_entries()
        active_ids = {e["id"] for e in entries}
        is_op_mode = self._is_operator_mode()

        for old_mid in list(self._archive_buttons.keys()):
            btn = self._archive_buttons[old_mid]
            if old_mid not in active_ids:
                self._archive_buttons.pop(old_mid, None)
                try:
                    btn.destroy()
                except Exception:
                    pass
            else:
                btn.pack_forget()

        for ent in entries:
            mid = ent["id"]
            spec = ent["spec"]
            arch = spec.get("archive") or {}
            btn_text = self._resolve_archive_text(arch, "button_label", mid, spec.get("name", mid))

            btn = self._archive_buttons.get(mid)
            if btn is None:
                btn = tk.Button(
                    self.ctrl_frame,
                    text=btn_text,
                    command=lambda m=mid: self.toggle_side_panel(m),
                    bg="#3a3a46",
                    fg="#ffffff",
                    font=("Segoe UI", 9, "bold"),
                    padx=8,
                    pady=3,
                    relief=tk.FLAT,
                    cursor="hand2",
                )
                self._archive_buttons[mid] = btn
            else:
                btn.configure(text=btn_text)

            is_on = bool((mods.get(mid) or {}).get("on", False))
            is_writer = (spec.get("kind") == "recorder") or bool(spec.get("writes_to_disk"))
            has_files = self._has_archive_files(mid)
            show_btn = has_files if (is_op_mode and is_writer) else (is_on or has_files)
            if show_btn:
                btn.pack(side=tk.LEFT, padx=3, before=self.status_lbl)
                if self.sidebar_visible and self.sidebar_mode == mid:
                    btn.configure(bg="#007acc")
                else:
                    btn.configure(bg="#3a3a46" if is_on else "#282832")

    def _toggle_fullscreen_window(self):
        try:
            st = str(self.win.state()).lower()
            if st == "zoomed":
                self.win.state("normal")
                self.btn_fullscreen.configure(text="⛶")
            else:
                self.win.state("zoomed")
                self.btn_fullscreen.configure(text="🗗")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Ізольована маршрутизація колеса миші в межах цього вікна
    # ------------------------------------------------------------------
    def _set_wheel_zone(self, zone: str):
        self._wheel_zone = zone

    def _clear_wheel_zone(self, zone: str):
        if self._wheel_zone == zone:
            self._wheel_zone = None

    def _route_mousewheel(self, event):
        if self._wheel_zone == "tree":
            return self._on_tree_mousewheel(event)
        if self._wheel_zone == "preview":
            return self._on_preview_mousewheel(event)

    def _on_tree_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "delta", 0):
            delta = int(-1 * (event.delta / 120))
        else:
            delta = 0

        if getattr(event, "state", 0) & 0x0001:
            self.items_tree.yview_scroll(delta * 2, "units")
        else:
            self._navigate_items(delta)
        return "break"

    def _on_preview_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        elif getattr(event, "delta", 0):
            delta = int(-1 * (event.delta / 120))
        else:
            delta = 0

        if delta != 0:
            self._navigate_items(delta)
        return "break"

    # ------------------------------------------------------------------
    # Автоприховування верхньої панелі саме для цього вікна
    # ------------------------------------------------------------------
    def _schedule_hide(self, delay=5000):
        self._cancel_hide_timer()
        self._hide_timer = self.win.after(delay, self._hide_panel)

    def _cancel_hide_timer(self):
        if self._hide_timer is not None:
            try:
                self.win.after_cancel(self._hide_timer)
            except Exception:
                pass
            self._hide_timer = None

    def _show_panel(self):
        if not self.panel_visible:
            self.ctrl_frame.pack(fill=tk.X, side=tk.TOP, before=self.main_body)
            self.panel_visible = True

    def _hide_panel(self):
        self._hide_timer = None
        if self.sidebar_visible:
            return
        if self.panel_visible:
            self.ctrl_frame.pack_forget()
            self.panel_visible = False

    def _on_mouse_motion(self, event):
        if self._is_closing:
            return
        try:
            top_y = self.win.winfo_rooty()
            rel_y = event.y_root - top_y
        except Exception:
            rel_y = 999
        if 0 <= rel_y <= 24:
            self._show_panel()
            self._cancel_hide_timer()
            return
        if self.panel_visible:
            panel_h = max(self.ctrl_frame.winfo_height(), 50)
            if 0 <= rel_y <= panel_h:
                self._cancel_hide_timer()
            elif self._hide_timer is None and not self.sidebar_visible:
                self._schedule_hide(5000)

    # ------------------------------------------------------------------
    # Потік рендеру живого основного потоку
    # ------------------------------------------------------------------
    def _update_status_bar_from_snapshot(self, snap: dict = None):
        if snap is None:
            if not hasattr(self, "status_lbl"):
                return
            self.status_lbl.configure(
                text=tr("ui.detail.status_connecting", "Connecting..."),
                fg="#ffcc00",
            )
            return
        st = snap.get("main_status", "standby")
        fps = float(snap.get("fps", 0.0))
        rec_on = bool(snap.get("recording", False))
        rec_enc = str(snap.get("encoder", "") or "")
        op_mode = (snap.get("operation_mode") == "operator") or self._is_operator_mode()

        rec_tag = (
            f"● REC ({rec_enc})"
            if rec_on else
            (
                tr("ui.detail.rec.operator", "👤 Operator (no recording)")
                if op_mode else
                tr("ui.detail.rec.off", "Recording off")
            )
        )
        if st == "active":
            st_txt = tr("ui.detail.status.active", "{rec_tag}  |  Main stream: {fps:.1f} fps", rec_tag=rec_tag, fps=fps)
            st_col = "#75e08b" if rec_on else ("#ffcc00" if op_mode else "#00e5ff")
        elif st in ("connecting", "reconnecting"):
            st_txt = tr("ui.detail.status.reconnecting", "{rec_tag}  |  Reconnecting stream...", rec_tag=rec_tag)
            st_col = "#ffcc00"
        elif st == "offline":
            st_txt = tr("ui.detail.status.offline", "{rec_tag}  |  Stream unavailable", rec_tag=rec_tag)
            st_col = "#ff5555"
        else:
            st_txt = tr("ui.detail.status.init", "{rec_tag}  |  Initializing...", rec_tag=rec_tag)
            st_col = "#ffcc00"

        self.status_lbl.configure(text=st_txt, fg=st_col)

    def _update_main_video_placeholder(self):
        if self.img_main_tk is not None or not hasattr(self, "main_video_canvas"):
            return
        if self._last_main_offline:
            msg = (
                tr("ui.detail.err.no_cv2", "OpenCV (cv2) is not installed")
                if cv2 is None else
                tr("ui.detail.err.no_signal_main", "No signal from main stream...\nWaiting for reconnection")
            )
        else:
            msg = tr("ui.detail.waiting_main", "Waiting for main stream...")
        self.main_video_canvas.delete("live_img")
        self.main_video_canvas.delete("live_crop_badge")
        self.main_video_canvas.delete("live_drag_rect")
        self._main_img_id = None
        self._main_img_off_x = 0
        self._main_img_off_y = 0
        self._main_disp_w = 1
        self._main_disp_h = 1
        cw = max(self.main_video_canvas.winfo_width(), 320)
        ch = max(self.main_video_canvas.winfo_height(), 180)
        if self._main_ph_id is not None:
            self.main_video_canvas.itemconfigure(self._main_ph_id, text=msg, state=tk.NORMAL)
            self.main_video_canvas.coords(self._main_ph_id, cw // 2, ch // 2)

    def _render_tick(self):
        if self._is_closing:
            return
        try:
            if self.win.state() == "iconic":
                self._render_after_id = self.win.after(250, self._render_tick)
                return
        except Exception:
            pass

        detail_fps = max(5.0, min(60.0, float(self.hw_cfg.get("display_detail_fps", 25.0))))
        interval_ms = max(16, int(round(1000.0 / detail_fps)))

        try:
            if self.camera_unit is not None:
                snap = self.camera_unit.get_detail_snapshot()
                self._last_detail_snap = snap
                self._update_status_bar_from_snapshot(snap)

                st = snap.get("main_status", "standby")
                frame_ref = snap.get("frame")
                seq = int(snap.get("seq", 0))
                now = time.time()
                if frame_ref is not None and (seq != self._last_drawn_seq or (now - self._last_draw_ts) >= 0.5):
                    self._last_drawn_seq = seq
                    self._last_draw_ts = now
                    self._last_main_offline = False
                    overlay = snap.get("overlay") or {}
                    self._live_last_raw_frame = frame_ref
                    self._live_last_overlay = overlay
                    self._live_last_now = float(now)
                    self._draw_overlays_and_present(frame_ref.copy(), overlay, now, cache_raw=False)
                elif frame_ref is None and st in ("offline", "reconnecting"):
                    self.img_main_tk = None
                    self._live_last_raw_frame = None
                    self._last_main_offline = True
                    self._update_main_video_placeholder()
        except Exception as e:
            print(f"[DetailWindow render error]: {e}")

        if not self._is_closing:
            self._render_after_id = self.win.after(interval_ms, self._render_tick)

    def _draw_overlays_and_present(self, frame: np.ndarray, overlay: dict, now: float, cache_raw: bool = True):
        if frame is None or getattr(frame, "size", 0) == 0:
            return

        if cache_raw:
            try:
                self._live_last_raw_frame = frame.copy()
                self._live_last_overlay = {
                    k: (dict(v) if isinstance(v, dict) else v)
                    for k, v in (overlay or {}).items()
                }
                self._live_last_now = float(now)
            except Exception:
                pass

        mods = (self.cam_cfg or {}).get("modules") or {}

        if cv2 is not None and self.registry is not None:
            for ent in self.registry.list_modules(kind="ai"):
                mid = ent["id"]
                spec = ent["spec"]
                m_cfg = mods.get(mid) or {}
                if not m_cfg.get("on", False):
                    continue
                if not is_date_range_active(m_cfg.get("date_from", ""), m_cfg.get("date_to", "")):
                    continue
                draw_fn = spec.get("draw_overlays")
                if callable(draw_fn):
                    try:
                        draw_fn(frame, (overlay or {}).get(mid) or {}, m_cfg, now)
                    except Exception:
                        pass

        view_frame = frame
        crop_norm = self._live_crop_norm
        if crop_norm is not None:
            fh, fw = frame.shape[:2]
            if fw > 1 and fh > 1:
                nx1, ny1, nx2, ny2 = crop_norm
                cx1 = max(0, min(fw - 1, int(round(nx1 * fw))))
                cy1 = max(0, min(fh - 1, int(round(ny1 * fh))))
                cx2 = max(cx1 + 1, min(fw, int(round(nx2 * fw))))
                cy2 = max(cy1 + 1, min(fh, int(round(ny2 * fh))))
                if cx2 > cx1 and cy2 > cy1:
                    view_frame = frame[cy1:cy2, cx1:cx2]

        frame_h, frame_w = view_frame.shape[:2]
        can_w = self.main_video_canvas.winfo_width()
        can_h = self.main_video_canvas.winfo_height()
        if can_w <= 16 or can_h <= 16:
            cont_w = self.video_container.winfo_width()
            cont_h = self.video_container.winfo_height()
            can_w = cont_w if cont_w > 100 else max(can_w, 640)
            can_h = cont_h if cont_h > 100 else max(can_h, 360)

        scale = min(can_w / float(frame_w), can_h / float(frame_h))
        new_w = max(1, int(round(frame_w * scale)))
        new_h = max(1, int(round(frame_h * scale)))

        if cv2 is not None:
            interp_name = str(self.hw_cfg.get("display_resize_method", "linear"))
            if interp_name == "nearest":
                interp = cv2.INTER_NEAREST
            elif interp_name == "lanczos" or (crop_norm is not None and scale > 1.0):
                interp = cv2.INTER_LANCZOS4
            else:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

            resized = cv2.resize(view_frame, (new_w, new_h), interpolation=interp)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        else:
            rgb = view_frame[:, :, ::-1] if view_frame.ndim == 3 and view_frame.shape[2] >= 3 else view_frame
            pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR)

        self.img_main_tk = ImageTk.PhotoImage(image=pil_img)
        off_x = max(0, (can_w - new_w) // 2)
        off_y = max(0, (can_h - new_h) // 2)
        self._main_disp_w = new_w
        self._main_disp_h = new_h
        self._main_img_off_x = off_x
        self._main_img_off_y = off_y

        if self._main_ph_id is not None:
            self.main_video_canvas.itemconfigure(self._main_ph_id, state=tk.HIDDEN)

        if self._main_img_id is None:
            self._main_img_id = self.main_video_canvas.create_image(
                off_x, off_y, anchor=tk.NW, image=self.img_main_tk, tags="live_img"
            )
        else:
            self.main_video_canvas.coords(self._main_img_id, off_x, off_y)
            self.main_video_canvas.itemconfigure(self._main_img_id, image=self.img_main_tk)

        if crop_norm is not None:
            self._draw_live_crop_badge()
        else:
            self.main_video_canvas.delete("live_crop_badge")

        if self._live_drag_moved:
            try:
                self.main_video_canvas.tag_raise("live_drag_rect")
            except Exception:
                pass

    def _draw_live_crop_badge(self):
        if not hasattr(self, "main_video_canvas"):
            return
        self.main_video_canvas.delete("live_crop_badge")
        if self._live_crop_norm is None or self.img_main_tk is None:
            return
        can_h = max(self.main_video_canvas.winfo_height(), 80)
        txt = tr("ui.detail.live_crop_badge", "🔍 Live Crop (RMB — reset)")
        bx = 16
        by = max(28, can_h - 14)
        txt_id = self.main_video_canvas.create_text(
            bx, by,
            text=txt,
            fill="#00e5ff",
            font=("Segoe UI", 9, "bold"),
            anchor=tk.SW,
            tags="live_crop_badge",
        )
        bbox = self.main_video_canvas.bbox(txt_id)
        if bbox:
            bg_id = self.main_video_canvas.create_rectangle(
                bbox[0] - 8, bbox[1] - 4, bbox[2] + 8, bbox[3] + 4,
                fill="#0d111a", outline="#00e5ff", width=1,
                tags="live_crop_badge",
            )
            self.main_video_canvas.tag_lower(bg_id, txt_id)

    def _rerender_live_cached(self):
        if self._is_closing:
            return
        if self._live_last_raw_frame is not None:
            self._draw_overlays_and_present(
                self._live_last_raw_frame.copy(),
                self._live_last_overlay,
                self._live_last_now,
                cache_raw=False,
            )
        else:
            self._update_main_video_placeholder()

    def _on_main_canvas_configure(self, event=None):
        if self._is_closing:
            return
        if self.img_main_tk is None and self._main_ph_id is not None and event is not None:
            try:
                self.main_video_canvas.coords(
                    self._main_ph_id, max(1, int(event.width)) // 2, max(1, int(event.height)) // 2
                )
            except Exception:
                pass
        if self._live_last_raw_frame is not None:
            if self._live_configure_after_id is not None:
                try:
                    self.win.after_cancel(self._live_configure_after_id)
                except Exception:
                    pass
            self._live_configure_after_id = self.win.after(16, self._on_main_canvas_configure_debounced)

    def _on_main_canvas_configure_debounced(self):
        self._live_configure_after_id = None
        self._rerender_live_cached()

    def _on_main_canvas_press(self, event):
        if self.img_main_tk is None or self._main_disp_w <= 1 or self._main_disp_h <= 1:
            self._live_drag_start_x = None
            self._live_drag_start_y = None
            self._live_drag_moved = False
            self._on_video_area_click(event)
            return
        self._live_drag_start_x = max(
            self._main_img_off_x, min(self._main_img_off_x + self._main_disp_w, event.x)
        )
        self._live_drag_start_y = max(
            self._main_img_off_y, min(self._main_img_off_y + self._main_disp_h, event.y)
        )
        self._live_drag_moved = False

    def _on_main_canvas_motion(self, event):
        if self._live_drag_start_x is None or self._live_drag_start_y is None:
            return
        cur_x = max(self._main_img_off_x, min(self._main_img_off_x + self._main_disp_w, event.x))
        cur_y = max(self._main_img_off_y, min(self._main_img_off_y + self._main_disp_h, event.y))
        if abs(cur_x - self._live_drag_start_x) >= 6 or abs(cur_y - self._live_drag_start_y) >= 6:
            self._live_drag_moved = True
        if self._live_drag_moved:
            self.main_video_canvas.delete("live_drag_rect")
            self.main_video_canvas.create_rectangle(
                self._live_drag_start_x, self._live_drag_start_y, cur_x, cur_y,
                outline="#00ff00", width=2, tags="live_drag_rect"
            )
            self.main_video_canvas.tag_raise("live_drag_rect")

    def _on_main_canvas_release(self, event):
        if self._live_drag_start_x is None or self._live_drag_start_y is None:
            self._live_drag_moved = False
            return

        end_x = max(self._main_img_off_x, min(self._main_img_off_x + self._main_disp_w, event.x))
        end_y = max(self._main_img_off_y, min(self._main_img_off_y + self._main_disp_h, event.y))
        x1 = min(self._live_drag_start_x, end_x) - self._main_img_off_x
        x2 = max(self._live_drag_start_x, end_x) - self._main_img_off_x
        y1 = min(self._live_drag_start_y, end_y) - self._main_img_off_y
        y2 = max(self._live_drag_start_y, end_y) - self._main_img_off_y

        was_moved = self._live_drag_moved
        self._live_drag_start_x = None
        self._live_drag_start_y = None
        self._live_drag_moved = False
        self.main_video_canvas.delete("live_drag_rect")

        if not was_moved or (x2 - x1) < 8 or (y2 - y1) < 8 or self._main_disp_w <= 1 or self._main_disp_h <= 1:
            self._on_video_area_click(event)
            return

        rx1 = max(0.0, min(1.0, x1 / float(self._main_disp_w)))
        ry1 = max(0.0, min(1.0, y1 / float(self._main_disp_h)))
        rx2 = max(0.0, min(1.0, x2 / float(self._main_disp_w)))
        ry2 = max(0.0, min(1.0, y2 / float(self._main_disp_h)))

        base_x1, base_y1, base_x2, base_y2 = self._live_crop_norm or (0.0, 0.0, 1.0, 1.0)
        span_w = base_x2 - base_x1
        span_h = base_y2 - base_y1
        nx1 = base_x1 + rx1 * span_w
        nx2 = base_x1 + rx2 * span_w
        ny1 = base_y1 + ry1 * span_h
        ny2 = base_y1 + ry2 * span_h

        min_span = 0.01
        if (nx2 - nx1) < min_span:
            mid_x = 0.5 * (nx1 + nx2)
            nx1 = max(0.0, min(1.0 - min_span, mid_x - 0.5 * min_span))
            nx2 = nx1 + min_span
        if (ny2 - ny1) < min_span:
            mid_y = 0.5 * (ny1 + ny2)
            ny1 = max(0.0, min(1.0 - min_span, mid_y - 0.5 * min_span))
            ny2 = ny1 + min_span

        self._live_crop_norm = (nx1, ny1, nx2, ny2)
        self._rerender_live_cached()

    def _on_main_canvas_right_click(self, event=None):
        self._live_drag_start_x = None
        self._live_drag_start_y = None
        self._live_drag_moved = False
        if hasattr(self, "main_video_canvas"):
            self.main_video_canvas.delete("live_drag_rect")
        if self._live_crop_norm is not None:
            self._live_crop_norm = None
            self._rerender_live_cached()
        return "break"

    # ------------------------------------------------------------------
    # Архівний сайдбар та кропер (повністю ізольовано на archive_dir камери)
    # ------------------------------------------------------------------
    def _navigate_items(self, direction):
        all_items = []
        for day in self.items_tree.get_children():
            for child in self.items_tree.get_children(day):
                if child in self.tree_item_map:
                    all_items.append(child)
        if not all_items:
            return
        sel = self.items_tree.selection()
        if sel and sel[0] in all_items:
            curr_idx = all_items.index(sel[0])
            new_idx = max(0, min(len(all_items) - 1, curr_idx + direction))
        else:
            new_idx = 0 if direction > 0 else len(all_items) - 1
        target = all_items[new_idx]
        self.items_tree.selection_set(target)
        self.items_tree.focus(target)
        self.items_tree.see(target)
        self._on_item_selected(None)

    def _cancel_single_click_play_toggle(self):
        if self._arch_click_after_id is not None:
            try:
                self.win.after_cancel(self._arch_click_after_id)
            except Exception:
                pass
            self._arch_click_after_id = None

    def _mark_double_click(self, tag=None):
        self._cancel_single_click_play_toggle()
        if tag:
            self._ignore_release_tag = tag
            self._ignore_release_ts = time.time()

    def _schedule_single_click_play_toggle(self, tag=None):
        self._cancel_single_click_play_toggle()
        if self._get_current_record_type() != "video" or self.arch_cap is None:
            return
        if tag and self._ignore_release_tag == tag and (time.time() - self._ignore_release_ts) < 0.35:
            self._ignore_release_tag = None
            return
        self._arch_click_after_id = self.win.after(220, self._run_single_click_play_toggle)

    def _run_single_click_play_toggle(self):
        self._arch_click_after_id = None
        if self._get_current_record_type() == "video" and self.arch_cap is not None:
            self._toggle_arch_play()

    def _on_screen_canvas_double_click(self, event=None):
        self._mark_double_click("screen_canvas")
        self._open_roi_viewer(crop_mode=False)

    def _on_roi_escape(self, event=None):
        if self.roi_maximized:
            self._toggle_roi_maximize()
        elif self.roi_visible:
            self._hide_roi_panel(clear_coords=True)
        return "break"

    def _on_roi_header_double_click(self, event=None):
        self.roi_drag_moved_win = False
        self._toggle_roi_maximize()
        return "break"

    def _on_roi_panel_configure(self, event=None):
        if event is not None and event.widget is not self.roi_panel:
            return
        w = event.width if (event is not None and getattr(event, "width", 0) > 0) else None
        self._update_roi_toolbar_labels(w)

    def _on_roi_canvas_configure(self, event=None):
        if self.roi_visible or self.roi_panel.winfo_ismapped():
            self._render_current_roi_view()

    def _on_roi_canvas_press(self, event):
        if self.current_screen_img_cv is None:
            return
        if self.roi_crop_mode:
            self.roi_canvas_drag_start_x = event.x
            self.roi_canvas_drag_start_y = event.y
            self.roi_canvas_drag_moved = False
            return
        self.roi_canvas_drag_start_x = max(self.roi_img_offset_x, min(self.roi_img_offset_x + self.roi_disp_w, event.x))
        self.roi_canvas_drag_start_y = max(self.roi_img_offset_y, min(self.roi_img_offset_y + self.roi_disp_h, event.y))
        self.roi_canvas_drag_moved = False

    def _on_roi_canvas_motion(self, event):
        if self.roi_canvas_drag_start_x is None or self.roi_canvas_drag_start_y is None or self.roi_crop_mode:
            return
        cur_x = max(self.roi_img_offset_x, min(self.roi_img_offset_x + self.roi_disp_w, event.x))
        cur_y = max(self.roi_img_offset_y, min(self.roi_img_offset_y + self.roi_disp_h, event.y))
        if abs(cur_x - self.roi_canvas_drag_start_x) >= 6 or abs(cur_y - self.roi_canvas_drag_start_y) >= 6:
            self.roi_canvas_drag_moved = True
        if self.roi_canvas_drag_moved:
            self.roi_canvas.delete("roi_drag")
            self.roi_canvas.create_rectangle(
                self.roi_canvas_drag_start_x, self.roi_canvas_drag_start_y, cur_x, cur_y,
                outline="#00ff00", width=2, tags="roi_drag"
            )
            self.roi_canvas.tag_raise("roi_drag")

    def _on_roi_canvas_release(self, event):
        if self.roi_crop_mode:
            self.roi_canvas_drag_start_x = None
            self.roi_canvas_drag_start_y = None
            self.roi_canvas_drag_moved = False
            self._schedule_single_click_play_toggle("roi_canvas")
            return

        if self.roi_canvas_drag_start_x is None or self.roi_canvas_drag_start_y is None:
            self._schedule_single_click_play_toggle("roi_canvas")
            return

        end_x = max(self.roi_img_offset_x, min(self.roi_img_offset_x + self.roi_disp_w, event.x))
        end_y = max(self.roi_img_offset_y, min(self.roi_img_offset_y + self.roi_disp_h, event.y))
        x1 = min(self.roi_canvas_drag_start_x, end_x) - self.roi_img_offset_x
        x2 = max(self.roi_canvas_drag_start_x, end_x) - self.roi_img_offset_x
        y1 = min(self.roi_canvas_drag_start_y, end_y) - self.roi_img_offset_y
        y2 = max(self.roi_canvas_drag_start_y, end_y) - self.roi_img_offset_y

        was_moved = self.roi_canvas_drag_moved
        self.roi_canvas_drag_start_x = None
        self.roi_canvas_drag_start_y = None
        self.roi_canvas_drag_moved = False
        self.roi_canvas.delete("roi_drag")

        if not was_moved or (x2 - x1) < 8 or (y2 - y1) < 8 or self.current_screen_img_cv is None:
            self._schedule_single_click_play_toggle("roi_canvas")
            return

        orig_h, orig_w = self.current_screen_img_cv.shape[:2]
        crop_x1 = max(0, min(orig_w - 1, int(round(x1 * self.roi_scale_x))))
        crop_y1 = max(0, min(orig_h - 1, int(round(y1 * self.roi_scale_y))))
        crop_x2 = max(crop_x1 + 1, min(orig_w, int(round(x2 * self.roi_scale_x))))
        crop_y2 = max(crop_y1 + 1, min(orig_h, int(round(y2 * self.roi_scale_y))))
        self._apply_roi_crop_coords((crop_x1, crop_y1, crop_x2, crop_y2), open_panel=True)

    def _on_roi_canvas_double_click(self, event=None):
        self._mark_double_click("roi_canvas")
        self._toggle_roi_maximize()

    def _on_roi_right_click(self, event=None):
        if self.roi_crop_mode or self.current_roi_coords is not None or self.last_roi_crop_bgr is not None:
            self._clear_roi_crop()
        return "break"

    def _on_video_area_click(self, event=None):
        if not self.roi_visible or self.roi_maximized:
            return
        try:
            vx = self.video_container.winfo_rootx()
            vy = self.video_container.winfo_rooty()
            vw = self.video_container.winfo_width()
            vh = self.video_container.winfo_height()
            rx = self.roi_panel.winfo_x()
            ry = self.roi_panel.winfo_y()
            rw = self.roi_panel.winfo_width()
            rh = self.roi_panel.winfo_height()
            cx = rx + rw // 2
            cy = ry + rh // 2
            if vx <= cx <= (vx + vw) and vy <= cy <= (vy + vh):
                self._hide_roi_panel()
        except Exception:
            pass

    def _save_roi_geometry_to_config(self):
        self.cfg_mgr.set_camera_window_geometry(
            self.cam_uuid,
            "roi_window",
            {
                "left": int(self.roi_user_left),
                "top": int(self.roi_user_top),
                "width": int(self.roi_user_width),
                "height": int(self.roi_user_height),
                "maximized": False,
            },
        )

    def _clamp_roi_to_virtual_screen(self):
        self.roi_user_left, self.roi_user_top, self.roi_user_width, self.roi_user_height = clamp_window_geometry(
            self.roi_user_left, self.roi_user_top, self.roi_user_width, self.roi_user_height,
            min_w=260, min_h=180, widget=self.win
        )

    def _on_roi_drag_start(self, event):
        if self.roi_maximized:
            return
        self._drag_start_x = event.x_root
        self._drag_start_y = event.y_root
        self.roi_drag_moved_win = False
        try:
            self._orig_left = int(self.roi_panel.winfo_x())
            self._orig_top = int(self.roi_panel.winfo_y())
        except Exception:
            self._orig_left = self.roi_user_left
            self._orig_top = self.roi_user_top

    def _on_roi_drag_motion(self, event):
        if self.roi_maximized or not hasattr(self, "_drag_start_x"):
            return
        dx = event.x_root - self._drag_start_x
        dy = event.y_root - self._drag_start_y
        if abs(dx) >= 2 or abs(dy) >= 2:
            self.roi_drag_moved_win = True
        self.roi_user_left = int(self._orig_left + dx)
        self.roi_user_top = int(self._orig_top + dy)
        self.roi_panel.geometry(f"{self.roi_user_width}x{self.roi_user_height}+{self.roi_user_left}+{self.roi_user_top}")

    def _on_roi_drag_release(self, event):
        if self.roi_maximized:
            return
        if self.roi_drag_moved_win:
            self.roi_drag_moved_win = False
            self._save_roi_geometry_to_config()

    def _on_roi_resize_start(self, event):
        if self.roi_maximized:
            return
        self._resize_start_x = event.x_root
        self._resize_start_y = event.y_root
        self._orig_w = self.roi_user_width
        self._orig_h = self.roi_user_height
        self.roi_resize_moved = False

    def _on_roi_resize_motion(self, event):
        if self.roi_maximized or not hasattr(self, "_resize_start_x"):
            return
        dx = event.x_root - self._resize_start_x
        dy = event.y_root - self._resize_start_y
        if abs(dx) >= 1 or abs(dy) >= 1:
            self.roi_resize_moved = True
        self.roi_user_width = max(260, self._orig_w + dx)
        self.roi_user_height = max(180, self._orig_h + dy)
        self.roi_panel.geometry(f"{self.roi_user_width}x{self.roi_user_height}+{self.roi_user_left}+{self.roi_user_top}")
        self._render_current_roi_view()

    def _on_roi_resize_release(self, event):
        if self.roi_maximized:
            return
        if getattr(self, "roi_resize_moved", False):
            self.roi_resize_moved = False
            self._save_roi_geometry_to_config()

    def _compute_sidebar_width(self, container_w=None):
        if container_w is None or container_w <= 100:
            try:
                container_w = int(self.main_body.winfo_width())
            except Exception:
                container_w = 0
        if container_w <= 100:
            try:
                container_w = int(self.win.winfo_width())
            except Exception:
                container_w = 1160
        max_by_ratio = max(220, int(round(container_w * 0.42)))
        target_w = max(290, min(460, int(round(container_w * 0.25))))
        return min(target_w, max_by_ratio)

    def _sync_sidebar_width(self, container_w=None):
        target_w = self._compute_sidebar_width(container_w)
        self.side_panel_width = target_w
        try:
            self.side_panel.configure(width=target_w)
            self.items_tree.column("#0", width=max(160, target_w - 30))
        except Exception:
            pass
        return target_w

    def _get_sidebar_preview_budgets(self, mode=None):
        mode = mode or self.sidebar_mode
        rtype = self._get_current_record_type(mode)
        try:
            pw = int(self.side_panel.winfo_width())
        except Exception:
            pw = 0
        if pw <= 60:
            pw = self.side_panel_width or self._compute_sidebar_width()
        slot_w = max(140, pw - 18)

        try:
            ph = int(self.side_panel.winfo_height())
        except Exception:
            ph = 0
        if ph <= 60:
            try:
                ph = int(self.main_body.winfo_height())
            except Exception:
                ph = 580
        if ph <= 60:
            ph = 580

        try:
            side_hdr_h = max(28, int(self.side_header.winfo_reqheight()))
            search_h = max(28, int(self.search_frame.winfo_reqheight()))
            scr_hdr_h = max(22, int(self.screen_header_frame.winfo_reqheight())) + 6
        except Exception:
            side_hdr_h, search_h, scr_hdr_h = 34, 34, 30
        top_chrome_h = side_hdr_h + search_h

        if rtype == "pair_image":
            try:
                prev_hdr_h = max(18, int(self.preview_info_lbl.winfo_reqheight())) + 2
            except Exception:
                prev_hdr_h = 22
            item_border_h = 4
            ctrl_h = 0
        elif rtype == "video":
            prev_hdr_h, item_border_h = 0, 0
            try:
                ctrl_h = max(32, int(self.video_controls_frame.winfo_reqheight())) + 6
            except Exception:
                ctrl_h = 38
        else:
            prev_hdr_h, item_border_h, ctrl_h = 0, 0, 0

        preview_chrome_h = 10 + prev_hdr_h + item_border_h + scr_hdr_h + 4 + ctrl_h
        min_tree_h = max(80, int(round(ph * (0.24 if ph < 500 else 0.28))))
        preview_budget = max(120, ph - top_chrome_h - min_tree_h - 6)
        img_budget = max(80, preview_budget - preview_chrome_h)

        if rtype == "pair_image":
            min_screen_floor = max(48, int(round(img_budget * 0.48)))
            base_item_max_h = max(
                36,
                min(int(round(img_budget * 0.38)), int(round(slot_w * 0.58)), max(36, img_budget - min_screen_floor)),
            )
            if self.current_item_preview_pil is not None:
                iw, ih = self.current_item_preview_pil.size
                if iw > 0 and ih > 0:
                    iscale = min(slot_w / float(iw), base_item_max_h / float(ih))
                    used_item_h = max(1, min(base_item_max_h, int(round(ih * iscale))))
                else:
                    used_item_h = min(base_item_max_h, 40)
            else:
                used_item_h = min(base_item_max_h, 38)
            max_screen_h = max(48, img_budget - used_item_h)
            return slot_w, base_item_max_h, max_screen_h
        return slot_w, 0, max(60, img_budget)

    def _on_main_body_configure(self, event=None):
        if event is not None and event.widget is not self.main_body:
            return
        mw = int(getattr(event, "width", 0) or self.main_body.winfo_width() or 0)
        mh = int(getattr(event, "height", 0) or self.main_body.winfo_height() or 0)
        if mw < 200 or mh < 150 or (mw, mh) == self._last_main_body_size:
            return
        self._last_main_body_size = (mw, mh)
        self._sync_sidebar_width(mw)
        if self.sidebar_visible:
            self._relayout_sidebar_previews()

    def _on_side_panel_configure(self, event=None):
        if event is not None and event.widget is not self.side_panel:
            return
        if not self.sidebar_visible:
            return
        pw = int(getattr(event, "width", 0) or self.side_panel.winfo_width() or 0)
        ph = int(getattr(event, "height", 0) or self.side_panel.winfo_height() or 0)
        if pw >= 60 and ph >= 60:
            self._relayout_sidebar_previews()

    def _relayout_sidebar_previews(self, force=False):
        if self._in_sidebar_relayout or not self.sidebar_visible:
            return
        try:
            pw = int(self.side_panel.winfo_width())
            ph = int(self.side_panel.winfo_height())
        except Exception:
            pw, ph = self.side_panel_width, 0

        budgets = self._get_sidebar_preview_budgets()
        state_key = (pw, ph, self.sidebar_mode, budgets, self._preview_revision)
        if not force and state_key == self._last_sidebar_layout_key:
            return

        self._in_sidebar_relayout = True
        try:
            self._last_sidebar_layout_key = state_key
            if self._get_current_record_type() == "pair_image" and self.current_item_preview_pil is not None:
                self._render_item_preview_image()
            if self.current_screen_img_cv is not None:
                self._display_cv_in_screen_canvas(self.current_screen_img_cv, keep_roi=True)
            else:
                ph_txt, ph_col = self._screen_placeholder_info
                self._set_screen_canvas_placeholder(ph_txt, color=ph_col)
            self._last_sidebar_layout_key = (
                pw, ph, self.sidebar_mode, self._get_sidebar_preview_budgets(), self._preview_revision
            )
        finally:
            self._in_sidebar_relayout = False

    def _update_preview_layout(self, mode):
        rtype = self._get_current_record_type(mode)
        if rtype != "pair_image":
            self.preview_info_lbl.pack_forget()
            self.preview_label.pack_forget()
        else:
            if str(self.preview_info_lbl.winfo_manager()) != "pack":
                self.preview_info_lbl.pack(side=tk.TOP, fill=tk.X, pady=(0, 2), before=self.screen_header_frame)
            if str(self.preview_label.winfo_manager()) != "pack":
                self.preview_label.pack(side=tk.TOP, fill=tk.X, before=self.screen_header_frame)

        if rtype == "video":
            self.screen_info_lbl.configure(text=tr("ui.detail.screen_header_video", "Video preview (crop / 2x click):"))
            if str(self.video_controls_frame.winfo_manager()) != "pack":
                self.video_controls_frame.pack(side=tk.TOP, fill=tk.X, pady=(4, 0), after=self.screen_canvas)
        else:
            self.video_controls_frame.pack_forget()
            self.screen_info_lbl.configure(text=tr("ui.detail.screen_header_photo", "Screenshot (crop / 2x click):"))

    def _clear_item_preview_by_key(self, key: str, default_text: str):
        self._item_placeholder_key = (key, default_text)
        self.current_item_preview_pil = None
        self.current_item_base_name = ""
        self.preview_photo_tk = None
        arch = self._get_archive_spec(self.sidebar_mode)
        lbl_type = self._resolve_archive_text(
            arch, "item_preview_label", self.sidebar_mode,
            tr("ui.detail.item_preview_default", "Snapshot:")
        )
        self.preview_info_lbl.configure(text=lbl_type)
        self.preview_label.configure(image="", text=tr(key, default_text), pady=8)
        self.current_item_disp_h = 38

    def _clear_item_preview(self, placeholder=None):
        if placeholder is None:
            self._clear_item_preview_by_key("ui.detail.ph.select_record", "[Select a record from list]")
            return
        self.current_item_preview_pil = None
        self.current_item_base_name = ""
        self.preview_photo_tk = None
        self.preview_info_lbl.configure(text=tr("ui.detail.item_preview_default", "Snapshot:"))
        self.preview_label.configure(image="", text=placeholder, pady=8)
        self.current_item_disp_h = 38

    def _set_screen_canvas_placeholder_key(self, key: str, default_text: str, color: str = "#707080"):
        self._screen_placeholder_spec = (key, default_text, color)
        self._set_screen_canvas_placeholder(tr(key, default_text), color=color)

    def _set_screen_canvas_placeholder(self, text=None, color="#707080"):
        if text is None:
            text = tr("ui.detail.ph.select_record", "[Select a record from list]")
        self._screen_placeholder_info = (text, color)
        self.screen_canvas.delete("all")
        self.screen_canvas_img_id = None
        self.current_screen_img_cv = None
        self.screen_photo_tk = None
        slot_w, _, max_screen_h = self._get_sidebar_preview_budgets()
        ph_w = slot_w
        ph_h = min(max_screen_h, max(72, int(round(ph_w * 9.0 / 16.0))))
        self.canvas_disp_w = ph_w
        self.canvas_disp_h = ph_h
        self.canvas_slot_w = ph_w
        self.canvas_slot_h = ph_h
        self.canvas_img_offset_x = 1
        self.canvas_img_offset_y = 1
        self.screen_canvas.config(width=ph_w, height=ph_h)
        self.screen_canvas.create_text(
            ph_w // 2 + 1, ph_h // 2 + 1,
            text=text, fill=color, font=("Segoe UI", 9),
            width=max(100, ph_w - 16), tags="placeholder",
        )

    def _clear_screen_preview(self, clear_item=True):
        if clear_item:
            self._clear_item_preview_by_key("ui.detail.ph.select_record", "[Select a record from list]")
        self.current_preview_fpath = None
        self._set_screen_canvas_placeholder_key("ui.detail.ph.select_record", "[Select a record from list]", color="#707080")

    def toggle_side_panel(self, mode=None):
        mode = mode or self.sidebar_mode or self._pick_default_sidebar_mode()
        if not mode:
            return
        if self.sidebar_visible and self.sidebar_mode == mode:
            self._stop_archive_player()
            self.side_panel.pack_forget()
            self.video_container.pack_forget()
            self.video_container.pack(fill=tk.BOTH, expand=True)
            self.sidebar_visible = False
            self._sync_module_buttons()
        else:
            prev_mode = self.sidebar_mode
            self.sidebar_mode = mode
            if self._get_current_record_type(mode) != "video":
                self._stop_archive_player()
            self.video_container.pack_forget()
            self._sync_sidebar_width()
            self.side_panel.pack(side=tk.LEFT, fill=tk.Y)
            self.video_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.sidebar_visible = True
            self._sync_module_buttons()

            arch = self._get_archive_spec(mode)
            header_title = self._resolve_archive_text(
                arch, "sidebar_title", mode,
                tr("ui.detail.archive_title_fmt", "Archive: {mod}", mod=mode),
            )
            self.side_header_lbl.configure(text=header_title)

            self._update_preview_layout(mode)
            if prev_mode != mode:
                self._hide_roi_panel(clear_coords=True)
                self._clear_screen_preview(clear_item=True)
                self.items_tree.delete(*self.items_tree.get_children())
                self.tree_item_map.clear()

            self._last_sidebar_layout_key = None
            self.refresh_saved_list()
            self._relayout_sidebar_previews(force=True)

    def _sync_sort_button_text(self):
        if self.sort_mode == 0:
            self.btn_sort.configure(text=tr("ui.detail.sort_time_desc", "▼ Time"))
        elif self.sort_mode == 1:
            self.btn_sort.configure(text=tr("ui.detail.sort_time_asc", "▲ Time"))
        else:
            self.btn_sort.configure(text=tr("ui.detail.sort_name", "🔤 Name"))

    def _toggle_sort(self):
        self.sort_mode = (self.sort_mode + 1) % 3
        self._sync_sort_button_text()
        self._apply_search_and_sort()

    def refresh_saved_list(self):
        self.raw_records.clear()
        if not self.sidebar_mode:
            self._apply_search_and_sort()
            return
        target_dir = os.path.join(self.archive_dir, self.sidebar_mode)
        if not target_dir or not os.path.exists(target_dir):
            self._apply_search_and_sort()
            return

        arch = self._get_archive_spec(self.sidebar_mode)
        rtype = str(arch.get("record_type") or "single_image")
        default_exts = (".mp4", ".avi", ".mkv") if rtype == "video" else (".jpg", ".jpeg", ".png")
        exts = tuple(str(e).lower() for e in (arch.get("extensions") or default_exts))
        parse_fn = arch.get("parse_record")

        for root_dir, _, files in os.walk(target_dir):
            if "_staging" in root_dir:
                continue
            for fname in files:
                if not fname.lower().endswith(exts):
                    continue
                fpath = os.path.join(root_dir, fname)
                if self.camera_unit and self.camera_unit.is_archive_file_busy(self.sidebar_mode, fpath):
                    continue

                base_name, _ = os.path.splitext(fname)
                rel_dir = os.path.relpath(root_dir, target_dir)
                try:
                    mtime = os.path.getmtime(fpath)
                except Exception:
                    continue

                day_key = rel_dir if rel_dir != "." else datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

                parsed = None
                if callable(parse_fn):
                    try:
                        parsed = parse_fn(fpath, base_name, root_dir, rel_dir)
                    except Exception:
                        parsed = None
                    if parsed is None:
                        continue
                else:
                    parsed = {"time": "--:--:--", "name": base_name}

                self.raw_records.append({
                    "day": day_key,
                    "time": str(parsed.get("time") or "--:--:--"),
                    "name": str(parsed.get("name") or base_name),
                    "fpath": fpath,
                    "mtime": mtime,
                })

        self._apply_search_and_sort()

    def _apply_search_and_sort(self):
        prev_day_nodes = self.items_tree.get_children()
        had_days = bool(prev_day_nodes)
        open_days = set()
        known_days = set()
        for d_id in prev_day_nodes:
            d_txt = self.items_tree.item(d_id, "text")
            if d_txt.startswith("📁 "):
                d_name = d_txt[2:].strip()
                known_days.add(d_name)
                if self.items_tree.item(d_id, "open"):
                    open_days.add(d_name)

        prev_sel = self.items_tree.selection()
        selected_fpath = self.tree_item_map.get(prev_sel[0]) if prev_sel else self.current_preview_fpath

        prev_yview_top = 0.0
        top_anchor = None
        if had_days:
            try:
                prev_yview_top = float(self.items_tree.yview()[0])
                if prev_yview_top > 0.001:
                    top_iid = self.items_tree.identify_row(4)
                    if top_iid:
                        if top_iid in self.tree_item_map:
                            top_anchor = ("item", self.tree_item_map[top_iid])
                        else:
                            t_txt = self.items_tree.item(top_iid, "text")
                            if t_txt.startswith("📁 "):
                                top_anchor = ("day", t_txt[2:].strip())
            except Exception:
                pass

        self._arch_tree_syncing = True
        try:
            if prev_day_nodes:
                self.items_tree.delete(*prev_day_nodes)
            self.tree_item_map.clear()

            query = self.search_var.get().strip().upper()
            filtered = [
                r for r in self.raw_records
                if not query or (query in r["name"].upper()) or (query in r["time"]) or (query in r["day"])
            ]

            if self.sort_mode == 0:
                filtered.sort(key=lambda x: x["mtime"], reverse=True)
            elif self.sort_mode == 1:
                filtered.sort(key=lambda x: x["mtime"], reverse=False)
            else:
                filtered.sort(key=lambda x: x["name"].upper())

            grouped = {}
            for r in filtered:
                grouped.setdefault(r["day"], []).append(r)

            arch = self._get_archive_spec(self.sidebar_mode)
            icon = str(arch.get("icon") or "📄")

            item_to_reselect = None
            for idx, day_str in enumerate(sorted(grouped.keys(), reverse=(self.sort_mode != 1))):
                if not had_days:
                    is_open = (idx == 0)
                else:
                    is_open = (day_str in open_days) or (idx == 0 and day_str not in known_days)

                day_node = self.items_tree.insert("", "end", text=f"📁 {day_str}", tags=("day_tag",), open=is_open)
                for rec in grouped[day_str]:
                    disp_str = f"🕒 {rec['time']}   {icon} {rec['name']}"
                    item_id = self.items_tree.insert(day_node, "end", text=f"  {disp_str}", tags=("item_tag",))
                    self.tree_item_map[item_id] = rec["fpath"]
                    if selected_fpath and rec["fpath"] == selected_fpath:
                        item_to_reselect = item_id
                        self.items_tree.item(day_node, open=True)

            if item_to_reselect is not None:
                self.items_tree.selection_set(item_to_reselect)
                self.items_tree.focus(item_to_reselect)

            if had_days and prev_yview_top > 0.001:
                visible_rows = 0
                anchor_idx = None
                for d_node in self.items_tree.get_children():
                    d_row = visible_rows
                    visible_rows += 1
                    dt = self.items_tree.item(d_node, "text")
                    dn = dt[2:].strip() if dt.startswith("📁 ") else ""
                    if top_anchor == ("day", dn):
                        anchor_idx = d_row
                    if self.items_tree.item(d_node, "open"):
                        for ch_id in self.items_tree.get_children(d_node):
                            c_row = visible_rows
                            visible_rows += 1
                            ch_fp = self.tree_item_map.get(ch_id)
                            if ch_fp and top_anchor == ("item", ch_fp):
                                anchor_idx = c_row
                self.items_tree.update_idletasks()
                if anchor_idx is not None and visible_rows > 0:
                    self.items_tree.yview_moveto(max(0.0, min(1.0, anchor_idx / float(visible_rows))))
                else:
                    self.items_tree.yview_moveto(prev_yview_top)

            self.items_tree.update_idletasks()
        finally:
            self._arch_tree_syncing = False

    def _show_item_in_preview(self, fpath):
        if not fpath or not os.path.exists(fpath):
            self._clear_item_preview_by_key("ui.detail.ph.item_missing", "[Snapshot missing]")
            return
        try:
            with Image.open(fpath) as pil_src:
                pil_src.load()
                self.current_item_preview_pil = pil_src.copy()
            self.current_item_base_name = os.path.splitext(os.path.basename(fpath))[0]
            self._render_item_preview_image()
        except Exception:
            self._clear_item_preview_by_key("ui.detail.ph.item_read_err", "[Error reading snapshot]")

    def _render_item_preview_image(self):
        if self.current_item_preview_pil is None:
            return
        slot_w, max_item_h, _ = self._get_sidebar_preview_budgets()
        orig_w, orig_h = self.current_item_preview_pil.size
        if orig_w <= 0 or orig_h <= 0:
            return
        scale = min(slot_w / float(orig_w), max_item_h / float(orig_h))
        target_w = max(1, int(round(orig_w * scale)))
        target_h = max(1, int(round(orig_h * scale)))
        pil_img = self.current_item_preview_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self.preview_photo_tk = ImageTk.PhotoImage(pil_img)
        self.current_item_disp_h = target_h

        arch = self._get_archive_spec(self.sidebar_mode)
        lbl_type = self._resolve_archive_text(
            arch, "item_preview_label", self.sidebar_mode,
            tr("ui.detail.item_preview_default", "Snapshot:"),
        )
        self.preview_info_lbl.configure(text=f"{lbl_type} {self.current_item_base_name}")
        self.preview_label.configure(image=self.preview_photo_tk, text="", pady=0)

    @staticmethod
    def _extract_roi_bgr(cv_img, coords):
        if cv_img is None or coords is None:
            return None
        orig_h, orig_w = cv_img.shape[:2]
        x1, y1, x2, y2 = coords
        cx1 = max(0, min(orig_w - 1, int(x1)))
        cy1 = max(0, min(orig_h - 1, int(y1)))
        cx2 = max(cx1 + 1, min(orig_w, int(x2)))
        cy2 = max(cy1 + 1, min(orig_h, int(y2)))
        crop = cv_img[cy1:cy2, cx1:cx2]
        return crop.copy() if crop.size > 0 else None

    def _draw_roi_rect_on_screen_canvas(self):
        self.screen_canvas.delete("roi_rect")
        if self.current_roi_coords is None or self.current_screen_img_cv is None:
            return
        if self.canvas_scale_x <= 0 or self.canvas_scale_y <= 0:
            return
        cx1, cy1, cx2, cy2 = self.current_roi_coords
        sx1 = self.canvas_img_offset_x + cx1 / self.canvas_scale_x
        sy1 = self.canvas_img_offset_y + cy1 / self.canvas_scale_y
        sx2 = self.canvas_img_offset_x + cx2 / self.canvas_scale_x
        sy2 = self.canvas_img_offset_y + cy2 / self.canvas_scale_y
        self.screen_canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#00ff00", width=2, tags="roi_rect")
        self.screen_canvas.tag_raise("roi_rect")

    def _display_cv_in_screen_canvas(self, cv_img, keep_roi=True):
        if cv_img is None or cv_img.size == 0:
            return
        try:
            self.current_screen_img_cv = cv_img
            orig_h, orig_w = cv_img.shape[:2]
            if orig_w <= 0 or orig_h <= 0:
                return

            slot_w, _, max_screen_h = self._get_sidebar_preview_budgets()
            scale = min(slot_w / float(orig_w), max_screen_h / float(orig_h))
            disp_w = max(1, int(round(orig_w * scale)))
            disp_h = max(1, int(round(orig_h * scale)))
            off_x = 1 + max(0, (slot_w - disp_w) // 2)
            off_y = 1

            if cv2 is not None:
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                resized = cv2.resize(cv_img, (disp_w, disp_h), interpolation=interp)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
            else:
                rgb = cv_img[:, :, ::-1] if cv_img.ndim == 3 and cv_img.shape[2] >= 3 else cv_img
                pil_img = Image.fromarray(rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)

            self.screen_photo_tk = ImageTk.PhotoImage(pil_img)

            slot_changed = (slot_w != self.canvas_slot_w) or (disp_h != self.canvas_slot_h)
            img_geom_changed = (
                slot_changed
                or disp_w != self.canvas_disp_w
                or disp_h != self.canvas_disp_h
                or off_x != self.canvas_img_offset_x
                or off_y != self.canvas_img_offset_y
            )

            self.canvas_scale_x = orig_w / float(disp_w)
            self.canvas_scale_y = orig_h / float(disp_h)
            self.canvas_disp_w = disp_w
            self.canvas_disp_h = disp_h
            self.canvas_img_offset_x = off_x
            self.canvas_img_offset_y = off_y
            self.canvas_slot_w = slot_w
            self.canvas_slot_h = disp_h

            self.screen_canvas.delete("placeholder")
            if slot_changed:
                self.screen_canvas.config(width=slot_w, height=disp_h)

            if self.screen_canvas_img_id is None or not self.screen_canvas.find_withtag("frame_img"):
                self.screen_canvas.delete("frame_img")
                self.screen_canvas_img_id = self.screen_canvas.create_image(
                    off_x, off_y, anchor=tk.NW, image=self.screen_photo_tk, tags="frame_img"
                )
            else:
                if img_geom_changed:
                    self.screen_canvas.coords(self.screen_canvas_img_id, off_x, off_y)
                self.screen_canvas.itemconfig(self.screen_canvas_img_id, image=self.screen_photo_tk)

            if keep_roi and self.current_roi_coords is not None:
                if self.drag_start_x is None:
                    self._draw_roi_rect_on_screen_canvas()
                crop_bgr = self._extract_roi_bgr(cv_img, self.current_roi_coords)
                if crop_bgr is not None:
                    self.last_roi_crop_bgr = crop_bgr
            else:
                self.screen_canvas.delete("roi_rect")

            if self.screen_canvas.find_withtag("roi_rect"):
                self.screen_canvas.tag_raise("roi_rect")

            if self.roi_visible or self.roi_panel.winfo_ismapped():
                self._render_current_roi_view()
        except Exception as e:
            print(f"[Preview render error]: {e}")

    def _show_screenshot_in_canvas(self, screen_path):
        if not screen_path or not os.path.exists(screen_path):
            self._set_screen_canvas_placeholder_key("ui.detail.ph.screen_missing", "[Screenshot missing]", color="#707080")
            if self.roi_visible or self.roi_panel.winfo_ismapped():
                self._sync_roi_controls_visibility()
                self._render_current_roi_view()
            return
        try:
            if cv2 is not None:
                cv_img = cv2.imread(screen_path)
            else:
                with Image.open(screen_path) as pim:
                    pim.load()
                    rgb_arr = np.asarray(pim.convert("RGB"))
                    cv_img = rgb_arr[:, :, ::-1].copy()

            if cv_img is None:
                self._set_screen_canvas_placeholder_key("ui.detail.ph.screen_read_err", "[Error reading screenshot]", color="#ff5555")
                return
            is_open = bool(self.roi_visible or self.roi_panel.winfo_ismapped())
            keep = bool(is_open and self.current_roi_coords is not None)
            if not is_open and self.current_roi_coords is not None:
                self._hide_roi_panel(clear_coords=True)
            self._display_cv_in_screen_canvas(cv_img, keep_roi=keep)
            if is_open:
                self._sync_roi_controls_visibility()
        except Exception as e:
            print(f"[Screenshot preview error]: {e}")

    @staticmethod
    def _format_sec(sec: float) -> str:
        s = max(0, int(round(sec)))
        if s >= 3600:
            h, rem = divmod(s, 3600)
            m, s2 = divmod(rem, 60)
            return f"{h:d}:{m:02d}:{s2:02d}"
        m, s2 = divmod(s, 60)
        return f"{m:02d}:{s2:02d}"

    def _update_arch_time_ui(self):
        total = max(1, self.arch_total_frames)
        curr = max(0, min(total - 1, self.arch_curr_frame_idx))
        fps = max(1.0, self.arch_fps)
        time_str = f"{self._format_sec(curr / fps)}/{self._format_sec(total / fps)}"
        self.lbl_arch_time.configure(text=time_str)
        self.roi_time_lbl.configure(text=time_str)
        self.arch_updating_slider = True
        try:
            update_scale_range_and_value(self.arch_slider, self.arch_slider_var, curr, max(1, total - 1))
            update_scale_range_and_value(self.roi_slider, self.roi_slider_var, curr, max(1, total - 1))
        finally:
            self.arch_updating_slider = False

    def _update_arch_play_buttons(self):
        icon = "⏸" if self.arch_playing else "▶"
        bg = "#d97706" if self.arch_playing else "#28a745"
        self.btn_arch_play.configure(text=icon, bg=bg)
        self.roi_btn_play.configure(text=icon, bg=bg)

    def _stop_archive_player(self):
        self._cancel_single_click_play_toggle()
        self.arch_playing = False
        if self.arch_after_id is not None:
            try:
                self.win.after_cancel(self.arch_after_id)
            except Exception:
                pass
            self.arch_after_id = None
        if self.arch_cap is not None:
            try:
                self.arch_cap.release()
            except Exception:
                pass
            self.arch_cap = None
        self.arch_fpath = None
        self.arch_total_frames = 0
        self.arch_curr_frame_idx = 0
        self._update_arch_play_buttons()
        self.lbl_arch_time.configure(text="00:00/00:00")
        self.roi_time_lbl.configure(text="00:00/00:00")
        self.arch_updating_slider = True
        try:
            update_scale_range_and_value(self.arch_slider, self.arch_slider_var, 0, 1)
            update_scale_range_and_value(self.roi_slider, self.roi_slider_var, 0, 1)
        finally:
            self.arch_updating_slider = False
        self._sync_roi_controls_visibility()

    def _load_archive_video(self, fpath, auto_play=True):
        self._stop_archive_player()
        if not fpath or not os.path.exists(fpath):
            self._set_screen_canvas_placeholder_key("ui.detail.ph.video_missing", "[Video file missing]", color="#707080")
            if self.roi_visible or self.roi_panel.winfo_ismapped():
                self._sync_roi_controls_visibility()
                self._render_current_roi_view()
            return

        if cv2 is None:
            self._set_screen_canvas_placeholder_key("ui.detail.ph.no_cv2", "[OpenCV not installed]", color="#ff5555")
            return

        if self.camera_unit and self.camera_unit.is_archive_file_busy(self.sidebar_mode, fpath):
            self._set_screen_canvas_placeholder_key("ui.detail.ph.video_recording", "[File is still recording...]", color="#ffcc00")
            if self.roi_visible or self.roi_panel.winfo_ismapped():
                self._sync_roi_controls_visibility()
                self._render_current_roi_view()
            return

        cap, fps, total_frames = open_video_capture_for_playback(cv2, fpath)
        if cap is None:
            self._set_screen_canvas_placeholder_key("ui.detail.ph.video_open_err", "[Failed to open video]", color="#ff5555")
            if self.roi_visible or self.roi_panel.winfo_ismapped():
                self._sync_roi_controls_visibility()
                self._render_current_roi_view()
            return

        self.arch_cap = cap
        self.arch_fpath = fpath
        self.arch_fps = fps
        self.arch_total_frames = total_frames
        self.arch_curr_frame_idx = 0

        ret, frame, landed_idx, updated_tot = seek_video_capture(
            cap,
            cv2,
            fps,
            total_frames,
            target_frame=0,
        )
        if ret and frame is not None:
            self.arch_curr_frame_idx = landed_idx
            self.arch_total_frames = updated_tot
            keep = bool(self.current_roi_coords is not None and (self.roi_visible or self.roi_panel.winfo_ismapped()))
            self._display_cv_in_screen_canvas(frame, keep_roi=keep)
        else:
            cap.release()
            self.arch_cap = None
            return

        self._update_arch_time_ui()
        self._sync_roi_controls_visibility()

        if auto_play:
            self.arch_playing = True
            self._update_arch_play_buttons()
            self._schedule_arch_tick(int(round(1000.0 / max(1.0, self.arch_fps))))
        else:
            self.arch_playing = False
            self._update_arch_play_buttons()

    def _schedule_arch_tick(self, delay_ms=None):
        if self.arch_after_id is not None:
            try:
                self.win.after_cancel(self.arch_after_id)
            except Exception:
                pass
            self.arch_after_id = None
        if not self.arch_playing or self._is_closing:
            return
        if delay_ms is None:
            effective_rate = self.arch_fps * (2 if self.arch_speed >= 2 else 1)
            delay_ms = max(16, int(round(1000.0 / max(1.0, effective_rate))))
        self.arch_after_id = self.win.after(delay_ms, self._arch_player_tick)

    def _arch_player_tick(self):
        self.arch_after_id = None
        if not self.arch_playing or self.arch_cap is None or self._is_closing:
            return

        t0 = time.perf_counter()
        step = 2 if self.arch_speed >= 4 else 1
        for _ in range(step - 1):
            if not self.arch_cap.grab():
                break
            self.arch_curr_frame_idx += 1

        ret, frame = self.arch_cap.read()
        if not ret or frame is None:
            self.arch_playing = False
            self._update_arch_play_buttons()
            return

        pos_f = float(self.arch_cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0) if cv2 is not None else 0.0
        if pos_f >= 1.0:
            self.arch_curr_frame_idx = max(0, int(round(pos_f)) - 1)
        else:
            self.arch_curr_frame_idx += 1
        if self.arch_curr_frame_idx >= self.arch_total_frames:
            self.arch_total_frames = self.arch_curr_frame_idx + 1

        self._display_cv_in_screen_canvas(frame, keep_roi=True)
        self._update_arch_time_ui()

        effective_rate = self.arch_fps * (2 if self.arch_speed >= 2 else 1)
        target_ms = 1000.0 / max(1.0, effective_rate)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._schedule_arch_tick(max(12, int(round(target_ms - elapsed_ms))))

    def _toggle_arch_play(self):
        if self.arch_cap is None:
            return
        if self.arch_playing:
            self.arch_playing = False
            if self.arch_after_id is not None:
                try:
                    self.win.after_cancel(self.arch_after_id)
                except Exception:
                    pass
                self.arch_after_id = None
            self._update_arch_play_buttons()
        else:
            if self.arch_curr_frame_idx >= max(1, self.arch_total_frames - 1):
                if cv2 is not None:
                    self.arch_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.arch_curr_frame_idx = 0
            self.arch_playing = True
            self._update_arch_play_buttons()
            self._schedule_arch_tick(16)

    def _seek_archive_video(self, frame_idx: int):
        if self.arch_cap is None or cv2 is None:
            return
        ret, frame, landed_idx, updated_tot = seek_video_capture(
            self.arch_cap,
            cv2,
            self.arch_fps,
            self.arch_total_frames,
            target_frame=int(frame_idx),
        )
        if ret and frame is not None:
            self.arch_total_frames = updated_tot
            self.arch_curr_frame_idx = landed_idx
            self._display_cv_in_screen_canvas(frame, keep_roi=True)
            self._update_arch_time_ui()

    def _on_arch_slider_move(self, val_str):
        if self.arch_updating_slider or self.arch_cap is None:
            return
        try:
            idx = int(float(val_str))
        except Exception:
            return
        if idx != self.arch_curr_frame_idx:
            self._seek_archive_video(idx)

    def _cycle_arch_speed(self):
        speeds = [1, 2, 4]
        idx = speeds.index(self.arch_speed) if self.arch_speed in speeds else 0
        self.arch_speed = speeds[(idx + 1) % len(speeds)]
        txt = f"{self.arch_speed}x"
        self.btn_arch_speed.configure(text=txt)
        self.roi_btn_speed.configure(text=txt)

    def _on_item_selected(self, event):
        if self._arch_tree_syncing:
            return
        sel = self.items_tree.selection()
        if not sel:
            return
        fpath = self.tree_item_map.get(sel[0])
        if not fpath:
            return
        rtype = self._get_current_record_type()
        already_loaded = (
            (self.arch_cap is not None and self.arch_fpath == fpath)
            if rtype == "video"
            else (self.current_screen_img_cv is not None)
        )
        if event is not None and fpath == self.current_preview_fpath and already_loaded:
            return

        self.current_preview_fpath = fpath
        self._preview_revision += 1

        if rtype == "video":
            self._load_archive_video(fpath, auto_play=True)
            return

        self._stop_archive_player()
        if rtype == "pair_image":
            self._show_item_in_preview(fpath)
            item_dir = os.path.dirname(fpath)
            time_prefix = os.path.basename(fpath).split("_")[0]
            cand = os.path.join(item_dir, f"{time_prefix}_full.jpg")
            screen_path = cand if os.path.exists(cand) else fpath
        else:
            screen_path = fpath

        self._show_screenshot_in_canvas(screen_path)

    def _on_item_double_clicked(self, event):
        sel = self.items_tree.selection()
        if not sel:
            return
        item_id = sel[0]
        if item_id not in self.tree_item_map:
            return
        fpath = self.tree_item_map[item_id]
        if self.current_preview_fpath != fpath or self.current_screen_img_cv is None:
            self._on_item_selected(None)
        if self.current_screen_img_cv is not None:
            self._open_roi_viewer(crop_mode=False)

    def _on_screen_canvas_press(self, event):
        if self.current_screen_img_cv is None:
            return
        min_x, max_x = self.canvas_img_offset_x, self.canvas_img_offset_x + self.canvas_disp_w
        min_y, max_y = self.canvas_img_offset_y, self.canvas_img_offset_y + self.canvas_disp_h
        self.drag_start_x = max(min_x, min(max_x, event.x))
        self.drag_start_y = max(min_y, min(max_y, event.y))
        self.drag_moved = False

    def _on_screen_canvas_motion(self, event):
        if self.drag_start_x is None or self.drag_start_y is None or self.current_screen_img_cv is None:
            return
        min_x, max_x = self.canvas_img_offset_x, self.canvas_img_offset_x + self.canvas_disp_w
        min_y, max_y = self.canvas_img_offset_y, self.canvas_img_offset_y + self.canvas_disp_h
        cur_x = max(min_x, min(max_x, event.x))
        cur_y = max(min_y, min(max_y, event.y))
        if abs(cur_x - self.drag_start_x) >= 4 or abs(cur_y - self.drag_start_y) >= 4:
            self.drag_moved = True
        if self.drag_moved:
            self.screen_canvas.delete("roi_rect")
            self.screen_canvas.create_rectangle(
                self.drag_start_x, self.drag_start_y, cur_x, cur_y, outline="#00ff00", width=2, tags="roi_rect"
            )
            self.screen_canvas.tag_raise("roi_rect")

    def _on_screen_canvas_release(self, event):
        if self.drag_start_x is None or self.drag_start_y is None or self.current_screen_img_cv is None:
            return
        min_x, max_x = self.canvas_img_offset_x, self.canvas_img_offset_x + self.canvas_disp_w
        min_y, max_y = self.canvas_img_offset_y, self.canvas_img_offset_y + self.canvas_disp_h
        end_x = max(min_x, min(max_x, event.x))
        end_y = max(min_y, min(max_y, event.y))

        x1 = min(self.drag_start_x, end_x) - min_x
        x2 = max(self.drag_start_x, end_x) - min_x
        y1 = min(self.drag_start_y, end_y) - min_y
        y2 = max(self.drag_start_y, end_y) - min_y

        was_dragged = self.drag_moved
        self.drag_start_x = None
        self.drag_start_y = None
        self.drag_moved = False

        if not was_dragged or (x2 - x1) < 4 or (y2 - y1) < 4:
            self._schedule_single_click_play_toggle("screen_canvas")
            return

        orig_h, orig_w = self.current_screen_img_cv.shape[:2]
        crop_x1 = max(0, min(orig_w - 1, int(round(x1 * self.canvas_scale_x))))
        crop_y1 = max(0, min(orig_h - 1, int(round(y1 * self.canvas_scale_y))))
        crop_x2 = max(crop_x1 + 1, min(orig_w, int(round(x2 * self.canvas_scale_x))))
        crop_y2 = max(crop_y1 + 1, min(orig_h, int(round(y2 * self.canvas_scale_y))))
        self._apply_roi_crop_coords((crop_x1, crop_y1, crop_x2, crop_y2), open_panel=True)

    def _apply_roi_crop_coords(self, coords, open_panel=True):
        if self.current_screen_img_cv is None or coords is None:
            return
        crop_bgr = self._extract_roi_bgr(self.current_screen_img_cv, coords)
        if crop_bgr is None:
            return
        self.current_roi_coords = coords
        self.last_roi_crop_bgr = crop_bgr
        self._draw_roi_rect_on_screen_canvas()
        if open_panel:
            self.roi_crop_mode = True
            self._show_roi_panel()
        elif self.roi_visible or self.roi_panel.winfo_ismapped():
            self._sync_roi_controls_visibility()
            self._render_current_roi_view()

    def _open_roi_viewer(self, crop_mode=False):
        if self.current_screen_img_cv is None:
            sel = self.items_tree.selection()
            if sel and sel[0] in self.tree_item_map:
                self._on_item_selected(None)
        if self.current_screen_img_cv is None:
            return
        use_crop = bool(crop_mode and (self.current_roi_coords is not None or self.last_roi_crop_bgr is not None))
        if not self.roi_visible:
            self.roi_maximized = False
        self.roi_crop_mode = use_crop
        self._show_roi_panel()

    def _show_roi_panel(self):
        was_visible = self.roi_visible
        self.roi_visible = True
        self._sync_roi_controls_visibility()

        if self.roi_maximized:
            mx, my, mw, mh = get_current_monitor_geometry(self.roi_panel)
            self.roi_panel.geometry(f"{mw}x{mh}+{mx}+{my}")
        else:
            self._clamp_roi_to_virtual_screen()
            self.roi_panel.geometry(f"{self.roi_user_width}x{self.roi_user_height}+{self.roi_user_left}+{self.roi_user_top}")

        self.roi_panel.deiconify()
        self.roi_panel.lift()
        if not was_visible:
            try:
                self.roi_panel.focus_set()
            except Exception:
                pass
        self.roi_panel.update_idletasks()
        self._update_roi_toolbar_labels()
        self._render_current_roi_view()
        if not self.roi_maximized:
            self.roi_resize_grip.lift()

    def _update_roi_toolbar_labels(self, panel_width=None):
        w = int(panel_width) if panel_width is not None else max(self.roi_user_width, int(self.roi_panel.winfo_width() or 0))
        wide = (w >= 460) or self.roi_maximized
        self.roi_btn_prev.configure(
            text=tr("ui.detail.roi.prev_wide", "◀ Previous") if wide else tr("ui.detail.roi.prev_short", "◀ Back")
        )
        self.roi_btn_next.configure(
            text=tr("ui.detail.roi.next_wide", "Next ▶") if wide else tr("ui.detail.roi.next_short", "Next ▶")
        )
        self.roi_btn_clear_crop.configure(
            text=tr("ui.detail.roi.reset_wide", "✕ Reset crop") if wide else tr("ui.detail.roi.reset_short", "✕ Reset")
        )
        if self.roi_crop_mode:
            self.roi_btn_crop_mode.configure(
                text=tr("ui.detail.roi.frame_wide", "🖼 Full frame") if wide else tr("ui.detail.roi.frame_short", "🖼 Frame")
            )
        else:
            self.roi_btn_crop_mode.configure(
                text=tr("ui.detail.roi.crop_wide", "🔍 Show crop") if wide else tr("ui.detail.roi.crop_short", "🔍 Crop")
            )

    def _sync_roi_controls_visibility(self):
        has_crop = self.current_roi_coords is not None
        if not has_crop and self.roi_crop_mode:
            self.roi_crop_mode = False

        if has_crop:
            if not self.roi_btn_crop_mode.winfo_ismapped():
                self.roi_btn_crop_mode.pack(side=tk.LEFT, padx=(0, 4))
            if not self.roi_btn_clear_crop.winfo_ismapped():
                self.roi_btn_clear_crop.pack(side=tk.LEFT)
        else:
            self.roi_btn_crop_mode.pack_forget()
            self.roi_btn_clear_crop.pack_forget()

        self._update_roi_toolbar_labels()
        self.roi_btn_max.configure(text="🗗" if self.roi_maximized else "⛶")

        drag_cursor = "" if self.roi_maximized else "fleur"
        for w in (self.roi_header, self.roi_title_lbl, self.roi_toolbar):
            if str(w.cget("cursor")) != drag_cursor:
                w.configure(cursor=drag_cursor)

        if self.roi_maximized:
            self.roi_resize_grip.place_forget()
            self.roi_btn_speed.pack_configure(padx=(0, 0))
        else:
            self.roi_resize_grip.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-1, y=-1)
            self.roi_btn_speed.pack_configure(padx=(0, 14))

        self.roi_canvas.configure(cursor="hand2" if self.roi_crop_mode else "cross")

        if self._get_current_record_type() == "video" and self.arch_cap is not None:
            if not self.roi_controls_frame.winfo_ismapped():
                self.roi_controls_frame.pack(fill=tk.X, side=tk.BOTTOM, before=self.roi_canvas)
            self._update_arch_play_buttons()
            self._update_arch_time_ui()
        else:
            self.roi_controls_frame.pack_forget()

        if not self.roi_maximized:
            self.roi_resize_grip.lift()

    def _render_current_roi_view(self):
        if not (self.roi_visible or self.roi_panel.winfo_ismapped()):
            return
        cw = self.roi_canvas.winfo_width()
        ch = self.roi_canvas.winfo_height()
        if cw < 20 or ch < 20:
            return

        if self.current_screen_img_cv is None:
            self.roi_canvas.delete("all")
            self.roi_canvas_img_id = None
            self.roi_canvas.create_text(
                cw // 2, ch // 2,
                text=tr("ui.detail.roi.no_image", "[No image]"),
                fill="#707080", font=("Segoe UI", 11), tags="placeholder",
            )
            self.roi_title_lbl.configure(text=tr("ui.detail.roi.view_title", "Preview"))
            return

        is_crop = False
        src_bgr = self.current_screen_img_cv
        if self.roi_crop_mode:
            crop_bgr = self._extract_roi_bgr(self.current_screen_img_cv, self.current_roi_coords) if self.current_roi_coords is not None else None
            if crop_bgr is not None:
                self.last_roi_crop_bgr = crop_bgr
                src_bgr = crop_bgr
                is_crop = True
            elif self.last_roi_crop_bgr is not None:
                src_bgr = self.last_roi_crop_bgr
                is_crop = True
            else:
                self.roi_crop_mode = False

        orig_h, orig_w = src_bgr.shape[:2]
        if orig_w <= 0 or orig_h <= 0:
            return
        scale = min(cw / float(orig_w), ch / float(orig_h))
        disp_w = max(1, int(round(orig_w * scale)))
        disp_h = max(1, int(round(orig_h * scale)))
        off_x = (cw - disp_w) // 2
        off_y = (ch - disp_h) // 2

        self.roi_scale_x = orig_w / float(disp_w)
        self.roi_scale_y = orig_h / float(disp_h)
        self.roi_img_offset_x = off_x
        self.roi_img_offset_y = off_y
        self.roi_disp_w = disp_w
        self.roi_disp_h = disp_h

        if cv2 is not None:
            interp = cv2.INTER_LANCZOS4 if is_crop else cv2.INTER_LINEAR
            resized = cv2.resize(src_bgr, (disp_w, disp_h), interpolation=interp)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
        else:
            rgb = src_bgr[:, :, ::-1] if src_bgr.ndim == 3 and src_bgr.shape[2] >= 3 else src_bgr
            res_alg = Image.Resampling.LANCZOS if is_crop else Image.Resampling.BILINEAR
            pil_img = Image.fromarray(rgb).resize((disp_w, disp_h), res_alg)

        self.roi_photo_tk = ImageTk.PhotoImage(pil_img)

        self.roi_canvas.delete("placeholder")
        if self.roi_canvas_img_id is None or not self.roi_canvas.find_withtag("roi_img"):
            self.roi_canvas_img_id = self.roi_canvas.create_image(off_x, off_y, anchor=tk.NW, image=self.roi_photo_tk, tags="roi_img")
        else:
            self.roi_canvas.coords(self.roi_canvas_img_id, off_x, off_y)
            self.roi_canvas.itemconfig(self.roi_canvas_img_id, image=self.roi_photo_tk)

        self.roi_canvas.delete("roi_existing")
        if (not is_crop) and (self.current_roi_coords is not None) and (self.roi_canvas_drag_start_x is None):
            cx1, cy1, cx2, cy2 = self.current_roi_coords
            rx1 = off_x + cx1 / self.roi_scale_x
            ry1 = off_y + cy1 / self.roi_scale_y
            rx2 = off_x + cx2 / self.roi_scale_x
            ry2 = off_y + cy2 / self.roi_scale_y
            self.roi_canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="#00ff00", width=2, tags="roi_existing")
            self.roi_canvas.tag_raise("roi_existing")

        if self.roi_canvas.find_withtag("roi_drag"):
            self.roi_canvas.tag_raise("roi_drag")

        fname = os.path.basename(self.current_preview_fpath) if self.current_preview_fpath else ""
        tag = (
            tr("ui.detail.roi.tag_crop", "Crop ({w}x{h})", w=orig_w, h=orig_h)
            if is_crop else
            tr("ui.detail.roi.tag_frame", "Full frame ({w}x{h})", w=orig_w, h=orig_h)
        )
        title_txt = f"{fname} — {tag}" if fname else tag
        if self.roi_title_lbl.cget("text") != title_txt:
            self.roi_title_lbl.configure(text=title_txt)

    def _toggle_roi_crop_mode(self):
        if self.current_roi_coords is None and self.last_roi_crop_bgr is None:
            return
        self.roi_crop_mode = not self.roi_crop_mode
        self._sync_roi_controls_visibility()
        self._render_current_roi_view()

    def _clear_roi_crop(self):
        self.current_roi_coords = None
        self.last_roi_crop_bgr = None
        self.roi_crop_mode = False
        self.screen_canvas.delete("roi_rect")
        if self.roi_visible or self.roi_panel.winfo_ismapped():
            self._sync_roi_controls_visibility()
            self._render_current_roi_view()

    def _toggle_roi_maximize(self):
        if not self.roi_visible:
            return
        if not self.roi_maximized:
            mx, my, mw, mh = get_current_monitor_geometry(self.roi_panel)
            self.roi_maximized = True
            self.roi_panel.geometry(f"{mw}x{mh}+{mx}+{my}")
        else:
            self.roi_maximized = False
            self._clamp_roi_to_virtual_screen()
            self.roi_panel.geometry(f"{self.roi_user_width}x{self.roi_user_height}+{self.roi_user_left}+{self.roi_user_top}")

        self.roi_panel.lift()
        self._sync_roi_controls_visibility()
        self.roi_panel.update_idletasks()
        self._update_roi_toolbar_labels()
        self._render_current_roi_view()
        if not self.roi_maximized:
            self.roi_resize_grip.lift()

    def _hide_roi_panel(self, clear_coords=True):
        self._cancel_single_click_play_toggle()
        self.roi_visible = False
        self.roi_maximized = False
        self.roi_crop_mode = False
        try:
            self.roi_panel.withdraw()
        except Exception:
            pass
        if clear_coords:
            self.current_roi_coords = None
            self.last_roi_crop_bgr = None
            try:
                self.screen_canvas.delete("roi_rect")
            except Exception:
                pass

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        self._cancel_hide_timer()
        if self._render_after_id is not None:
            try:
                self.win.after_cancel(self._render_after_id)
            except Exception:
                pass
            self._render_after_id = None
        if self._refresh_debounce_id is not None:
            try:
                self.win.after_cancel(self._refresh_debounce_id)
            except Exception:
                pass
            self._refresh_debounce_id = None
        if self._live_configure_after_id is not None:
            try:
                self.win.after_cancel(self._live_configure_after_id)
            except Exception:
                pass
            self._live_configure_after_id = None

        self._hide_roi_panel(clear_coords=False)
        self._stop_archive_player()

        if self.camera_unit is not None:
            self.camera_unit.remove_detail_subscriber()

        try:
            self.geom_tracker.close()
        except Exception:
            pass

        try:
            self.roi_panel.destroy()
        except Exception:
            pass
        try:
            self.win.destroy()
        except Exception:
            pass

        if self.on_close_cb:
            try:
                self.on_close_cb(self.cam_uuid)
            except Exception:
                pass
