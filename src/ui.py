from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from database import ProfileDatabase
from detect import FaceDetectionError, FaceDetector
from enroll import EnrollmentManager, EnrollmentProfile
from recognize import EmbeddingModelError, FaceRecognizer, create_embedder
from utils import (
    DEFAULT_DEEPFACE_MODEL_NAME,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_MATCH_MARGIN,
    DEFAULT_OPENCV_FACE_MODEL_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K_SCORES,
    draw_face_label,
    ensure_directories,
    open_camera,
    read_camera_frame,
    save_profile_image,
    slugify,
    rotate_frame,
    resolve_project_path,
)


class FaceRecognitionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("FaceID • Intelligent Recognition System")
        self.root.geometry("1200x760")
        self.root.minsize(1080, 680)

        self.stop_recognition = threading.Event()
        self.recognition_thread: threading.Thread | None = None

        self.database = ProfileDatabase()
        self.detector: FaceDetector | None = None
        self.enroller: EnrollmentManager | None = None
        self.recognizer: FaceRecognizer | None = None

        self.status_var = tk.StringVar(value="Initializing system...")
        self.people_var = tk.StringVar(value="Registered identities: loading...")
        self.identity_var = tk.StringVar(value="No recognition running.")
        self.result_state_var = tk.StringVar(value="STANDBY")
        self.similarity_var = tk.StringVar(value="Confidence: —")
        self.profile_name_var = tk.StringVar(value="—")
        self.profile_age_var = tk.StringVar(value="—")
        self.profile_nationality_var = tk.StringVar(value="—")
        self.profile_career_var = tk.StringVar(value="—")
        self.profile_role_var = tk.StringVar(value="—")
        self.profile_note_var = tk.StringVar(value="Activate recognition to display identity details.")
        self.camera_photo: tk.PhotoImage | None = None
        self.headshot_photo: tk.PhotoImage | None = None
        self.people_tree: ttk.Treeview | None = None
        self.database_people_tree: ttk.Treeview | None = None
        self.database_selected_id_var = tk.StringVar(value="No identity selected")
        self.database_name_var = tk.StringVar()
        self.database_age_var = tk.StringVar()
        self.database_nationality_var = tk.StringVar()
        self.database_career_var = tk.StringVar()
        self.database_role_var = tk.StringVar()
        self.database_image_var = tk.StringVar(value="No profile image selected.")
        self.database_preview_photo: tk.PhotoImage | None = None

        self.camera_index_var = tk.StringVar(value="0")
        self.camera_backend_var = tk.StringVar(value="auto")
        self.rotation_var = tk.StringVar(value="none")
        self.samples_var = tk.StringVar(value="15")
        self.threshold_var = tk.StringVar(value=f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}")
        self.margin_var = tk.StringVar(value=f"{DEFAULT_MATCH_MARGIN:.2f}")
        self.mirror_var = tk.BooleanVar(value=True)

        # ─── Dark, high-tech color palette ───────────────────────────────────
        self.colors = {
            "bg":           "#0a0e1a",
            "panel":        "#0f1524",
            "card":         "#141c2e",
            "card_alt":     "#1a2236",
            "border":       "#1e2d47",
            "border_glow":  "#1e4d8c",
            "accent":       "#00b4ff",
            "accent2":      "#0077cc",
            "accent_dim":   "#003a66",
            "green":        "#00e676",
            "green_dim":    "#003322",
            "orange":       "#ff9800",
            "orange_dim":   "#3d2200",
            "red":          "#ff3d3d",
            "red_dim":      "#330000",
            "text":         "#e2e8f8",
            "text_muted":   "#5a7090",
            "text_dim":     "#8a9bb5",
            "title":        "#00b4ff",
            "header_bg":    "#07101f",
        }

        self.style = ttk.Style()
        self._configure_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)
        self.root.after(100, self.initialize_services)

    def _configure_style(self) -> None:
        self.root.configure(bg=self.colors["bg"])
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        bg        = self.colors["bg"]
        card      = self.colors["card"]
        card_alt  = self.colors["card_alt"]
        border    = self.colors["border"]
        accent    = self.colors["accent"]
        text      = self.colors["text"]
        muted     = self.colors["text_muted"]
        dim       = self.colors["text_dim"]
        green     = self.colors["green"]
        orange    = self.colors["orange"]
        red_col   = self.colors["red"]
        panel     = self.colors["panel"]
        header_bg = self.colors["header_bg"]

        self.style.configure(".", font=("Courier New", 10), background=bg, foreground=text)

        # Frames
        self.style.configure("App.TFrame",    background=bg)
        self.style.configure("Panel.TFrame",  background=panel)
        self.style.configure("Card.TFrame",   background=card)
        self.style.configure("CardAlt.TFrame",background=card_alt)
        self.style.configure("Header.TFrame", background=header_bg)

        # Labels
        self.style.configure("HeaderTitle.TLabel",
            background=header_bg, foreground=accent,
            font=("Courier New", 18, "bold"))
        self.style.configure("HeaderSub.TLabel",
            background=header_bg, foreground=muted,
            font=("Courier New", 9))
        self.style.configure("HeaderTag.TLabel",
            background=header_bg, foreground=dim,
            font=("Courier New", 8))

        self.style.configure("Section.TLabel",
            background=card, foreground=accent,
            font=("Courier New", 11, "bold"))
        self.style.configure("SectionAlt.TLabel",
            background=card_alt, foreground=accent,
            font=("Courier New", 11, "bold"))
        self.style.configure("Muted.TLabel",
            background=card, foreground=muted,
            font=("Courier New", 9))
        self.style.configure("MutedAlt.TLabel",
            background=card_alt, foreground=muted,
            font=("Courier New", 9))
        self.style.configure("Value.TLabel",
            background=card, foreground=text,
            font=("Courier New", 10, "bold"))
        self.style.configure("Dim.TLabel",
            background=card, foreground=dim,
            font=("Courier New", 9))
        self.style.configure("Status.TLabel",
            background=card, foreground=dim,
            font=("Courier New", 20, "bold"))
        self.style.configure("Recognized.TLabel",
            background=card, foreground=green,
            font=("Courier New", 20, "bold"))
        self.style.configure("Unknown.TLabel",
            background=card, foreground=orange,
            font=("Courier New", 20, "bold"))
        self.style.configure("Error.TLabel",
            background=card, foreground=red_col,
            font=("Courier New", 20, "bold"))
        self.style.configure("Badge.TLabel",
            background=self.colors["accent_dim"], foreground=accent,
            font=("Courier New", 10, "bold"), padding=(10, 5))
        self.style.configure("FieldKey.TLabel",
            background=card, foreground=muted,
            font=("Courier New", 9), width=12)
        self.style.configure("FieldVal.TLabel",
            background=card, foreground=text,
            font=("Courier New", 10, "bold"))

        # Buttons ─ Primary
        self.style.configure("Primary.TButton",
            font=("Courier New", 10, "bold"),
            padding=(14, 9),
            background=self.colors["accent2"],
            foreground="#ffffff",
            relief="flat", borderwidth=0)
        self.style.map("Primary.TButton",
            background=[("active", accent), ("pressed", self.colors["accent_dim"])],
            foreground=[("active", "#ffffff")])

        # Buttons ─ Secondary
        self.style.configure("Secondary.TButton",
            font=("Courier New", 9),
            padding=(10, 7),
            background=self.colors["card_alt"],
            foreground=dim,
            relief="flat", borderwidth=0)
        self.style.map("Secondary.TButton",
            background=[("active", border), ("pressed", panel)],
            foreground=[("active", text)])

        # Danger button
        self.style.configure("Danger.TButton",
            font=("Courier New", 9),
            padding=(10, 7),
            background=self.colors["red_dim"] if "red_dim" in self.colors else "#330000",
            foreground=red_col,
            relief="flat", borderwidth=0)
        self.style.map("Danger.TButton",
            background=[("active", "#550000")],
            foreground=[("active", "#ff6666")])

        # Notebook / Tabs
        self.style.configure("TNotebook",
            background=bg, borderwidth=0,
            tabmargins=[0, 0, 0, 0])
        self.style.configure("TNotebook.Tab",
            padding=(16, 8),
            font=("Courier New", 9, "bold"),
            background=panel,
            foreground=muted)
        self.style.map("TNotebook.Tab",
            background=[("selected", card)],
            foreground=[("selected", accent)])

        # Treeview
        self.style.configure("Treeview",
            background=panel,
            fieldbackground=panel,
            foreground=text,
            font=("Courier New", 9),
            rowheight=26,
            borderwidth=0)
        self.style.configure("Treeview.Heading",
            background=self.colors["card_alt"],
            foreground=accent,
            font=("Courier New", 9, "bold"),
            relief="flat")
        self.style.map("Treeview",
            background=[("selected", self.colors["accent_dim"])],
            foreground=[("selected", accent)])

        # Entry
        self.style.configure("TEntry",
            fieldbackground=panel,
            foreground=text,
            insertcolor=accent,
            borderwidth=1,
            relief="flat",
            padding=(6, 4))
        self.style.map("TEntry",
            fieldbackground=[("focus", self.colors["card_alt"])])

        # Combobox
        self.style.configure("TCombobox",
            fieldbackground=panel,
            background=panel,
            foreground=text,
            selectbackground=self.colors["accent_dim"],
            selectforeground=accent)

        # Checkbutton
        self.style.configure("TCheckbutton",
            background=card_alt,
            foreground=dim,
            font=("Courier New", 9))
        self.style.map("TCheckbutton",
            foreground=[("active", text)])

        # Separator
        self.style.configure("TSeparator", background=border)

        # Scrollbar
        self.style.configure("Vertical.TScrollbar",
            background=panel,
            troughcolor=bg,
            arrowcolor=muted,
            relief="flat", borderwidth=0)
        self.style.map("Vertical.TScrollbar",
            background=[("active", border)])

        # Statusbar
        self.style.configure("Status.TLabel",
            background=self.colors["header_bg"],
            foreground=muted,
            font=("Courier New", 8))

    # ─────────────────────────────────────────────────────────────────────────
    def _build_layout(self) -> None:
        self._build_header()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=0, pady=0)

        live_tab     = ttk.Frame(self.notebook, style="App.TFrame")
        enroll_tab   = ttk.Frame(self.notebook, style="App.TFrame")
        status_tab   = ttk.Frame(self.notebook, style="App.TFrame")
        settings_tab = ttk.Frame(self.notebook, style="App.TFrame")
        database_tab = ttk.Frame(self.notebook, style="App.TFrame")
        exit_tab     = ttk.Frame(self.notebook, style="App.TFrame")

        self.notebook.add(live_tab,     text="  ⬡  LIVE DEMO  ")
        self.notebook.add(enroll_tab,   text="  ⊕  ENROLL  ")
        self.notebook.add(status_tab,   text="  ◈  STATUS  ")
        self.notebook.add(settings_tab, text="  ⚙  ADVANCED  ")
        self.notebook.add(database_tab, text="  ⊞  DATABASE  ")
        self.notebook.add(exit_tab,     text="  ✕  EXIT  ")

        self._build_live_tab(    self._create_scrollable_tab(live_tab))
        self._build_enroll_tab(  self._create_scrollable_tab(enroll_tab))
        self._build_status_tab(  self._create_scrollable_tab(status_tab))
        self._build_settings_tab(self._create_scrollable_tab(settings_tab))
        self._build_database_tab(self._create_scrollable_tab(database_tab))
        self._build_exit_tab(    self._create_scrollable_tab(exit_tab))

        # Status bar
        statusbar = tk.Frame(self.root, bg=self.colors["header_bg"], height=22)
        statusbar.pack(fill="x", side="bottom")
        tk.Label(
            statusbar,
            textvariable=self.status_var,
            bg=self.colors["header_bg"],
            fg=self.colors["text_muted"],
            font=("Courier New", 8),
            anchor="w",
            padx=16,
        ).pack(fill="x", side="left")
        tk.Label(
            statusbar,
            text="FaceID System  •  AI Recognition Engine",
            bg=self.colors["header_bg"],
            fg=self.colors["border"],
            font=("Courier New", 7),
            anchor="e",
            padx=16,
        ).pack(fill="x", side="right")

    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=self.colors["header_bg"], height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Left: logo + title
        left = tk.Frame(hdr, bg=self.colors["header_bg"])
        left.pack(side="left", padx=20, pady=10)

        # Accent bar
        bar = tk.Frame(left, bg=self.colors["accent"], width=3)
        bar.pack(side="left", fill="y", padx=(0, 12))

        title_block = tk.Frame(left, bg=self.colors["header_bg"])
        title_block.pack(side="left")
        tk.Label(
            title_block,
            text="FACE·ID",
            bg=self.colors["header_bg"],
            fg=self.colors["accent"],
            font=("Courier New", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="INTELLIGENT RECOGNITION SYSTEM",
            bg=self.colors["header_bg"],
            fg=self.colors["text_muted"],
            font=("Courier New", 7),
        ).pack(anchor="w")

        # Right: info tags
        right = tk.Frame(hdr, bg=self.colors["header_bg"])
        right.pack(side="right", padx=20)
        for tag_text, tag_color in [
            ("● NEURAL ENGINE", self.colors["green"]),
            ("● LOCAL DATABASE", self.colors["accent"]),
            ("● SECURE", self.colors["text_muted"]),
        ]:
            tk.Label(
                right,
                text=tag_text,
                bg=self.colors["header_bg"],
                fg=tag_color,
                font=("Courier New", 8),
            ).pack(anchor="e")

    # ─────────────────────────────────────────────────────────────────────────
    def _create_scrollable_tab(self, parent: ttk.Frame) -> ttk.Frame:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        canvas = tk.Canvas(parent, bg=self.colors["bg"], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        content = ttk.Frame(canvas, style="App.TFrame")
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def update_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def match_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event):
            delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")

        def bind_mousewheel(_event=None):
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(_event=None):
            canvas.unbind_all("<MouseWheel>")

        content.bind("<Configure>", update_scrollregion)
        canvas.bind("<Configure>", match_width)
        content.bind("<Enter>", bind_mousewheel)
        content.bind("<Leave>", unbind_mousewheel)
        return content

    # ─── Helpers ─────────────────────────────────────────────────────────────
    def _card(self, parent, **kwargs) -> tk.Frame:
        """Dark card with border."""
        f = tk.Frame(
            parent,
            bg=self.colors["card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            **kwargs,
        )
        return f

    def _section_label(self, parent, text: str) -> None:
        row = tk.Frame(parent, bg=self.colors["card"])
        row.pack(fill="x", pady=(0, 14))
        tk.Label(row, text="▸ " + text,
                 bg=self.colors["card"], fg=self.colors["accent"],
                 font=("Courier New", 11, "bold")).pack(side="left")
        tk.Frame(row, bg=self.colors["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(10, 0), pady=6)

    def _muted_label(self, parent, text: str = "", textvariable=None,
                     wraplength: int = 0) -> tk.Label:
        kw = dict(bg=self.colors["card"], fg=self.colors["text_muted"],
                  font=("Courier New", 9))
        if wraplength:
            kw["wraplength"] = wraplength
        if textvariable:
            return tk.Label(parent, textvariable=textvariable, **kw)
        return tk.Label(parent, text=text, **kw)

    def _primary_btn(self, parent, text: str, command) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command,
            bg=self.colors["accent2"], fg="#ffffff",
            font=("Courier New", 10, "bold"),
            relief="flat", bd=0, padx=16, pady=9,
            activebackground=self.colors["accent"],
            activeforeground="#ffffff",
            cursor="hand2",
        )

    def _secondary_btn(self, parent, text: str, command) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command,
            bg=self.colors["card_alt"], fg=self.colors["text_dim"],
            font=("Courier New", 9),
            relief="flat", bd=0, padx=12, pady=7,
            activebackground=self.colors["border"],
            activeforeground=self.colors["text"],
            cursor="hand2",
        )

    def _danger_btn(self, parent, text: str, command) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command,
            bg="#1a0000", fg=self.colors["red"],
            font=("Courier New", 9),
            relief="flat", bd=0, padx=12, pady=7,
            activebackground="#330000",
            activeforeground="#ff6666",
            cursor="hand2",
        )

    # ─── LIVE TAB ─────────────────────────────────────────────────────────────
    def _build_live_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=0)

        wrapper = tk.Frame(parent, bg=self.colors["bg"])
        wrapper.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=18, pady=18)
        wrapper.columnconfigure(0, weight=1)
        wrapper.columnconfigure(1, weight=0)

        # ── Camera card ────────────────────────────────────────────────────
        cam_card = self._card(wrapper, width=760, height=530)
        cam_card.grid(row=0, column=0, sticky="nw", padx=(0, 14))
        cam_card.grid_propagate(False)
        cam_card.columnconfigure(0, weight=1)
        cam_card.rowconfigure(1, weight=1)

        cam_hdr = tk.Frame(cam_card, bg=self.colors["card"])
        cam_hdr.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 10))
        cam_hdr.columnconfigure(0, weight=1)

        tk.Label(cam_hdr, text="▸ CAMERA PREVIEW",
                 bg=self.colors["card"], fg=self.colors["accent"],
                 font=("Courier New", 11, "bold")).grid(row=0, column=0, sticky="w")

        self.recognition_button = self._primary_btn(
            cam_hdr, "  ▶  START DEMO", self.toggle_recognition)
        self.recognition_button.grid(row=0, column=1, sticky="e")

        preview = tk.Frame(
            cam_card, bg="#050a10",
            highlightbackground=self.colors["border_glow"],
            highlightthickness=1,
            width=722, height=455,
        )
        preview.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        preview.grid_propagate(False)
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.camera_label = tk.Label(
            preview,
            text="◈  Press START DEMO to activate live feed",
            bg="#050a10", fg=self.colors["border_glow"],
            font=("Courier New", 13, "bold"),
            wraplength=340,
        )
        self.camera_label.pack(fill="both", expand=True)

        # ── Result card ────────────────────────────────────────────────────
        res_card = self._card(wrapper, width=310, height=530)
        res_card.grid(row=0, column=1, sticky="nw")
        res_card.grid_propagate(False)

        inner = tk.Frame(res_card, bg=self.colors["card"])
        inner.pack(fill="both", expand=True, padx=18, pady=18)

        self._section_label(inner, "RECOGNITION RESULT")

        # State label
        self.result_label = tk.Label(
            inner, textvariable=self.result_state_var,
            bg=self.colors["card"], fg=self.colors["text_muted"],
            font=("Courier New", 18, "bold"))
        self.result_label.pack(anchor="w", pady=(0, 8))

        # Confidence badge
        badge_frame = tk.Frame(inner, bg=self.colors["accent_dim"],
                               highlightbackground=self.colors["accent"],
                               highlightthickness=1)
        badge_frame.pack(anchor="w", pady=(0, 14))
        tk.Label(
            badge_frame, textvariable=self.similarity_var,
            bg=self.colors["accent_dim"], fg=self.colors["accent"],
            font=("Courier New", 9, "bold"), padx=10, pady=4,
        ).pack()

        # Headshot
        self.headshot_label = tk.Label(
            inner, text="[ NO IMAGE ]",
            bg=self.colors["panel"], fg=self.colors["border"],
            font=("Courier New", 9),
            width=22, height=9,
            relief="flat",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        self.headshot_label.pack(anchor="w", pady=(0, 12))

        # Separator
        tk.Frame(inner, bg=self.colors["border"], height=1).pack(fill="x", pady=8)

        # Profile rows
        self._add_profile_row(inner, "NAME",        self.profile_name_var)
        self._add_profile_row(inner, "AGE",         self.profile_age_var)
        self._add_profile_row(inner, "NATIONALITY", self.profile_nationality_var)
        self._add_profile_row(inner, "CAREER",      self.profile_career_var)
        self._add_profile_row(inner, "ROLE",        self.profile_role_var)

        tk.Frame(inner, bg=self.colors["border"], height=1).pack(fill="x", pady=10)
        tk.Label(
            inner, textvariable=self.profile_note_var,
            bg=self.colors["card"], fg=self.colors["text_muted"],
            font=("Courier New", 8), wraplength=260, justify="left",
        ).pack(anchor="w")

    # ─── ENROLL TAB ───────────────────────────────────────────────────────────
    def _build_enroll_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        outer = tk.Frame(parent, bg=self.colors["bg"])
        outer.pack(fill="x", padx=60, pady=40)

        card = self._card(outer)
        card.pack(fill="x")

        inner = tk.Frame(card, bg=self.colors["card"])
        inner.pack(fill="both", padx=28, pady=24)

        self._section_label(inner, "ENROLL A NEW IDENTITY")

        self._muted_label(
            inner,
            text="Enter profile details and capture webcam samples.\n"
                 "The neural engine stores only identity embeddings; "
                 "all profile fields are saved to the local SQLite database.",
            wraplength=720,
        ).pack(anchor="w", pady=(0, 20))

        btn_row = tk.Frame(inner, bg=self.colors["card"])
        btn_row.pack(anchor="w")
        self._primary_btn(
            btn_row, "  ⊕  OPEN ENROLLMENT FORM",
            self.open_enrollment_dialog,
        ).pack(side="left")

        # Separator
        tk.Frame(inner, bg=self.colors["border"], height=1).pack(
            fill="x", pady=24)

        # Registered list label
        tk.Label(
            inner, textvariable=self.people_var,
            bg=self.colors["card"], fg=self.colors["text_muted"],
            font=("Courier New", 9), justify="left", wraplength=720,
        ).pack(anchor="w")

    # ─── STATUS TAB ───────────────────────────────────────────────────────────
    def _build_status_tab(self, parent: ttk.Frame) -> None:
        outer = tk.Frame(parent, bg=self.colors["bg"])
        outer.pack(fill="both", expand=True, padx=60, pady=40)

        card = self._card(outer)
        card.pack(fill="both", expand=True)

        inner = tk.Frame(card, bg=self.colors["card"])
        inner.pack(fill="both", padx=28, pady=24)

        self._section_label(inner, "SYSTEM STATUS")

        tk.Label(
            inner, textvariable=self.status_var,
            bg=self.colors["card"], fg=self.colors["text"],
            font=("Courier New", 11, "bold"), wraplength=820, justify="left",
        ).pack(anchor="w", pady=(0, 12))

        tk.Label(
            inner, textvariable=self.people_var,
            bg=self.colors["card"], fg=self.colors["text_muted"],
            font=("Courier New", 9), justify="left", wraplength=820,
        ).pack(anchor="w", pady=(0, 24))

        tk.Frame(inner, bg=self.colors["border"], height=1).pack(fill="x", pady=(0, 20))

        self._primary_btn(inner, "  ◈  REFRESH STATUS", self.refresh_status).pack(anchor="w")

    # ─── SETTINGS TAB ─────────────────────────────────────────────────────────
    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        outer = tk.Frame(parent, bg=self.colors["bg"])
        outer.pack(fill="x", padx=60, pady=40)

        card = self._card(outer)
        card.pack(fill="x")

        inner = tk.Frame(card, bg=self.colors["card"])
        inner.pack(fill="both", padx=28, pady=24)

        self._section_label(inner, "ADVANCED SETTINGS")

        form = tk.Frame(inner, bg=self.colors["card"])
        form.pack(anchor="w", fill="x")
        form.columnconfigure(1, weight=0)

        self._add_setting(form, "CAMERA INDEX",  self.camera_index_var,  0)
        self._add_combo(  form, "BACKEND",       self.camera_backend_var, ("auto", "dshow", "msmf"), 1)
        self._add_combo(  form, "ROTATE",        self.rotation_var,       ("none", "cw", "ccw", "180"), 2)
        self._add_setting(form, "SAMPLES",       self.samples_var,        3)
        self._add_setting(form, "THRESHOLD",     self.threshold_var,      4)
        self._add_setting(form, "MARGIN",        self.margin_var,         5)

        chk = tk.Checkbutton(
            form, text="  Mirror Preview",
            variable=self.mirror_var,
            bg=self.colors["card"], fg=self.colors["text_dim"],
            selectcolor=self.colors["panel"],
            activebackground=self.colors["card"],
            activeforeground=self.colors["text"],
            font=("Courier New", 9),
            relief="flat", bd=0,
        )
        chk.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        tk.Frame(inner, bg=self.colors["border"], height=1).pack(fill="x", pady=(18, 16))

        self._secondary_btn(inner, "  ⚙  APPLY & REFRESH", self.refresh_status).pack(anchor="w")

    # ─── DATABASE TAB ─────────────────────────────────────────────────────────
    def _build_database_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        outer = tk.Frame(parent, bg=self.colors["bg"])
        outer.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        outer.columnconfigure(0, weight=1)

        # ── Table card ────────────────────────────────────────────────────
        list_card = self._card(outer)
        list_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        list_card.columnconfigure(0, weight=1)

        lhdr = tk.Frame(list_card, bg=self.colors["card"])
        lhdr.pack(fill="x", padx=18, pady=(14, 10))
        lhdr.columnconfigure(0, weight=1)

        tk.Label(lhdr, text="▸ DATABASE RECORDS",
                 bg=self.colors["card"], fg=self.colors["accent"],
                 font=("Courier New", 11, "bold")).pack(side="left")
        self._secondary_btn(lhdr, "  ↻  REFRESH", self.refresh_database_tab).pack(side="right")

        tk.Frame(list_card, bg=self.colors["border"], height=1).pack(fill="x")

        tree_frame = tk.Frame(list_card, bg=self.colors["card"])
        tree_frame.pack(fill="x", padx=18, pady=14)
        tree_frame.columnconfigure(0, weight=1)

        columns = ("id", "name", "age", "nationality", "career", "role", "samples")
        self.database_people_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=10)
        headings = {
            "id": "ID", "name": "NAME", "age": "AGE",
            "nationality": "NATIONALITY", "career": "CAREER",
            "role": "ROLE", "samples": "SAMPLES",
        }
        widths = {"id": 50, "name": 140, "age": 55, "nationality": 110,
                  "career": 160, "role": 110, "samples": 70}
        for col, heading in headings.items():
            self.database_people_tree.heading(col, text=heading)
            self.database_people_tree.column(col, width=widths.get(col, 90), anchor="center")

        self.database_people_tree.grid(row=0, column=0, sticky="nsew")
        tsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self.database_people_tree.yview)
        tsb.grid(row=0, column=1, sticky="ns")
        self.database_people_tree.configure(yscrollcommand=tsb.set)
        self.database_people_tree.bind("<<TreeviewSelect>>", self._on_database_person_select)
        self.database_people_tree.bind("<ButtonRelease-1>",  self._on_database_row_click)

        # ── Editor card ───────────────────────────────────────────────────
        editor_card = self._card(outer)
        editor_card.grid(row=1, column=0, sticky="ew", pady=(0, 18))

        editor_inner = tk.Frame(editor_card, bg=self.colors["card"])
        editor_inner.pack(fill="both", padx=18, pady=18)
        editor_inner.columnconfigure(0, weight=1)
        editor_inner.columnconfigure(1, weight=0)

        tk.Label(editor_inner, text="▸ EDIT SELECTED IDENTITY",
                 bg=self.colors["card"], fg=self.colors["accent"],
                 font=("Courier New", 11, "bold")).grid(
                     row=0, column=0, columnspan=2, sticky="w")
        tk.Label(editor_inner, textvariable=self.database_selected_id_var,
                 bg=self.colors["card"], fg=self.colors["text_muted"],
                 font=("Courier New", 8)).grid(
                     row=1, column=0, columnspan=2, sticky="w", pady=(4, 14))

        # Form + preview side by side
        form = tk.Frame(editor_inner, bg=self.colors["card"])
        form.grid(row=2, column=0, sticky="nw")

        fields_def = [
            ("NAME",        self.database_name_var),
            ("AGE",         self.database_age_var),
            ("NATIONALITY", self.database_nationality_var),
            ("CAREER",      self.database_career_var),
            ("ROLE",        self.database_role_var),
        ]
        for field_label, field_var in fields_def:
            row_f = tk.Frame(form, bg=self.colors["card"])
            row_f.pack(fill="x", pady=4)
            tk.Label(row_f, text=field_label, width=13, anchor="w",
                     bg=self.colors["card"], fg=self.colors["text_muted"],
                     font=("Courier New", 9)).pack(side="left")
            entry = tk.Entry(
                row_f, textvariable=field_var, width=32,
                bg=self.colors["panel"], fg=self.colors["text"],
                insertbackground=self.colors["accent"],
                relief="flat", bd=0,
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                font=("Courier New", 10),
            )
            entry.pack(side="left", fill="x", expand=True, ipady=4)

        # Action buttons
        actions = tk.Frame(form, bg=self.colors["card"])
        actions.pack(fill="x", pady=(18, 0))
        self._primary_btn(actions, "  ✔  SAVE CHANGES",
                          self._save_database_changes).pack(side="left", padx=(0, 8))
        self._secondary_btn(actions, "  ⊞  SET PICTURE",
                            self._set_professional_picture).pack(side="left", padx=(0, 8))
        self._danger_btn(actions, "  ✕  DELETE",
                         self._delete_database_person).pack(side="left")

        # Picture preview (right side)
        pic_card = tk.Frame(editor_inner, bg=self.colors["card"])
        pic_card.grid(row=2, column=1, sticky="ne", padx=(24, 0))

        tk.Label(pic_card, text="PROFILE IMAGE",
                 bg=self.colors["card"], fg=self.colors["text_muted"],
                 font=("Courier New", 8)).pack(anchor="w")

        self.database_preview_label = tk.Label(
            pic_card, text="[ NO IMAGE ]",
            bg=self.colors["panel"], fg=self.colors["border"],
            font=("Courier New", 9),
            width=22, height=11, relief="flat",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        self.database_preview_label.pack(anchor="w", pady=(8, 8))
        tk.Label(
            pic_card, textvariable=self.database_image_var,
            bg=self.colors["card"], fg=self.colors["text_muted"],
            font=("Courier New", 8), wraplength=240,
        ).pack(anchor="w")

    # ─── EXIT TAB ─────────────────────────────────────────────────────────────
    def _build_exit_tab(self, parent: ttk.Frame) -> None:
        outer = tk.Frame(parent, bg=self.colors["bg"])
        outer.pack(fill="x", padx=60, pady=40)

        card = self._card(outer)
        card.pack(fill="x")

        inner = tk.Frame(card, bg=self.colors["card"])
        inner.pack(fill="both", padx=28, pady=28)

        self._section_label(inner, "EXIT APPLICATION")

        self._muted_label(
            inner,
            text="This will stop all recognition threads, release the camera device, "
                 "and close all application windows cleanly.",
            wraplength=700,
        ).pack(anchor="w", pady=(0, 24))

        self._danger_btn(inner, "  ✕  EXIT CLEANLY", self.exit_app).pack(anchor="w")

    # ─── Profile row helper ────────────────────────────────────────────────────
    def _add_profile_row(self, parent: tk.Frame, label: str,
                         variable: tk.StringVar) -> None:
        row = tk.Frame(parent, bg=self.colors["card"])
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, width=12, anchor="w",
                 bg=self.colors["card"], fg=self.colors["text_muted"],
                 font=("Courier New", 8)).pack(side="left")
        tk.Label(row, textvariable=variable,
                 bg=self.colors["card"], fg=self.colors["text"],
                 font=("Courier New", 10, "bold")).pack(side="left")

    # ─── Settings helpers ──────────────────────────────────────────────────────
    def _add_setting(self, parent, label: str, variable: tk.StringVar,
                     row: int) -> None:
        tk.Label(parent, text=label,
                 bg=self.colors["card"], fg=self.colors["text_muted"],
                 font=("Courier New", 9), width=16, anchor="w").grid(
                     row=row, column=0, sticky="w", padx=10, pady=5)
        entry = tk.Entry(
            parent, textvariable=variable, width=14,
            bg=self.colors["panel"], fg=self.colors["text"],
            insertbackground=self.colors["accent"],
            relief="flat", bd=0,
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            font=("Courier New", 10),
        )
        entry.grid(row=row, column=1, sticky="ew", padx=10, pady=5, ipady=4)

    def _add_combo(self, parent, label: str, variable: tk.StringVar,
                   values: tuple, row: int) -> None:
        tk.Label(parent, text=label,
                 bg=self.colors["card"], fg=self.colors["text_muted"],
                 font=("Courier New", 9), width=16, anchor="w").grid(
                     row=row, column=0, sticky="w", padx=10, pady=5)
        ttk.Combobox(parent, textvariable=variable, values=values,
                     state="readonly", width=12).grid(
                         row=row, column=1, sticky="ew", padx=10, pady=5)

    # ─── Service init ──────────────────────────────────────────────────────────
    def initialize_services(self) -> None:
        try:
            ensure_directories()
            self.database.initialize()
            detector = FaceDetector()
            embedder = create_embedder(
                backend=DEFAULT_EMBEDDING_BACKEND,
                deepface_model=DEFAULT_DEEPFACE_MODEL_NAME,
                opencv_model_path=DEFAULT_OPENCV_FACE_MODEL_PATH,
            )
            self.detector  = detector
            self.enroller  = EnrollmentManager(self.database, detector, embedder)
            self.recognizer = FaceRecognizer(
                database=self.database,
                detector=detector,
                embedder=embedder,
                threshold=float(self.threshold_var.get()),
                match_margin=float(self.margin_var.get()),
                top_k_scores=DEFAULT_TOP_K_SCORES,
            )
            self.status_var.set("SYSTEM READY  •  Database initialized.")
            self.refresh_status()
            self.refresh_database_tab()
        except Exception as error:
            self.status_var.set("STARTUP FAILED")
            messagebox.showerror("Startup Error", str(error))

    def refresh_services(self) -> None:
        if self.detector is None or self.enroller is None:
            self.initialize_services()
            return
        if self.recognizer is not None:
            self.recognizer.threshold    = self._float_from_var(self.threshold_var, "Threshold")
            self.recognizer.match_margin = self._float_from_var(self.margin_var,    "Margin")

    def refresh_status(self) -> None:
        try:
            self.database.initialize()
            people = self.database.get_all_people()
            if not people:
                self.people_var.set("Registered identities: none")
            else:
                lines = ["Registered identities:"]
                for person in people:
                    lines.append(
                        f"  [{person['person_id']}]  {person['name']}  "
                        f"({person['sample_count']} samples)"
                    )
                self.people_var.set("\n".join(lines))
            self._populate_people_table(people)
            self.status_var.set(f"SYSTEM READY  •  {len(people)} registered identities.")
        except Exception as error:
            messagebox.showerror("Database Error", str(error))

    def _populate_people_table(self, people: list[dict]) -> None:
        if self.people_tree is None:
            return
        for item in self.people_tree.get_children():
            self.people_tree.delete(item)
        for person in people:
            self.people_tree.insert(
                "", "end",
                values=(
                    person["person_id"],
                    person["name"],
                    person["age"],
                    person["nationality"],
                    person["career"],
                    person["sample_count"],
                ),
            )

    # ─── Database tab logic ────────────────────────────────────────────────────
    def refresh_database_tab(self) -> None:
        self.database.initialize()
        people = self.database.get_all_people()
        if self.database_people_tree is not None:
            for item in self.database_people_tree.get_children():
                self.database_people_tree.delete(item)
            for person in people:
                self.database_people_tree.insert(
                    "", "end",
                    values=(
                        person["person_id"],
                        person["name"],
                        person["age"],
                        person["nationality"],
                        person["career"],
                        person.get("role", ""),
                        person["sample_count"],
                    ),
                )
        self.refresh_status()

    def _on_database_person_select(self, _event=None) -> None:
        self.root.after_idle(self._load_selected_database_person)

    def _on_database_row_click(self, _event=None) -> None:
        self.root.after_idle(self._load_selected_database_person)

    def _load_selected_database_person(self) -> None:
        person_id = self._get_selected_database_person_id()
        if person_id is None:
            messagebox.showinfo("Database", "Select a person first.")
            return
        person = self.database.get_person_by_id(person_id)
        if person is None:
            messagebox.showerror("Database", "Selected person no longer exists.")
            self.refresh_database_tab()
            return
        self.database_selected_id_var.set(f"Editing  person_id = {person_id}")
        self.database_name_var.set(str(person["name"]))
        self.database_age_var.set(str(person["age"]))
        self.database_nationality_var.set(str(person["nationality"]))
        self.database_career_var.set(str(person["career"]))
        self.database_role_var.set(str(person.get("role", "")))
        image_path = person.get("profile_image_path")
        self.database_image_var.set(
            f"File: {Path(image_path).name}" if image_path
            else "No profile image selected."
        )
        self._show_database_preview(person.get("profile_image_path"))

    def _save_database_changes(self) -> None:
        person_id = self._get_selected_database_person_id()
        if person_id is None:
            messagebox.showinfo("Database", "Select a person first.")
            return
        try:
            name        = self.database_name_var.get().strip()
            age         = self._parse_age(self.database_age_var.get())
            nationality = self.database_nationality_var.get().strip()
            career      = self.database_career_var.get().strip()
            role        = self.database_role_var.get().strip()
            if not role:
                role = "visitor"
            if not name or not nationality or not career:
                raise ValueError("Name, nationality, and career are required.")
            self.database.update_person(person_id, name, age, nationality, career, role)
        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))
            return
        except Exception as error:
            messagebox.showerror("Database Error", str(error))
            return
        self.refresh_database_tab()
        self._select_database_person(person_id)
        self._load_selected_database_person()
        self.status_var.set(f"Updated  person_id={person_id}")

    def _set_professional_picture(self) -> None:
        person_id = self._get_selected_database_person_id()
        if person_id is None:
            messagebox.showinfo("Database", "Select a person first.")
            return
        image_path = filedialog.askopenfilename(
            title="Choose profile picture",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            return
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Database", "Could not read the selected image.")
            return
        person = self.database.get_person_by_id(person_id)
        if person is None:
            messagebox.showerror("Database", "Selected person no longer exists.")
            return
        prefix     = f"{person_id}_{slugify(str(person['name']))}_profile"
        saved_path = save_profile_image(image, prefix)
        self.database.update_profile_image(person_id, str(saved_path))
        self.refresh_database_tab()
        self._select_database_person(person_id)
        self._load_selected_database_person()
        self.status_var.set(f"Updated profile picture  person_id={person_id}")

    def _delete_database_person(self) -> None:
        person_id = self._get_selected_database_person_id()
        if person_id is None:
            messagebox.showinfo("Database", "Select a person first.")
            return
        person = self.database.get_person_by_id(person_id)
        label = person["name"] if person else f"person_id={person_id}"
        if not messagebox.askyesno(
                "Confirm Delete",
                f"Delete '{label}'?\nThis removes the identity and all face samples."):
            return
        self.database.delete_person(person_id)
        self.database_selected_id_var.set("No identity selected")
        self.database_name_var.set("")
        self.database_age_var.set("")
        self.database_nationality_var.set("")
        self.database_career_var.set("")
        self.database_role_var.set("")
        self.database_image_var.set("No profile image selected.")
        self._show_database_preview(None)
        self.refresh_database_tab()
        self.status_var.set(f"Deleted  person_id={person_id}")

    def _show_database_preview(self, image_path) -> None:
        resolved_path = resolve_project_path(image_path) if image_path else None
        image = cv2.imread(str(resolved_path)) if resolved_path else None
        if image is None:
            self.database_preview_photo = None
            if hasattr(self, "database_preview_label"):
                self.database_preview_label.configure(image="", text="[ NO IMAGE ]")
            return
        image = self._center_square(image)
        image = cv2.resize(image, (220, 220))
        self.database_preview_photo = self._cv_to_photo(image)
        self.database_preview_label.configure(image=self.database_preview_photo, text="")

    def _select_database_person(self, person_id: int) -> None:
        if self.database_people_tree is None:
            return
        for item in self.database_people_tree.get_children():
            values = self.database_people_tree.item(item, "values")
            if values and int(values[0]) == person_id:
                self.database_people_tree.selection_set(item)
                self.database_people_tree.see(item)
                break

    def _get_selected_database_person_id(self) -> int | None:
        if self.database_people_tree is None:
            return None
        selection = self.database_people_tree.selection()
        if not selection:
            return None
        values = self.database_people_tree.item(selection[0], "values")
        if not values:
            return None
        try:
            return int(values[0])
        except (TypeError, ValueError):
            return None

    # ─── Enrollment dialog ─────────────────────────────────────────────────────
    def open_enrollment_dialog(self) -> None:
        if self.enroller is None:
            messagebox.showerror("Not Ready", "Services are not initialized yet.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Enroll New Identity")
        dialog.geometry("420x380")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.colors["bg"])

        # Header
        hdr = tk.Frame(dialog, bg=self.colors["header_bg"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="⊕  ENROLL NEW IDENTITY",
                 bg=self.colors["header_bg"], fg=self.colors["accent"],
                 font=("Courier New", 12, "bold"), padx=18, pady=10).pack(anchor="w")

        body = tk.Frame(dialog, bg=self.colors["card"],
                        highlightbackground=self.colors["border"],
                        highlightthickness=1)
        body.pack(fill="both", expand=True, padx=18, pady=14)
        inner = tk.Frame(body, bg=self.colors["card"])
        inner.pack(fill="both", padx=18, pady=18)

        fields = {
            "Name":        tk.StringVar(),
            "Age":         tk.StringVar(),
            "Nationality": tk.StringVar(),
            "Career":      tk.StringVar(),
            "Role":        tk.StringVar(),
        }

        for field_label, field_var in fields.items():
            row_f = tk.Frame(inner, bg=self.colors["card"])
            row_f.pack(fill="x", pady=4)
            tk.Label(row_f, text=field_label.upper(), width=12, anchor="w",
                     bg=self.colors["card"], fg=self.colors["text_muted"],
                     font=("Courier New", 9)).pack(side="left")
            entry = tk.Entry(
                row_f, textvariable=field_var, width=26,
                bg=self.colors["panel"], fg=self.colors["text"],
                insertbackground=self.colors["accent"],
                relief="flat", bd=0,
                highlightbackground=self.colors["border"],
                highlightthickness=1,
                font=("Courier New", 10),
            )
            entry.pack(side="left", fill="x", expand=True, ipady=4)

        tk.Frame(inner, bg=self.colors["border"], height=1).pack(fill="x", pady=(12, 8))

        tk.Label(
            inner,
            text="After clicking START, an OpenCV window opens.\n"
                 "Press SPACE to capture each sample frame.",
            bg=self.colors["card"], fg=self.colors["text_muted"],
            font=("Courier New", 8), justify="left",
        ).pack(anchor="w", pady=(0, 12))

        def submit() -> None:
            try:
                profile = EnrollmentProfile(
                    name=fields["Name"].get().strip(),
                    age=self._parse_age(fields["Age"].get()),
                    nationality=fields["Nationality"].get().strip(),
                    career=fields["Career"].get().strip(),
                    role=fields["Role"].get().strip(),
                )
                if not profile.name or not profile.nationality or not profile.career:
                    raise ValueError("Name, nationality, and career are required.")
                sample_count = self._int_from_var(self.samples_var, "Samples")
            except ValueError as error:
                messagebox.showerror("Invalid Input", str(error), parent=dialog)
                return
            dialog.destroy()
            self._start_worker("Enrollment", self._run_enrollment, profile, sample_count)

        self._primary_btn(inner, "  ▶  START CAPTURE", submit).pack(anchor="w")

    def _run_enrollment(self, profile: EnrollmentProfile, sample_count: int) -> None:
        if self.enroller is None:
            raise RuntimeError("Enrollment service is not initialized.")
        person_id = self.enroller.enroll_from_webcam(
            profile=profile,
            sample_count=sample_count,
            camera_index=self._int_from_var(self.camera_index_var, "Camera index"),
            camera_backend=self.camera_backend_var.get(),
            rotation=self.rotation_var.get(),
            mirror=self.mirror_var.get(),
        )
        self.root.after(0, lambda: self._enrollment_finished(person_id))

    def _enrollment_finished(self, person_id: int) -> None:
        self.status_var.set(f"Enrollment complete  •  person_id={person_id}")
        self.refresh_status()
        messagebox.showinfo("Enrollment Complete", f"Identity saved  •  person_id={person_id}")

    # ─── Recognition loop ──────────────────────────────────────────────────────
    def toggle_recognition(self) -> None:
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.stop_recognition.set()
            self.status_var.set("Stopping recognition...")
            return
        if self.recognizer is None:
            messagebox.showerror("Not Ready", "Services are not initialized yet.")
            return
        try:
            self.refresh_services()
            camera_index = self._int_from_var(self.camera_index_var, "Camera index")
        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))
            return

        self.stop_recognition.clear()
        self.recognition_button.configure(text="  ■  STOP RECOGNITION")
        self.recognition_thread = threading.Thread(
            target=self._recognition_loop, args=(camera_index,), daemon=True)
        self.recognition_thread.start()

    def _recognition_loop(self, camera_index: int) -> None:
        capture = None
        try:
            capture = open_camera(camera_index, self.camera_backend_var.get())
            if not capture.isOpened():
                raise RuntimeError("Could not open webcam.")
            self.root.after(0, lambda: self.status_var.set(
                "RECOGNITION ACTIVE  •  Press STOP to end demo."))

            while not self.stop_recognition.is_set():
                frame  = read_camera_frame(capture)
                frame  = rotate_frame(frame, self.rotation_var.get())
                if self.mirror_var.get():
                    frame = cv2.flip(frame, 1)

                display = frame.copy()
                try:
                    assert self.recognizer is not None
                    output = self.recognizer.recognize_image(frame)
                    lines  = [f"Similarity: {output.match.similarity:.3f}"]
                    display = draw_face_label(display, output.bbox, output.match.name, lines)
                    self.root.after(0, self._show_match, output.match)
                except FaceDetectionError as error:
                    self.root.after(0, self._show_unknown, str(error))
                    cv2.putText(display, str(error), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)

                self.root.after(0, self._show_camera_frame, display)
        except Exception as error:
            message = str(error)
            self.root.after(0, lambda: messagebox.showerror("Recognition Error", message))
        finally:
            if capture is not None:
                capture.release()
            self.root.after(0, self._recognition_stopped)

    def _show_match(self, match) -> None:
        if not match.is_known or not match.profile:
            self._show_unknown("Unknown")
            return
        profile = match.profile
        self.identity_var.set(f"Recognized: {profile['name']}")
        self.result_state_var.set("RECOGNIZED")
        self.result_label.configure(fg=self.colors["green"])
        self.similarity_var.set(f"Confidence  {match.similarity:.3f}")
        self.profile_name_var.set(str(profile["name"]))
        self.profile_age_var.set(str(profile["age"]))
        self.profile_nationality_var.set(str(profile["nationality"]))
        self.profile_career_var.set(str(profile["career"]))
        self.profile_role_var.set(str(profile.get("role", "N/A")))
        self.profile_note_var.set("Profile data retrieved from local database.")
        self._show_headshot(profile.get("profile_image_path"))

    def _show_unknown(self, reason: str) -> None:
        self.identity_var.set("Unknown")
        self.result_state_var.set("UNKNOWN")
        self.result_label.configure(fg=self.colors["orange"])
        self.similarity_var.set("Confidence  —")
        self.profile_name_var.set("—")
        self.profile_age_var.set("—")
        self.profile_nationality_var.set("—")
        self.profile_career_var.set("—")
        self.profile_role_var.set("—")
        self.profile_note_var.set(reason)
        self._clear_headshot()

    def _recognition_stopped(self) -> None:
        self.recognition_button.configure(text="  ▶  START DEMO")
        self.status_var.set("Recognition stopped.")

    def _show_camera_frame(self, frame) -> None:
        if not hasattr(self, "camera_label"):
            return
        parent = self.camera_label.master
        target_width  = max(1, parent.winfo_width()  - 4 if parent.winfo_width()  > 1 else parent.winfo_reqwidth()  - 4)
        target_height = max(1, parent.winfo_height() - 4 if parent.winfo_height() > 1 else parent.winfo_reqheight() - 4)
        fitted = self._fit_image(frame, target_width, target_height)
        self.camera_photo = self._cv_to_photo(fitted)
        self.camera_label.configure(image=self.camera_photo, text="")

    def _show_headshot(self, image_path) -> None:
        resolved_path = resolve_project_path(image_path) if image_path else None
        image = cv2.imread(str(resolved_path)) if resolved_path else None
        if image is None:
            self._clear_headshot()
            return
        image = self._center_square(image)
        image = cv2.resize(image, (190, 190))
        self.headshot_photo = self._cv_to_photo(image)
        self.headshot_label.configure(image=self.headshot_photo, text="", width=190, height=190)

    def _clear_headshot(self) -> None:
        self.headshot_photo = None
        if hasattr(self, "headshot_label"):
            self.headshot_label.configure(image="", text="[ NO IMAGE ]", width=22, height=9)

    # ─── Static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _fit_image(image, max_width: int, max_height: int):
        height, width = image.shape[:2]
        scale      = max(max_width / width, max_height / height)
        new_width  = max(1, int(width  * scale))
        new_height = max(1, int(height * scale))
        resized = cv2.resize(image, (new_width, new_height))
        y1 = max(0, (new_height - max_height) // 2)
        x1 = max(0, (new_width  - max_width)  // 2)
        return resized[y1:y1 + max_height, x1:x1 + max_width]

    @staticmethod
    def _center_square(image):
        height, width = image.shape[:2]
        side = min(height, width)
        x1 = max(0, (width  - side) // 2)
        y1 = max(0, (height - side) // 2)
        return image[y1:y1 + side, x1:x1 + side]

    @staticmethod
    def _cv_to_photo(image) -> tk.PhotoImage:
        ok, encoded = cv2.imencode(".ppm", image)
        if not ok:
            raise RuntimeError("Failed to render camera frame.")
        return tk.PhotoImage(data=encoded.tobytes(), format="PPM")

    def show_people_window(self) -> None:
        self.refresh_status()
        if hasattr(self, "notebook"):
            self.notebook.select(2)

    def _start_worker(self, label: str, target, *args) -> None:
        self.status_var.set(f"{label} running...")

        def run() -> None:
            try:
                target(*args)
            except (FaceDetectionError, EmbeddingModelError, RuntimeError, ValueError) as error:
                message = str(error)
                self.root.after(0, lambda: messagebox.showerror(f"{label} Error", message))
                self.root.after(0, lambda: self.status_var.set(f"{label} failed."))

        threading.Thread(target=run, daemon=True).start()

    def exit_app(self) -> None:
        self.stop_recognition.set()
        cv2.destroyAllWindows()
        self.root.destroy()

    @staticmethod
    def _parse_age(value: str) -> int:
        value = value.strip()
        if not value.isdigit():
            raise ValueError("Age must be a valid integer.")
        return int(value)

    @staticmethod
    def _int_from_var(variable: tk.StringVar, label: str) -> int:
        try:
            return int(variable.get().strip())
        except ValueError as error:
            raise ValueError(f"{label} must be an integer.") from error

    @staticmethod
    def _float_from_var(variable: tk.StringVar, label: str) -> float:
        try:
            return float(variable.get().strip())
        except ValueError as error:
            raise ValueError(f"{label} must be a number.") from error


def main() -> None:
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
