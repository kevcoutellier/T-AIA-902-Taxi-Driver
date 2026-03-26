"""
gui.py — Interface graphique pour Taxi Driver (Q-Learning).
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from bruteforce import run_bruteforce, BruteForceAgent
from qlearning import train_qlearning, test_qlearning, QLearningAgent
from environment import create_env, get_env_info

# ── Palette ──────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c3aed"
ACCENT2 = "#06b6d4"
SUCCESS = "#22c55e"
WARNING = "#f59e0b"
DANGER  = "#ef4444"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
BORDER  = "#3f3f5a"

# Couleurs des 4 stations
LOC_COLORS = {"R": "#ef4444", "G": "#22c55e", "Y": "#eab308", "B": "#3b82f6"}
# Positions (row, col) des 4 stations dans Taxi-v3
LOC_POS = [(0, 0), (0, 4), (4, 0), (4, 3)]
LOC_LABELS = ["R", "G", "Y", "B"]

# Murs verticaux : liste de (row, col_gauche) — mur entre col et col+1
WALLS = [(0, 1), (1, 1), (3, 0), (4, 0), (3, 2), (4, 2)]


def decode_state(state):
    """Décode un état Taxi-v3 (0-499) → (taxi_row, taxi_col, pass_idx, dest_idx)."""
    dest_idx = state % 4
    pass_idx = (state // 4) % 5
    taxi_col = (state // 20) % 5
    taxi_row = state // 100
    return taxi_row, taxi_col, pass_idx, dest_idx


# ─────────────────────────────────────────────────────
#  Canvas de grille animée
# ─────────────────────────────────────────────────────

class GridCanvas:
    CELL  = 56
    PAD   = 14
    ROWS  = 5
    COLS  = 5

    def __init__(self, parent):
        size = self.CELL * self.COLS + self.PAD * 2
        self.canvas = tk.Canvas(parent, width=size, height=size,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self._draw_static()

    def _xy(self, row, col):
        x = self.PAD + col * self.CELL
        y = self.PAD + row * self.CELL
        return x, y

    def _draw_static(self):
        c = self.canvas
        sz = self.CELL
        # Cellules
        for r in range(self.ROWS):
            for col in range(self.COLS):
                x, y = self._xy(r, col)
                c.create_rectangle(x, y, x + sz, y + sz,
                                   fill=SURFACE, outline=BORDER, width=1,
                                   tags="static")

        # Stations (R, G, Y, B)
        for i, (r, col) in enumerate(LOC_POS):
            x, y = self._xy(r, col)
            lbl = LOC_LABELS[i]
            clr = LOC_COLORS[lbl]
            c.create_oval(x + 10, y + 10, x + sz - 10, y + sz - 10,
                          fill=clr, outline="", tags="static")
            c.create_text(x + sz // 2, y + sz // 2,
                          text=lbl, fill="white",
                          font=("Courier New", 13, "bold"), tags="static")

        # Murs
        for r, col in WALLS:
            x, y = self._xy(r, col)
            c.create_line(x + sz, y + 2, x + sz, y + sz - 2,
                          fill=DANGER, width=3, tags="static")

    def update(self, state, reward=None, total_reward=None, step=None):
        """Met à jour la grille avec le nouvel état."""
        self.canvas.delete("dynamic")
        taxi_row, taxi_col, pass_idx, dest_idx = decode_state(state)
        c = self.canvas
        sz = self.CELL

        # Destination : halo cyan
        dr, dc = LOC_POS[dest_idx]
        dx, dy = self._xy(dr, dc)
        c.create_rectangle(dx + 3, dy + 3, dx + sz - 3, dy + sz - 3,
                           outline=ACCENT2, width=3, dash=(4, 3), tags="dynamic")

        # Passager (si pas dans le taxi)
        if pass_idx < 4:
            pr, pc = LOC_POS[pass_idx]
            px, py = self._xy(pr, pc)
            c.create_oval(px + sz - 20, py + sz - 20,
                          px + sz - 6,  py + sz - 6,
                          fill=TEXT, outline="", tags="dynamic")
            c.create_text(px + sz - 13, py + sz - 13,
                          text="P", fill=BG,
                          font=("Courier New", 7, "bold"), tags="dynamic")

        # Taxi
        tx, ty = self._xy(taxi_row, taxi_col)
        # Carosserie
        taxi_fill = WARNING if pass_idx == 4 else "#d97706"
        c.create_rectangle(tx + 6, ty + 14, tx + sz - 6, ty + sz - 8,
                           fill=taxi_fill, outline="#92400e", width=2, tags="dynamic")
        # Toit
        c.create_rectangle(tx + 12, ty + 8, tx + sz - 12, ty + 18,
                           fill=taxi_fill, outline="#92400e", width=1, tags="dynamic")
        # Roues
        for wx, wy in [(tx + 9, ty + sz - 14), (tx + sz - 18, ty + sz - 14)]:
            c.create_oval(wx, wy, wx + 9, wy + 9,
                          fill="#1e1e2e", outline="", tags="dynamic")
        # Fenêtre
        c.create_rectangle(tx + 15, ty + 10, tx + sz - 15, ty + 17,
                           fill="#bae6fd", outline="", tags="dynamic")

        # Info étape
        if step is not None:
            info = f"Step {step}"
            if total_reward is not None:
                info += f"   reward: {total_reward:+.0f}"
            c.create_text(
                self.PAD + self.CELL * self.COLS // 2,
                self.PAD + self.CELL * self.ROWS + 6,
                text=info, fill=MUTED,
                font=("Courier New", 9), tags="dynamic"
            )

    def clear(self):
        self.canvas.delete("dynamic")


# ─────────────────────────────────────────────────────
#  Widgets de paramètres
# ─────────────────────────────────────────────────────

class ParamRow:
    def __init__(self, parent, label, from_, to, resolution, default, row, fmt="{:.3f}"):
        self.fmt = fmt
        self.var = tk.DoubleVar(value=default)

        tk.Label(parent, text=label, bg=SURFACE, fg=TEXT,
                 font=("Courier New", 11), anchor="w", width=18).grid(
            row=row, column=0, sticky="w", padx=(12, 4), pady=5)

        self.slider = tk.Scale(
            parent, from_=from_, to=to, resolution=resolution,
            orient="horizontal", variable=self.var,
            bg=SURFACE, fg=TEXT, troughcolor=BG, activebackground=ACCENT,
            highlightthickness=0, sliderlength=18, length=220,
            command=self._on_slide
        )
        self.slider.grid(row=row, column=1, padx=4)

        self.entry_var = tk.StringVar(value=fmt.format(default))
        self.entry = tk.Entry(
            parent, textvariable=self.entry_var, width=9,
            bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat",
            font=("Courier New", 11),
            highlightthickness=1, highlightcolor=ACCENT, highlightbackground=BORDER
        )
        self.entry.grid(row=row, column=2, padx=(4, 12))
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

    def _on_slide(self, _=None):
        self.entry_var.set(self.fmt.format(self.var.get()))

    def _on_entry(self, _=None):
        try:
            self.var.set(float(self.entry_var.get()))
        except ValueError:
            self.entry_var.set(self.fmt.format(self.var.get()))

    def get(self):
        return self.var.get()


class IntParamRow:
    def __init__(self, parent, label, from_, to, default, row):
        self.var = tk.IntVar(value=default)

        tk.Label(parent, text=label, bg=SURFACE, fg=TEXT,
                 font=("Courier New", 11), anchor="w", width=18).grid(
            row=row, column=0, sticky="w", padx=(12, 4), pady=5)

        self.slider = tk.Scale(
            parent, from_=from_, to=to, resolution=100,
            orient="horizontal", variable=self.var,
            bg=SURFACE, fg=TEXT, troughcolor=BG, activebackground=ACCENT,
            highlightthickness=0, sliderlength=18, length=220,
            command=self._on_slide
        )
        self.slider.grid(row=row, column=1, padx=4)

        self.entry_var = tk.StringVar(value=str(default))
        self.entry = tk.Entry(
            parent, textvariable=self.entry_var, width=9,
            bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat",
            font=("Courier New", 11),
            highlightthickness=1, highlightcolor=ACCENT, highlightbackground=BORDER
        )
        self.entry.grid(row=row, column=2, padx=(4, 12))
        self.entry.bind("<Return>", self._on_entry)
        self.entry.bind("<FocusOut>", self._on_entry)

    def _on_slide(self, _=None):
        self.entry_var.set(str(self.var.get()))

    def _on_entry(self, _=None):
        try:
            self.var.set(int(self.entry_var.get()))
        except ValueError:
            self.entry_var.set(str(self.var.get()))

    def get(self):
        return self.var.get()


# ─────────────────────────────────────────────────────
#  Application principale
# ─────────────────────────────────────────────────────

class TaxiGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Taxi Driver — Q-Learning GUI")
        self.root.configure(bg=BG)
        self.root.minsize(1200, 720)
        self._training   = False
        self._animating  = False
        self._agent      = None
        self._build_ui()

    # ── Layout ───────────────────────────────────────

    def _build_ui(self):
        tk.Frame(self.root, bg=ACCENT, height=4).pack(fill="x")

        title_f = tk.Frame(self.root, bg=BG)
        title_f.pack(fill="x", padx=20, pady=(12, 6))
        tk.Label(title_f, text="🚕  Taxi Driver", bg=BG, fg=TEXT,
                 font=("Courier New", 22, "bold")).pack(side="left")
        tk.Label(title_f, text="Q-Learning configurator", bg=BG, fg=MUTED,
                 font=("Courier New", 12)).pack(side="left", padx=12)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=6)

        self._build_left(body)
        self._build_center(body)
        self._build_right(body)

    # ── Panneau gauche : contrôles ────────────────────

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG)
        left.pack(side="left", fill="y", padx=(0, 10))

        # Mode
        mc = self._card(left, "Mode")
        mc.pack(fill="x", pady=(0, 8))
        self.mode_var = tk.StringVar(value="user")
        for txt, val in [("👤  User (paramètres manuels)", "user"),
                         ("⏱  Time-limited (budget temps)", "time"),
                         ("📊  Benchmark (grid search)", "benchmark")]:
            tk.Radiobutton(
                mc, text=txt, variable=self.mode_var, value=val,
                bg=SURFACE, fg=TEXT, selectcolor=ACCENT,
                activebackground=SURFACE, activeforeground=TEXT,
                font=("Courier New", 11), command=self._on_mode_change
            ).pack(anchor="w", padx=12, pady=2)
        tk.Frame(mc, height=6, bg=SURFACE).pack()

        # Hyperparamètres
        pc = self._card(left, "Hyperparamètres")
        pc.pack(fill="x", pady=(0, 8))
        pg = self._gf(pc)
        self.p_alpha   = ParamRow(pg, "Alpha (lr)",       0.01, 1.0,  0.01,  0.5,   0)
        self.p_gamma   = ParamRow(pg, "Gamma (discount)", 0.5,  1.0,  0.01,  0.95,  1)
        self.p_eps     = ParamRow(pg, "Epsilon (init)",   0.0,  1.0,  0.01,  1.0,   2)
        self.p_eps_min = ParamRow(pg, "Epsilon min",      0.0,  0.5,  0.001, 0.01,  3)
        self.p_eps_dec = ParamRow(pg, "Epsilon decay",    0.9,  1.0,  0.001, 0.995, 4)

        # Épisodes
        ec = self._card(left, "Épisodes")
        ec.pack(fill="x", pady=(0, 8))
        eg = self._gf(ec)
        self.p_train = IntParamRow(eg, "Train episodes", 100, 20000, 5000, 0)
        self.p_test  = IntParamRow(eg, "Test episodes",  10,  500,   100,  1)

        # Temps
        tc = self._card(left, "Budget temps (s)")
        tc.pack(fill="x", pady=(0, 8))
        tg = self._gf(tc)
        self.p_time = IntParamRow(tg, "Durée max (s)", 5, 120, 30, 0)

        # Boutons
        bf = tk.Frame(left, bg=BG)
        bf.pack(fill="x", pady=4)
        self.btn_run = tk.Button(
            bf, text="▶  Lancer l'entraînement",
            bg=ACCENT, fg="white", activebackground="#6d28d9",
            font=("Courier New", 12, "bold"), relief="flat",
            padx=12, pady=9, cursor="hand2",
            command=self._start_training
        )
        self.btn_run.pack(fill="x", pady=(0, 5))
        tk.Button(
            bf, text="↺  Réinitialiser",
            bg=SURFACE, fg=MUTED, activebackground=BORDER,
            font=("Courier New", 11), relief="flat",
            padx=12, pady=7, cursor="hand2",
            command=self._reset
        ).pack(fill="x")

        # Progress
        pf = tk.Frame(left, bg=BG)
        pf.pack(fill="x", pady=8)
        self.progress = ttk.Progressbar(pf, mode="indeterminate", length=320)
        self.progress.pack(fill="x")
        self.status_var = tk.StringVar(value="Prêt.")
        tk.Label(pf, textvariable=self.status_var, bg=BG, fg=MUTED,
                 font=("Courier New", 10)).pack(anchor="w", pady=3)

        self._on_mode_change()

    # ── Panneau central : grille animée ──────────────

    def _build_center(self, parent):
        center = tk.Frame(parent, bg=BG)
        center.pack(side="left", fill="y", padx=(0, 10))

        gc = self._card(center, "Simulation")
        gc.pack(fill="x")

        # Canvas de la grille
        canvas_size = GridCanvas.CELL * 5 + GridCanvas.PAD * 2 + 20
        self.grid_canvas = GridCanvas(gc)

        # Légende
        leg = tk.Frame(gc, bg=SURFACE)
        leg.pack(fill="x", padx=8, pady=(0, 4))
        for lbl, clr in LOC_COLORS.items():
            f = tk.Frame(leg, bg=clr, width=14, height=14)
            f.pack(side="left", padx=(6, 2))
            tk.Label(leg, text=lbl, bg=SURFACE, fg=TEXT,
                     font=("Courier New", 9)).pack(side="left", padx=(0, 8))
        tk.Label(leg, text="── destination", bg=SURFACE, fg=ACCENT2,
                 font=("Courier New", 9)).pack(side="left", padx=4)
        tk.Label(leg, text="P passager", bg=SURFACE, fg=TEXT,
                 font=("Courier New", 9)).pack(side="left", padx=4)

        # Contrôles animation
        ctrl = tk.Frame(gc, bg=SURFACE)
        ctrl.pack(fill="x", padx=8, pady=(4, 8))

        self.btn_watch = tk.Button(
            ctrl, text="👁  Regarder épisode (Q-Learning)",
            bg=ACCENT2, fg=BG, activebackground="#0891b2",
            font=("Courier New", 10, "bold"), relief="flat",
            padx=8, pady=6, cursor="hand2",
            command=lambda: self._watch_episode(mode="ql")
        )
        self.btn_watch.pack(fill="x", pady=(0, 4))

        self.btn_watch_bf = tk.Button(
            ctrl, text="🎲  Regarder épisode (Brute-Force)",
            bg=SURFACE, fg=WARNING, activebackground=BORDER,
            font=("Courier New", 10), relief="flat",
            padx=8, pady=5, cursor="hand2",
            command=lambda: self._watch_episode(mode="bf")
        )
        self.btn_watch_bf.pack(fill="x", pady=(0, 6))

        # Vitesse
        spd = tk.Frame(ctrl, bg=SURFACE)
        spd.pack(fill="x")
        tk.Label(spd, text="Vitesse :", bg=SURFACE, fg=MUTED,
                 font=("Courier New", 10)).pack(side="left")
        self.speed_var = tk.DoubleVar(value=0.3)
        speed_slider = tk.Scale(
            spd, from_=0.05, to=1.0, resolution=0.05,
            orient="horizontal", variable=self.speed_var,
            bg=SURFACE, fg=TEXT, troughcolor=BG, activebackground=ACCENT,
            highlightthickness=0, sliderlength=14, length=140,
            showvalue=False
        )
        speed_slider.pack(side="left", padx=4)
        tk.Label(spd, text="← rapide · lent →", bg=SURFACE, fg=MUTED,
                 font=("Courier New", 8)).pack(side="left")

        # Stats épisode
        self.ep_stats_var = tk.StringVar(value="")
        tk.Label(gc, textvariable=self.ep_stats_var, bg=SURFACE, fg=SUCCESS,
                 font=("Courier New", 10)).pack(anchor="w", padx=12, pady=(0, 8))

    # ── Panneau droit : console + graphiques ──────────

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        con_card = self._card(right, "Console")
        con_card.pack(fill="both", expand=True, pady=(0, 8))
        self.console = scrolledtext.ScrolledText(
            con_card, bg="#0f0f1a", fg="#a0f0a0",
            font=("Courier New", 10), relief="flat",
            insertbackground=TEXT, wrap="word", height=12, state="disabled"
        )
        self.console.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.console.tag_config("header",  foreground=ACCENT2,  font=("Courier New", 10, "bold"))
        self.console.tag_config("success", foreground=SUCCESS)
        self.console.tag_config("warn",    foreground=WARNING)

        graph_card = self._card(right, "Courbes d'entraînement")
        graph_card.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(7, 3), facecolor=SURFACE)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18, wspace=0.4)
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax2 = self.fig.add_subplot(1, 3, 2)
        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Helpers UI ───────────────────────────────────

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=SURFACE, highlightthickness=1,
                         highlightbackground=BORDER, relief="flat")
        tk.Label(outer, text=title, bg=SURFACE, fg=ACCENT2,
                 font=("Courier New", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(0, 6))
        return outer

    def _gf(self, card):
        """Inner frame pour grid geometry manager."""
        f = tk.Frame(card, bg=SURFACE)
        f.pack(fill="x", pady=(0, 6))
        return f

    def _style_axes(self):
        for ax, t in zip((self.ax1, self.ax2, self.ax3),
                         ["Reward / épisode", "Steps / épisode", "Epsilon"]):
            ax.set_facecolor(BG)
            ax.tick_params(colors=MUTED, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.set_title(t, color=TEXT, fontsize=9)

    def _on_mode_change(self):
        mode = self.mode_var.get()
        ep_state   = "normal" if mode in ("user", "benchmark") else "disabled"
        time_state = "normal" if mode == "time" else "disabled"
        for w in (self.p_train.slider, self.p_train.entry):
            w.configure(state=ep_state)
        for w in (self.p_time.slider, self.p_time.entry):
            w.configure(state=time_state)

    def _log(self, msg, tag=None):
        def _w():
            self.console.configure(state="normal")
            self.console.insert("end", msg + "\n", tag or "")
            self.console.see("end")
            self.console.configure(state="disabled")
        self.root.after(0, _w)

    def _set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def _reset(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        for ax in (self.ax1, self.ax2, self.ax3):
            ax.cla()
        self._style_axes()
        self.canvas.draw()
        self.grid_canvas.clear()
        self.ep_stats_var.set("")
        self._set_status("Réinitialisé.")

    # ── Animation épisode ─────────────────────────────

    def _watch_episode(self, mode="ql"):
        if self._animating:
            return
        if mode == "ql" and self._agent is None:
            self.ep_stats_var.set("⚠  Entraîne d'abord un agent !")
            return
        self._animating = True
        self.btn_watch.configure(state="disabled")
        self.btn_watch_bf.configure(state="disabled")
        self.ep_stats_var.set("Simulation en cours…")
        threading.Thread(target=self._animate_episode, args=(mode,), daemon=True).start()

    def _animate_episode(self, mode):
        delay = self.speed_var.get()
        env = create_env(render_mode=None)
        state, _ = env.reset()

        if mode == "ql":
            select_fn = self._agent.select_best_action
        else:
            bf_agent = BruteForceAgent()
            select_fn = bf_agent.select_action

        total_reward = 0
        step = 0

        # Afficher état initial
        self.root.after(0, lambda s=state: self.grid_canvas.update(s, step=0, total_reward=0))
        time.sleep(delay)

        for step in range(1, 51):
            action = select_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Capture pour le lambda (éviter closure sur variables mutables)
            s, r, t = state, total_reward, step
            self.root.after(0, lambda s=s, r=r, t=t: self.grid_canvas.update(s, total_reward=r, step=t))
            time.sleep(delay)

            if terminated or truncated:
                break

        env.close()

        label = "Q-Learning" if mode == "ql" else "Brute-Force"
        outcome = "✓ Succès" if total_reward > 0 else "✗ Timeout"
        msg = f"{label} — {step} steps | reward: {total_reward:+.0f} | {outcome}"
        self.root.after(0, lambda m=msg: self.ep_stats_var.set(m))

        self._animating = False
        self.root.after(0, lambda: self.btn_watch.configure(state="normal"))
        self.root.after(0, lambda: self.btn_watch_bf.configure(state="normal"))

    # ── Entraînement ─────────────────────────────────

    def _start_training(self):
        if self._training:
            return
        self._training = True
        self.btn_run.configure(state="disabled", text="⏳  En cours…")
        self.progress.start(12)
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            mode = self.mode_var.get()
            if mode == "user":
                self._run_user()
            elif mode == "time":
                self._run_time()
            elif mode == "benchmark":
                self._run_benchmark()
        except Exception as e:
            self._log(f"\n[ERREUR] {e}", "warn")
        finally:
            self._training = False
            self.root.after(0, self._done)

    def _done(self):
        self.progress.stop()
        self.btn_run.configure(state="normal", text="▶  Lancer l'entraînement")
        self._set_status("Terminé ✓")

    def _params(self):
        return dict(
            alpha=self.p_alpha.get(), gamma=self.p_gamma.get(),
            epsilon=self.p_eps.get(), epsilon_min=self.p_eps_min.get(),
            epsilon_decay=self.p_eps_dec.get(),
        )

    def _run_user(self):
        p = self._params()
        n_train, n_test = self.p_train.get(), self.p_test.get()

        self._log("═" * 52, "header")
        self._log("  MODE UTILISATEUR", "header")
        for k, v in p.items():
            self._log(f"  {k:<16}: {v}")
        self._log(f"  train={n_train}  test={n_test}")
        self._log("═" * 52, "header")

        self._set_status("Brute-force…")
        self._log("\n[1/3] Brute-force…")
        bf = run_bruteforce(n_test)
        self._log(f"   → steps: {bf['mean_steps']:.1f} | reward: {bf['mean_reward']:.1f}")

        self._set_status(f"Entraînement ({n_train} épisodes)…")
        self._log(f"\n[2/3] Q-Learning ({n_train} épisodes)…")
        agent, history = train_qlearning(n_train, **p, verbose=False)
        self._agent = agent

        self._set_status("Test…")
        self._log(f"\n[3/3] Test ({n_test} épisodes)…")
        ql = test_qlearning(agent, n_test, verbose=False)

        self._log("\n" + "═" * 52, "success")
        self._log(f"  Brute-Force → steps: {bf['mean_steps']:.1f} | reward: {bf['mean_reward']:.1f}")
        self._log(f"  Q-Learning  → steps: {ql['mean_steps']:.1f} | reward: {ql['mean_reward']:.1f}", "success")
        self._log("═" * 52, "success")
        self._log("\n  ▶ Clique sur 'Regarder épisode' pour voir l'agent !", "success")

        self.root.after(0, lambda: self._plot(history, bf, ql))

    def _run_time(self):
        p = self._params()
        time_limit, n_test = self.p_time.get(), self.p_test.get()

        self._log("═" * 52, "header")
        self._log(f"  MODE TEMPS LIMITÉ — {time_limit}s", "header")
        self._log("═" * 52, "header")

        env = create_env(render_mode=None)
        n_states, n_actions = get_env_info(env)
        agent = QLearningAgent(n_states, n_actions, **p)
        all_rewards, all_steps = [], []
        start = time.time()

        while (time.time() - start) < time_limit:
            state, _ = env.reset()
            total_reward, steps = 0, 0
            for _ in range(200):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.learn(state, action, reward, next_state)
                total_reward += reward
                steps += 1
                state = next_state
                if terminated or truncated:
                    break
            agent.decay_epsilon()
            all_rewards.append(total_reward)
            all_steps.append(steps)
            n = len(all_rewards)
            if n % 50 == 0:
                e = time.time() - start
                self._set_status(f"{e:.0f}s / {time_limit}s — {n} épisodes")

        env.close()
        self._agent = agent
        elapsed = time.time() - start
        self._log(f"\n  {len(all_rewards)} épisodes en {elapsed:.1f}s")

        ql = test_qlearning(agent, n_test, verbose=False)
        self._log(f"\n  steps: {ql['mean_steps']:.1f} | reward: {ql['mean_reward']:.1f}", "success")
        self._log("\n  ▶ Clique sur 'Regarder épisode' pour voir l'agent !", "success")

        history = {"rewards": all_rewards, "steps": all_steps,
                   "epsilons": [p["epsilon"]] * len(all_rewards)}
        self.root.after(0, lambda: self._plot(history, None, ql))

    def _run_benchmark(self):
        from benchmark import run_grid_search
        n_train, n_test = self.p_train.get(), self.p_test.get()

        self._log("═" * 52, "header")
        self._log(f"  MODE BENCHMARK  train={n_train}  test={n_test}", "header")
        self._log("═" * 52, "header")

        self._set_status("Brute-force…")
        self._log("\n[1/4] Brute-force…")
        bf = run_bruteforce(n_test)
        self._log(f"   → steps: {bf['mean_steps']:.1f} | reward: {bf['mean_reward']:.1f}")

        self._set_status("Grid search alpha…")
        self._log("\n[2/4] Grid search ALPHA…")
        base = {"gamma": 0.99, "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995}
        a_res = run_grid_search("alpha", [0.1, 0.3, 0.5, 0.7, 0.9], base,
                                train_episodes=n_train, test_episodes=n_test)
        best_alpha = min(a_res, key=lambda r: r["mean_steps"])["value"]
        self._log(f"   → Meilleur alpha: {best_alpha}")

        self._set_status("Grid search gamma…")
        self._log("\n[3/4] Grid search GAMMA…")
        base = {"alpha": best_alpha, "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.995}
        g_res = run_grid_search("gamma", [0.7, 0.85, 0.95, 0.99], base,
                                train_episodes=n_train, test_episodes=n_test)
        best_gamma = min(g_res, key=lambda r: r["mean_steps"])["value"]
        self._log(f"   → Meilleur gamma: {best_gamma}")

        self._set_status("Entraînement final…")
        self._log("\n[4/4] Entraînement optimisé…")
        agent, history = train_qlearning(
            n_train, alpha=best_alpha, gamma=best_gamma,
            epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, verbose=False
        )
        self._agent = agent
        ql = test_qlearning(agent, n_test, verbose=False)

        self._log("\n" + "═" * 52, "success")
        self._log(f"  Brute-Force → steps: {bf['mean_steps']:.1f} | reward: {bf['mean_reward']:.1f}")
        self._log(f"  Q-Learning  → steps: {ql['mean_steps']:.1f} | reward: {ql['mean_reward']:.1f}", "success")
        self._log(f"  α={best_alpha}  γ={best_gamma}", "success")
        self._log("═" * 52, "success")
        self._log("\n  ▶ Clique sur 'Regarder épisode' pour voir l'agent !", "success")

        self.root.after(0, lambda: self.p_alpha.var.set(best_alpha))
        self.root.after(0, lambda: self.p_gamma.var.set(best_gamma))
        self.root.after(0, lambda: self.p_alpha._on_slide())
        self.root.after(0, lambda: self.p_gamma._on_slide())
        self.root.after(0, lambda: self._plot(history, bf, ql))

    # ── Graphiques ───────────────────────────────────

    def _plot(self, history, bf, ql):
        def smooth(d, w=50):
            return np.convolve(d, np.ones(w) / w, mode="valid") if len(d) >= w else d

        for ax in (self.ax1, self.ax2, self.ax3):
            ax.cla()
        self._style_axes()

        self.ax1.plot(smooth(history["rewards"]), color=ACCENT2, linewidth=1.4)
        if bf:
            self.ax1.axhline(bf["mean_reward"], color=WARNING, linestyle="--",
                             linewidth=1, label=f"BF {bf['mean_reward']:.0f}")
        if ql:
            self.ax1.axhline(ql["mean_reward"], color=SUCCESS, linestyle="--",
                             linewidth=1, label=f"QL {ql['mean_reward']:.0f}")
        self.ax1.legend(fontsize=7, facecolor=SURFACE, labelcolor=TEXT)

        self.ax2.plot(smooth(history["steps"]), color=ACCENT, linewidth=1.4)
        if ql:
            self.ax2.axhline(ql["mean_steps"], color=SUCCESS, linestyle="--",
                             linewidth=1, label=f"QL {ql['mean_steps']:.0f}")
        self.ax2.legend(fontsize=7, facecolor=SURFACE, labelcolor=TEXT)

        self.ax3.plot(history["epsilons"], color=WARNING, linewidth=1.4)
        self.canvas.draw()


def main():
    root = tk.Tk()
    TaxiGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
