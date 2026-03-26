import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import gymnasium as gym
import numpy as np
import threading
import time

from agents import RandomAgent, QLearningAgent, SARSAAgent, DQNAgent
from utils import HyperparameterAnalyzer, MetricsCalculator
from .taxi_grid import TaxiGridVisualization
from .tooltips import add_tooltip


class TaxiRLApp:
    """
    Application principale pour l'entraînement et le test d'agents RL sur Taxi-v3.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Taxi Driver - Reinforcement Learning Interactive")

        # Configuration responsive
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1600, int(screen_width * 0.9))
        window_height = min(950, int(screen_height * 0.9))

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.root.configure(bg='#1E1E1E')

        # Style moderne
        self.setup_styles()

        # Variables
        self.env = gym.make('Taxi-v3', render_mode=None)
        self.agent = None
        self.is_training = False
        self.is_testing = False
        self.current_episode = 0
        self.training_thread = None

        # Analyseur et calculateur
        self.analyzer = HyperparameterAnalyzer()
        self.metrics_calc = MetricsCalculator()

        # Variables Tkinter
        self.agent_var = tk.StringVar(value="Q-Learning")
        self.mode_var = tk.StringVar(value="train")
        self.episodes_var = tk.StringVar(value="1000")
        self.speed_var = tk.IntVar(value=50)

        # Hyperparamètres
        self.alpha_var = tk.DoubleVar(value=0.1)
        self.gamma_var = tk.DoubleVar(value=0.99)
        self.epsilon_var = tk.DoubleVar(value=1.0)
        self.epsilon_decay_var = tk.DoubleVar(value=0.995)
        self.epsilon_min_var = tk.DoubleVar(value=0.01)
        self.batch_size_var = tk.IntVar(value=32)
        self.buffer_size_var = tk.IntVar(value=10000)
        self.target_update_var = tk.IntVar(value=100)

        # Construction de l'interface
        self.create_ui()
        self.create_agent()

    def setup_styles(self):
        """Configure les styles modernes."""
        style = ttk.Style()
        style.theme_use('clam')

        # Colors
        self.colors = {
            'bg_dark': '#1E1E1E',
            'bg_medium': '#2D2D2D',
            'bg_light': '#3C3C3C',
            'accent': '#007ACC',
            'accent_hover': '#005A9E',
            'success': '#4CAF50',
            'danger': '#F44336',
            'warning': '#FF9800',
            'text': '#E0E0E0',
            'text_dark': '#A0A0A0',
            'border': '#555555'
        }

        # Configure styles
        style.configure('Modern.TFrame', background=self.colors['bg_dark'])
        style.configure('Card.TFrame', background=self.colors['bg_medium'], relief='flat')
        style.configure('Modern.TLabel', background=self.colors['bg_dark'], foreground=self.colors['text'], font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=self.colors['bg_dark'], foreground=self.colors['text'], font=('Segoe UI', 16, 'bold'))
        style.configure('Card.TLabel', background=self.colors['bg_medium'], foreground=self.colors['text'], font=('Segoe UI', 9))

        # Progress bar
        style.configure('Modern.Horizontal.TProgressbar', background=self.colors['accent'], troughcolor=self.colors['bg_light'], borderwidth=0)

    def create_ui(self):
        """Crée l'interface utilisateur responsive."""
        # Frame principal avec scrollbar
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)

        # Canvas pour le scroll
        canvas = tk.Canvas(main_container, bg=self.colors['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_dark'])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # ============ HEADER ============
        header = tk.Frame(scrollable_frame, bg=self.colors['bg_dark'])
        header.pack(fill=tk.X, pady=(0, 20))

        title = tk.Label(
            header,
            text="🚕 TAXI DRIVER - RL Interactive",
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['accent']
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Entraînez et testez des algorithmes de Reinforcement Learning",
            font=('Segoe UI', 11),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_dark']
        )
        subtitle.pack()

        # ============ LAYOUT RESPONSIVE ============
        content = tk.Frame(scrollable_frame, bg=self.colors['bg_dark'])
        content.pack(fill=tk.BOTH, expand=True)

        # Left column (40%)
        left_col = tk.Frame(content, bg=self.colors['bg_dark'])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right column (60%)
        right_col = tk.Frame(content, bg=self.colors['bg_dark'])
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ============ LEFT: Visualisation + Contrôles ============
        self.create_visualization_section(left_col)
        self.create_controls_section(left_col)

        # ============ RIGHT: Hyperparamètres + Résultats ============
        self.create_hyperparameters_section(right_col)
        self.create_results_section(right_col)

        # Initialise l'affichage
        self.update_agent_info()

    def create_card(self, parent, title, height=None):
        """Crée une carte stylée."""
        card = tk.Frame(parent, bg=self.colors['bg_medium'], relief='solid', borderwidth=1, bd=1)
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Header
        header = tk.Frame(card, bg=self.colors['bg_light'], height=35)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_label = tk.Label(
            header,
            text=title,
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            anchor='w'
        )
        title_label.pack(side=tk.LEFT, padx=15, pady=8)

        # Content
        content = tk.Frame(card, bg=self.colors['bg_medium'])
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        if height:
            content.configure(height=height)
            content.pack_propagate(False)

        return content

    def create_visualization_section(self, parent):
        """Crée la section visualisation."""
        viz_content = self.create_card(parent, "🎮 Environnement Taxi 5x5", height=470)

        # Grille
        self.grid_canvas = TaxiGridVisualization(viz_content, cell_size=70)
        self.grid_canvas.pack(pady=10)

        # Légende compacte
        legend_frame = tk.Frame(viz_content, bg=self.colors['bg_medium'])
        legend_frame.pack(pady=(10, 0))

        legend_items = [
            ("🟨", "Taxi"),
            ("🔵", "Passager"),
            ("⭐", "Destination"),
            ("🅁🅖🅨🅑", "Lieux")
        ]

        for emoji, label in legend_items:
            item = tk.Frame(legend_frame, bg=self.colors['bg_medium'])
            item.pack(side=tk.LEFT, padx=8)

            tk.Label(
                item,
                text=f"{emoji} {label}",
                font=('Segoe UI', 9),
                bg=self.colors['bg_medium'],
                fg=self.colors['text']
            ).pack()

    def create_controls_section(self, parent):
        """Crée la section contrôles."""
        ctrl_content = self.create_card(parent, "⚙️ Contrôles")

        # Agent selection
        self.create_control_row(ctrl_content, "Agent:", self.agent_var,
                               ["Random", "Q-Learning", "SARSA", "DQN"],
                               'agent_qlearning')

        # Mode
        self.create_control_row(ctrl_content, "Mode:", self.mode_var,
                               [("train", "Entraînement"), ("test", "Test")],
                               'mode_train')

        # Episodes
        episodes_frame = tk.Frame(ctrl_content, bg=self.colors['bg_medium'])
        episodes_frame.pack(fill=tk.X, pady=8)

        tk.Label(
            episodes_frame,
            text="Épisodes:",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            width=12,
            anchor='w'
        ).pack(side=tk.LEFT)

        episodes_entry = tk.Entry(
            episodes_frame,
            textvariable=self.episodes_var,
            font=('Segoe UI', 10),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief='flat',
            width=15
        )
        episodes_entry.pack(side=tk.LEFT, padx=5)

        info_btn = tk.Label(
            episodes_frame,
            text="ℹ️",
            font=('Segoe UI', 12),
            bg=self.colors['bg_medium'],
            fg=self.colors['accent'],
            cursor='hand2'
        )
        info_btn.pack(side=tk.LEFT, padx=2)
        add_tooltip(info_btn, 'episodes')

        # Speed
        speed_frame = tk.Frame(ctrl_content, bg=self.colors['bg_medium'])
        speed_frame.pack(fill=tk.X, pady=8)

        tk.Label(
            speed_frame,
            text="Vitesse:",
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            width=12,
            anchor='w'
        ).pack(side=tk.LEFT)

        speed_scale = tk.Scale(
            speed_frame,
            from_=1,
            to=100,
            variable=self.speed_var,
            orient=tk.HORIZONTAL,
            font=('Segoe UI', 8),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            troughcolor=self.colors['bg_light'],
            highlightthickness=0,
            length=200
        )
        speed_scale.pack(side=tk.LEFT, padx=5)

        # Buttons
        btn_frame = tk.Frame(ctrl_content, bg=self.colors['bg_medium'])
        btn_frame.pack(fill=tk.X, pady=(15, 5))

        self.start_button = tk.Button(
            btn_frame,
            text="▶ DÉMARRER",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['success'],
            fg='white',
            activebackground='#45A049',
            command=self.start_training,
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10
        )
        self.start_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.stop_button = tk.Button(
            btn_frame,
            text="⬛ ARRÊTER",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['danger'],
            fg='white',
            activebackground='#D32F2F',
            command=self.stop_training,
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Progress
        progress_frame = tk.Frame(ctrl_content, bg=self.colors['bg_medium'])
        progress_frame.pack(fill=tk.X, pady=(10, 0))

        self.progress_label = tk.Label(
            progress_frame,
            text="Prêt à démarrer",
            font=('Segoe UI', 9),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dark']
        )
        self.progress_label.pack(pady=(0, 5))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            style='Modern.Horizontal.TProgressbar',
            orient=tk.HORIZONTAL,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X)

    def create_control_row(self, parent, label, variable, options, tooltip_key):
        """Crée une ligne de contrôle avec radio buttons."""
        frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        frame.pack(fill=tk.X, pady=8)

        lbl = tk.Label(
            frame,
            text=label,
            font=('Segoe UI', 10, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            width=12,
            anchor='w'
        )
        lbl.pack(side=tk.LEFT)

        radio_frame = tk.Frame(frame, bg=self.colors['bg_medium'])
        radio_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        for option in options:
            if isinstance(option, tuple):
                value, text = option
            else:
                value = text = option

            rb = tk.Radiobutton(
                radio_frame,
                text=text,
                variable=variable,
                value=value,
                font=('Segoe UI', 9),
                bg=self.colors['bg_medium'],
                fg=self.colors['text'],
                selectcolor=self.colors['bg_light'],
                activebackground=self.colors['bg_medium'],
                activeforeground=self.colors['accent'],
                command=self.on_agent_changed if label == "Agent:" else None
            )
            rb.pack(side=tk.LEFT, padx=5)

            # Add tooltip
            if label == "Agent:":
                if value == "Random":
                    add_tooltip(rb, 'agent_random')
                elif value == "Q-Learning":
                    add_tooltip(rb, 'agent_qlearning')
                elif value == "SARSA":
                    add_tooltip(rb, 'agent_sarsa')
                elif value == "DQN":
                    add_tooltip(rb, 'agent_dqn')
            elif label == "Mode:":
                if value == "train":
                    add_tooltip(rb, 'mode_train')
                else:
                    add_tooltip(rb, 'mode_test')

    def create_hyperparameters_section(self, parent):
        """Crée la section hyperparamètres."""
        hyper_content = self.create_card(parent, "🎛️ Hyperparamètres", height=400)

        # Scroll container
        scroll_frame = tk.Frame(hyper_content, bg=self.colors['bg_medium'])
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Common params
        self.create_param_slider(scroll_frame, "Alpha (α)", "Learning Rate",
                                self.alpha_var, 0.01, 1.0, 0.01, 'alpha')
        self.create_param_slider(scroll_frame, "Gamma (γ)", "Discount Factor",
                                self.gamma_var, 0.5, 1.0, 0.01, 'gamma')
        self.create_param_slider(scroll_frame, "Epsilon (ε)", "Exploration Rate",
                                self.epsilon_var, 0.0, 1.0, 0.01, 'epsilon')
        self.create_param_slider(scroll_frame, "Epsilon Decay", "",
                                self.epsilon_decay_var, 0.9, 1.0, 0.001, 'epsilon_decay')
        self.create_param_slider(scroll_frame, "Epsilon Min", "",
                                self.epsilon_min_var, 0.0, 0.5, 0.01, 'epsilon_min')

        # DQN params
        self.dqn_params_frame = tk.Frame(scroll_frame, bg=self.colors['bg_medium'])
        self.dqn_params_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            self.dqn_params_frame,
            text="--- DQN Spécifique ---",
            font=('Segoe UI', 9, 'italic'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text_dark']
        ).pack(pady=5)

        self.create_param_slider(self.dqn_params_frame, "Batch Size", "",
                                self.batch_size_var, 8, 256, 8, 'batch_size')
        self.create_param_slider(self.dqn_params_frame, "Buffer Size", "",
                                self.buffer_size_var, 1000, 50000, 1000, 'buffer_size')
        self.create_param_slider(self.dqn_params_frame, "Target Update", "",
                                self.target_update_var, 50, 1000, 50, 'target_update')

        self.update_hyperparameters_visibility()

        # Reset button
        reset_btn = tk.Button(
            hyper_content,
            text="🔄 Réinitialiser (Valeurs recommandées)",
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_hover'],
            command=self.reset_hyperparameters,
            relief='flat',
            cursor='hand2',
            pady=8
        )
        reset_btn.pack(fill=tk.X, pady=(10, 0))

    def create_param_slider(self, parent, name, subtitle, variable, from_, to, resolution, tooltip_key):
        """Crée un slider d'hyperparamètre avec info button."""
        container = tk.Frame(parent, bg=self.colors['bg_medium'])
        container.pack(fill=tk.X, pady=5)

        # Header with name and info button
        header = tk.Frame(container, bg=self.colors['bg_medium'])
        header.pack(fill=tk.X)

        name_label = tk.Label(
            header,
            text=name,
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            anchor='w'
        )
        name_label.pack(side=tk.LEFT)

        # Info button (?)
        info_btn = tk.Label(
            header,
            text="❓",
            font=('Segoe UI', 10),
            bg=self.colors['bg_medium'],
            fg=self.colors['accent'],
            cursor='hand2'
        )
        info_btn.pack(side=tk.LEFT, padx=5)
        add_tooltip(info_btn, tooltip_key)

        if subtitle:
            subtitle_label = tk.Label(
                header,
                text=f"({subtitle})",
                font=('Segoe UI', 8),
                bg=self.colors['bg_medium'],
                fg=self.colors['text_dark']
            )
            subtitle_label.pack(side=tk.LEFT, padx=5)

        # Value display
        value_label = tk.Label(
            header,
            textvariable=variable,
            font=('Segoe UI', 9, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['accent'],
            width=8,
            relief='flat',
            padx=5
        )
        value_label.pack(side=tk.RIGHT)

        # Slider
        scale = tk.Scale(
            container,
            from_=from_,
            to=to,
            resolution=resolution,
            variable=variable,
            orient=tk.HORIZONTAL,
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            troughcolor=self.colors['bg_light'],
            highlightthickness=0,
            showvalue=False,
            command=self.on_hyperparameter_changed
        )
        scale.pack(fill=tk.X, pady=(2, 0))

    def create_results_section(self, parent):
        """Crée la section résultats."""
        # Agent info
        info_content = self.create_card(parent, "ℹ️ Informations Agent", height=150)

        self.info_text = scrolledtext.ScrolledText(
            info_content,
            font=('Consolas', 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief='flat',
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Metrics
        metrics_content = self.create_card(parent, "📊 Métriques Performance", height=200)

        self.metrics_text = scrolledtext.ScrolledText(
            metrics_content,
            font=('Consolas', 9),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief='flat',
            wrap=tk.WORD
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Recommendations
        recomm_content = self.create_card(parent, "💡 Recommandations & Optimisations")

        self.recomm_text = scrolledtext.ScrolledText(
            recomm_content,
            font=('Consolas', 9),
            bg='#2C2416',
            fg='#FFE4B5',
            insertbackground='#FFE4B5',
            relief='flat',
            wrap=tk.WORD
        )
        self.recomm_text.pack(fill=tk.BOTH, expand=True)

    def update_hyperparameters_visibility(self):
        """Affiche/cache les hyperparamètres DQN selon l'agent."""
        if self.agent_var.get() == "DQN":
            self.dqn_params_frame.pack(fill=tk.X, pady=(10, 0))
        else:
            self.dqn_params_frame.pack_forget()

    def on_agent_changed(self):
        """Appelé quand l'agent est changé."""
        self.update_hyperparameters_visibility()
        self.create_agent()
        self.update_agent_info()
        self.reset_hyperparameters()

    def on_hyperparameter_changed(self, value=None):
        """Appelé quand un hyperparamètre est modifié."""
        if self.agent:
            hyperparams = {
                'alpha': self.alpha_var.get(),
                'gamma': self.gamma_var.get(),
                'epsilon': self.epsilon_var.get(),
                'epsilon_decay': self.epsilon_decay_var.get(),
                'epsilon_min': self.epsilon_min_var.get(),
            }

            if self.agent_var.get() == "DQN":
                hyperparams['batch_size'] = self.batch_size_var.get()
                hyperparams['buffer_size'] = self.buffer_size_var.get()
                hyperparams['target_update'] = self.target_update_var.get()

            self.agent.set_hyperparameters(**hyperparams)

    def reset_hyperparameters(self):
        """Réinitialise les hyperparamètres aux valeurs recommandées."""
        agent_type = self.agent_var.get()
        agent_class_map = {
            "Q-Learning": "QLearningAgent",
            "SARSA": "SARSAAgent",
            "DQN": "DQNAgent"
        }

        if agent_type in agent_class_map:
            recommended = self.analyzer.get_recommended_hyperparameters(
                agent_class_map[agent_type],
                'default'
            )

            self.alpha_var.set(recommended.get('alpha', 0.1))
            self.gamma_var.set(recommended.get('gamma', 0.99))
            self.epsilon_var.set(recommended.get('epsilon', 1.0))
            self.epsilon_decay_var.set(recommended.get('epsilon_decay', 0.995))
            self.epsilon_min_var.set(recommended.get('epsilon_min', 0.01))

            if agent_type == "DQN":
                self.batch_size_var.set(recommended.get('batch_size', 32))
                self.buffer_size_var.set(recommended.get('buffer_size', 10000))
                self.target_update_var.set(recommended.get('target_update', 100))

            self.on_hyperparameter_changed()

    def create_agent(self):
        """Crée l'agent sélectionné."""
        agent_type = self.agent_var.get()
        n_actions = self.env.action_space.n
        n_states = self.env.observation_space.n

        if agent_type == "Random":
            self.agent = RandomAgent(n_actions, n_states)
        elif agent_type == "Q-Learning":
            self.agent = QLearningAgent(
                n_actions, n_states,
                alpha=self.alpha_var.get(),
                gamma=self.gamma_var.get(),
                epsilon=self.epsilon_var.get(),
                epsilon_decay=self.epsilon_decay_var.get(),
                epsilon_min=self.epsilon_min_var.get()
            )
        elif agent_type == "SARSA":
            self.agent = SARSAAgent(
                n_actions, n_states,
                alpha=self.alpha_var.get(),
                gamma=self.gamma_var.get(),
                epsilon=self.epsilon_var.get(),
                epsilon_decay=self.epsilon_decay_var.get(),
                epsilon_min=self.epsilon_min_var.get()
            )
        elif agent_type == "DQN":
            self.agent = DQNAgent(
                n_actions, n_states,
                alpha=self.alpha_var.get(),
                gamma=self.gamma_var.get(),
                epsilon=self.epsilon_var.get(),
                epsilon_decay=self.epsilon_decay_var.get(),
                epsilon_min=self.epsilon_min_var.get(),
                batch_size=self.batch_size_var.get(),
                buffer_size=self.buffer_size_var.get(),
                target_update=self.target_update_var.get()
            )

        self.agent.reset_stats()

    def update_agent_info(self):
        """Met à jour les informations sur l'agent."""
        if not self.agent:
            return

        agent_type = self.agent.__class__.__name__
        policy_type = self.agent.get_policy_type()

        info_text = f"Agent: {agent_type}\n"
        info_text += f"Type: {policy_type.upper()}\n"
        info_text += "="*50 + "\n\n"

        if policy_type == "off-policy":
            info_text += "OFF-POLICY: Apprend la politique optimale\n"
            info_text += "tout en explorant. Separe apprentissage\n"
            info_text += "et action. Convergence rapide.\n"
        elif policy_type == "on-policy":
            info_text += "ON-POLICY: Apprend la politique suivie.\n"
            info_text += "Plus stable et prudent. Apprend de\n"
            info_text += "ses propres actions.\n"
        else:
            info_text += "Pas d'apprentissage - Baseline.\n"

        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', info_text)

    def start_training(self):
        """Démarre l'entraînement/test."""
        if self.is_training or self.is_testing:
            return

        try:
            episodes = int(self.episodes_var.get())
            if episodes <= 0:
                raise ValueError
        except:
            messagebox.showerror("Erreur", "Nombre d'episodes invalide")
            return

        self.create_agent()
        self.is_training = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        mode = self.mode_var.get()
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(episodes, mode),
            daemon=True
        )
        self.training_thread.start()

    def stop_training(self):
        """Arrête l'entraînement."""
        self.is_training = False
        self.is_testing = False

    def run_training(self, episodes, mode):
        """Exécute l'entraînement/test dans un thread séparé."""
        training = (mode == "train")

        for episode in range(episodes):
            if not self.is_training:
                break

            state, _ = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0

            while not done and not truncated:
                if not self.is_training:
                    break

                if episode % max(1, episodes // 20) == 0:
                    self.root.after(0, self.grid_canvas.draw_state, state)
                    time.sleep(0.05 * (100 - self.speed_var.get()) / 100)

                action = self.agent.select_action(state, training=training)
                next_state, reward, done, truncated, _ = self.env.step(action)

                if training:
                    self.agent.update(state, action, reward, next_state, done or truncated)

                total_reward += reward
                steps += 1
                state = next_state

            success = (total_reward > 0)
            self.agent.add_episode_stats(total_reward, steps, success)

            if episode % max(1, episodes // 100) == 0:
                progress = (episode + 1) / episodes * 100
                self.root.after(0, self.update_progress, episode + 1, episodes, progress)

        self.root.after(0, self.training_finished)

    def update_progress(self, current, total, progress):
        """Met à jour la barre de progression."""
        self.progress_bar['value'] = progress
        self.progress_label.config(text=f"Episode {current}/{total} ({progress:.1f}%)")

        metrics = self.metrics_calc.calculate_metrics(self.agent.training_stats)
        metrics_text = self.metrics_calc.format_metrics_text(metrics)
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert('1.0', metrics_text)

    def training_finished(self):
        """Appelé quand l'entraînement est terminé."""
        self.is_training = False
        self.is_testing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Termine!")

        self.generate_recommendations()
        messagebox.showinfo("Termine", "Entrainement/Test termine!")

    def generate_recommendations(self):
        """Génère les recommandations d'hyperparamètres avec valeurs exactes."""
        if not self.agent or self.agent.__class__.__name__ == "RandomAgent":
            self.recomm_text.delete('1.0', tk.END)
            self.recomm_text.insert('1.0', "Aucune recommandation pour l'agent aleatoire.")
            return

        hyperparams = self.agent.get_hyperparameters()
        recommendations = self.analyzer.analyze_performance(
            self.agent,
            self.agent.training_stats,
            hyperparams
        )

        recomm_text = f"""
{'='*60}
  ANALYSE & RECOMMANDATIONS D'OPTIMISATION
{'='*60}

STATUS: {recommendations['status']}

"""

        if 'metrics' in recommendations:
            m = recommendations['metrics']
            recomm_text += f"""RESULTATS ACTUELS:
  Recompense moyenne: {m['avg_reward']:.2f}
  Steps moyens: {m['avg_steps']:.2f}
  Taux de succes: {m['success_rate']*100:.1f}%

"""

        if recommendations['warnings']:
            recomm_text += "="*60 + "\n"
            recomm_text += "AVERTISSEMENTS:\n"
            recomm_text += "="*60 + "\n"
            for warning in recommendations['warnings']:
                recomm_text += f"{warning}\n\n"

        if recommendations['suggestions']:
            recomm_text += "="*60 + "\n"
            recomm_text += "QUOI OPTIMISER? (Actions concretes)\n"
            recomm_text += "="*60 + "\n\n"
            for i, suggestion in enumerate(recommendations['suggestions'], 1):
                recomm_text += f"{i}. {suggestion}\n\n"

        # Add exact values to optimize
        if 'metrics' in recommendations:
            m = recommendations['metrics']
            recomm_text += "="*60 + "\n"
            recomm_text += "ACTIONS A FAIRE MAINTENANT:\n"
            recomm_text += "="*60 + "\n\n"

            if m['success_rate'] < 0.3:
                recomm_text += "1. AJUSTEZ les hyperparametres suivants:\n\n"

                if hyperparams.get('epsilon', 0) < 0.1:
                    recomm_text += f"   -> EPSILON: Mettez a 0.5\n"
                    recomm_text += f"      (actuellement: {hyperparams.get('epsilon', 0):.3f})\n\n"

                if hyperparams.get('alpha', 0) < 0.05:
                    recomm_text += f"   -> ALPHA: Mettez a 0.2\n"
                    recomm_text += f"      (actuellement: {hyperparams.get('alpha', 0):.3f})\n\n"

                if hyperparams.get('gamma', 0) < 0.95:
                    recomm_text += f"   -> GAMMA: Mettez a 0.99\n"
                    recomm_text += f"      (actuellement: {hyperparams.get('gamma', 0):.3f})\n\n"

                recomm_text += "2. Relancez l'entrainement avec 2000+ episodes\n\n"

            elif m['success_rate'] < 0.7:
                recomm_text += "1. Performance moyenne - Ajustements mineurs:\n\n"

                if hyperparams.get('epsilon', 0) > 0.3:
                    recomm_text += f"   -> EPSILON DECAY: Augmentez a 0.997\n"
                    recomm_text += f"      (actuellement: {hyperparams.get('epsilon_decay', 0):.3f})\n\n"

                recomm_text += "2. Continuez l'entrainement 1000+ episodes\n\n"

            else:
                recomm_text += "1. TRES BIEN! Optimisation finale:\n\n"

                if hyperparams.get('epsilon', 0) > 0.05:
                    recomm_text += f"   -> EPSILON: Reduisez a 0.01 pour max exploitation\n"
                    recomm_text += f"      (actuellement: {hyperparams.get('epsilon', 0):.3f})\n\n"

                recomm_text += "2. Testez en mode TEST pour valider\n\n"

            recomm_text += "="*60 + "\n"
            recomm_text += "OBJECTIF FINAL:\n"
            recomm_text += "="*60 + "\n"
            recomm_text += "  Recompense moyenne: > +5\n"
            recomm_text += "  Steps moyens: < 30\n"
            recomm_text += "  Taux de succes: > 90%\n"

        self.recomm_text.delete('1.0', tk.END)
        self.recomm_text.insert('1.0', recomm_text)
