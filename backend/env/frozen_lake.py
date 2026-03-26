"""
Wrapper autour de gymnasium FrozenLake-v1.
Expose une interface simple utilisée par tous les agents.
"""
import gymnasium as gym
import numpy as np
from typing import Optional


# Maps disponibles
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

# Actions
ACTION_NAMES = {0: "←", 1: "↓", 2: "→", 3: "↑"}


class FrozenLakeEnv:
    """
    Wrapper FrozenLake avec helpers pour les agents RL.

    Paramètres
    ----------
    map_name : "4x4" | "8x8" | None (custom)
    custom_map : liste de strings si map_name is None
    is_slippery : si True, déplacements stochastiques (défaut bootstrap)
    render_mode : "human" | "rgb_array" | None
    """

    def __init__(
        self,
        map_name: str = "4x4",
        custom_map: Optional[list] = None,
        is_slippery: bool = True,
        render_mode: Optional[str] = None,
    ):
        desc = custom_map if custom_map else MAPS.get(map_name)
        if desc is None:
            raise ValueError(f"map_name inconnu: {map_name}. Choisir parmi {list(MAPS)}")

        self.env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            is_slippery=is_slippery,
            render_mode=render_mode,
        )

        self.n_states: int = self.env.observation_space.n
        self.n_actions: int = self.env.action_space.n
        self.map_name = map_name
        self.desc = desc
        self.is_slippery = is_slippery

        # Calcul des dimensions de la grille
        self.nrow = len(desc)
        self.ncol = len(desc[0])

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def reset(self) -> int:
        state, _ = self.env.reset()
        return int(state)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return int(obs), float(reward), terminated, truncated, info

    def sample_action(self) -> int:
        return self.env.action_space.sample()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def state_to_pos(self, state: int) -> tuple[int, int]:
        """Convertit un état (entier) en position (row, col)."""
        return divmod(state, self.ncol)

    def pos_to_state(self, row: int, col: int) -> int:
        return row * self.ncol + col

    def is_hole(self, state: int) -> bool:
        row, col = self.state_to_pos(state)
        return self.desc[row][col] == "H"

    def is_goal(self, state: int) -> bool:
        row, col = self.state_to_pos(state)
        return self.desc[row][col] == "G"

    def grid_repr(self) -> str:
        return "\n".join(self.desc)

    def get_transition_probs(self) -> dict:
        """Retourne la table de transition P[s][a] = [(prob, next_s, reward, done)]."""
        return self.env.unwrapped.P

    def __repr__(self) -> str:
        return (
            f"FrozenLakeEnv(map={self.map_name}, "
            f"states={self.n_states}, actions={self.n_actions}, "
            f"slippery={self.is_slippery})"
        )
