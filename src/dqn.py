"""
dqn.py — Agent Deep Q-Network (DQN) avec réseau de neurones NumPy.

DQN remplace la Q-table par un réseau de neurones pour approximer Q(s, a).
Composants clés :
  - Main network   : prédit Q(s, a) et reçoit les mises à jour de gradient
  - Target network : cible stable, copiée depuis main toutes les N étapes
  - Replay buffer  : mémoire circulaire d'expériences (s, a, r, s', done)
  - Epsilon-greedy : exploration décroissante identique à Q-Learning

Mise à jour (mini-batch) :
  target  = r + γ × max_a' Q_target(s', a')   (0 si terminal)
  loss    = MSE(Q_main(s, a), target)
  → descente de gradient sur main_network
"""

import numpy as np
from tqdm import tqdm
from environment import create_env, get_env_info


# ─────────────────────────────────────────────────────
#  Réseau de neurones (NumPy pur)
# ─────────────────────────────────────────────────────

class _MLP:
    """MLP minimaliste : forward, backward, predict."""

    def __init__(self, sizes, lr=0.001):
        self.lr = lr
        rng = np.random.default_rng(0)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])).astype(np.float32))
            self.b.append(np.zeros((1, sizes[i + 1]), dtype=np.float32))

    def forward(self, x):
        self._a = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = self._a[-1] @ w + b
            self._a.append(np.maximum(0, z) if i < len(self.W) - 1 else z)
        return self._a[-1]

    def backward(self, grad):
        d = grad
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                d = d * (self._a[i + 1] > 0)
            gw = self._a[i].T @ d
            gb = d.sum(axis=0, keepdims=True)
            d  = d @ self.W[i].T
            self.W[i] -= self.lr * np.clip(gw, -1.0, 1.0)
            self.b[i]  -= self.lr * np.clip(gb, -1.0, 1.0)

    def predict(self, x):
        a = x
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = np.maximum(0, z) if i < len(self.W) - 1 else z
        return a

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


# ─────────────────────────────────────────────────────
#  Replay buffer
# ─────────────────────────────────────────────────────

class ReplayBuffer:
    """Buffer circulaire numpy — accès O(1) au lieu de O(n) avec deque."""

    def __init__(self, maxlen=10_000):
        self.maxlen  = maxlen
        self.states  = np.zeros(maxlen, dtype=np.int32)
        self.actions = np.zeros(maxlen, dtype=np.int32)
        self.rewards = np.zeros(maxlen, dtype=np.float32)
        self.nexts   = np.zeros(maxlen, dtype=np.int32)
        self.dones   = np.zeros(maxlen, dtype=np.float32)
        self._size   = 0
        self._idx    = 0

    def push(self, s, a, r, ns, done):
        self.states [self._idx] = s
        self.actions[self._idx] = a
        self.rewards[self._idx] = r
        self.nexts  [self._idx] = ns
        self.dones  [self._idx] = float(done)
        self._idx  = (self._idx + 1) % self.maxlen
        self._size = min(self._size + 1, self.maxlen)

    def sample(self, n):
        idx = np.random.choice(self._size, n, replace=True)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.nexts[idx],  self.dones[idx])

    def __len__(self):
        return self._size


# ─────────────────────────────────────────────────────
#  Agent DQN
# ─────────────────────────────────────────────────────

class DQNAgent:
    """Deep Q-Network : réseau de neurones + experience replay + target network."""

    def __init__(self, n_states=500, n_actions=6,
                 gamma=0.99, lr=5e-4,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, memory_size=10_000, target_update=200):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self._steps        = 0

        sizes = [n_states, 8, 128, n_actions]
        self.main   = _MLP(sizes, lr=lr)
        self.target = _MLP(sizes, lr=lr)
        self.target.copy_from(self.main)

        self.memory = ReplayBuffer(memory_size)
        self._eye   = np.eye(n_states, dtype=np.float32)  # one-hot lookup
        self._bidx  = np.arange(batch_size)               # pré-alloué

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.main.predict(self._eye[state : state + 1])))

    def select_best_action(self, state):
        return int(np.argmax(self.main.predict(self._eye[state : state + 1])))

    def learn(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.memory.sample(self.batch_size)
        s_oh  = self._eye[s]
        ns_oh = self._eye[ns]

        q_next  = self.target.predict(ns_oh)
        targets = r + self.gamma * q_next.max(axis=1) * (1.0 - d)

        q_pred  = self.main.forward(s_oh)
        err     = q_pred[self._bidx, a] - targets

        grad = np.zeros_like(q_pred)
        grad[self._bidx, a] = 2.0 * err / self.batch_size
        self.main.backward(grad)

        self._steps += 1
        if self._steps % self.target_update == 0:
            self.target.copy_from(self.main)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# ─────────────────────────────────────────────────────
#  Train / Test
# ─────────────────────────────────────────────────────

def _find_convergence_episode(rewards, window=50, threshold=0.25):
    success = np.array(rewards) > 0
    for i in range(window, len(success) + 1):
        if np.mean(success[i - window : i]) >= threshold:
            return i - window + 1
    return None


def train_dqn(n_episodes, gamma=0.99, lr=5e-4,
              epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
              batch_size=64, memory_size=10_000, target_update=200,
              verbose=True):
    env = create_env(render_mode=None)
    n_states, n_actions = get_env_info(env)

    agent = DQNAgent(
        n_states=n_states, n_actions=n_actions,
        gamma=gamma, lr=lr,
        epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
        batch_size=batch_size, memory_size=memory_size, target_update=target_update,
    )

    history = {"rewards": [], "steps": [], "epsilons": []}
    it = tqdm(range(n_episodes), desc="Training DQN") if verbose else range(n_episodes)

    for _ in it:
        state, _ = env.reset()
        total_r, steps = 0, 0
        for _ in range(200):
            action = agent.select_action(state)
            next_s, reward, terminated, truncated, _ = env.step(action)
            agent.learn(state, action, reward, next_s, terminated or truncated)
            total_r += reward
            steps += 1
            state = next_s
            if terminated or truncated:
                break
        agent.decay_epsilon()
        history["rewards"].append(total_r)
        history["steps"].append(steps)
        history["epsilons"].append(agent.epsilon)

    env.close()
    history["convergence_episode"] = _find_convergence_episode(history["rewards"])
    return agent, history


def test_dqn(agent, n_episodes, verbose=True):
    env = create_env(render_mode=None)
    all_rewards, all_steps, all_illegal = [], [], []
    it = tqdm(range(n_episodes), desc="Testing DQN") if verbose else range(n_episodes)

    for _ in it:
        state, _ = env.reset()
        total_r, steps, illegal = 0, 0, 0
        for _ in range(200):
            action = agent.select_best_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_r += reward
            steps += 1
            if reward == -10:
                illegal += 1
            if terminated or truncated:
                break
        all_rewards.append(total_r)
        all_steps.append(steps)
        all_illegal.append(illegal)

    env.close()
    rewards   = np.array(all_rewards)
    steps_arr = np.array(all_steps)
    success   = rewards > 0
    return {
        "rewards": all_rewards,
        "steps":   all_steps,
        "mean_reward":     float(np.mean(rewards)),
        "std_reward":      float(np.std(rewards)),
        "mean_steps":      float(np.mean(steps_arr)),
        "std_steps":       float(np.std(steps_arr)),
        "success_rate":    float(np.mean(success) * 100),
        "reward_per_step": float(np.sum(rewards) / np.sum(steps_arr)),
        "illegal_actions": int(np.sum(all_illegal)),
        "convergence_episode": None,
    }
