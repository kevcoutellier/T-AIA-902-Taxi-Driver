"""
API FastAPI + WebSocket pour FindYourIcePath.

Endpoints REST :
  GET  /maps          → liste des maps disponibles
  GET  /agents        → liste des agents disponibles
  POST /train         → entraîne un agent (body JSON)
  POST /reset         → remet à zéro un agent

WebSocket :
  WS /ws/episode      → stream un épisode step par step
  WS /ws/train        → stream les stats d'entraînement en live
"""
import asyncio
import json
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.frozen_lake import FrozenLakeEnv, MAPS
from agents.random_agent import RandomAgent
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent

app = FastAPI(title="FindYourIcePath API")


def to_py(obj):
    """Convertit récursivement les types numpy en types Python natifs (JSON-safe)."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# State global (sessions légères, 1 utilisateur)
# ------------------------------------------------------------------

_agents: dict = {}   # key = "{agent_name}_{map_name}"
_envs:   dict = {}   # key = map_name


def get_env(map_name: str, is_slippery: bool = True) -> FrozenLakeEnv:
    key = f"{map_name}_{is_slippery}"
    if key not in _envs:
        _envs[key] = FrozenLakeEnv(map_name=map_name, is_slippery=is_slippery)
    return _envs[key]


def get_agent(agent_name: str, map_name: str, is_slippery: bool = True):
    key = f"{agent_name}_{map_name}_{is_slippery}"
    if key not in _agents:
        env = get_env(map_name, is_slippery)
        if agent_name == "random":
            _agents[key] = RandomAgent(env)
        elif agent_name == "qlearning":
            _agents[key] = QLearningAgent(env)
        elif agent_name == "sarsa":
            _agents[key] = SARSAAgent(env)
        elif agent_name == "dqn":
            _agents[key] = DQNAgent(env)
        else:
            raise ValueError(f"Agent inconnu: {agent_name}")
    return _agents[key]


# ------------------------------------------------------------------
# REST
# ------------------------------------------------------------------

@app.get("/maps")
def list_maps():
    return {
        "maps": [
            {"name": k, "grid": v, "rows": len(v), "cols": len(v[0])}
            for k, v in MAPS.items()
        ]
    }


@app.get("/agents")
def list_agents():
    return {
        "agents": [
            {"name": "random",    "label": "Random",    "trainable": False},
            {"name": "qlearning", "label": "Q-Learning","trainable": True},
            {"name": "sarsa",     "label": "SARSA",     "trainable": True},
            {"name": "dqn",       "label": "DQN",       "trainable": True},
        ]
    }


class TrainRequest(BaseModel):
    agent: str
    map_name: str = "4x4"
    is_slippery: bool = True
    n_episodes: int = 1000


@app.post("/train")
def train_agent(req: TrainRequest):
    agent = get_agent(req.agent, req.map_name, req.is_slippery)
    if req.agent == "random":
        return {"error": "Random agent is not trainable"}
    result = agent.train(req.n_episodes)
    return {
        "agent": req.agent,
        "episodes": result.episodes,
        "win_rate": result.win_rate,
        "avg_reward": result.avg_reward,
        "avg_steps": result.avg_steps,
        "train_duration": result.train_duration,
    }


@app.post("/reset")
def reset_agent(agent: str, map_name: str = "4x4", is_slippery: bool = True):
    key = f"{agent}_{map_name}_{is_slippery}"
    _agents.pop(key, None)
    return {"status": "reset", "agent": agent}


@app.get("/qtable")
def get_qtable(agent: str, map_name: str = "4x4", is_slippery: bool = True):
    """Retourne la Q-table pour visualisation (Q-Learning / SARSA)."""
    ag = get_agent(agent, map_name, is_slippery)
    if not hasattr(ag, "Q"):
        return {"error": "Cet agent n'a pas de Q-table"}
    return {"qtable": ag.Q.tolist()}


# ------------------------------------------------------------------
# WebSocket : stream d'un épisode step by step
# ------------------------------------------------------------------

@app.websocket("/ws/episode")
async def ws_episode(
    websocket: WebSocket,
    agent: str = "qlearning",
    map_name: str = "4x4",
    is_slippery: bool = True,
    speed: float = 0.3,  # secondes entre chaque step
):
    await websocket.accept()
    try:
        ag = get_agent(agent, map_name, is_slippery)
        env = ag.env

        # Replay toujours greedy (epsilon=0) pour voir la vraie politique apprise
        saved_epsilon = getattr(ag, "epsilon", 0)
        if hasattr(ag, "epsilon"):
            ag.epsilon = 0.0

        state = env.reset()
        row, col = env.state_to_pos(state)

        await websocket.send_json(to_py({
            "type": "start",
            "state": state,
            "row": row,
            "col": col,
            "grid": env.desc,
            "n_states": env.n_states,
            "n_actions": env.n_actions,
        }))

        total_reward = 0.0
        steps = 0

        while True:
            action = ag.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            nr, nc = env.state_to_pos(next_state)
            cell = env.desc[nr][nc]

            await websocket.send_json(to_py({
                "type": "step",
                "step": steps,
                "action": action,
                "state": next_state,
                "row": nr,
                "col": nc,
                "cell": cell,
                "reward": reward,
                "total_reward": total_reward,
                "done": done,
            }))

            await asyncio.sleep(speed)
            state = next_state

            if done:
                await websocket.send_json(to_py({
                    "type": "end",
                    "won": reward > 0,
                    "steps": steps,
                    "total_reward": total_reward,
                }))
                break

        # Restaure epsilon après le replay
        if hasattr(ag, "epsilon"):
            ag.epsilon = saved_epsilon

    except WebSocketDisconnect:
        if hasattr(ag, "epsilon"):
            ag.epsilon = saved_epsilon


# ------------------------------------------------------------------
# WebSocket : stream d'entraînement en live
# ------------------------------------------------------------------

@app.websocket("/ws/train")
async def ws_train(
    websocket: WebSocket,
    agent: str = "qlearning",
    map_name: str = "4x4",
    is_slippery: bool = True,
    n_episodes: int = 500,
    report_every: int = 10,
    alpha: float = 0.8,
    gamma: float = 0.95,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
):
    await websocket.accept()
    try:
        # Crée un agent frais avec les hyperparamètres choisis
        key = f"{agent}_{map_name}_{is_slippery}"
        env = get_env(map_name, is_slippery)

        if agent == "qlearning":
            from agents.qlearning import QLearningAgent
            ag = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
        elif agent == "sarsa":
            from agents.sarsa import SARSAAgent
            ag = SARSAAgent(env, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
        elif agent == "dqn":
            from agents.dqn import DQNAgent
            ag = DQNAgent(env, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
        else:
            ag = get_agent(agent, map_name, is_slippery)

        # Sauvegarde immédiatement pour que ws_episode puisse l'utiliser
        _agents[key] = ag

        await websocket.send_json(to_py({
            "type": "train_start",
            "agent": agent,
            "n_episodes": n_episodes,
        }))

        async def run_batch(start_ep: int, count: int) -> tuple[list, list]:
            """Exécute `count` épisodes dans un thread — libère l'event loop."""
            def _blocking():
                rewards, wins = [], []
                for _ in range(count):
                    r = ag.run_episode(train=True)
                    rewards.append(r.reward)
                    wins.append(r.won)
                return rewards, wins
            return await asyncio.to_thread(_blocking)

        ep = 0
        while ep < n_episodes:
            batch = min(report_every, n_episodes - ep)
            rewards, wins = await run_batch(ep, batch)
            ep += batch

            await websocket.send_json(to_py({
                "type": "train_progress",
                "episode": ep,
                "avg_reward": sum(rewards) / len(rewards),
                "win_rate": sum(wins) / len(wins),
                "epsilon": getattr(ag, "epsilon", 0),
            }))

        # Test final greedy dans un thread aussi
        test_res = await asyncio.to_thread(lambda: ag.test(100))
        await websocket.send_json(to_py({
            "type": "train_end",
            "episodes": n_episodes,
            "test_win_rate": test_res.win_rate,
            "test_avg_reward": test_res.avg_reward,
            "test_avg_steps": test_res.avg_steps,
        }))

    except WebSocketDisconnect:
        pass
