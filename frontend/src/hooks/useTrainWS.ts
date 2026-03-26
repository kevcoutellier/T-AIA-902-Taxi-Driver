import { useCallback, useRef, useState } from "react";
import type { TrainEvent, TrainPoint, AgentName, MapName } from "../types";
import { WS_BASE, API_BASE } from "../config";
import type { Hyperparams } from "../components/Controls";

const BASE_WS = `${WS_BASE}/ws/train`;

export interface TrainState {
  running: boolean;
  progress: number;
  total: number;
  points: TrainPoint[];
  finalResult: {
    win_rate: number;
    avg_reward: number;
    avg_steps: number;
  } | null;
}

const DEFAULT: TrainState = {
  running: false,
  progress: 0,
  total: 0,
  points: [],
  finalResult: null,
};

export function useTrainWS() {
  const [state, setState] = useState<TrainState>(DEFAULT);
  const wsRef = useRef<WebSocket | null>(null);

  const startTraining = useCallback(
    (agent: AgentName, map: MapName, slippery: boolean, nEpisodes: number, hp: Hyperparams) => {
      if (wsRef.current) wsRef.current.close();
      setState({ ...DEFAULT, running: true });

      const url =
        `${BASE_WS}?agent=${agent}&map_name=${map}&is_slippery=${slippery}` +
        `&n_episodes=${nEpisodes}&alpha=${hp.alpha}&gamma=${hp.gamma}` +
        `&epsilon_decay=${hp.epsilon_decay}&epsilon_min=${hp.epsilon_min}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        const msg: TrainEvent = JSON.parse(e.data);

        if (msg.type === "train_start") {
          setState((s) => ({ ...s, total: msg.n_episodes ?? 0 }));
        } else if (msg.type === "train_progress") {
          setState((s) => ({
            ...s,
            progress: msg.episode ?? s.progress,
            points: [
              ...s.points,
              {
                episode: msg.episode ?? 0,
                win_rate: (msg.win_rate ?? 0) * 100,
                avg_reward: msg.avg_reward ?? 0,
                epsilon: msg.epsilon ?? 0,
              },
            ],
          }));
        } else if (msg.type === "train_end") {
          setState((s) => ({
            ...s,
            running: false,
            finalResult: {
              win_rate: (msg.test_win_rate ?? 0) * 100,
              avg_reward: msg.test_avg_reward ?? 0,
              avg_steps: msg.test_avg_steps ?? 0,
            },
          }));
          ws.close();
        }
      };

      ws.onerror = () => setState((s) => ({ ...s, running: false }));
    },
    []
  );

  const reset = useCallback(async (agent: AgentName, map: MapName, slippery: boolean) => {
    wsRef.current?.close();
    setState({ ...DEFAULT });
    await fetch(`${API_BASE}/reset?agent=${agent}&map_name=${map}&is_slippery=${slippery}`, { method: "POST" });
  }, []);

  const stop = useCallback(() => {
    wsRef.current?.close();
    setState((s) => ({ ...s, running: false }));
  }, []);

  return { state, startTraining, stop, reset };
}
