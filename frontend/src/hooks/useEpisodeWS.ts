import { useCallback, useRef, useState } from "react";
import type { StepEvent, AgentName, MapName } from "../types";
import { WS_BASE } from "../config";

const BASE_WS = `${WS_BASE}/ws/episode`;

export interface EpisodeState {
  grid: string[];
  agentRow: number;
  agentCol: number;
  currentStep: number;
  totalReward: number;
  running: boolean;
  result: { won: boolean; steps: number } | null;
  history: { row: number; col: number }[];
}

const DEFAULT: EpisodeState = {
  grid: [],
  agentRow: 0,
  agentCol: 0,
  currentStep: 0,
  totalReward: 0,
  running: false,
  result: null,
  history: [],
};

export function useEpisodeWS() {
  const [state, setState] = useState<EpisodeState>(DEFAULT);
  const wsRef = useRef<WebSocket | null>(null);

  const play = useCallback(
    (agent: AgentName, map: MapName, slippery: boolean, speed: number) => {
      if (wsRef.current) wsRef.current.close();

      setState({ ...DEFAULT, running: true });

      const url = `${BASE_WS}?agent=${agent}&map_name=${map}&is_slippery=${slippery}&speed=${speed}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (e) => {
        const msg: StepEvent = JSON.parse(e.data);

        if (msg.type === "start") {
          setState((s) => ({
            ...s,
            grid: msg.grid ?? [],
            agentRow: msg.row ?? 0,
            agentCol: msg.col ?? 0,
            history: [{ row: msg.row ?? 0, col: msg.col ?? 0 }],
          }));
        } else if (msg.type === "step") {
          setState((s) => ({
            ...s,
            agentRow: msg.row ?? s.agentRow,
            agentCol: msg.col ?? s.agentCol,
            currentStep: msg.step ?? s.currentStep,
            totalReward: msg.total_reward ?? s.totalReward,
            history: [...s.history, { row: msg.row ?? 0, col: msg.col ?? 0 }],
          }));
        } else if (msg.type === "end") {
          setState((s) => ({
            ...s,
            running: false,
            result: { won: msg.won ?? false, steps: msg.steps ?? 0 },
          }));
          ws.close();
        }
      };

      ws.onerror = () => setState((s) => ({ ...s, running: false }));
    },
    []
  );

  const stop = useCallback(() => {
    wsRef.current?.close();
    setState((s) => ({ ...s, running: false }));
  }, []);

  const reset = useCallback(() => {
    wsRef.current?.close();
    setState({ ...DEFAULT });
  }, []);

  return { state, play, stop, reset };
}
