import { useCallback, useState } from "react";
import { API_BASE } from "../config";
import type { AgentName, MapName } from "../types";

export function useQTable() {
  const [qtable, setQtable] = useState<number[][] | null>(null);

  const fetch_ = useCallback(async (agent: AgentName, map: MapName, slippery: boolean) => {
    try {
      const res = await fetch(`${API_BASE}/qtable?agent=${agent}&map_name=${map}&is_slippery=${slippery}`);
      const data = await res.json();
      if (data.qtable) setQtable(data.qtable);
    } catch {
      // agent sans Q-table (DQN, Random) → on ignore
    }
  }, []);

  const clear = useCallback(() => setQtable(null), []);

  return { qtable, fetch: fetch_, clear };
}
