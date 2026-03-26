export type CellType = "S" | "F" | "H" | "G";
export type AgentName = "random" | "qlearning" | "sarsa" | "dqn";
export type MapName = "4x4" | "8x8";

export interface StepEvent {
  type: "start" | "step" | "end";
  state?: number;
  row?: number;
  col?: number;
  cell?: CellType;
  action?: number;
  reward?: number;
  total_reward?: number;
  done?: boolean;
  won?: boolean;
  step?: number;   // dans les events "step" (compteur courant)
  steps?: number;  // dans l'event "end" (total)
  grid?: string[];
  n_states?: number;
  n_actions?: number;
}

export interface TrainEvent {
  type: "train_start" | "train_progress" | "train_end";
  episode?: number;
  n_episodes?: number;
  avg_reward?: number;
  win_rate?: number;
  epsilon?: number;
  test_win_rate?: number;
  test_avg_reward?: number;
  test_avg_steps?: number;
}

export interface TrainPoint {
  episode: number;
  win_rate: number;
  avg_reward: number;
  epsilon: number;
}
