import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { TrainPoint } from "../types";

interface Props {
  points: TrainPoint[];
  running: boolean;
  finalResult: { win_rate: number; avg_reward: number; avg_steps: number } | null;
}

export default function TrainChart({ points, running, finalResult }: Props) {
  return (
    <div className="train-panel">
      <h3 className="panel-title">
        Courbe d'entraînement
        {running && <span className="pill pill-blue">En cours…</span>}
        {finalResult && <span className="pill pill-green">Terminé</span>}
      </h3>

      {points.length === 0 ? (
        <div className="chart-placeholder">Lance l'entraînement pour voir les courbes</div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={points} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis
              dataKey="episode"
              tick={{ fill: "#8ecdf7", fontSize: 11 }}
              label={{ value: "épisodes", position: "insideBottomRight", offset: -5, fill: "#8ecdf7", fontSize: 11 }}
            />
            <YAxis tick={{ fill: "#8ecdf7", fontSize: 11 }} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ background: "#0d1f35", border: "1px solid #1e4976", borderRadius: 8 }}
              labelStyle={{ color: "#8ecdf7" }}
              itemStyle={{ color: "#fff" }}
              formatter={(v, name) =>
                name === "win_rate" ? [`${Number(v).toFixed(1)}%`, "Win Rate"] : [Number(v).toFixed(3), "Avg Reward"]
              }
            />
            <Legend wrapperStyle={{ color: "#8ecdf7", fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="win_rate"
              name="win_rate"
              stroke="#38bdf8"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {finalResult && (
        <div className="final-stats">
          <div className="stat-card">
            <span className="stat-label">Win Rate</span>
            <span className="stat-value">{finalResult.win_rate.toFixed(1)}%</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Avg Reward</span>
            <span className="stat-value">{finalResult.avg_reward.toFixed(3)}</span>
          </div>
          <div className="stat-card">
            <span className="stat-label">Avg Steps</span>
            <span className="stat-value">{finalResult.avg_steps.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
