import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { CellType } from "../types";

interface Props {
  grid: string[];
  agentRow: number;
  agentCol: number;
  showAgent?: boolean;
  history: { row: number; col: number }[];
  result: { won: boolean; steps: number } | null;
  qtable?: number[][] | null;
  ncols?: number;
}

const CELL_ICONS: Record<CellType, string> = { S: "🏁", F: "", H: "🕳️", G: "🏆" };
const CELL_CLASS: Record<CellType, string> = {
  S: "cell-start", F: "cell-frozen", H: "cell-hole", G: "cell-goal",
};

// Actions : 0=← 1=↓ 2=→ 3=↑
const ACTIONS = [
  { label: "←", dir: "left",  idx: 0 },
  { label: "↓", dir: "down",  idx: 1 },
  { label: "→", dir: "right", idx: 2 },
  { label: "↑", dir: "up",    idx: 3 },
];

function QOverlay({ qvals }: { qvals: number[] }) {
  const max = Math.max(...qvals);
  const min = Math.min(...qvals);
  const range = max - min || 1;

  return (
    <div className="q-overlay">
      {/* Flèche haut */}
      <div className="q-overlay-grid">
        <div />
        <QArrow label="↑" value={qvals[3]} isMax={qvals[3] === max} range={range} min={min} />
        <div />
        <QArrow label="←" value={qvals[0]} isMax={qvals[0] === max} range={range} min={min} />
        <div className="q-center">
          <span className="q-best-label">{ACTIONS.find(a => a.idx === qvals.indexOf(max))?.label}</span>
        </div>
        <QArrow label="→" value={qvals[2]} isMax={qvals[2] === max} range={range} min={min} />
        <div />
        <QArrow label="↓" value={qvals[1]} isMax={qvals[1] === max} range={range} min={min} />
        <div />
      </div>
    </div>
  );
}

function QArrow({ label, value, isMax, range, min }: {
  label: string; value: number; isMax: boolean; range: number; min: number;
}) {
  const intensity = (value - min) / range; // 0 → 1
  return (
    <div
      className={`q-arrow ${isMax ? "q-arrow-best" : ""}`}
      style={{ "--intensity": intensity } as React.CSSProperties}
    >
      <span className="q-arrow-icon">{label}</span>
      <span className="q-arrow-val">{value.toFixed(2)}</span>
    </div>
  );
}

export default function Grid({ grid, agentRow, agentCol, showAgent = true, history, result, qtable }: Props) {
  const [hoveredState, setHoveredState] = useState<number | null>(null);

  if (!grid.length) {
    return (
      <div className="grid-placeholder">
        <span>Sélectionne un agent et lance un épisode</span>
      </div>
    );
  }

  const cols = grid[0].length;

  return (
    <div className="grid-wrapper">
      {qtable && (
        <div className="qtable-hint">🔍 Survole une case pour voir les Q-values</div>
      )}
      <div className="grid" style={{ "--cols": cols } as React.CSSProperties}>
        {grid.map((row, r) =>
          row.split("").map((cell, c) => {
            const ct = cell as CellType;
            const isAgent = r === agentRow && c === agentCol;
            const visitCount = history.filter((h) => h.row === r && h.col === c).length;
            const stateIdx = r * cols + c;
            const qvals = qtable?.[stateIdx];
            const isHovered = hoveredState === stateIdx;
            const bestAction = qvals ? qvals.indexOf(Math.max(...qvals)) : -1;

            // Couleur de fond selon la meilleure Q-value (heatmap légère)
            const maxQ = qvals ? Math.max(...qvals) : 0;
            const qIntensity = qvals && maxQ > 0 ? Math.min(maxQ, 1) : 0;

            return (
              <div
                key={`${r}-${c}`}
                className={`cell ${CELL_CLASS[ct]} ${visitCount > 0 ? "cell-visited" : ""} ${isHovered && qvals ? "cell-hovered" : ""}`}
                data-visited={Math.min(visitCount, 5)}
                style={qvals && ct !== "H" ? {
                  "--q-intensity": qIntensity,
                } as React.CSSProperties : undefined}
                onMouseEnter={() => qvals && setHoveredState(stateIdx)}
                onMouseLeave={() => setHoveredState(null)}
              >
                <span className="cell-icon">{CELL_ICONS[ct]}</span>

                {visitCount > 0 && ct === "F" && <div className="cell-trail" />}

                {/* Flèche de la meilleure action (visible en permanence si qtable) */}
                {qvals && ct === "F" && !isHovered && bestAction >= 0 && (
                  <div className="cell-best-action">
                    {ACTIONS[bestAction].label}
                  </div>
                )}

                {/* Q-values détaillées au hover */}
                {isHovered && qvals && ct !== "H" && (
                  <QOverlay qvals={qvals} />
                )}

                <AnimatePresence>
                  {isAgent && showAgent && (
                    <motion.div
                      className={`agent ${result && !result.won ? "agent-dead" : ""} ${result?.won ? "agent-win" : ""}`}
                      layoutId="agent"
                      initial={{ scale: 0.5, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1, rotate: result?.won ? [0, 10, -10, 0] : 0 }}
                      exit={{ scale: 0, opacity: 0 }}
                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                    >
                      🎿
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })
        )}
      </div>

      <AnimatePresence>
        {result && (
          <motion.div
            className={`result-overlay ${result.won ? "result-win" : "result-lose"}`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
          >
            <span className="result-emoji">{result.won ? "🏆" : "💀"}</span>
            <span className="result-text">{result.won ? "Goal atteint !" : "Dans le trou..."}</span>
            <span className="result-steps">{result.steps} steps</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
