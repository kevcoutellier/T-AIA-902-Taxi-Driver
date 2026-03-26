import { useState, useEffect } from "react";
import type { AgentName, MapName } from "./types";
import { useEpisodeWS } from "./hooks/useEpisodeWS";
import { useTrainWS } from "./hooks/useTrainWS";
import { useQTable } from "./hooks/useQTable";
import Grid from "./components/Grid";
import Controls, { type Hyperparams } from "./components/Controls";
import TrainChart from "./components/TrainChart";
import Analysis from "./components/Analysis";
import { DEFAULT_MAPS } from "./config";
import "./App.css";

const DEFAULT_HP: Hyperparams = { alpha: 0.8, gamma: 0.95, epsilon_decay: 0.995, epsilon_min: 0.01 };
const DEFAULT_AGENT: AgentName = "qlearning";
const DEFAULT_MAP: MapName = "4x4";

// Presets : configs recommandées par situation
const PRESETS: { label: string; emoji: string; desc: string; agent: AgentName; map: MapName; slippery: boolean; nEpisodes: number; hp: Hyperparams }[] = [
  {
    label: "Débutant", emoji: "🟢", desc: "4×4 déterministe — facile à converger",
    agent: "qlearning", map: "4x4", slippery: false, nEpisodes: 1000,
    hp: { alpha: 0.8, gamma: 0.95, epsilon_decay: 0.99, epsilon_min: 0.01 },
  },
  {
    label: "Standard", emoji: "🔵", desc: "4×4 glissant — config équilibrée",
    agent: "qlearning", map: "4x4", slippery: true, nEpisodes: 3000,
    hp: { alpha: 0.8, gamma: 0.99, epsilon_decay: 0.995, epsilon_min: 0.01 },
  },
  {
    label: "SARSA", emoji: "🟣", desc: "On-policy prudent en mode glissant",
    agent: "sarsa", map: "4x4", slippery: true, nEpisodes: 4000,
    hp: { alpha: 0.5, gamma: 0.99, epsilon_decay: 0.997, epsilon_min: 0.01 },
  },
  {
    label: "Expert", emoji: "🔴", desc: "8×8 glissant — long mais instructif",
    agent: "qlearning", map: "8x8", slippery: true, nEpisodes: 10000,
    hp: { alpha: 0.5, gamma: 0.99, epsilon_decay: 0.9995, epsilon_min: 0.01 },
  },
];

export default function App() {
  const [agent, setAgent] = useState<AgentName>("qlearning");
  const [map, setMap] = useState<MapName>("4x4");
  const [slippery, setSlippery] = useState(true);
  const [speed, setSpeed] = useState(0.3);
  const [nEpisodes, setNEpisodes] = useState(2000);
  const [hyperparams, setHyperparams] = useState<Hyperparams>(DEFAULT_HP);

  // Détecte si les params ont changé depuis le dernier entraînement
  const [trainedSnapshot, setTrainedSnapshot] = useState<string | null>(null);
  const currentSnapshot = JSON.stringify({ agent, map, slippery, nEpisodes, hyperparams });
  const paramsChanged = trainedSnapshot !== null && trainedSnapshot !== currentSnapshot;

  const { state: epState, play, stop: stopEp, reset: resetEp } = useEpisodeWS();
  const { state: trainState, startTraining, stop: stopTrain, reset: resetTrain } = useTrainWS();
  const { qtable, fetch: fetchQTable, clear: clearQTable } = useQTable();

  // Fetch Q-table once training completes (only for tabular agents)
  useEffect(() => {
    if (trainState.finalResult) {
      fetchQTable(agent, map, slippery);
    }
  }, [trainState.finalResult]);

  const handlePlay  = () => play(agent, map, slippery, speed);
  const handleTrain = () => {
    setTrainedSnapshot(currentSnapshot);
    startTraining(agent, map, slippery, nEpisodes, hyperparams);
  };
  const handleStop  = () => { stopEp(); stopTrain(); };
  const handleReset = async () => {
    stopEp(); stopTrain();
    resetEp();
    clearQTable();
    await resetTrain(agent, map, slippery);
    setTrainedSnapshot(null);
    setAgent(DEFAULT_AGENT);
    setMap(DEFAULT_MAP);
    setSlippery(true);
    setNEpisodes(2000);
    setHyperparams(DEFAULT_HP);
  };

  // Grille à afficher : épisode en cours sinon map par défaut
  const displayGrid = epState.grid.length > 0 ? epState.grid : DEFAULT_MAPS[map] ?? [];
  const displayAgentRow = epState.grid.length > 0 ? epState.agentRow : 0;
  const displayAgentCol = epState.grid.length > 0 ? epState.agentCol : 0;
  const showAgent = epState.grid.length > 0 || !trainState.finalResult;

  const applyPreset = (p: typeof PRESETS[0]) => {
    setAgent(p.agent);
    setMap(p.map);
    setSlippery(p.slippery);
    setNEpisodes(p.nEpisodes);
    setHyperparams(p.hp);
  };

  // Epsilon courant estimé (affiché pendant le training)
  const lastPoint = trainState.points[trainState.points.length - 1];
  const currentEpsilon = lastPoint?.epsilon;

  return (
    <div className="app">
      <div className="bg-particles">
        {Array.from({ length: 20 }).map((_, i) => (
          <div key={i} className="particle" style={{ "--i": i } as React.CSSProperties} />
        ))}
      </div>

      <header className="app-header">
        <h1 className="app-title">
          <span className="title-ice">🧊</span>
          Find Your Ice Path
          <span className="title-ice">🧊</span>
        </h1>
        <p className="app-subtitle">Reinforcement Learning — FrozenLake</p>
      </header>

      <main className="app-main">
        <aside className="sidebar">

          {/* ── Reset global ── */}
          <button
            className="btn-reset-global"
            onClick={handleReset}
            disabled={trainState.running || epState.running}
          >
            🔄 Tout remettre à zéro
          </button>

          {/* ── Workflow guide ── */}
          <div className="workflow-guide">
            <div className={`workflow-step ${trainedSnapshot === null ? "step-active" : "step-done"}`}>
              <span className="step-num">1</span>
              <span>Choisis un preset ou configure les paramètres</span>
            </div>
            <div className={`workflow-step ${trainState.running ? "step-active" : trainState.finalResult ? "step-done" : ""}`}>
              <span className="step-num">2</span>
              <span>Lance l'entraînement</span>
              {trainState.running && currentEpsilon !== undefined && (
                <span className="epsilon-live">ε = {currentEpsilon.toFixed(3)}</span>
              )}
            </div>
            <div className={`workflow-step ${trainState.finalResult && !paramsChanged ? "step-active" : ""}`}>
              <span className="step-num">3</span>
              <span>Joue un épisode pour voir l'agent</span>
            </div>
          </div>

          {/* ── Bannière params modifiés ── */}
          {paramsChanged && (
            <div className="params-changed-banner">
              ⚠️ Paramètres modifiés — <strong>relance l'entraînement</strong> pour les appliquer
            </div>
          )}

          {/* ── Presets ── */}
          <div className="control-group">
            <label className="control-label">⚡ Presets recommandés</label>
            <div className="presets-grid">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  className="btn-preset"
                  onClick={() => applyPreset(p)}
                  title={p.desc}
                >
                  <span className="preset-emoji">{p.emoji}</span>
                  <span className="preset-label">{p.label}</span>
                  <span className="preset-desc">{p.desc}</span>
                </button>
              ))}
            </div>
          </div>

          <Controls
            agent={agent}
            map={map}
            slippery={slippery}
            speed={speed}
            nEpisodes={nEpisodes}
            hyperparams={hyperparams}
            episodeRunning={epState.running}
            trainRunning={trainState.running}
            onAgentChange={setAgent}
            onMapChange={setMap}
            onSlipperyChange={setSlippery}
            onSpeedChange={setSpeed}
            onEpisodesChange={setNEpisodes}
            onHyperparamsChange={setHyperparams}
            onPlay={handlePlay}
            onTrain={handleTrain}
            onStop={handleStop}
          />
        </aside>

        <section className="center">
          <div className="episode-meta">
            {epState.running && (
              <span className="pill pill-blue">Step {epState.currentStep}</span>
            )}
            {epState.totalReward > 0 && (
              <span className="pill pill-green">Reward {epState.totalReward.toFixed(2)}</span>
            )}
          </div>
          <Grid
            grid={displayGrid}
            agentRow={displayAgentRow}
            agentCol={displayAgentCol}
            showAgent={showAgent}
            history={epState.history}
            result={epState.result}
            qtable={qtable}
          />
        </section>

        <aside className="right-panel">
          {(trainState.running || trainState.points.length > 0) && (
            <div className="progress-bar-wrap">
              <div
                className="progress-bar"
                style={{ width: `${(trainState.progress / trainState.total) * 100}%` }}
              />
              <span className="progress-label">
                {trainState.progress} / {trainState.total} épisodes
              </span>
            </div>
          )}
          <TrainChart
            points={trainState.points}
            running={trainState.running}
            finalResult={trainState.finalResult}
          />
          <Analysis
            agent={agent}
            map={map}
            slippery={slippery}
            hyperparams={hyperparams}
            nEpisodes={nEpisodes}
            episodeResult={epState.result}
            trainResult={trainState.finalResult}
          />
        </aside>
      </main>
    </div>
  );
}
