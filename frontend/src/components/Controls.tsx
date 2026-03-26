import type { AgentName, MapName } from "../types";
import InfoBubble from "./InfoBubble";

export interface Hyperparams {
  alpha: number;
  gamma: number;
  epsilon_decay: number;
  epsilon_min: number;
}

interface Props {
  agent: AgentName;
  map: MapName;
  slippery: boolean;
  speed: number;
  nEpisodes: number;
  hyperparams: Hyperparams;
  episodeRunning: boolean;
  trainRunning: boolean;
  onAgentChange: (a: AgentName) => void;
  onMapChange: (m: MapName) => void;
  onSlipperyChange: (v: boolean) => void;
  onSpeedChange: (v: number) => void;
  onEpisodesChange: (v: number) => void;
  onHyperparamsChange: (h: Hyperparams) => void;
  onPlay: () => void;
  onTrain: () => void;
  onStop: () => void;
}

const AGENTS: { name: AgentName; label: string; trainable: boolean; tag: string }[] = [
  { name: "random",    label: "🎲 Random",    trainable: false, tag: "baseline" },
  { name: "qlearning", label: "📊 Q-Learning", trainable: true,  tag: "off-policy" },
  { name: "sarsa",     label: "🔄 SARSA",      trainable: true,  tag: "on-policy" },
  { name: "dqn",       label: "🧠 DQN",        trainable: true,  tag: "deep" },
];

const AGENT_DESCRIPTIONS: Record<AgentName, React.ReactNode> = {
  random: (
    <>
      <p>Choisit une action au hasard à chaque step. Aucun apprentissage.</p>
      <p>Sert de <strong>baseline</strong> : tout bon agent doit faire mieux que ça.</p>
    </>
  ),
  qlearning: (
    <>
      <p><strong>Off-policy</strong> : apprend la meilleure stratégie possible, même si ses actions actuelles sont différentes.</p>
      <p>Mise à jour :<br /><code>Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s') - Q(s,a)]</code></p>
      <p>Il utilise le <strong>max</strong> sur s' → il imagine toujours qu'il fera le meilleur choix ensuite, même pendant l'exploration.</p>
    </>
  ),
  sarsa: (
    <>
      <p><strong>On-policy</strong> : apprend depuis les actions qu'il prend réellement, y compris les erreurs d'exploration.</p>
      <p>Mise à jour :<br /><code>Q(s,a) ← Q(s,a) + α·[r + γ·Q(s',a') - Q(s,a)]</code></p>
      <p>Il utilise <strong>a'</strong>, l'action réellement choisie en s' → plus prudent, apprend le comportement réel de sa politique.</p>
    </>
  ),
  dqn: (
    <>
      <p><strong>Deep Q-Network</strong> : remplace la Q-table par un réseau de neurones.</p>
      <p>Utile quand l'espace d'états est grand (ici on l'utilise pour l'exercice).</p>
      <p>Ajout clé : <strong>Replay Buffer</strong> (mémoire d'expériences) + <strong>Target Network</strong> (stabilité de l'entraînement).</p>
    </>
  ),
};

function SliderRow({
  label, value, min, max, step, format, onChange, info,
}: {
  label: string; value: number; min: number; max: number; step: number;
  format: (v: number) => string; onChange: (v: number) => void; info: React.ReactNode;
}) {
  return (
    <div className="control-group">
      <div className="control-label-row">
        <label className="control-label">{label}</label>
        <InfoBubble title={label}>{info}</InfoBubble>
      </div>
      <div className="slider-row">
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        <span className="slider-val">{format(value)}</span>
      </div>
    </div>
  );
}

export default function Controls({
  agent, map, slippery, speed, nEpisodes, hyperparams,
  episodeRunning, trainRunning,
  onAgentChange, onMapChange, onSlipperyChange, onSpeedChange,
  onEpisodesChange, onHyperparamsChange, onPlay, onTrain, onStop,
}: Props) {
  const selectedAgent = AGENTS.find((a) => a.name === agent)!;
  const hp = hyperparams;
  const setHp = (patch: Partial<Hyperparams>) => onHyperparamsChange({ ...hp, ...patch });

  return (
    <div className="controls">

      {/* ── Agent ── */}
      <div className="control-group">
        <div className="control-label-row">
          <label className="control-label">Agent</label>
        </div>
        <div className="btn-group">
          {AGENTS.map((a) => (
            <div key={a.name} className="agent-btn-wrap">
              <button
                className={`btn-agent ${agent === a.name ? "active" : ""}`}
                onClick={() => onAgentChange(a.name)}
              >
                {a.label}
                <span className={`agent-tag tag-${a.tag}`}>{a.tag}</span>
              </button>
            </div>
          ))}
        </div>
        {/* Description de l'agent sélectionné */}
        <div className="agent-desc">
          <InfoBubble title={`${selectedAgent.label} — comment ça marche ?`}>
            {AGENT_DESCRIPTIONS[agent]}
          </InfoBubble>
          <span className="agent-desc-text">Comment fonctionne cet agent ?</span>
        </div>
      </div>

      {/* ── Map ── */}
      <div className="control-group">
        <div className="control-label-row">
          <label className="control-label">Map</label>
          <InfoBubble title="Taille de la grille">
            <p><strong>4×4</strong> : 16 états, plus simple à apprendre. Bon pour comprendre les algos.</p>
            <p><strong>8×8</strong> : 64 états, plus difficile — augmenter N épisodes et diminuer ε-decay.</p>
          </InfoBubble>
        </div>
        <div className="btn-group">
          {(["4x4", "8x8"] as MapName[]).map((m) => (
            <button key={m} className={`btn-map ${map === m ? "active" : ""}`} onClick={() => onMapChange(m)}>
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* ── Slippery ── */}
      <div className="control-group">
        <div className="control-label-row">
          <label className="control-label">Mode</label>
          <InfoBubble title="Glissant vs Déterministe">
            <p><strong>Glissant 🧊</strong> : chaque action n'a qu'1/3 de chance de se faire dans la direction choisie. Les 2/3 restants partent aléatoirement. Reflète l'incertitude réelle.</p>
            <p><strong>Déterministe 🔒</strong> : l'agent va exactement où il choisit. Win rate bien plus élevé — utile pour valider les algos.</p>
          </InfoBubble>
        </div>
        <label className="toggle">
          <input type="checkbox" checked={slippery} onChange={(e) => onSlipperyChange(e.target.checked)} />
          <span className="toggle-slider" />
          <span className="toggle-label">{slippery ? "🧊 Glissant" : "🔒 Déterministe"}</span>
        </label>
      </div>

      {/* ── Hyperparamètres (agents entraînables seulement) ── */}
      {selectedAgent.trainable && (
        <div className="hyperparam-section">
          <div className="section-title">
            ⚙️ Hyperparamètres
          </div>

          <SliderRow
            label="α — Learning Rate"
            value={hp.alpha} min={0.01} max={1.0} step={0.01}
            format={(v) => v.toFixed(2)}
            onChange={(v) => setHp({ alpha: v })}
            info={
              <>
                <p><strong>Taux d'apprentissage</strong> — à quelle vitesse l'agent met à jour ses croyances.</p>
                <p>🔺 Élevé (0.8–1.0) : apprend vite mais instable, peut "oublier" ce qu'il a appris.</p>
                <p>🔻 Faible (0.01–0.1) : apprend lentement mais stable. Nécessite plus d'épisodes.</p>
                <p>💡 Commence à <strong>0.8</strong> sur FrozenLake 4×4.</p>
              </>
            }
          />

          <SliderRow
            label="γ — Discount Factor"
            value={hp.gamma} min={0.5} max={0.999} step={0.01}
            format={(v) => v.toFixed(3)}
            onChange={(v) => setHp({ gamma: v })}
            info={
              <>
                <p><strong>Facteur d'actualisation</strong> — combien l'agent valorise les récompenses futures vs immédiates.</p>
                <p>γ = 0.99 → l'agent pense <em>loin</em>, planifie sur le long terme.</p>
                <p>γ = 0.5 → l'agent est myope, préfère les petites récompenses immédiates.</p>
                <p>💡 Sur FrozenLake, le goal est loin → prendre <strong>γ ≥ 0.95</strong>.</p>
              </>
            }
          />

          <SliderRow
            label="ε-decay — Vitesse d'exploitation"
            value={hp.epsilon_decay} min={0.9} max={0.9999} step={0.001}
            format={(v) => v.toFixed(4)}
            onChange={(v) => setHp({ epsilon_decay: v })}
            info={
              <>
                <p><strong>Décroissance d'epsilon</strong> — à quelle vitesse l'agent passe de l'exploration à l'exploitation.</p>
                <p>ε commence à 1.0 (100% aléatoire) et est multiplié par ε-decay après chaque épisode.</p>
                <p>🔺 Proche de 1 (0.999) : explore longtemps → nécessite plus d'épisodes.</p>
                <p>🔻 Plus faible (0.99) : exploite rapidement sa stratégie → peut se bloquer sur une solution sous-optimale.</p>
                <p>💡 Règle d'or : ε doit atteindre ~0.1 vers la <em>moitié</em> des épisodes.</p>
              </>
            }
          />

          <SliderRow
            label="ε-min — Plancher d'exploration"
            value={hp.epsilon_min} min={0.001} max={0.2} step={0.001}
            format={(v) => v.toFixed(3)}
            onChange={(v) => setHp({ epsilon_min: v })}
            info={
              <>
                <p><strong>Valeur minimale d'epsilon</strong> — l'agent n'explorera jamais moins que ça, même après beaucoup d'épisodes.</p>
                <p>🔺 Élevé (0.1–0.2) : l'agent continue d'explorer en permanence. Utile si l'environnement change.</p>
                <p>🔻 Faible (0.001–0.01) : l'agent exploite quasi-totalement sa stratégie apprise.</p>
                <p>💡 Sur FrozenLake statique : <strong>0.01</strong> suffit. Le replay est toujours à ε=0.</p>
              </>
            }
          />

          <SliderRow
            label="N — Épisodes d'entraînement"
            value={nEpisodes} min={100} max={10000} step={100}
            format={(v) => v.toLocaleString()}
            onChange={onEpisodesChange}
            info={
              <>
                <p>Nombre d'épisodes pour entraîner l'agent.</p>
                <p>Trop peu → variance trop élevée, l'agent n'a pas vu assez de situations.</p>
                <p>Trop → temps de calcul inutile une fois convergé.</p>
                <p>💡 4×4 glissant : <strong>2 000–5 000</strong> épisodes. 8×8 : <strong>5 000–20 000</strong>.</p>
              </>
            }
          />
        </div>
      )}

      {/* ── Vitesse de replay ── */}
      <div className="control-group">
        <div className="control-label-row">
          <label className="control-label">Vitesse épisode</label>
        </div>
        <div className="slider-row">
          <input type="range" min={0.05} max={1.0} step={0.05} value={speed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))} />
          <span className="slider-val">{speed.toFixed(2)}s</span>
        </div>
      </div>

      {/* ── Boutons ── */}
      <div className="action-btns">
        <button className="btn-play" onClick={onPlay} disabled={episodeRunning || trainRunning}>
          ▶ Jouer un épisode
        </button>
        {selectedAgent.trainable && (
          <button className="btn-train" onClick={onTrain} disabled={episodeRunning || trainRunning}>
            🏋️ Entraîner
          </button>
        )}
        {(episodeRunning || trainRunning) && (
          <button className="btn-stop" onClick={onStop}>⏹ Stop</button>
        )}
      </div>
    </div>
  );
}
