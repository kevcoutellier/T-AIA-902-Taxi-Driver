import type { AgentName, MapName } from "../types";
import type { Hyperparams } from "./Controls";

interface Props {
  agent: AgentName;
  map: MapName;
  slippery: boolean;
  hyperparams: Hyperparams;
  nEpisodes: number;
  episodeResult: { won: boolean; steps: number } | null;
  trainResult: { win_rate: number; avg_reward: number; avg_steps: number } | null;
}

interface Tip {
  level: "error" | "warn" | "ok" | "info";
  icon: string;
  title: string;
  body: string;
  fix?: string;
}

function analyze(props: Props): Tip[] {
  const { agent, map, slippery, hyperparams: hp, nEpisodes, episodeResult, trainResult } = props;
  const tips: Tip[] = [];
  const wr = trainResult?.win_rate ?? 0;

  // Pas encore entraîné
  if (!trainResult) {
    if (agent === "random") {
      tips.push({
        level: "info", icon: "🎲",
        title: "Agent aléatoire",
        body: "Random ne peut pas apprendre. Il sert de baseline : tout bon agent doit faire mieux.",
      });
    } else {
      tips.push({
        level: "warn", icon: "⚠️",
        title: "Pas encore entraîné",
        body: "L'agent joue avec une Q-table vide — il choisit au hasard.",
        fix: "Lance l'entraînement avant de jouer un épisode.",
      });
    }
    return tips;
  }

  // Win rate global
  if (wr < 20) {
    tips.push({
      level: "error", icon: "📉",
      title: `Win rate trop faible (${wr.toFixed(0)}%)`,
      body: "L'agent n'a pas appris grand-chose. Plusieurs causes possibles.",
    });
  } else if (wr < 60) {
    tips.push({
      level: "warn", icon: "📊",
      title: `Win rate moyen (${wr.toFixed(0)}%)`,
      body: "L'agent apprend mais la politique n'est pas encore stable.",
    });
  } else {
    tips.push({
      level: "ok", icon: "✅",
      title: `Bon win rate (${wr.toFixed(0)}%)`,
      body: "L'agent a bien appris. Si il tombe encore en épisode, c'est une lacune de la Q-table.",
    });
  }

  // Mode glissant
  if (slippery && wr < 50) {
    tips.push({
      level: "warn", icon: "🧊",
      title: "Mode glissant difficile",
      body: "En mode glissant, chaque action n'a que 1/3 de chance d'aller dans la bonne direction.",
      fix: "Commence par valider en mode Déterministe, puis repasse en Glissant.",
    });
  }

  // Nombre d'épisodes
  const minEp = map === "8x8" ? 5000 : 2000;
  if (nEpisodes < minEp && wr < 70) {
    tips.push({
      level: "warn", icon: "🔢",
      title: "Pas assez d'épisodes",
      body: `Sur une map ${map}${slippery ? " glissante" : ""}, l'agent a besoin de voir plus de situations.`,
      fix: `Augmente N à ${minEp.toLocaleString()}${map === "8x8" ? "–20 000" : "–5 000"}.`,
    });
  }

  // Gamma trop faible
  if (hp.gamma < 0.9 && wr < 60) {
    tips.push({
      level: "warn", icon: "🔭",
      title: "γ trop faible — agent myope",
      body: `γ = ${hp.gamma.toFixed(2)} : l'agent ne valorise pas assez les récompenses futures. Le goal est loin, il faut voir loin.`,
      fix: "Monte γ à 0.95 minimum, 0.99 recommandé.",
    });
  }

  // Alpha inadapté
  if (hp.alpha < 0.1 && wr < 50) {
    tips.push({
      level: "warn", icon: "🐌",
      title: "α trop faible — apprentissage lent",
      body: `α = ${hp.alpha.toFixed(2)} : l'agent met trop de temps à intégrer chaque expérience.`,
      fix: "Monte α à 0.5–0.8 pour FrozenLake.",
    });
  }

  // Epsilon decay trop lent → pas assez d'exploitation
  const epsilonAtHalf = Math.pow(hp.epsilon_decay, nEpisodes / 2);
  if (epsilonAtHalf > 0.5 && wr < 60) {
    tips.push({
      level: "warn", icon: "🎲",
      title: "ε encore trop élevé à mi-entraînement",
      body: `Avec ε-decay = ${hp.epsilon_decay.toFixed(4)}, epsilon vaut encore ${(epsilonAtHalf * 100).toFixed(0)}% après ${(nEpisodes / 2).toLocaleString()} épisodes — l'agent explore trop longtemps.`,
      fix: `Baisse ε-decay à ${(Math.pow(0.1, 2 / nEpisodes)).toFixed(4)} pour que ε atteigne 0.1 à mi-parcours.`,
    });
  }

  // Epsilon decay trop rapide → exploitation prématurée
  const epsilonAt10pct = Math.pow(hp.epsilon_decay, nEpisodes * 0.1);
  if (epsilonAt10pct < 0.15 && wr < 50) {
    tips.push({
      level: "warn", icon: "⚡",
      title: "ε-decay trop rapide — exploitation prématurée",
      body: `L'agent arrête d'explorer trop tôt (ε < 0.15 dès ${(nEpisodes * 0.1).toFixed(0)} épisodes) avant d'avoir vu assez de l'environnement.`,
      fix: "Monte ε-decay vers 0.995–0.999.",
    });
  }

  // Chute en greedy malgré bon win rate
  if (episodeResult && !episodeResult.won && wr >= 60) {
    tips.push({
      level: "info", icon: "🕳️",
      title: "Chute malgré bon entraînement",
      body: "La Q-table a une lacune sur cet état précis. C'est normal : l'agent n'a pas forcément visité tous les chemins.",
      fix: "Augmente N épisodes pour couvrir plus de trajectoires, ou relance plusieurs fois l'entraînement.",
    });
  }

  // Conseil SARSA vs Q-Learning
  if (agent === "sarsa" && slippery && wr < 40) {
    tips.push({
      level: "info", icon: "🔄",
      title: "SARSA on-policy en mode glissant",
      body: "SARSA apprend la politique qu'il suit réellement, y compris ses erreurs. En mode glissant, il est naturellement plus prudent mais converge plus lentement.",
      fix: "Compare avec Q-Learning (off-policy) sur les mêmes paramètres.",
    });
  }

  return tips;
}

const LEVEL_STYLE: Record<Tip["level"], string> = {
  error: "tip-error",
  warn:  "tip-warn",
  ok:    "tip-ok",
  info:  "tip-info",
};

export default function Analysis(props: Props) {
  const { episodeResult } = props;

  // N'affiche l'analyse que si un épisode a été joué ou si un entraînement est terminé
  if (!props.trainResult && !episodeResult) return null;

  const tips = analyze(props);
  if (!tips.length) return null;

  return (
    <div className="analysis-panel">
      <h3 className="panel-title">🔍 Analyse & Conseils</h3>
      <div className="tips-list">
        {tips.map((tip, i) => (
          <div key={i} className={`tip ${LEVEL_STYLE[tip.level]}`}>
            <div className="tip-header">
              <span className="tip-icon">{tip.icon}</span>
              <span className="tip-title">{tip.title}</span>
            </div>
            <p className="tip-body">{tip.body}</p>
            {tip.fix && (
              <p className="tip-fix">💡 {tip.fix}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
