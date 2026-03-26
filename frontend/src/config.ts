// Configuré via variables d'env Vite au build
// En dev local (npm run dev) : localhost:8000 par défaut
// En Docker : injecté via VITE_API_URL / VITE_WS_URL (build args)

export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const WS_BASE = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000";

// Grilles par défaut (identiques au backend)
export const DEFAULT_MAPS: Record<string, string[]> = {
  "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
  "8x8": [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
  ],
};
