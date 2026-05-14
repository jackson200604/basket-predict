# ============================================================
# config.py — Clés API & Constantes globales
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

# ── APIs ─────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "")

# Jina AI Reader : sans clé
JINA_BASE_URL   = "https://r.jina.ai/"

# ── Paramètres Moteur ─────────────────────────────────────────
MONTE_CARLO_SIMS = 20_000
ELO_K_FACTOR     = 20
ELO_HOME_BOOST   = 100
TIME_DECAY_ALPHA = 0.92
ROLLING_WINDOW   = 15

# ── Ligues supportées ─────────────────────────────────────────
SUPPORTED_LEAGUES = ["NBA", "EuroLeague", "NCAA", "BSL", "LNB"]

# ── Pondération Four Factors (Dean Oliver) ────────────────────
FOUR_FACTORS_WEIGHTS = {
    "eFG":  0.40,
    "TOV":  0.25,
    "ORB":  0.20,
    "FTr":  0.15,
}

# ── Gemini ───────────────────────────────────────────────────
GEMINI_MODEL      = "gemini-3-flash-preview"
GEMINI_MAX_TOKENS = 30000
