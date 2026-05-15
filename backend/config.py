# ============================================================
# config.py — Clés API & Constantes globales
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

# ── APIs existantes ───────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "")

# Jina AI Reader : sans clé
JINA_BASE_URL   = "https://r.jina.ai/"

# ── NEW : The Odds API ────────────────────────────────────────
ODDS_API_KEY    = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE   = "https://api.the-odds-api.com/v4"
ODDS_SPORT      = "basketball_nba"          # slug NBA sur The Odds API
ODDS_REGIONS    = "us"                      # us | uk | eu | au
ODDS_MARKETS    = "h2h"                     # moneyline
ODDS_BOOKMAKERS = ["draftkings", "fanduel", "betmgm"]  # priorité consensus

# ── NEW : Tank01 Fantasy Stats (RapidAPI) ────────────────────
TANK01_API_KEY  = os.getenv("TANK01_API_KEY", "")
TANK01_BASE     = "https://tank01-fantasy-stats.p.rapidapi.com"
TANK01_HOST     = "tank01-fantasy-stats.p.rapidapi.com"

# ── NEW : NBA Stats public (stats.nba.com) ───────────────────
NBA_STATS_BASE  = "https://stats.nba.com/stats"
NBA_STATS_HEADERS = {
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection":      "keep-alive",
    "Host":            "stats.nba.com",
    "Origin":          "https://www.nba.com",
    "Referer":         "https://www.nba.com/",
    "User-Agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}
# Saison courante (format NBA : "2024-25")
NBA_CURRENT_SEASON   = "2024-25"
NBA_SEASON_TYPE      = "Regular Season"
# Nombre de derniers matchs pour recent_form
NBA_FORM_LAST_N      = 10

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
GEMINI_MAX_TOKENS = 60000
