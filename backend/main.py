# ============================================================
# main.py — FastAPI Application principale
# ============================================================
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import MatchRequest, PredictionResponse
from data_ingestion import collect_match_data
from analytics_engine import run_analytics
from monte_carlo import run_full_simulation

# ── Initialisation ────────────────────────────────────────────
app = FastAPI(
    title       = "🏀 BasketPredictAI",
    description = "Prédiction de matchs basketball par IA",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# ROUTES
# =============================================================

@app.get("/")
def root():
    return {
        "app":     "BasketPredictAI",
        "version": "1.0.0",
        "status":  "online",
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: MatchRequest):
    """
    Endpoint principal de prédiction.
    Reçoit deux équipes et retourne l'analyse complète.
    """
    try:
        # ── 1. Collecte données ───────────────────────────────
        data = await collect_match_data(
            team_home  = req.team_home,
            team_away  = req.team_away,
            league     = req.league or "NBA",
            match_date = req.match_date,
        )

        # ── 2. Moteur analytique ──────────────────────────────
        engine = run_analytics(
            home = data["home_stats"],
            away = data["away_stats"],
        )

        # ── 3. Monte Carlo ────────────────────────────────────
        simulation = run_full_simulation(
            engine    = engine,
            team_home = req.team_home,
            team_away = req.team_away,
        )

        # ── 4. Réponse complète ───────────────────────────────
        return PredictionResponse(
            team_home    = req.team_home,
            team_away    = req.team_away,
            league       = req.league or "NBA",
            match_date   = req.match_date,

            home_stats   = data["home_stats"],
            away_stats   = data["away_stats"],
            injuries_home = data["injuries_home"],
            injuries_away = data["injuries_away"],
            key_factor   = data["key_factor"],
            h2h_summary  = data["h2h_summary"],

            monte_carlo  = simulation["monte_carlo"],
            four_factors = engine.four_factors,
            scenarios    = simulation["scenarios"],

            bet_recommendations = simulation["bets"],
            confidence_global   = engine.confidence,
            data_sources        = data["data_sources"],
        )

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Erreur prédiction : {str(e)}",
        )


@app.get("/leagues")
def get_leagues():
    """Retourne les ligues supportées."""
    from config import SUPPORTED_LEAGUES
    return {"leagues": SUPPORTED_LEAGUES}


@app.get("/example")
def get_example():
    """Exemple de requête pour tester l'API."""
    return {
        "example_request": {
            "team_home":  "Los Angeles Lakers",
            "team_away":  "Boston Celtics",
            "league":     "NBA",
            "match_date": "2025-05-15",
        },
        "curl": (
            'curl -X POST "http://localhost:8000/predict" '
            '-H "Content-Type: application/json" '
            '-d \'{"team_home":"Lakers","team_away":"Celtics","league":"NBA"}\''
        ),
        }
