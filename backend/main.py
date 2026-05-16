# ============================================================
# main.py — FastAPI Application principale
# ============================================================
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import MatchRequest, PredictionResponse, MLResult, LiveOdds
from data_ingestion import collect_match_data
from analytics_engine import run_analytics
from monte_carlo import run_full_simulation, compute_implied_probability
from ml_model import predict_ml
from lstm_model import predict_lstm, load_lstm_at_startup
from stacking_ensemble import run_stacking, stacking_log
from api_clients import get_odds

app = FastAPI(
    title       = "🏀 BasketPredictAI",
    description = "Prédiction de matchs basketball par IA",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    import asyncio
    from ml_model import _load_model
    loop = asyncio.get_event_loop()
    # XGBoost : chargement rapide, synchrone OK
    _load_model()
    # LSTM : peut entraîner ~30s si .keras absent → thread séparé
    await loop.run_in_executor(None, load_lstm_at_startup)


@app.get("/")
def root():
    return {"app": "BasketPredictAI", "version": "1.0.0", "status": "online", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: MatchRequest):
    try:
        # ── 1. Données & cotes en parallèle ──────────────────
        data, odds_raw = await asyncio.gather(
            collect_match_data(
                team_home=req.team_home, team_away=req.team_away,
                league=req.league or "NBA", match_date=req.match_date,
            ),
            get_odds(req.team_home, req.team_away),
        )

        # ── 2. Moteur analytique ──────────────────────────────
        engine = run_analytics(
            home=data["home_stats"], away=data["away_stats"],
        )

        # ── 3. ML XGBoost + LSTM → Stacking Ensemble ─────────
        ml_output   = predict_ml(
            engine=engine,
            form_home=engine.form_score_home,
            form_away=engine.form_score_away,
        )
        lstm_output  = predict_lstm(
            form_home=data["home_stats"].recent_form,
            form_away=data["away_stats"].recent_form,
        )
        stack_output = run_stacking(ml_output, lstm_output)
        print(stacking_log(stack_output))

        ml_result = MLResult(
            home_win_prob   = stack_output["stacked_prob_home"],
            away_win_prob   = stack_output["stacked_prob_away"],
            model_used      = stack_output["model_used"],
            top_features    = stack_output["top_features"],
            form_score_home = engine.form_score_home,
            form_score_away = engine.form_score_away,
            stacking_detail = stack_output["stacking_detail"],
        )

        # ── 4. Cotes réelles (avec fallback -110/-110) ────────
        odds_home = -110.0
        odds_away = -110.0
        live_odds = None

        if odds_raw:
            odds_home = odds_raw["odds_home"]
            odds_away = odds_raw["odds_away"]
            live_odds = LiveOdds(
                odds_home    = odds_home,
                odds_away    = odds_away,
                bookmaker    = odds_raw["bookmaker"],
                implied_home = round(compute_implied_probability(odds_home), 4),
                implied_away = round(compute_implied_probability(odds_away), 4),
            )

        # ── 5. Monte Carlo (blend + value bet sur vraies cotes)
        simulation = run_full_simulation(
            engine             = engine,
            team_home          = req.team_home,
            team_away          = req.team_away,
            odds_home          = odds_home,
            odds_away          = odds_away,
            stacking_prob_home = stack_output["stacked_prob_home"],
            use_stacking       = True,
        )

        # ── 6. Réponse ────────────────────────────────────────
        return PredictionResponse(
            team_home     = req.team_home,
            team_away     = req.team_away,
            league        = req.league or "NBA",
            match_date    = req.match_date,
            home_stats    = data["home_stats"],
            away_stats    = data["away_stats"],
            injuries_home = data["injuries_home"],
            injuries_away = data["injuries_away"],
            key_factor    = data["key_factor"],
            h2h_summary   = data["h2h_summary"],
            monte_carlo   = simulation["monte_carlo"],
            four_factors  = engine.four_factors,
            scenarios     = simulation["scenarios"],
            ml_result     = ml_result,
            live_odds     = live_odds,
            bet_recommendations = simulation["bets"],
            confidence_global   = engine.confidence,
            data_sources        = data["data_sources"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")


@app.get("/leagues")
def get_leagues():
    from config import SUPPORTED_LEAGUES
    return {"leagues": SUPPORTED_LEAGUES}


@app.get("/example")
def get_example():
    return {
        "example_request": {
            "team_home": "Los Angeles Lakers", "team_away": "Boston Celtics",
            "league": "NBA", "match_date": "2025-05-15",
        },
        "curl": (
            'curl -X POST "http://localhost:8000/predict" '
            '-H "Content-Type: application/json" '
            '-d \'{"team_home":"Lakers","team_away":"Celtics","league":"NBA"}\''
        ),
    }