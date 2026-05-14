# ============================================================
# models.py — Schémas Pydantic
# ============================================================
from pydantic import BaseModel, Field
from typing import Optional, List


# ── Input utilisateur ─────────────────────────────────────────
class MatchRequest(BaseModel):
    team_home:  str           = Field(..., example="Los Angeles Lakers")
    team_away:  str           = Field(..., example="Boston Celtics")
    league:     Optional[str] = Field("NBA", example="NBA")
    match_date: Optional[str] = Field(None,  example="2025-05-15")


# ── Stats d'une équipe ────────────────────────────────────────
class TeamStats(BaseModel):
    team_name:    str
    elo_rating:   float = 1500.0
    net_rating:   float = 0.0
    pace:         float = 100.0
    efg_pct:      float = 0.50
    tov_pct:      float = 13.0
    orb_pct:      float = 25.0
    ftr:          float = 0.25
    ts_pct:       float = 0.55
    ppp:          float = 1.10
    rest_days:    int   = 2
    back_to_back: bool  = False


# ── Rapport blessures ─────────────────────────────────────────
class InjuryReport(BaseModel):
    player_name: str
    status:      str
    impact_pts:  float = 0.0


# ── Four Factors ──────────────────────────────────────────────
class FourFactorsResult(BaseModel):
    home_score: float
    away_score: float
    advantage:  str


# ── Monte Carlo ───────────────────────────────────────────────
class MonteCarloResult(BaseModel):
    home_win_pct:   float
    away_win_pct:   float
    avg_score_home: float
    avg_score_away: float
    std_home:       float
    std_away:       float
    simulations:    int


# ── Scénario de score ─────────────────────────────────────────
class ScoreScenario(BaseModel):
    label:      str
    score_home: int
    score_away: int
    confidence: float


# ── Recommandation pari ───────────────────────────────────────
class BetRecommendation(BaseModel):
    label:        str
    bet_type:     str
    description:  str
    probability:  float
    edge:         Optional[float] = None
    is_value_bet: bool = False


# ── Réponse complète ──────────────────────────────────────────
class PredictionResponse(BaseModel):
    team_home:    str
    team_away:    str
    league:       str
    match_date:   Optional[str]

    home_stats:   TeamStats
    away_stats:   TeamStats
    injuries_home: List[InjuryReport] = []
    injuries_away: List[InjuryReport] = []
    key_factor:   str
    h2h_summary:  str

    monte_carlo:  MonteCarloResult
    four_factors: FourFactorsResult
    scenarios:    List[ScoreScenario]

    bet_recommendations:  List[BetRecommendation]
    confidence_global:    float
    data_sources:         List[str] = []
