# ============================================================
# models.py — Schémas Pydantic
# ============================================================
from pydantic import BaseModel, Field
from typing import Optional, List


class MatchRequest(BaseModel):
    team_home:  str           = Field(..., example="Los Angeles Lakers")
    team_away:  str           = Field(..., example="Boston Celtics")
    league:     Optional[str] = Field("NBA", example="NBA")
    match_date: Optional[str] = Field(None,  example="2025-05-15")


class TeamStats(BaseModel):
    team_name:    str
    elo_rating:   float       = 1500.0
    net_rating:   float       = 0.0
    pace:         float       = 100.0
    efg_pct:      float       = 0.50
    tov_pct:      float       = 13.0
    orb_pct:      float       = 25.0
    ftr:          float       = 0.25
    ts_pct:       float       = 0.55
    ppp:          float       = 1.10
    rest_days:    int         = 2
    back_to_back: bool        = False
    recent_form:  List[float] = Field(default_factory=list)


class InjuryReport(BaseModel):
    player_name: str
    status:      str
    impact_pts:  float = 0.0


class FourFactorsResult(BaseModel):
    home_score: float
    away_score: float
    advantage:  str


class MonteCarloResult(BaseModel):
    home_win_pct:   float
    away_win_pct:   float
    avg_score_home: float
    avg_score_away: float
    std_home:       float
    std_away:       float
    simulations:    int


class ScoreScenario(BaseModel):
    label:      str
    score_home: int
    score_away: int
    confidence: float


class BetRecommendation(BaseModel):
    label:        str
    bet_type:     str
    description:  str
    probability:  float
    edge:         Optional[float] = None
    is_value_bet: bool = False


class MLResult(BaseModel):
    home_win_prob:   float
    away_win_prob:   float
    model_used:      str
    top_features:    List[str] = []
    form_score_home: float     = 0.0
    form_score_away: float     = 0.0


# ── NEW : cotes en temps réel ─────────────────────────────────
class LiveOdds(BaseModel):
    odds_home:  float            # américain, ex: -150
    odds_away:  float            # américain, ex: +130
    bookmaker:  str
    implied_home: float          # probabilité implicite home
    implied_away: float          # probabilité implicite away


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
    ml_result:    Optional[MLResult]  = None

    # ── NEW ───────────────────────────────────────────────────
    live_odds:    Optional[LiveOdds]  = None

    bet_recommendations: List[BetRecommendation]
    confidence_global:   float
    data_sources:        List[str] = []
