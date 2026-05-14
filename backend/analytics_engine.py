# ============================================================
# analytics_engine.py — Elo · Four Factors · PPP · Time-decay
# ============================================================
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from models import TeamStats, FourFactorsResult
from config import (
    ELO_K_FACTOR, ELO_HOME_BOOST, TIME_DECAY_ALPHA,
    FOUR_FACTORS_WEIGHTS,
)


# =============================================================
# 0. STRUCTURES
# =============================================================

@dataclass
class MatchContext:
    rest_advantage:  float = 0.0
    travel_penalty:  float = 0.0
    altitude_factor: float = 0.0
    home_court:      float = 3.5
    pace_mismatch:   float = 0.0


@dataclass
class EngineResult:
    home_elo:             float = 1500.0
    away_elo:             float = 1500.0
    elo_win_prob_home:    float = 0.50
    elo_win_prob_away:    float = 0.50
    four_factors:         Optional[FourFactorsResult] = None
    four_factors_edge:    float = 0.0
    expected_pace:        float = 100.0
    home_ppp_projected:   float = 1.10
    away_ppp_projected:   float = 1.10
    home_expected_score:  float = 108.0
    away_expected_score:  float = 105.0
    home_std:             float = 10.0
    away_std:             float = 10.0
    context:              MatchContext = field(default_factory=MatchContext)
    confidence:           float = 0.65


# =============================================================
# 1. ELO DYNAMIQUE
# =============================================================

def expected_elo_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin: float, elo_diff: float) -> float:
    margin = max(margin, 1.0)
    return (math.log(margin + 1.0) * 2.2) / (elo_diff * 0.001 + 2.2)


def update_elo(
    rating_winner: float,
    rating_loser:  float,
    margin:        float,
    k:             float = ELO_K_FACTOR,
) -> tuple[float, float]:
    expected_w = expected_elo_score(rating_winner, rating_loser)
    mov_mult   = mov_multiplier(margin, rating_winner - rating_loser)
    delta      = k * mov_mult * (1.0 - expected_w)
    return (
        round(rating_winner + delta, 2),
        round(rating_loser  - delta, 2),
    )


def compute_elo_win_probability(
    home: TeamStats,
    away: TeamStats,
) -> tuple[float, float]:
    home_elo_adj = home.elo_rating + ELO_HOME_BOOST
    prob_home    = expected_elo_score(home_elo_adj, away.elo_rating)
    return round(prob_home, 4), round(1.0 - prob_home, 4)


# =============================================================
# 2. FOUR FACTORS
# =============================================================

def compute_four_factors(home: TeamStats, away: TeamStats) -> FourFactorsResult:
    w = FOUR_FACTORS_WEIGHTS

    home_score = (
        w["eFG"] *  home.efg_pct
      + w["TOV"] * (1.0 - home.tov_pct / 100.0)
      + w["ORB"] *  home.orb_pct / 100.0
      + w["FTr"] *  home.ftr
    )
    away_score = (
        w["eFG"] *  away.efg_pct
      + w["TOV"] * (1.0 - away.tov_pct / 100.0)
      + w["ORB"] *  away.orb_pct / 100.0
      + w["FTr"] *  away.ftr
    )

    diff = home_score - away_score
    if   diff >  0.02: advantage = "home"
    elif diff < -0.02: advantage = "away"
    else:              advantage = "even"

    return FourFactorsResult(
        home_score=round(home_score, 4),
        away_score=round(away_score, 4),
        advantage=advantage,
    )


# =============================================================
# 3. PPP
# =============================================================

def project_ppp(off_ppp: float, def_rating: float) -> float:
    defense_factor = 1.0 - (def_rating / 200.0)
    projected      = off_ppp * max(defense_factor, 0.75)
    return round(min(projected, 1.40), 4)


def expected_score_from_ppp(ppp: float, possessions: float) -> float:
    return round(ppp * possessions, 1)


# =============================================================
# 4. TIME-DECAY
# =============================================================

def time_decay_weights(n_games: int, alpha: float = TIME_DECAY_ALPHA) -> np.ndarray:
    weights = np.array([alpha ** i for i in range(n_games - 1, -1, -1)])
    return weights / weights.sum()


def weighted_average(values: list[float], alpha: float = TIME_DECAY_ALPHA) -> float:
    if not values:
        return 0.0
    n       = len(values)
    weights = time_decay_weights(n, alpha)
    return float(np.dot(weights, values[-n:]))


# =============================================================
# 5. PACE
# =============================================================

def compute_expected_pace(
    home_pace: float,
    away_pace: float,
    league_avg: float = 100.0,
) -> float:
    raw     = (home_pace + away_pace) / 2.0
    blended = 0.70 * raw + 0.30 * league_avg
    return round(blended, 1)


# =============================================================
# 6. CONTEXTE
# =============================================================

def compute_context(home: TeamStats, away: TeamStats) -> MatchContext:
    rest_diff      = home.rest_days - away.rest_days
    rest_advantage = max(-3.0, min(3.0, rest_diff * 1.0))

    travel_penalty = 0.0
    if home.back_to_back: travel_penalty -= 3.5
    if away.back_to_back: travel_penalty += 3.5

    pace_mismatch  = abs(home.pace - away.pace) * 0.05

    return MatchContext(
        rest_advantage=round(rest_advantage, 2),
        travel_penalty=round(travel_penalty, 2),
        home_court=3.5,
        pace_mismatch=round(pace_mismatch, 2),
    )


def apply_context_to_scores(
    home_score: float,
    away_score: float,
    ctx: MatchContext,
) -> tuple[float, float]:
    home_adj = home_score + ctx.home_court + ctx.rest_advantage + ctx.travel_penalty
    away_adj = away_score - ctx.rest_advantage - ctx.travel_penalty
    return round(home_adj, 1), round(away_adj, 1)


# =============================================================
# 7. REGRESSION TO THE MEAN
# =============================================================

def regress_to_mean(value: float, mean: float, weight: float = 0.15) -> float:
    return round(value * (1.0 - weight) + mean * weight, 4)


def apply_regression(home: TeamStats, away: TeamStats) -> tuple[TeamStats, TeamStats]:
    LEAGUE_AVG = {
        "net_rating": 0.0,
        "efg_pct":    0.52,
        "tov_pct":    13.0,
        "ppp":        1.12,
    }
    for team in (home, away):
        team.net_rating = regress_to_mean(team.net_rating, LEAGUE_AVG["net_rating"])
        team.efg_pct    = regress_to_mean(team.efg_pct,    LEAGUE_AVG["efg_pct"])
        team.tov_pct    = regress_to_mean(team.tov_pct,    LEAGUE_AVG["tov_pct"])
        team.ppp        = regress_to_mean(team.ppp,         LEAGUE_AVG["ppp"])
    return home, away


# =============================================================
# 8. MOTEUR PRINCIPAL
# =============================================================

def run_analytics(home: TeamStats, away: TeamStats) -> EngineResult:
    result = EngineResult()

    # Régression
    home, away = apply_regression(home, away)

    # Elo
    result.home_elo = home.elo_rating
    result.away_elo = away.elo_rating
    result.elo_win_prob_home, result.elo_win_prob_away = \
        compute_elo_win_probability(home, away)

    # Four Factors
    result.four_factors      = compute_four_factors(home, away)
    result.four_factors_edge = round(
        result.four_factors.home_score - result.four_factors.away_score, 4
    )

    # Pace & PPP
    result.expected_pace        = compute_expected_pace(home.pace, away.pace)
    result.home_ppp_projected   = project_ppp(home.ppp, away.net_rating)
    result.away_ppp_projected   = project_ppp(away.ppp, home.net_rating)

    # Scores bruts
    raw_home = expected_score_from_ppp(result.home_ppp_projected, result.expected_pace)
    raw_away = expected_score_from_ppp(result.away_ppp_projected, result.expected_pace)

    # Contexte
    result.context = compute_context(home, away)
    result.home_expected_score, result.away_expected_score = \
        apply_context_to_scores(raw_home, raw_away, result.context)

    # Écart-type
    pace_factor   = result.expected_pace / 100.0
    result.home_std = round(10.0 * pace_factor, 2)
    result.away_std = round(10.0 * pace_factor, 2)

    # Confiance
    elo_gap    = abs(home.elo_rating - away.elo_rating)
    ff_gap     = abs(result.four_factors_edge)
    score_gap  = abs(result.home_expected_score - result.away_expected_score)
    raw_conf   = (
        0.40 * min(elo_gap / 200.0,  1.0)
      + 0.30 * min(ff_gap / 0.10,   1.0)
      + 0.30 * min(score_gap / 15.0, 1.0)
    )
    result.confidence = round(0.50 + raw_conf * 0.40, 3)

    return result
