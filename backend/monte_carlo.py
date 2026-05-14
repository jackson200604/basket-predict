# ============================================================
# monte_carlo.py — 20 000 simulations Monte Carlo
# ============================================================
import numpy as np
from typing import List
from models import MonteCarloResult, ScoreScenario, BetRecommendation
from analytics_engine import EngineResult
from config import MONTE_CARLO_SIMS


# =============================================================
# 1. SIMULATION PRINCIPALE
# =============================================================

def run_monte_carlo(
    engine: EngineResult,
    n_sims: int = MONTE_CARLO_SIMS,
    seed:   int = 42,
) -> MonteCarloResult:

    rng = np.random.default_rng(seed)

    μ_home = engine.home_expected_score
    μ_away = engine.away_expected_score
    σ_home = engine.home_std
    σ_away = engine.away_std

    # Scores de base
    scores_home = rng.normal(μ_home, σ_home, n_sims)
    scores_away = rng.normal(μ_away, σ_away, n_sims)

    # Variance adresse au tir
    scores_home += rng.normal(0, μ_home * 0.05, n_sims)
    scores_away += rng.normal(0, μ_away * 0.05, n_sims)

    # Variance fautes
    scores_home += rng.normal(0, 2.0, n_sims)
    scores_away += rng.normal(0, 2.0, n_sims)

    # Bruit Elo
    elo_noise    = rng.normal(0, 3.0, n_sims)
    scores_home += elo_noise
    scores_away -= elo_noise

    # Plancher réaliste
    scores_home = np.maximum(scores_home, 70.0)
    scores_away = np.maximum(scores_away, 70.0)

    home_wins = int(np.sum(scores_home > scores_away))
    away_wins = n_sims - home_wins

    return MonteCarloResult(
        home_win_pct   = round(home_wins / n_sims, 4),
        away_win_pct   = round(away_wins / n_sims, 4),
        avg_score_home = round(float(np.mean(scores_home)), 1),
        avg_score_away = round(float(np.mean(scores_away)), 1),
        std_home       = round(float(np.std(scores_home)), 2),
        std_away       = round(float(np.std(scores_away)), 2),
        simulations    = n_sims,
    )


# =============================================================
# 2. SCÉNARIOS DE SCORE
# =============================================================

def build_score_scenarios(
    mc:     MonteCarloResult,
    engine: EngineResult,
) -> List[ScoreScenario]:

    # Principal
    main = ScoreScenario(
        label      = "⚡ Principal (Baseline)",
        score_home = int(round(mc.avg_score_home)),
        score_away = int(round(mc.avg_score_away)),
        confidence = round(engine.confidence, 3),
    )

    # Défensif Low Pace
    defensive = ScoreScenario(
        label      = "🛡️ Défensif (Low Pace)",
        score_home = int(round(mc.avg_score_home * 0.92 * 0.96)),
        score_away = int(round(mc.avg_score_away * 0.92 * 0.96)),
        confidence = round(engine.confidence * 0.80, 3),
    )

    # Offensif High Pace
    offensive = ScoreScenario(
        label      = "🔥 Offensif (High Pace)",
        score_home = int(round(mc.avg_score_home * 1.08 * 1.03)),
        score_away = int(round(mc.avg_score_away * 1.08 * 1.03)),
        confidence = round(engine.confidence * 0.75, 3),
    )

    return [main, defensive, offensive]


# =============================================================
# 3. BLEND PROBABILITÉS
# =============================================================

def blend_win_probability(
    mc:     MonteCarloResult,
    engine: EngineResult,
) -> tuple[float, float]:

    ff_prob_home = 0.50 + engine.four_factors_edge * 1.5
    ff_prob_home = max(0.05, min(0.95, ff_prob_home))

    p_home = (
        0.50 * mc.home_win_pct
      + 0.30 * engine.elo_win_prob_home
      + 0.20 * ff_prob_home
    )
    return round(p_home, 4), round(1.0 - p_home, 4)


# =============================================================
# 4. VALUE BET — sans scipy (numpy pur)
# =============================================================

def _normal_cdf(x: float) -> float:
    """
    Approximation CDF normale standard via numpy.
    Remplace scipy.stats.norm.cdf()
    """
    return float(0.5 * (1.0 + np.sign(x) * np.sqrt(
        1.0 - np.exp(-2.0 / np.pi * x ** 2)
    )))


def compute_implied_probability(american_odds: float) -> float:
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    return 100 / (american_odds + 100)


def detect_value_bet(
    prob_home_ai: float,
    prob_away_ai: float,
    mc:           MonteCarloResult,
    engine:       EngineResult,
    odds_home:    float = -110,
    odds_away:    float = -110,
    team_home:    str   = "Domicile",
    team_away:    str   = "Extérieur",
) -> List[BetRecommendation]:

    impl_home = compute_implied_probability(odds_home)
    impl_away = compute_implied_probability(odds_away)
    edge_home = round(prob_home_ai - impl_home, 4)
    edge_away = round(prob_away_ai - impl_away, 4)

    bets: List[BetRecommendation] = []

    # Principal Moneyline
    if prob_home_ai >= prob_away_ai:
        bets.append(BetRecommendation(
            label        = "🏆 Principal",
            bet_type     = "Moneyline",
            description  = f"Victoire {team_home}",
            probability  = prob_home_ai,
            edge         = edge_home if edge_home > 0 else None,
            is_value_bet = edge_home > 0.03,
        ))
    else:
        bets.append(BetRecommendation(
            label        = "🏆 Principal",
            bet_type     = "Moneyline",
            description  = f"Victoire {team_away}",
            probability  = prob_away_ai,
            edge         = edge_away if edge_away > 0 else None,
            is_value_bet = edge_away > 0.03,
        ))

    # Spread
    score_diff  = mc.avg_score_home - mc.avg_score_away
    spread      = round(score_diff * 0.90, 1)
    spread_desc = (
        f"{team_home} -{abs(spread):.1f}"
        if spread > 0
        else f"{team_away} -{abs(spread):.1f}"
    )
    z           = (0 - (score_diff - spread)) / max(engine.home_std, 1.0)
    spread_prob = round(max(0.35, min(0.75, _normal_cdf(z))), 3)

    bets.append(BetRecommendation(
        label        = "🛡️ Prudente 1 (Spread)",
        bet_type     = "Spread",
        description  = spread_desc,
        probability  = spread_prob,
        is_value_bet = False,
    ))

    # Total Over/Under
    total_proj = mc.avg_score_home + mc.avg_score_away
    ou_line    = round(total_proj * 0.98, 1)

    bets.append(BetRecommendation(
        label        = "📊 Prudente 2 (Total)",
        bet_type     = "Total",
        description  = (
            f"Over {ou_line:.0f} pts"
            if total_proj > ou_line
            else f"Under {ou_line:.0f} pts"
        ),
        probability  = round(0.52 if total_proj > ou_line else 0.48, 3),
        is_value_bet = False,
    ))

    # Value Bet
    best_edge = max(edge_home, edge_away)
    if best_edge > 0.03:
        value_team = team_home if edge_home > edge_away else team_away
        value_prob = prob_home_ai if edge_home > edge_away else prob_away_ai
        bets.append(BetRecommendation(
            label        = "💎 VALUE BET",
            bet_type     = "Moneyline",
            description  = f"EDGE +{best_edge*100:.1f}% — {value_team} sous-évalué",
            probability  = round(value_prob, 3),
            edge         = round(best_edge, 4),
            is_value_bet = True,
        ))

    return bets


# =============================================================
# 5. ORCHESTRATEUR
# =============================================================

def run_full_simulation(
    engine:    EngineResult,
    team_home: str   = "Domicile",
    team_away: str   = "Extérieur",
    odds_home: float = -110,
    odds_away: float = -110,
    n_sims:    int   = MONTE_CARLO_SIMS,
) -> dict:

    mc                   = run_monte_carlo(engine, n_sims=n_sims)
    prob_home, prob_away = blend_win_probability(mc, engine)
    mc.home_win_pct      = prob_home
    mc.away_win_pct      = prob_away
    scenarios            = build_score_scenarios(mc, engine)
    bets                 = detect_value_bet(
        prob_home, prob_away, mc, engine,
        odds_home, odds_away, team_home, team_away,
    )

    return {
        "monte_carlo": mc,
        "scenarios":   scenarios,
        "bets":        bets,
        "prob_home":   prob_home,
        "prob_away":   prob_away,
    }

