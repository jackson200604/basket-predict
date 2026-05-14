# ============================================================
# ml_model.py — Modèle XGBoost de prédiction de match
# ============================================================
"""
Architecture :
  - Features : 14 variables issues de EngineResult + form scores
  - Cible    : home_win (0/1)
  - Fallback : si aucun modèle .joblib n'est trouvé, on entraîne
               sur 5 000 matchs synthétiques cohérents avec les
               distributions NBA réelles.

Pour entraîner sur de vraies données :
  1. Construire un CSV avec les colonnes FEATURE_COLS + "home_win"
  2. Appeler train_and_save("path/to/data.csv", "model.joblib")
  3. Redéployer avec model.joblib dans le dossier backend/
"""
import os
import numpy as np
import joblib
from dataclasses import dataclass
from typing import Optional

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from analytics_engine import EngineResult

# ── Constantes ────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.joblib")
FEATURE_COLS = [
    "elo_diff",          # elo domicile - elo extérieur
    "elo_home_boost",    # avantage domicile fixe (100 pts)
    "ff_edge",           # Four Factors edge
    "ppp_diff",          # PPP projeté home - away
    "pace",              # rythme attendu
    "score_diff_raw",    # score attendu home - away (avant contexte)
    "rest_advantage",    # jours de repos home - away
    "travel_penalty",    # back-to-back penalty
    "home_court",        # avantage terrain (pts)
    "pace_mismatch",     # écart de pace
    "form_home",         # score de forme pondéré domicile
    "form_away",         # score de forme pondéré extérieur
    "form_diff",         # form_home - form_away
    "confidence_engine", # confiance moteur analytique
]


# =============================================================
# 1. EXTRACTION DES FEATURES
# =============================================================

def extract_features(
    engine: EngineResult,
    form_home: float,
    form_away: float,
) -> np.ndarray:
    """Transforme un EngineResult en vecteur de features (1 × 14)."""
    ctx = engine.context
    x = np.array([
        engine.home_elo - engine.away_elo,
        100.0,                                          # ELO_HOME_BOOST constant
        engine.four_factors_edge,
        engine.home_ppp_projected - engine.away_ppp_projected,
        engine.expected_pace,
        engine.home_expected_score - engine.away_expected_score,
        ctx.rest_advantage,
        ctx.travel_penalty,
        ctx.home_court,
        ctx.pace_mismatch,
        form_home,
        form_away,
        form_home - form_away,
        engine.confidence,
    ], dtype=np.float32).reshape(1, -1)
    return x


# =============================================================
# 2. DONNÉES SYNTHÉTIQUES
# =============================================================

def _generate_synthetic_data(n: int = 5000, seed: int = 42) -> tuple:
    """
    Génère n matchs synthétiques cohérents avec les distributions NBA.
    La probabilité de victoire domicile suit un modèle logistique sur
    elo_diff + ff_edge + form_diff, ce qui garantit que XGBoost apprend
    des relations réalistes même sans données réelles.
    """
    rng = np.random.default_rng(seed)

    elo_diff      = rng.normal(0,    150,  n).astype(np.float32)
    elo_boost     = np.full(n, 100.0,       dtype=np.float32)
    ff_edge       = rng.normal(0,    0.04, n).astype(np.float32)
    ppp_diff      = rng.normal(0,    0.05, n).astype(np.float32)
    pace          = rng.normal(100,  4,    n).astype(np.float32)
    score_diff    = rng.normal(0,    8,    n).astype(np.float32)
    rest_adv      = rng.choice([-2,-1,0,1,2], n).astype(np.float32)
    travel_pen    = rng.choice([-3.5,0,3.5],  n).astype(np.float32)
    home_court    = np.full(n, 3.5,            dtype=np.float32)
    pace_mismatch = np.abs(rng.normal(0, 3, n)).astype(np.float32)
    form_home     = rng.normal(2,  5, n).astype(np.float32)
    form_away     = rng.normal(-2, 5, n).astype(np.float32)
    form_diff     = (form_home - form_away).astype(np.float32)
    confidence    = rng.uniform(0.50, 0.85, n).astype(np.float32)

    X = np.stack([
        elo_diff, elo_boost, ff_edge, ppp_diff, pace,
        score_diff, rest_adv, travel_pen, home_court,
        pace_mismatch, form_home, form_away, form_diff, confidence,
    ], axis=1)

    # Logit réaliste : domicile gagne ~60 % sur données neutres
    logit = (
        0.003 * elo_diff
      + 0.002 * elo_boost
      + 4.0   * ff_edge
      + 6.0   * ppp_diff
      + 0.05  * score_diff
      + 0.3   * rest_adv
      + 0.15  * travel_pen
      + 0.08  * home_court
      + 0.04  * form_diff
      + rng.normal(0, 0.3, n)   # bruit irréductible
    )
    prob_home = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob_home).astype(np.int32)

    return X, y


# =============================================================
# 3. ENTRAÎNEMENT & SAUVEGARDE
# =============================================================

def train_and_save(csv_path: Optional[str] = None, out_path: str = MODEL_PATH) -> "xgb.XGBClassifier":
    """
    Entraîne XGBoost.
    - Si csv_path fourni : utilise les vraies données (colonnes = FEATURE_COLS + 'home_win')
    - Sinon             : données synthétiques (fallback)
    Sauvegarde le modèle dans out_path.
    """
    if not XGB_AVAILABLE:
        raise RuntimeError("xgboost non installé")

    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        df  = pd.read_csv(csv_path)
        X   = df[FEATURE_COLS].values.astype(np.float32)
        y   = df["home_win"].values.astype(np.int32)
        tag = "xgboost_trained"
    else:
        X, y = _generate_synthetic_data()
        tag  = "xgboost_synthetic"

    model = xgb.XGBClassifier(
        n_estimators      = 400,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.80,
        colsample_bytree  = 0.80,
        use_label_encoder = False,
        eval_metric       = "logloss",
        random_state      = 42,
        n_jobs            = -1,
    )
    model.fit(X, y)
    joblib.dump({"model": model, "tag": tag}, out_path)
    print(f"[ML] Modèle sauvegardé → {out_path} ({tag})")
    return model


# =============================================================
# 4. CHARGEMENT (lazy, une seule fois au démarrage)
# =============================================================

_cache: dict = {}

def _load_model() -> tuple["xgb.XGBClassifier", str]:
    if not XGB_AVAILABLE:
        return None, "unavailable"
    if "model" not in _cache:
        if os.path.exists(MODEL_PATH):
            payload       = joblib.load(MODEL_PATH)
            _cache["model"] = payload["model"]
            _cache["tag"]   = payload["tag"]
            print(f"[ML] Modèle chargé depuis {MODEL_PATH} ({_cache['tag']})")
        else:
            print("[ML] Aucun modèle trouvé — entraînement synthétique…")
            train_and_save()
            payload         = joblib.load(MODEL_PATH)
            _cache["model"] = payload["model"]
            _cache["tag"]   = payload["tag"]
    return _cache["model"], _cache["tag"]


# =============================================================
# 5. INFÉRENCE PRINCIPALE
# =============================================================

def predict_ml(
    engine:    EngineResult,
    form_home: float,
    form_away: float,
) -> dict:
    """
    Retourne :
      home_win_prob  : float [0, 1]
      away_win_prob  : float [0, 1]
      model_used     : str
      top_features   : list[str]  (3 features les plus importantes)
    """
    model, tag = _load_model()

    if model is None:
        # XGBoost non disponible : fallback neutre
        return {
            "home_win_prob": round(engine.elo_win_prob_home, 4),
            "away_win_prob": round(engine.elo_win_prob_away, 4),
            "model_used":    "fallback_elo",
            "top_features":  [],
        }

    X          = extract_features(engine, form_home, form_away)
    prob_home  = float(model.predict_proba(X)[0][1])
    prob_home  = max(0.05, min(0.95, prob_home))

    # Top 3 features par importance
    importances  = model.feature_importances_
    top_idx      = np.argsort(importances)[::-1][:3]
    top_features = [FEATURE_COLS[i] for i in top_idx]

    return {
        "home_win_prob": round(prob_home, 4),
        "away_win_prob": round(1.0 - prob_home, 4),
        "model_used":    tag,
        "top_features":  top_features,
    }
