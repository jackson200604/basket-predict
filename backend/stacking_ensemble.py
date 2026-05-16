# ============================================================
# stacking_ensemble.py — Méta-modèle Stacking LSTM + XGBoost
# ============================================================
import numpy as np
from typing import Tuple

LSTM_WEIGHT   = 0.60
XGB_WEIGHT    = 0.40
LSTM_MIN_CONF = 0.15


def _adaptive_weights(lstm_conf: float, xgb_conf: float) -> Tuple[float, float]:
    w_lstm = LSTM_WEIGHT * (0.4 + 0.6 * max(lstm_conf, LSTM_MIN_CONF))
    w_xgb  = XGB_WEIGHT  * (0.4 + 0.6 * xgb_conf)
    total  = w_lstm + w_xgb
    return w_lstm / total, w_xgb / total


def _stack(p_xgb: float, p_lstm: float, lstm_conf: float, xgb_conf: float, lstm_ok: bool) -> dict:
    if not lstm_ok:
        prob = 0.5 + (p_xgb - 0.5) * 0.90
        return {
            "stacked_prob_home": round(max(0.05, min(0.95, prob)), 4),
            "w_lstm": 0.0, "w_xgb": 1.0,
            "p_xgb": round(p_xgb, 4), "p_lstm": None,
            "agreement": None, "divergence": None,
            "mode": "xgb_only",
        }
    w_lstm, w_xgb = _adaptive_weights(lstm_conf, xgb_conf)
    blended = w_lstm * p_lstm + w_xgb * p_xgb
    prob    = 0.5 + (blended - 0.5) * 0.92
    prob    = max(0.05, min(0.95, prob))
    return {
        "stacked_prob_home": round(prob, 4),
        "w_lstm":     round(w_lstm, 3),
        "w_xgb":      round(w_xgb, 3),
        "p_xgb":      round(p_xgb, 4),
        "p_lstm":     round(p_lstm, 4),
        "agreement":  round(1.0 - abs(p_lstm - p_xgb), 3),
        "divergence": round(abs(p_lstm - p_xgb), 3),
        "mode": "lstm_xgb_stack",
    }


def run_stacking(ml_output: dict, lstm_output: dict) -> dict:
    p_xgb     = float(ml_output.get("home_win_prob", 0.50))
    p_lstm    = float(lstm_output.get("lstm_prob_home", 0.50))
    lstm_ok   = bool(lstm_output.get("lstm_available", False))
    lstm_conf = float(lstm_output.get("lstm_confidence", 0.0))
    xgb_conf  = min(1.0, abs(p_xgb - 0.5) * 2)
    detail    = _stack(p_xgb, p_lstm, lstm_conf, xgb_conf, lstm_ok)
    prob_home = detail["stacked_prob_home"]
    return {
        "stacked_prob_home": prob_home,
        "stacked_prob_away": round(1.0 - prob_home, 4),
        "stacking_detail":   detail,
        "model_used":        "stacking_lstm_xgb" if lstm_ok else "stacking_xgb_only",
        "top_features":      ml_output.get("top_features", []),
    }


def stacking_log(result: dict) -> str:
    d    = result.get("stacking_detail", {})
    mode = d.get("mode", "?")
    if mode == "xgb_only":
        return f"[STACK] Mode: XGB seul (LSTM absent) | P(home)={result['stacked_prob_home']:.1%}"
    warn = " ⚠️ désaccord élevé" if (d.get("divergence") or 0) > 0.15 else ""
    return (
        f"[STACK] XGB={d['p_xgb']:.1%}(w={d['w_xgb']:.0%}) "
        f"LSTM={d['p_lstm']:.1%}(w={d['w_lstm']:.0%}) "
        f"→ {result['stacked_prob_home']:.1%} | accord={d['agreement']:.0%}{warn}"
    )