# ============================================================
# lstm_model.py — Modèle LSTM sur séries temporelles de forme
# ============================================================
import os
import numpy as np
from typing import List, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

SEQ_LEN   = 10
N_FEAT    = 2
LSTM_PATH = os.path.join(os.path.dirname(__file__), "lstm_model.keras")


def _pad_sequence(form: List[float], seq_len: int = SEQ_LEN) -> np.ndarray:
    arr = np.clip(np.array(form, dtype=np.float32) / 25.0, -1.0, 1.0)
    if len(arr) >= seq_len:
        return arr[-seq_len:]
    return np.concatenate([np.zeros(seq_len - len(arr), dtype=np.float32), arr])


def build_lstm_input(form_home: List[float], form_away: List[float]) -> np.ndarray:
    seq = np.stack([_pad_sequence(form_home), _pad_sequence(form_away)], axis=1)
    return seq[np.newaxis, ...].astype(np.float32)


def _generate_lstm_data(n: int = 4000, seed: int = 7) -> tuple:
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    for _ in range(n):
        base_h = rng.normal(0, 8)
        base_a = rng.normal(0, 8)
        form_h = rng.normal(base_h, 5, SEQ_LEN).astype(np.float32)
        form_a = rng.normal(base_a, 5, SEQ_LEN).astype(np.float32)
        seq_h  = np.clip(form_h / 25.0, -1.0, 1.0)
        seq_a  = np.clip(form_a / 25.0, -1.0, 1.0)
        X      = np.stack([seq_h, seq_a], axis=1)
        logit  = (
            float(np.mean(form_h[-3:])) * 0.04
          - float(np.mean(form_a[-3:])) * 0.04
          + 0.15 + rng.normal(0, 0.1)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        X_list.append(X)
        y_list.append(int(rng.uniform() < prob))
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
    )


def _build_model() -> "keras.Model":
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, N_FEAT)),
        keras.layers.LSTM(64, return_sequences=True,  dropout=0.20, recurrent_dropout=0.10),
        keras.layers.LSTM(32, return_sequences=False, dropout=0.15, recurrent_dropout=0.10),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.10),
        keras.layers.Dense(1,  activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_save_lstm(out_path: str = LSTM_PATH) -> Optional["keras.Model"]:
    if not TF_AVAILABLE:
        print("[LSTM] TensorFlow non disponible — entraînement ignoré.")
        return None
    print("[LSTM] Génération données synthétiques…")
    X, y  = _generate_lstm_data(n=4000)
    split = int(len(X) * 0.85)
    model = _build_model()
    model.fit(
        X[:split], y[:split],
        validation_data=(X[split:], y[split:]),
        epochs=80, batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ],
        verbose=0,
    )
    val_loss, val_acc = model.evaluate(X[split:], y[split:], verbose=0)
    print(f"[LSTM] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
    model.save(out_path)
    print(f"[LSTM] Modèle sauvegardé → {out_path}")
    return model


_cache: dict = {}


def _load_lstm() -> Optional["keras.Model"]:
    if not TF_AVAILABLE:
        return None
    if "model" not in _cache:
        if os.path.exists(LSTM_PATH):
            print(f"[LSTM] Chargement depuis {LSTM_PATH}")
            _cache["model"] = keras.models.load_model(LSTM_PATH)
        else:
            print("[LSTM] Aucun modèle trouvé — entraînement synthétique…")
            _cache["model"] = train_and_save_lstm()
    return _cache.get("model")


def load_lstm_at_startup():
    """Appelé dans startup_event() de main.py via run_in_executor."""
    _load_lstm()


def predict_lstm(form_home: List[float], form_away: List[float]) -> dict:
    model = _load_lstm()
    if model is None:
        return _fallback_form(form_home, form_away)
    X    = build_lstm_input(form_home, form_away)
    raw  = float(model.predict(X, verbose=0)[0][0])
    prob = 0.5 + (max(0.05, min(0.95, raw)) - 0.5) * 0.85
    return {
        "lstm_prob_home":  round(prob, 4),
        "lstm_available":  True,
        "lstm_confidence": round(abs(prob - 0.5) * 2, 4),
    }


def _fallback_form(form_home: List[float], form_away: List[float]) -> dict:
    def _wmean(form: List[float], n: int = 5) -> float:
        if not form:
            return 0.0
        recent  = form[-n:]
        weights = np.linspace(0.5, 1.0, len(recent))
        return float(np.dot(weights / weights.sum(), recent))
    logit = (_wmean(form_home) - _wmean(form_away)) / 20.0 + 0.15
    prob  = max(0.10, min(0.90, 1.0 / (1.0 + np.exp(-logit))))
    return {
        "lstm_prob_home":  round(prob, 4),
        "lstm_available":  False,
        "lstm_confidence": round(abs(prob - 0.5) * 2, 4),
    }