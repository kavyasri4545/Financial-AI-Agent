"""
prediction_model.py  –  Fast, cached price prediction.

Strategy
--------
* A lightweight sklearn Ridge regression (+ polynomial features) trains in
  milliseconds and is used as the *default fast path*.
* The heavy LSTM+Attention model is trained **once per symbol**, cached in
  memory, and re-used on subsequent calls within the same session.
* If TensorFlow is unavailable the system falls back to the Ridge model
  transparently.
"""

import hashlib
import logging
import threading
import time
from functools import lru_cache

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

# ─── In-memory model cache ────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}          # symbol -> (timestamp, model, scaler)
_MODEL_LOCK  = threading.Lock()
MODEL_TTL    = 3600              # seconds – retrain after 1 hour

# ─── Try importing TensorFlow lazily ─────────────────────────────────────────
_TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Layer
    from tensorflow.keras import backend as K
    tf.get_logger().setLevel("ERROR")
    _TF_AVAILABLE = True
except Exception:
    logger.info("TensorFlow not available – using fast Ridge model.")


# ─── Attention layer (only defined when TF is present) ───────────────────────
if _TF_AVAILABLE:
    class AttentionLayer(Layer):
        def build(self, input_shape):
            self.W = self.add_weight("aw", shape=(input_shape[-1], 1),
                                     initializer="random_normal")
            self.b = self.add_weight("ab", shape=(input_shape[1], 1),
                                     initializer="zeros")
            super().build(input_shape)

        def call(self, x):
            score   = K.tanh(K.dot(x, self.W) + self.b)
            weights = K.softmax(score, axis=1)
            return K.sum(x * weights, axis=1)


# ─── Fast Ridge predictor ─────────────────────────────────────────────────────
def _predict_ridge(close_values: np.ndarray) -> float:
    """Train a tiny Ridge regression on the last N days and predict next close."""
    window = min(60, len(close_values))
    data   = close_values[-window:].reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data).flatten()

    X = np.array([scaled[i - 20: i] for i in range(20, len(scaled))])
    y = scaled[20:]

    if len(X) < 5:
        return float(close_values[-1])

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(X, y)

    last_seq = scaled[-20:].reshape(1, -1)
    pred_scaled = model.predict(last_seq)[0]
    pred = scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred)


# ─── Heavy LSTM predictor (cached) ───────────────────────────────────────────
def _build_lstm_model(seq_len: int) -> "Model":
    inputs  = Input(shape=(seq_len, 1))
    x       = LSTM(64, return_sequences=True)(inputs)
    x       = LSTM(32, return_sequences=True)(x)
    x       = AttentionLayer()(x)
    outputs = Dense(1)(x)
    model   = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _predict_lstm(symbol: str, close_values: np.ndarray) -> float:
    SEQ = 60
    if len(close_values) < SEQ + 10:
        return _predict_ridge(close_values)

    data   = close_values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X = np.array([scaled[i - SEQ: i] for i in range(SEQ, len(scaled))])
    y = scaled[SEQ:]

    now     = time.time()
    cache_key = symbol

    with _MODEL_LOCK:
        entry = _MODEL_CACHE.get(cache_key)

    if entry and (now - entry[0]) < MODEL_TTL:
        model = entry[1]
        logger.debug("LSTM cache hit for %s", symbol)
    else:
        logger.debug("Training LSTM for %s (%d samples) …", symbol, len(X))
        model = _build_lstm_model(SEQ)
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        with _MODEL_LOCK:
            _MODEL_CACHE[cache_key] = (now, model)

    last_seq = scaled[-SEQ:].reshape(1, SEQ, 1)
    pred     = scaler.inverse_transform(model.predict(last_seq, verbose=0))[0][0]
    return float(pred)


# ─── Public API ──────────────────────────────────────────────────────────────
def predict_price(df, symbol: str = "__unknown__", use_lstm: bool = False) -> float:
    """
    Predict the next-day closing price.

    Parameters
    ----------
    df        : DataFrame with at least a "Close" column.
    symbol    : Ticker symbol – used to key the model cache.
    use_lstm  : Force LSTM even on the first call (slower but more accurate).

    Returns
    -------
    float : predicted price
    """
    close = df["Close"].values.astype(float).flatten()

    if not _TF_AVAILABLE or not use_lstm:
        return _predict_ridge(close)

    try:
        return _predict_lstm(symbol, close)
    except Exception as e:
        logger.warning("LSTM failed (%s), falling back to Ridge: %s", symbol, e)
        return _predict_ridge(close)


def warm_cache(symbol: str, df) -> None:
    """
    Pre-train and cache the LSTM model in a background thread.
    Call this right after loading data so the model is ready when the user clicks.
    """
    if not _TF_AVAILABLE:
        return

    def _train():
        try:
            close = df["Close"].values.astype(float).flatten()
            _predict_lstm(symbol, close)
            logger.debug("LSTM warm-cache done for %s", symbol)
        except Exception as e:
            logger.debug("Warm-cache failed for %s: %s", symbol, e)

    threading.Thread(target=_train, daemon=True).start()
