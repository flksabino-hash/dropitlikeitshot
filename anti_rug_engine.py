# ===================================================================
# APEX ANTI-RUG ML ENGINE v2026.3 (Port 8003)
# Hardened: deterministic synthetic training, model persistence, health endpoint
# ===================================================================

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier

from apex_common.logging import get_logger

load_dotenv()
log = get_logger("anti_rug")

app = FastAPI(title="Apex Anti-Rug ML Engine", version="2026.3")

DATA_DIR = Path(os.getenv("ANTI_RUG_DATA_DIR", ".")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = DATA_DIR / "anti_rug_model.pkl"

def train_model(model_path: Path) -> RandomForestClassifier:
    log.info("🧠 [ML] Training synthetic Anti-Rug model...")
    np.random.seed(42)

    # Rug-like: low liquidity, high concentration, dev spams tx, very new, moderate volume
    rugs_X = np.column_stack([
        np.random.uniform(500, 15000, 800),     # liquidity_usd
        np.random.uniform(35, 97, 800),         # top_holder_pct
        np.random.randint(10, 200, 800),        # dev_wallet_tx_count
        np.random.uniform(0.1, 12, 800),        # age_hours
        np.random.uniform(1000, 80000, 800)     # volume_24h
    ])
    rugs_y = np.ones(800)

    # Non-rug: higher liquidity, lower concentration, quieter dev, older, higher volume
    succ_X = np.column_stack([
        np.random.uniform(50000, 800000, 800),
        np.random.uniform(3, 30, 800),
        np.random.randint(0, 10, 800),
        np.random.uniform(24, 2400, 800),
        np.random.uniform(100000, 15000000, 800)
    ])
    succ_y = np.zeros(800)

    X = np.vstack([rugs_X, succ_X])
    y = np.concatenate([rugs_y, succ_y])

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=7,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    joblib.dump(clf, model_path)
    log.info(f"✅ [ML] Model saved: {model_path}")
    return clf

if MODEL_FILE.exists():
    predator_model: RandomForestClassifier = joblib.load(MODEL_FILE)
    log.info(f"[ML] Loaded model: {MODEL_FILE}")
else:
    predator_model = train_model(MODEL_FILE)

class TokenMetrics(BaseModel):
    liquidity_usd: float = Field(..., ge=0)
    top_holder_pct: float = Field(..., ge=0, le=100)
    dev_wallet_tx_count: int = Field(..., ge=0)
    age_hours: float = Field(..., ge=0)
    volume_24h: float = Field(..., ge=0)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "anti_rug", "version": app.version, "model": str(MODEL_FILE)}

@app.post("/analyze_token")
async def analyze_token(metrics: TokenMetrics):
    features = np.array([[
        metrics.liquidity_usd,
        metrics.top_holder_pct,
        metrics.dev_wallet_tx_count,
        metrics.age_hours,
        metrics.volume_24h
    ]], dtype=float)

    rug_prob = float(predator_model.predict_proba(features)[0][1])
    status = "REJEITADO" if rug_prob > 0.40 else "APROVADO"

    return {
        "status": status,
        "rug_probability_pct": round(rug_prob * 100, 2),
        "edge_directive": "Risco de honeypot/rug detectado." if status == "REJEITADO" else "Estrutura on-chain limpa."
    }
