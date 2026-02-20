# ===================================================================
# APEX CCXT ASYNC EXECUTIONER NODE v2026.3 (Port 8002)
# Hardened: dynamic equity sizing, bracket SL/TP, rollback on partial failure
# ===================================================================

from __future__ import annotations

import os
from typing import Literal, Optional

import ccxt.async_support as ccxt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apex_common.config import ExecutionerConfig
from apex_common.logging import get_logger

load_dotenv()
log = get_logger("executioner")
cfg = ExecutionerConfig()

app = FastAPI(title="Apex CCXT Executioner", version="2026.3")

Side = Literal["buy", "sell"]
Venue = Literal["binance", "bybit", "okx"]

class StrikePayload(BaseModel):
    symbol: str = Field(..., description="Unified symbol per ccxt, ex: BTC/USDT:USDT")
    side: Side
    venue: Venue

    # sizing: choose ONE
    size_usd: Optional[float] = Field(None, gt=0, description="Notional size in USD")
    risk_pct: Optional[float] = Field(None, gt=0, le=0.05, description="Fraction of free USDT equity to allocate")

    # protection
    sl_pct: float = Field(default=0.015, gt=0, lt=0.25)
    tp_pct: float = Field(default=0.045, gt=0, lt=1.00)

    # safety
    reduce_only_brackets: bool = True

def _get_exchange(venue: str):
    venue_l = venue.lower()

    if venue_l == "binance":
        ex_cls = ccxt.binance
        opts = {"defaultType": "swap"}
        params = {}
    elif venue_l == "bybit":
        ex_cls = ccxt.bybit
        opts = {"defaultType": "swap"}
        params = {}
    elif venue_l == "okx":
        ex_cls = ccxt.okx
        opts = {"defaultType": "swap"}
        params = {"password": os.getenv("OKX_PASSWORD", "")}
    else:
        raise ValueError("venue inválida")

    api_key = os.getenv(f"{venue_l.upper()}_API_KEY", "")
    secret = os.getenv(f"{venue_l.upper()}_API_SECRET", "")

    exchange = ex_cls({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "options": opts,
        **params
    })

    if cfg.use_testnet:
        try:
            exchange.set_sandbox_mode(True)
        except Exception:
            pass
    return exchange

async def _fetch_free_usdt(exchange) -> float:
    balance = await exchange.fetch_balance()
    # ccxt differences: sometimes 'USDT' in balance, sometimes 'total' etc.
    if "USDT" in balance:
        usdt = balance["USDT"]
        if isinstance(usdt, dict):
            return float(usdt.get("free", 0.0) or 0.0)
    # fallback: try 'free' dict
    free = balance.get("free", {})
    if isinstance(free, dict) and "USDT" in free:
        return float(free.get("USDT", 0.0) or 0.0)
    return 0.0

def _validate_payload(p: StrikePayload):
    if (p.size_usd is None) == (p.risk_pct is None):
        raise HTTPException(status_code=400, detail="Envie exatamente um: size_usd OU risk_pct.")
    if p.size_usd is not None and p.size_usd > cfg.max_notional_usd:
        raise HTTPException(status_code=400, detail=f"size_usd acima do limite de segurança ({cfg.max_notional_usd}).")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "executioner", "version": app.version, "testnet": cfg.use_testnet}

@app.get("/get_equity/{venue}")
async def get_equity(venue: Venue):
    exchange = None
    try:
        exchange = _get_exchange(venue)
        free_usdt = await _fetch_free_usdt(exchange)
        return {"venue": venue.upper(), "free_usdt": free_usdt, "status": "SUCCESS", "testnet": cfg.use_testnet}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if exchange:
            await exchange.close()

@app.post("/execute_strike")
async def execute_strike(payload: StrikePayload):
    _validate_payload(payload)
    exchange = None
    entry_order = None
    sl_order = None
    tp_order = None

    try:
        exchange = _get_exchange(payload.venue)
        await exchange.load_markets()

        # Resolve notional sizing
        if payload.risk_pct is not None:
            free_usdt = await _fetch_free_usdt(exchange)
            size_usd = float(free_usdt) * float(payload.risk_pct)
        else:
            size_usd = float(payload.size_usd)

        if size_usd <= 0:
            raise HTTPException(status_code=400, detail="Equity insuficiente para sizing.")

        if size_usd > cfg.max_notional_usd:
            raise HTTPException(status_code=400, detail=f"Tamanho calculado excede limite ({cfg.max_notional_usd}).")

        # Fetch price
        ticker = await exchange.fetch_ticker(payload.symbol)
        current_price = float(ticker.get("last") or ticker.get("mark") or ticker.get("close") or 0.0)
        if current_price <= 0:
            raise HTTPException(status_code=502, detail="Não foi possível obter preço atual.")

        raw_amount = size_usd / current_price
        amount = float(exchange.amount_to_precision(payload.symbol, raw_amount))

        # Entry
        entry_order = await exchange.create_order(payload.symbol, "market", payload.side, amount)
        fill_price = float(entry_order.get("average") or entry_order.get("price") or current_price)

        # Brackets: unify stop/tp sides
        close_side = "sell" if payload.side == "buy" else "buy"

        if payload.side == "buy":
            sl_price = fill_price * (1 - payload.sl_pct)
            tp_price = fill_price * (1 + payload.tp_pct)
        else:
            sl_price = fill_price * (1 + payload.sl_pct)
            tp_price = fill_price * (1 - payload.tp_pct)

        sl_price = float(exchange.price_to_precision(payload.symbol, sl_price))
        tp_price = float(exchange.price_to_precision(payload.symbol, tp_price))

        params = {}
        if payload.reduce_only_brackets:
            params["reduceOnly"] = True

        # Create SL/TP (best-effort across venues). If any fails, rollback.
        # ccxt unified types:
        # - 'stop_market' with 'stopPrice'
        # - 'take_profit_market' with 'stopPrice'
        sl_order = await exchange.create_order(
            payload.symbol, "stop_market", close_side, amount, None, {"stopPrice": sl_price, **params}
        )
        tp_order = await exchange.create_order(
            payload.symbol, "take_profit_market", close_side, amount, None, {"stopPrice": tp_price, **params}
        )

        return {
            "status": "SUCCESS",
            "venue": payload.venue.upper(),
            "symbol": payload.symbol,
            "notional_usd": round(size_usd, 2),
            "amount": amount,
            "entry_fill_price": fill_price,
            "stop_loss_set": sl_price,
            "take_profit_set": tp_price,
            "testnet": cfg.use_testnet,
            "orders": {
                "entry": entry_order.get("id"),
                "sl": sl_order.get("id"),
                "tp": tp_order.get("id"),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Execution failed: {e}")
        # rollback best-effort
        try:
            if exchange and sl_order and sl_order.get("id"):
                await exchange.cancel_order(sl_order["id"], payload.symbol)
        except Exception:
            pass
        try:
            if exchange and tp_order and tp_order.get("id"):
                await exchange.cancel_order(tp_order["id"], payload.symbol)
        except Exception:
            pass
        # NOTE: rolling back market entry is non-trivial; user may need manual close.
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if exchange:
            await exchange.close()
