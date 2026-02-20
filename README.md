# Apex Orchestrator (Citadel) — v2026.3 hardened

Este pacote contém **4 microserviços FastAPI** (portas 8000–8003) para pesquisa/automação de trading e análise on-chain.
Ele inclui **hardening**, logs estruturados, healthchecks, e padrões mais robustos de reconexão/execução.

> ⚠️ Aviso: automação de trade envolve risco financeiro e risco operacional. Rode sempre primeiro em **testnet/sandbox**.

## Serviços

- **8000** `brain_engine.py` — motor de decisão (tri-layer) com:
  - *Hill tail-index* (risco de cauda),
  - *AMH memory multiplier* (memória de desempenho),
  - integração de microestrutura (micro-price shift/imbalance/long-short ratio) para ajuste de confiança.
- **8001** `app_shadowglass.py` — ingestão Binance Futures via websocket (openInterest, markPrice, depth) e scrape do **globalLongShortAccountRatio**.
- **8002** `app_executioner.py` — execução via `ccxt.async_support` com:
  - sizing por `size_usd` **ou** `risk_pct` baseado em equity,
  - criação de bracket (SL/TP) com `reduceOnly`,
  - rollback/cancelamento se falhar a criação de proteção.
- **8003** `anti_rug_engine.py` — modelo scikit-learn (RandomForest) treinado sinteticamente no primeiro boot.

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 4 terminais:
uvicorn brain_engine:app --host 0.0.0.0 --port 8000
uvicorn app_shadowglass:app --host 0.0.0.0 --port 8001
uvicorn app_executioner:app --host 0.0.0.0 --port 8002
uvicorn anti_rug_engine:app --host 0.0.0.0 --port 8003
```

## Endpoints úteis

- Health:
  - `GET /health` em todos os serviços.
- Brain:
  - `POST /process_tick`
- Shadowglass:
  - `GET /get_market_state/{symbol}`
- Executioner:
  - `GET /get_equity/{venue}`
  - `POST /execute_strike`
- Anti-rug:
  - `POST /analyze_token`

## Segurança/Boas práticas

- Use `.env` e **nunca** commite chaves.
- Mantenha `USE_TESTNET=TRUE` até validar totalmente.
- Rode em usuário sem privilégios e coloque rate-limits/retries do lado do cliente também.
