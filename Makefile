# ===========================
# Quant-ML: CLI entry points
# ===========================
# Usage:
#   make ingest_h1 YEARS="2024 2025" SYMBOLS="NVDA TSLA RKLB"
#   make make_features_h1 YEARS="2024 2025"
#   make make_labels_h1 YEARS="2024 2025"
#   make train_wf YEARS="2018 2019 ... 2025"
#   make backtest_h1 RUN_ID=run_20250101_000000 YEARS="2018 ... 2025"
#   make report RUN_ID=run_...
#   make nightly  (full chain)

# ---------- config ----------
export PYTHONPATH := .
export TZ ?= US/Eastern

# universe
SYMBOLS_FILE ?= src/config/symbols.yaml
# default universe (kept small per your note)
SYMBOLS ?= NVDA TSLA RKLB

# years
YEARS ?= 2018 2019 2020 2021 2022 2023 2024 2025
CUR_YEAR ?= $(shell date -u +%Y)

# run id
RUN_ID ?= $(shell date -u +"run_%Y%m%d_%H%M%S")

# market-wide regime instrument
QQQ_SYMBOL ?= QQQ

# costs & controls
COMMISSION_PER_SHARE ?= 0.0038
SLIPPAGE_BPS ?= 5
DD_HALT_PCT ?= 0.02
ATR_VOL_CAP ?= 0.08
TARGET_VOL ?= 0.10

# Alpaca feed (iex|sip)
ALPACA_DATA_FEED ?= iex

# Paths
PARAMS_CFG ?= src/config/params.yaml

# ---------- helpers ----------
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make ingest_h1           # Fetch 1H bars from Alpaca -> R2 monthly partitions"
	@echo "  make make_features_h1    # Build 1H features for ALL symbols (current year by default)"
	@echo "  make make_labels_h1      # Build 1H triple-barrier labels"
	@echo "  make train_wf            # Train XGB walk-forward (push OOS probs to R2)"
	@echo "  make backtest_h1         # Run 1H engine with gates + costs (push results to R2)"
	@echo "  make report              # Portfolio aggregation + metrics"
	@echo "  make nightly             # Full pipeline end-to-end"
	@echo ""
	@echo "Vars: SYMBOLS, YEARS, RUN_ID, ALPACA_DATA_FEED, COMMISSION_PER_SHARE, SLIPPAGE_BPS"
	@echo "      TARGET_VOL, DD_HALT_PCT, ATR_VOL_CAP, PARAMS_CFG, QQQ_SYMBOL"

# ---------- 12) CLI Targets ----------
.PHONY: ingest_h1
ingest_h1:
	@for s in $(SYMBOLS); do \
		echo "== Ingest 1H: $$s :: $(YEARS) feed=$(ALPACA_DATA_FEED)"; \
		python -m src.ingest.alpaca_h1_to_r2 --symbol $$s --years $(YEARS) --feed $(ALPACA_DATA_FEED); \
	done

.PHONY: make_features_h1
make_features_h1:
	# Current year by default (fast nightly)
	python -m src.features.make_features_h1_to_cloud --symbol ALL --symbols-cfg $(SYMBOLS_FILE) --years $(CUR_YEAR)

.PHONY: make_labels_h1
make_labels_h1:
	@for s in $(SYMBOLS); do \
		echo "== Labels 1H: $$s :: $(YEARS)"; \
		python -m src.labels.h1_triple_barrier --symbol $$s --years $(YEARS); \
	done

.PHONY: train_wf
train_wf:
	@for s in $(SYMBOLS); do \
		echo "== Train WF: $$s :: $(YEARS)"; \
		python -m src.models.xgb_walkforward --symbol $$s --years $(YEARS) \
			--min-train-days 252 --min-test-days 30 --push-to-r2 \
			--scale-pos-weight auto --calibration-dual; \
	done

.PHONY: backtest_h1
backtest_h1:
	@for s in $(SYMBOLS); do \
		echo "== Backtest 1H: $$s :: $(YEARS) RUN_ID=$(RUN_ID)"; \
		python -m src.backtest.engine --symbol $$s --years $(YEARS) \
			--run-id $(RUN_ID) --commission-per-share $(COMMISSION_PER_SHARE) \
			--slippage-bps $(SLIPPAGE_BPS) --use-param-cfg $(PARAMS_CFG) \
			--qqq-regime --hmm-regime --qqq-symbol $(QQQ_SYMBOL) \
			--dd-halt-pct $(DD_HALT_PCT) --atr-vol-cap $(ATR_VOL_CAP) \
			--push-to-r2; \
	done

.PHONY: report
report:
	python -m src.reporting.eval_suite --run-id $(RUN_ID) --universe $(SYMBOLS) \
		--enable-hrp --enable-target-vol --target-vol $(TARGET_VOL)

.PHONY: nightly
nightly: ingest_h1 make_features_h1 make_labels_h1 train_wf backtest_h1 report
	@echo "[DONE] RUN_ID=$(RUN_ID)"

# quick sanity check for env
.PHONY: check_env
check_env:
	@python - <<'PY'
import os
keys = ["ALPACA_API_KEY_ID","ALPACA_SECRET_KEY","APCA_API_KEY_ID","APCA_API_SECRET_KEY","R2_ENDPOINT","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY","R2_BUCKET"]
missing = [k for k in keys if not os.getenv(k)]
print("Missing:", missing if missing else "OK")
PY
