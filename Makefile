.PHONY: help dev stop logs simulate grafana train evaluate export-onnx test lint fmt typecheck clean

POETRY := poetry run
ARTIFACTS_DIR := artifacts/
DATA_DIR := data/

help:
	@echo ""
	@echo "SensorWatch — Developer Commands"
	@echo "────────────────────────────────────────────────────"
	@echo "  make dev        Start core services (TimescaleDB + Redis + app)"
	@echo "  make simulate   Start core + sensor simulator"
	@echo "  make grafana    Start core + Grafana dashboard"
	@echo "  make stop       Stop all containers"
	@echo "  make logs       Tail container logs"
	@echo ""
	@echo "  make train      Download CMAPSS + train model + calibrate thresholds"
	@echo "  make evaluate     Evaluate model precision/recall on test set"
	@echo "  make export-onnx  Export model to ONNX for edge deployment (Jetson / RPi)"
	@echo ""
	@echo "  make test       Run all tests with coverage"
	@echo "  make lint       Run ruff linter"
	@echo "  make fmt        Auto-format with ruff"
	@echo "  make typecheck  Run mypy"
	@echo "  make clean      Remove Python cache files"
	@echo ""

dev:
	docker compose up --build

simulate:
	docker compose --profile simulate up --build

grafana:
	docker compose --profile grafana up --build

stop:
	docker compose down

logs:
	docker compose logs -f

train:
	$(POETRY) python -m ml.scripts.train \
		--subset FD001 \
		--data-dir $(DATA_DIR) \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--window-size 30 \
		--latent-dim 32 \
		--hidden-size 64 \
		--epochs 100 \
		--batch-size 256 \
		--patience 10 \
		--warning-percentile 90 \
		--critical-percentile 99

evaluate:
	$(POETRY) python -m ml.scripts.evaluate \
		--subset FD001 \
		--data-dir $(DATA_DIR) \
		--artifacts-dir $(ARTIFACTS_DIR)

export-onnx:
	poetry install --with export --no-root -q
	$(POETRY) python -m ml.scripts.export_onnx \
		--artifacts-dir $(ARTIFACTS_DIR) \
		--opset 17 \
		--verify

test:
	$(POETRY) pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	$(POETRY) ruff check app/ ml/ simulator/ tests/

fmt:
	$(POETRY) ruff format app/ ml/ simulator/ tests/

typecheck:
	$(POETRY) mypy app/ ml/ simulator/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov
