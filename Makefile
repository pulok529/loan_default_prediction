.PHONY: install train test lint serve docker-build docker-run help

help:
	@echo "Available targets:"
	@echo "  make install       Install dependencies and editable package"
	@echo "  make train         Train models and generate artifacts"
	@echo "  make test          Run pytest suite"
	@echo "  make lint          Run ruff linter"
	@echo "  make serve         Start FastAPI development server"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"

install:
	pip install -r requirements-dev.txt
	pip install -e .

train:
	python -m loan_default.train --n 30000 --out artifacts/

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t loan-default:latest .

docker-run:
	docker run -p 8000:8000 loan-default:latest
