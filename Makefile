.PHONY: help install dev api lint format test clean check

# Default target
help:
	@echo "ML Training Platform - Development Commands"
	@echo ""
	@echo "install     Install dependencies with uv"
	@echo "dev         Start development servers (backend + frontend)"
	@echo "api         Start FastAPI backend only"
	@echo "ui          Start React frontend only"
	@echo "lint        Run linting checks"
	@echo "format      Format code with black and ruff"
	@echo "test        Run tests"
	@echo "clean       Clean cache and build files"
	@echo "check       Run all checks (lint + test)"

# Install dependencies
install:
	uv sync --group dev
	@if [ ! -d "ui/node_modules" ]; then \
		cd ui && npm install; \
	fi

# Start both development servers
dev:
	@echo "Starting FastAPI backend on http://localhost:8000"
	@echo "Starting React frontend on http://localhost:5173"
	@make api &
	@make ui &
	@wait

# Start FastAPI backend only
api:
	@echo "Starting FastAPI backend..."
	uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start React frontend only
ui:
	@echo "Starting React frontend..."
	cd ui && npm run dev

# Run linting checks
lint:
	@echo "Running linting checks..."
	uv run ruff check backend/ ml_core/ data_pipeline/ tests/
	@echo "Note: mypy checks disabled for bootstrap phase"

# Format code
format:
	@echo "Formatting code..."
	uv run black backend/ ml_core/ data_pipeline/ tests/
	uv run ruff check --fix backend/ ml_core/ data_pipeline/ tests/
	@echo "Note: mypy checks disabled for bootstrap phase"

# Run tests
test:
	@echo "Running tests..."
	uv run pytest

# Clean cache and build files
clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage .pytest_cache .mypy_cache
	rm -rf build/ dist/ *.egg-info/
	cd ui && rm -rf node_modules/ dist/ || true

# Run all checks
check: lint test
	@echo "All checks completed!"