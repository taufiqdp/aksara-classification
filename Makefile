.PHONY: dev prod format lint test clean

# Development
dev:
	python3 -m uvicorn app.main:app --port 8000 --host 0.0.0.0 --reload

dev-docker:
	docker compose -f docker-compose.dev.yml up -d

dev-rebuild:
	docker compose -f docker-compose.dev.yml down && docker compose -f docker-compose.dev.yml up --build -d

dev-stop:
	docker compose -f docker-compose.dev.yml down

# Production
prod:
	docker compose -f docker-compose.yml up -d

prod-rebuild:
	docker compose -f docker-compose.yml down && docker compose -f docker-compose.yml up --build -d

prod-stop:
	docker compose -f docker-compose.yml down

# Code quality
format:
	python3 -m black .
	python3 -m isort .

lint:
	python3 -m flake8 .
	python3 -m mypy .

test:
	python3 -m pytest

# Utility
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +

# Help
help:
	@echo "Available commands:"
	@echo "Development:"
	@echo "  make dev              - Run development server locally"
	@echo "  make dev-docker       - Run development environment in Docker"
	@echo "  make dev-rebuild      - Rebuild and run development Docker containers"
	@echo "  make dev-stop         - Stop development Docker containers"
	@echo "Production:"
	@echo "  make prod             - Run production environment in Docker"
	@echo "  make prod-rebuild     - Rebuild and run production Docker containers"
	@echo "  make prod-stop        - Stop production Docker containers"
	@echo "Code quality:"
	@echo "  make format           - Format code using black and isort"
	@echo "  make lint             - Run linting tools"
	@echo "  make test             - Run tests"
	@echo "Utility:"
	@echo "  make clean            - Remove Python cache files"
	@echo "  make help             - Show this help message"

