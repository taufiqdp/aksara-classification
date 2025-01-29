run:
	python3 -m uvicorn app.main:app --port 8000 --host 0.0.0.0

format:
	python3 -m black .

restart:
	docker compose down && docker compose up -d

