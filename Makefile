dev:
	python3 -m uvicorn app.main:app --port 8000 --host 0.0.0.0 --reload

format:
	python3 -m black .

restart:
	docker compose down && docker compose up -d

rebuild:
	docker compose down && docker compose up --build -d

stop:
	docker compose down

