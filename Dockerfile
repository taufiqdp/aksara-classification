# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --no-warn-script-location --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY app/ app/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
