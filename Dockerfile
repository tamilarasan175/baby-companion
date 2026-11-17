FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg openssl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/certs && \
    openssl req -x509 -newkey rsa:2048 -keyout /app/certs/key.pem -out /app/certs/cert.pem -days 365 -nodes -subj "/CN=baby-companion"

RUN python -c "from app.main import app; print('Import OK')"

EXPOSE 8443

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8443", "--ssl-keyfile", "/app/certs/key.pem", "--ssl-certfile", "/app/certs/cert.pem"]
