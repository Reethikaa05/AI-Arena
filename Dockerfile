FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables (override at runtime)
ENV HF_TOKEN=""
ENV ANTHROPIC_API_KEY=""
ENV PORT=8000

EXPOSE 8000

CMD ["python", "app.py"]
