FROM python:3.11-slim

# Install system deps for numpy & sklearn
RUN apt-get update && apt-get install -y build-essential libatlas-base-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Uvicorn will be started by Vercel
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8080"]