FROM python:3.11-slim

# Install system dependencies for patchright (Chromium-based stealth browser)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates \
    libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpangocairo-1.0-0 \
    libpango-1.0-0 libcairo2 libatspi2.0-0 libx11-6 libxext6 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install patchright's Chromium browser binary
RUN python -m patchright install chromium --with-deps

COPY . .

ENV ORCHESTRATOR_URL=http://host.docker.internal:8000
ENV BROWSER_BACKEND=playwright
ENV LLM_PROVIDER=anthropic

CMD ["python", "main.py"]
