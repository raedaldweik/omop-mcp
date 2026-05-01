# OMOP MCP tool server — slim, non-root, multi-stage.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OS-level deps. ca-certificates is needed for TLS to Supabase.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the server.
COPY server.py .

# Run as a non-privileged user.
RUN useradd --create-home --shell /bin/bash mcp
USER mcp

# Streamable HTTP transport listens on this port.
EXPOSE 8000

CMD ["python", "server.py"]
