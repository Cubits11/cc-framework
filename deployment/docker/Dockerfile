FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN useradd --create-home --shell /usr/sbin/nologin ccuser \
    && chown -R ccuser:ccuser /app
USER ccuser

CMD ["python", "-c", "import cc; print('cc-framework ready')"]
