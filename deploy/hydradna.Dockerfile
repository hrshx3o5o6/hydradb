FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY causal_memory/ causal_memory/

RUN pip install --no-cache-dir .

ENV HYDRADB_URL=http://hydradb:8443 \
    HYDRADB_ADMIN_URL=http://hydradb:9090 \
    HYDRADB_TOKEN=local-dev-auth-token-32-characters-long

ENTRYPOINT ["hydradna"]