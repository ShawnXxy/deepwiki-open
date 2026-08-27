# syntax=docker/dockerfile:1-labs

# Build argument for custom certificates directory
ARG CUSTOM_CERT_DIR="certs"
ARG NPM_REGISTRY="https://packagefeedproxy.microsoft.io/npm/"
ARG PYPI_REGISTRY="https://packagefeedproxy.microsoft.io/pypi/simple/"
ARG POETRY_VERSION="2.0.1"
ARG POETRY_MIRROR_PLUGIN_VERSION="0.5.0"

FROM node:20-alpine3.22 AS node_base

FROM node_base AS node_deps
ARG NPM_REGISTRY
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm config set registry "${NPM_REGISTRY}" && \
    npm config set replace-registry-host always && \
    npm ci --legacy-peer-deps --no-audit --no-fund

FROM node_base AS node_builder
WORKDIR /app
COPY --from=node_deps /app/node_modules ./node_modules
# Copy only necessary files for Next.js build
COPY package.json package-lock.json next.config.ts tsconfig.json tailwind.config.js postcss.config.mjs ./
COPY src/ ./src/
COPY img/public/ ./public/
# Increase Node.js memory limit for build and disable telemetry
ENV NODE_OPTIONS="--max-old-space-size=4096"
ENV NEXT_TELEMETRY_DISABLED=1
RUN NODE_ENV=production npm run build

FROM python:3.11-slim AS py_deps
ARG PYPI_REGISTRY
ARG POETRY_VERSION
ARG POETRY_MIRROR_PLUGIN_VERSION
WORKDIR /app
COPY pyproject.toml .
COPY poetry.lock .
RUN python -m pip install \
        --index-url "${PYPI_REGISTRY}" \
        --no-cache-dir \
        "poetry==${POETRY_VERSION}" \
        "poetry-plugin-pypi-mirror==${POETRY_MIRROR_PLUGIN_VERSION}" && \
    poetry config virtualenvs.create true --local && \
    poetry config virtualenvs.in-project true --local && \
    poetry config virtualenvs.options.always-copy --local true && \
    POETRY_PYPI_MIRROR_URL="${PYPI_REGISTRY}" \
        POETRY_MAX_WORKERS=10 \
        poetry install --no-interaction --no-ansi --only main && \
    poetry cache clear --all .

# Use Python 3.11 as final image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Node.js, npm, nginx, and Azure CLI (for local dev authentication)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    ca-certificates \
    lsb-release \
    nginx \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    # Install Azure CLI for AzureCliCredential support in local development
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Create nginx temp directories
    && mkdir -p /tmp/nginx_client_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi \
    && mkdir -p /var/log/nginx \
    && chown -R root:root /var/log/nginx

# Update certificates if custom ones were provided and copied successfully
RUN if [ -n "${CUSTOM_CERT_DIR}" ]; then \
        mkdir -p /usr/local/share/ca-certificates && \
        if [ -d "${CUSTOM_CERT_DIR}" ]; then \
            cp -r ${CUSTOM_CERT_DIR}/* /usr/local/share/ca-certificates/ 2>/dev/null || true; \
            update-ca-certificates; \
            echo "Custom certificates installed successfully."; \
        else \
            echo "Warning: ${CUSTOM_CERT_DIR} not found. Skipping certificate installation."; \
        fi \
    fi

ENV PATH="/opt/venv/bin:$PATH"

# Copy Python dependencies
COPY --from=py_deps /app/.venv /opt/venv
COPY backend/ ./backend/

# Note: Environment-specific configs are handled by deployment scripts:
# - Local Docker: test-local.ps1 creates .local/ and mounts it at runtime
# - Azure Cloud: deploy-azure.ps1 creates .cloud/ and copies it before build
# The .cloud folder (if exists) overrides default config for Azure deployments
COPY backend/config/.clou[d]/ ./backend/config/

# Copy Node app
COPY --from=node_builder /app/public ./public
COPY --from=node_builder /app/.next/standalone ./
COPY --from=node_builder /app/.next/static ./.next/static

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose the port the app runs on (nginx on 3000)
EXPOSE 3000

# Create a script to run both backend and frontend
RUN echo '#!/bin/bash\n\
# Load environment variables from backend/.env file if it exists (silently)\n\
if [ -f backend/.env ]; then\n\
  set -a\n\
  source backend/.env 2>/dev/null || true\n\
  set +a\n\
fi\n\
\n\
# Configuration status (never log actual keys or values)\n\
echo "📋 Azure Configuration Status:"\n\
echo "  • AZURE_CLIENT_ID (MSI): $([ -n \"$AZURE_CLIENT_ID\" ] && echo \"✓ Set\" || echo \"✗ Not set\")"\n\
echo "  • AZURE_OPENAI_API_KEY: $([ -n \"$AZURE_OPENAI_API_KEY\" ] && echo \"✓ Set\" || echo \"✗ Not set\")"\n\
echo ""\n\
\n\
# Check Azure authentication method\n\
has_azure_msi=false\n\
has_azure_key=false\n\
\n\
if [ -n "$AZURE_CLIENT_ID" ]; then\n\
  has_azure_msi=true\n\
fi\n\
if [ -n "$AZURE_OPENAI_API_KEY" ]; then\n\
  has_azure_key=true\n\
fi\n\
\n\
# Validate configuration\n\
if [ "$has_azure_msi" = false ] && [ "$has_azure_key" = false ]; then\n\
  echo "⚠️  Warning: No Azure authentication configured!"\n\
  echo "Please configure one of the following:"\n\
  echo "  • AZURE_CLIENT_ID (Managed Identity - recommended for Azure)"\n\
  echo "  • AZURE_OPENAI_API_KEY (API Key - for local development)"\n\
  echo ""\n\
fi\n\
\n\
if [ "$has_azure_msi" = true ]; then\n\
  echo "🚀 Starting DeepWiki with Azure Managed Identity..."\n\
else\n\
  echo "🚀 Starting DeepWiki with Azure OpenAI API Key..."\n\
fi\n\
\n\
# Start nginx in the foreground (after backgrounding other services)\n\
# Next.js on port 3001 (internal), FastAPI on port 8001 (internal)\n\
# nginx on port 3000 (external) proxies to both\n\
echo "Starting FastAPI backend on port 8001..."\n\
python -m backend.main --port 8001 &\n\
BACKEND_PID=$!\n\
\n\
echo "Starting Next.js frontend on port 3001..."\n\
PORT=3001 HOSTNAME=127.0.0.1 node server.js &\n\
NEXTJS_PID=$!\n\
\n\
# Wait for services to start\n\
sleep 3\n\
\n\
echo "Starting nginx reverse proxy on port 3000..."\n\
nginx -g "daemon off;" &\n\
NGINX_PID=$!\n\
\n\
# Wait for any process to exit\n\
wait -n $BACKEND_PID $NEXTJS_PID $NGINX_PID\n\
\n\
# Exit with the status of the process that exited first\n\
exit $?' > /app/start.sh && chmod +x /app/start.sh

# Set environment variables
ENV PORT=8001
ENV NODE_ENV=production
ENV SERVER_BASE_URL=http://localhost:${PORT:-8001}

# Supported environment variables (set via .env file or docker run -e):
# Azure OpenAI (required):
#   AZURE_OPENAI_API_KEY - Azure OpenAI API key (or use Managed Identity)
#   AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint
#   AZURE_OPENAI_DEPLOYMENT - Azure deployment name
#   AZURE_OPENAI_VERSION - API version (default: 2024-12-01-preview)
# Azure Managed Identity (recommended for Azure deployments):
#   AZURE_CLIENT_ID - Managed Identity client ID (replaces API key auth)
# Azure Blob Storage (optional, for data persistence):
#   Configured via backend/config/infra.json
# Configuration:
#   LOG_LEVEL - Logging level (default: INFO)
#   DEEPWIKI_CONFIG_DIR - Custom config directory path

# Create empty .env file (will be overridden if one exists at runtime)
RUN touch backend/.env

# Command to run the application
CMD ["/app/start.sh"]
