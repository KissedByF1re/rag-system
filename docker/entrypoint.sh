#!/bin/bash
set -e

# Function to log messages
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
  log "Waiting for PostgreSQL at ${POSTGRES_HOST}:${POSTGRES_PORT}..."
  
  # Install netcat if not available
  if ! command -v nc >/dev/null 2>&1; then
    log "Installing netcat..."
    apt-get update && apt-get install -y netcat-openbsd
  fi
  
  # Check if PostgreSQL is ready
  until nc -z "${POSTGRES_HOST}" "${POSTGRES_PORT}"; do
    log "PostgreSQL is unavailable - sleeping 2 seconds"
    sleep 2
  done
  
  log "PostgreSQL is up and running!"
}

# Set up watchdog for improved hot reloading
setup_watchdog() {
  log "Setting up watchdog for hot reloading..."
  pip install --no-cache-dir watchdog[watchmedo] >>/dev/null 2>&1
  log "Watchdog setup complete"
}

# Main execution
log "Starting container initialization..."

# Set environment variables from container environment
export POSTGRES_HOST=${POSTGRES_HOST:-bread-postgres}
export POSTGRES_PORT=${POSTGRES_PORT:-5432}
export DB_HOST=${POSTGRES_HOST}
export DB_PORT=${POSTGRES_PORT}
export DB_USER=${POSTGRES_USER:-postgres}
export DB_PASSWORD=${POSTGRES_PASSWORD:-postgres}
export DB_NAME=${POSTGRES_DB:-agent_service}
export SEARCH_DB_NAME=${SEARCH_DB:-postgres_search}

# Print environment for debugging
log "Using POSTGRES_HOST: ${POSTGRES_HOST}"
log "Using POSTGRES_PORT: ${POSTGRES_PORT}"

# Wait for PostgreSQL to be ready
wait_for_postgres

# Initialize ChromaDB if it doesn't exist
initialize_chroma_db() {
  log "Checking ChromaDB initialization..."
  
  if [ ! -d "/app/data/chroma_db" ]; then
    log "ChromaDB not found. Initializing..."
    cd /app
    python scripts/create_chroma_db.py
    log "ChromaDB initialization complete"
  else
    log "ChromaDB already exists"
  fi
}

# Initialize ChromaDB
initialize_chroma_db

# Setup watchdog for hot reloading
setup_watchdog

# Check if we should enable enhanced hot reload
if [ "${MODE}" = "development" ] || [ "${DEV_MODE}" = "true" ]; then
  log "Starting the main application in DEVELOPMENT mode with enhanced hot reloading..."
  # Set environment variable to enable development mode
  export MODE=development
  export DEV_MODE=true
  
  # Start with watchdog monitoring
  if [[ "$@" == *"run_service.py"* ]]; then
    log "Running with enhanced hot reload for Python service"
    exec python "$@" --hot-reload
  else
    log "Running with standard hot reload"
    exec "$@"
  fi
else
  log "Starting the main application in PRODUCTION mode..."
  exec "$@"
fi