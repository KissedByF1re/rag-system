FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1

# Install client dependencies directly with pip
COPY pyproject.toml .
RUN pip install --no-cache-dir streamlit httpx pydantic python-dotenv watchdog

COPY src/client/ ./client/
COPY src/schema/ ./schema/
COPY src/streamlit_app.py .

# Run with auto-reloading enabled
CMD ["streamlit", "run", "streamlit_app.py", "--server.runOnSave=true", "--server.fileWatcherType=poll"]
