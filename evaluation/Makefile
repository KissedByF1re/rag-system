.PHONY: help build test run clean setup

help:
	@echo "Available commands:"
	@echo "  make setup       - Set up development environment"
	@echo "  make build       - Build Docker images"
	@echo "  make test        - Run tests in Docker"
	@echo "  make run         - Run the platform in Docker"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make shell       - Open shell in container"
	@echo "  make logs        - View container logs"

setup:
	@echo "Setting up development environment..."
	cp .env.example .env
	@echo "Please edit .env file with your API keys"

build:
	@echo "Building Docker images..."
	docker-compose build

test:
	@echo "Running tests..."
	docker-compose run --rm test-runner pytest tests/unit -v
	@echo "Unit tests completed. To run integration tests with real APIs:"
	@echo "docker-compose run --rm test-runner pytest tests/integration -v --integration"

run:
	@echo "Starting RAG platform..."
	docker-compose up -d rag-platform chromadb mlflow
	@echo "Platform is running. Use 'make shell' to interact with it."

shell:
	@echo "Opening shell in RAG platform container..."
	docker-compose exec rag-platform /bin/bash

logs:
	docker-compose logs -f

clean:
	@echo "Cleaning up..."
	docker-compose down -v
	rm -rf chroma_db test_chroma_db mlruns test_mlruns results test_results

# Development commands
test-local:
	poetry run pytest tests/unit -v

run-local:
	poetry run rag --help

# Quick test commands
test-config:
	docker-compose run --rm rag-platform python -m rag_platform.cli check-env

test-components:
	docker-compose run --rm rag-platform python -m rag_platform.cli list-components

create-sample-config:
	docker-compose run --rm rag-platform python -m rag_platform.cli create-config /app/configs/sample.yaml

# Run a sample experiment
run-sample:
	docker-compose run --rm rag-platform python -m rag_platform.cli run /app/configs/basic_rag.yaml --sample-size 10