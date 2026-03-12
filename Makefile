.PHONY: install sync lint test test-unit test-integration coverage generate-samples docs docs-serve docs-deploy build publish publish-test infra-up infra-down infra-logs infra-status

install:
	@uv sync --dev

sync:
	@uv sync --dev

lint:
	@uv run ruff check --fix src/ tests/
	@uv run ruff format src/ tests/

test:
	@uv run pytest

test-unit:
	@uv run pytest tests/unit/

test-integration:
	@uv run pytest tests/integration/ --stt-url=ws://localhost:45120 --tts-url=http://localhost:45130/v1/audio/speech

coverage:
	@uv run coverage run -m pytest tests/unit/ -q --no-header
	@uv run coverage report

generate-samples:
	@uv run python scripts/generate_samples.py

##### DOCUMENTATION #####

docs:
	@uv run mkdocs build --strict

docs-serve:
	@uv run mkdocs serve

docs-deploy:
	@uv run mkdocs gh-deploy --force

##### PUBLISHING #####

build:
	@rm -rf dist/
	@uv build

publish-test:
	@uv publish --publish-url https://test.pypi.org/legacy/

publish:
	@uv publish

##### DOCKER — INTEGRATION SERVICES #####

infra-up:
	@docker compose -f compose.integration.yml up -d
	@echo "Waiting for services to become healthy..."
	@docker compose -f compose.integration.yml ps

infra-down:
	@docker compose -f compose.integration.yml down

infra-logs:
	@docker compose -f compose.integration.yml logs -f

infra-status:
	@docker compose -f compose.integration.yml ps
