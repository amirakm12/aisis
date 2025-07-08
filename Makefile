# Makefile for aisis project

.PHONY: help install test lint run clean docs apidocs coverage docker-build

help:
	@echo "Available targets:"
	@echo "  install   Install dependencies"
	@echo "  test      Run tests"
	@echo "  lint      Run linter (flake8)"
	@echo "  run       Run the main application"
	@echo "  clean     Remove Python cache and build artifacts"
	@echo "  docs      Build documentation"
	@echo "  apidocs   Build API documentation"
	@echo "  coverage  Run coverage tests"
	@echo "  docker-build Build Docker image"

install:
	pip install -r requirements.txt

lint:
	flake8 src/ tests/

test:
	pytest tests/

run:
	python main.py

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache/ .tox/ *.pyc *.pyo *.egg-info build/ dist/ htmlcov/ .coverage

docs:
	cd docs && make html

apidocs:
	cd docs && make html

coverage:
	pytest --cov=src --cov-report=html

docker-build:
	docker build -t aisis . 