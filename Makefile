# make sure this is executed with bash
SHELL := /bin/bash


YELLOW := "\e[1;33m"
NC := "\e[0m"

# Logger function
INFO := @bash -c '\
  printf $(YELLOW); \
  echo "=> $$1"; \
  printf $(NC)' SOME_VALUE

.venv:  # creates .venv folder if does not exist
	python3.10 -m venv .venv


.venv/bin/uv: .venv # installs latest pip
	.venv/bin/pip install -U uv

.PHONY: install download_data_from_s3 shinka

install: .venv/bin/uv
	# before running install cmake
	.venv/bin/uv sync
	# after installing source .venv/bin/activate in your shell

download_data_from_s3:
	.venv/bin/python3 -m download_data

shinka:
	@set -a; \
	if [ -f .env ]; then \
		source .env; \
	fi; \
	set +a; \
	if [ -z "$${OPENAI_API_KEY:-}" ] && [ -z "$${LOCAL_OPENAI_API_KEY:-}" ]; then \
		echo "Set OPENAI_API_KEY or LOCAL_OPENAI_API_KEY in .env before running shinka"; \
		exit 1; \
	fi; \
	export OPENAI_API_KEY="$${OPENAI_API_KEY:-$${LOCAL_OPENAI_API_KEY}}"; \
	export LOCAL_OPENAI_API_KEY="$${LOCAL_OPENAI_API_KEY:-$${OPENAI_API_KEY}}"; \
	shinka_run --task-dir shinka --results_dir results/shinka_simple_ml --num_generations 100 --config-fname shinka.yaml
