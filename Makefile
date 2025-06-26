PYTHON_VERSION := 3

venv:
	python$(PYTHON_VERSION) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -r requirements.txt
	. .venv/bin/activate && python -m pip install -r test_requirements.txt

test:
	sox -n -r 16000 -c 1 -b 16 input.wav synth 1 whitenoise
	for model in $(shell ls models); do \
		echo "Testing $$model"; \
		export HF_HOME=$(shell pwd)/.cache/huggingface; \
		. .venv/bin/activate && python infer.py --model models/$$model --input input.wav --output $$model.z; \
	done

test-clean:
	rm -f input.wav *.z

format:
	. .venv/bin/activate && ruff check --fix *.py
	. .venv/bin/activate && ruff format *.py

.PHONY: venv test format
