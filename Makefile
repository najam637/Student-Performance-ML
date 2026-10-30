.PHONY: train test run
train:
	python train_models.py

test:
	pytest -q

run:
	uvicorn app:app --host 127.0.0.1 --port 8000 --reload
