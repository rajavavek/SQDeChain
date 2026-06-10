.PHONY: install sample smoke test clean zip

install:
	pip install -r requirements.txt
	pip install -e .

sample:
	python scripts/make_synthetic_data.py --output data/sample --n-reviews 500 --n-images 80 --n-users 200

smoke: sample
	python scripts/run_all_experiments.py --config configs/smoke.yaml --outdir results/smoke

test:
	pytest -q

clean:
	rm -rf build dist *.egg-info .pytest_cache __pycache__

zip:
	cd .. && zip -r sqdechain_github_repo.zip sqdechain_github_repo -x "*/__pycache__/*" "*.pyc"
