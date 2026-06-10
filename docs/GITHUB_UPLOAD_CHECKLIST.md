# GitHub Upload Checklist

1. Create a new GitHub repository.
2. Unzip this package.
3. Remove or keep the paper copy under `docs/paper/` according to your journal/data-sharing policy.
4. Do not commit large raw datasets or checkpoints.
5. Run:

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_all_experiments.py --config configs/smoke.yaml --outdir results/smoke
```

6. Commit:

```bash
git init
git add .
git commit -m "Initial SQDeChain reproducibility code release"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

7. Add a GitHub release and connect the release to Zenodo if you need a DOI for software.
