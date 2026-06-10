# GitHub Repository Suggestions for SQDeChain / SQcoin

## Recommended repository name

`sqdechain-sqcoin`

Alternative names:

- `sqdechain-tourism5-ai-blockchain`
- `sqcoin-tourism-kpi-chain`
- `sqdechain-reproducibility`

## GitHub description

Reproducible implementation of SQDeChain/SQcoin, a lightweight multimodal AI and sharded BFT blockchain framework for verifiable Tourism 5.0 KPIs.

## Suggested topics

```text
smart-tourism
tourism-5-0
blockchain
bft
sharding
multimodal-ai
pytorch
transformers
mobilevit
gru
reproducibility
```

## What to upload

Upload source code, configs, documentation, tests, a small sample dataset, and smoke-test outputs.

Do not upload large raw Zenodo files, private datasets, API tokens, trained checkpoints, or environment folders.

## Minimum files for a professional repository

- `README.md`
- `LICENSE`
- `CITATION.cff`
- `requirements.txt`
- `pyproject.toml`
- `.gitignore`
- `configs/`
- `src/`
- `scripts/`
- `tests/`
- `docs/`

## Suggested first commit

```bash
git init
git add .
git commit -m "Initial release of SQDeChain reproducibility code"
git branch -M main
git remote add origin https://github.com/<your-username>/sqdechain-sqcoin.git
git push -u origin main
```

## Suggested release workflow

1. Push the cleaned repository to GitHub.
2. Run the smoke test and unit tests.
3. Create a GitHub release named `v0.1.0`.
4. Archive the GitHub release on Zenodo if you want a software DOI.
5. Update `CITATION.cff` with the final DOI.
