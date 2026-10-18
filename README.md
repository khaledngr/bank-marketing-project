# Bank Marketing: who should the bank call?

Group project for **Supervised Learning & Model Engineering** (Ilana Sebag, Albert School, Fall 2026).
Team: Khaled Elnagar, Alzira Ferreira.

## Goal

A Portuguese bank phones clients to sell a term deposit. Every call costs money; every
subscription earns money. We build an end-to-end proof of concept, from the raw data to a served
prediction endpoint, that scores each client's probability of subscribing so the bank calls only
the clients for whom a call is expected to pay off.

- **Dataset:** UCI Bank Marketing, OpenML id 1461 (45,211 clients, 11.7% subscribed).
- **Target:** `y = 1` if the client subscribed (OpenML `Class == "2"`).
- **Business assumptions** (in [`config.yaml`](config.yaml)): EUR 5 per call, EUR 50 margin per
  subscription, so a call is worth it when the predicted probability is above 0.10.

## Setup

Requires Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No secrets are needed. [`.env.example`](.env.example) only lists the variables you would need for a
remote MLflow server.

## Project structure

```
config.yaml        paths, seed, split sizes, business costs
src/               all project code (run each step as a module, see below)
reports/figures/   plots used in the report
reports/tables/    CSV tables used in the report
artifacts/         fitted models (git-ignored)
data/              OpenML download cache (git-ignored)
```

## How to run

All commands run from the project root with the virtual environment active.

_Steps are added here as they are implemented._

## Reproducibility

- One seed (`seed: 42` in `config.yaml`) is passed to every split, model and sampler.
- The test set is split off once with a fixed seed and is only scored at the very end.
- All preprocessing lives inside the scikit-learn Pipeline, so it is refitted inside every CV fold.
