# NBA player points: does a model beat the sportsbooks?

A paper-trading experiment, run entirely by GitHub Actions. A model projects
each player's points; every sportsbook points line is logged up to 6 hours before
tip-off along with the model's probability, and a hypothetical $100 bet is
placed whenever the model's side has at least +3% expected value. Nothing is
edited after the fact, so the log is an honest out-of-sample record.

The NBA has 5-10 games most nights, so the experiment reaches a meaningful
sample (300+ bets) in a few weeks.

## How it works

| Step | When | What |
|---|---|---|
| `model.py` | Daily 13:30 UTC | Retrain on ESPN box scores through last night (2021-22 season on). |
| `autobet.py` | Every 30 min, 15:15-03:45 UTC | For games tipping off within 6 hours: pull every book's points lines, project each player, log the line and any paper bet. For logged games tipping off within 90 minutes: snapshot the closing lines. |
| `report.py` | Mondays | Post the season-to-date report as a GitHub issue. |

**Model.** An average of XGBoost and ridge regression on the player's recent
scoring, minutes, shot volume, usage, starter role, rest, home/away, and the
opponent's points allowed and pace. Over/under probabilities come from
held-out errors scaled by the square root of the projection, so the model's
uncertainty grows with the projection (a 28-point scorer's range is wider than
an 8-point bench player's).

**Evaluation** (on settled lines; DNPs are void, as at sportsbooks):

- *Model rule*: the pre-registered strategy (EV ≥ 3%, best price across books). This is the real test.
- *Shadow Kelly*: the same bets, sized by the Kelly fraction, scaled to the same total stake.
- *Controls*: always over, always under, the model's side on every line.
- *Probability accuracy*: Brier score and log loss vs. the no-vig market probability.
- *Closing line value*: whether the line moves toward the bets before tip. The fastest reliable sign of a real edge.
- *Calibration*: when the model says 60%, does it hit 60%?

## Dashboard

`streamlit_app.py` is a phone-friendly dashboard: profit and record, tonight's
bets, each night's results, profit over time, and the evaluation details. It
reads the experiment log straight from GitHub, so it's never more than 10
minutes behind. Deploy it free at [share.streamlit.io](https://share.streamlit.io)
(repo `shannoj/nba_props_project`, main file `streamlit_app.py`).

## Setup

1. Get an API key at [the-odds-api.com](https://the-odds-api.com) and add it as the
   repository secret `ODDS_API_KEY` (Settings → Secrets and variables → Actions).
2. Run **Train and report** once from the Actions tab (or wait for 13:30 UTC).

Credits: each game costs 2 (log + closing), about 450 a month during the season.
The free tier is 500 a month, so use a separate key from any other project.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python model.py                       # train
ODDS_API_KEY=... .venv/bin/python autobet.py    # log lines for games tipping off soon
.venv/bin/python report.py report.md            # experiment report
```

## Data

- Box scores: ESPN via [sportsdataverse-data](https://github.com/sportsdataverse/sportsdataverse-data) releases.
- Odds: [The Odds API](https://the-odds-api.com), `player_points` market.

## Known limitations

- The model doesn't know about teammates being out tonight. When a star sits,
  everyone else's usage rises, and the books adjust for that before tip. This is
  the most likely place the books have an edge, and the first thing to add.
- Early-season projections lean on last season's form.
