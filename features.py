"""Pre-game features for player points.

Every feature uses only games before the one being predicted. Training rows and
upcoming games go through the same code: upcoming rows are appended with no
result, so their rolling stats are the player's form going into tonight.
"""
import numpy as np
import pandas as pd

TARGET = "points"

FEATURES = [
    # Scoring and role
    "pts_5", "pts_10", "pts_ewm", "pts_season", "pts_per_min_20",
    "min_5", "min_10", "min_ewm", "min_last", "starter_5",
    "fga_10", "tpa_10", "fta_10", "usage_10",
    "games_prior", "games_season",
    # Schedule
    "is_home", "days_rest", "b2b",
    # Team and opponent (last 10 games)
    "team_pts_10", "opp_allowed_10", "opp_pace_10",
]


def _roll(g, col, n, minp=1):
    return g[col].transform(lambda s: s.shift(1).rolling(n, min_periods=minp).mean())


def _player_features(df):
    df = df.sort_values(["player_id", "tipoff"]).copy()
    g = df.groupby("player_id", sort=False)
    df["pts_5"], df["pts_10"] = _roll(g, "points", 5), _roll(g, "points", 10)
    df["pts_ewm"] = g["points"].transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    df["min_5"], df["min_10"] = _roll(g, "minutes", 5), _roll(g, "minutes", 10)
    df["min_ewm"] = g["minutes"].transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())
    df["min_last"] = g["minutes"].shift(1)
    df["starter_5"] = _roll(g, "starter_f", 5)
    for c in ("fga", "tpa", "fta"):
        df[f"{c}_10"] = _roll(g, c, 10)
    poss = df["fga"] + 0.44 * df["fta"] + df["turnovers"]
    df["_poss"] = poss
    sums = {c: g[c].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).sum())
            for c, n in (("points", 20), ("minutes", 20), ("_poss", 10))}
    min10 = g["minutes"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).sum())
    df["pts_per_min_20"] = sums["points"] / sums["minutes"].replace(0, np.nan)
    df["usage_10"] = sums["_poss"] / min10.replace(0, np.nan)
    df["games_prior"] = g.cumcount()
    gs = df.groupby(["player_id", "season"], sort=False)
    df["games_season"] = gs.cumcount()
    df["pts_season"] = gs["points"].transform(lambda s: s.shift(1).expanding().mean())
    prev_date = g["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - prev_date).dt.days.clip(upper=7)
    df["b2b"] = (df["days_rest"] == 1).astype(float)
    df["is_home"] = (df["home_away"] == "home").astype(float)
    return df.drop(columns="_poss")


def _utc(s):
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def _team_games(box):
    """One row per team-game with rolling scoring 'after' that game (shifted in by merge_asof)."""
    t = (box.groupby(["game_id", "team"], as_index=False)
         .agg(tipoff=("tipoff", "first"), scored=("team_score", "first"), allowed=("opponent_team_score", "first"))
         .sort_values(["team", "tipoff"]))
    g = t.groupby("team", sort=False)
    t["pts_10"] = g["scored"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    t["allowed_10"] = g["allowed"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    t["pace_10"] = t["pts_10"] + t["allowed_10"]
    return t.sort_values("tipoff")


def _asof(rows, teams, key, cols):
    """Attach the latest team-game values strictly before each row's tipoff."""
    right = teams[["team", "tipoff", *cols]].rename(columns={"team": key, **{c: f"{key}_{c}" for c in cols}})
    rows = rows.sort_values("tipoff")
    return pd.merge_asof(rows, right.sort_values("tipoff"), on="tipoff", by=key,
                         allow_exact_matches=False, direction="backward")


def build(box, upcoming=None):
    """Feature rows for every played game (and `upcoming`, if given).

    `upcoming` needs: player_id, player, team, opponent, home_away, tipoff, season,
    plus any extra columns to carry through (e.g. the event id).
    """
    hist = box[box["played"]].copy()
    hist["upcoming"] = False
    rows = hist
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["upcoming"] = True
        up["played"] = True
        up["game_date"] = pd.to_datetime(up["tipoff"].dt.tz_convert("America/New_York").dt.date)
        rows = pd.concat([hist, up], ignore_index=True)
    rows["tipoff"] = _utc(rows["tipoff"])
    rows["starter_f"] = rows["starter"].astype(float)
    rows = _player_features(rows)

    teams = _team_games(box.assign(tipoff=_utc(box["tipoff"])))
    rows = _asof(rows, teams, "team", ["pts_10"])
    rows = _asof(rows, teams, "opponent", ["allowed_10", "pace_10"])
    rows = rows.rename(columns={"team_pts_10": "team_pts_10", "opponent_allowed_10": "opp_allowed_10",
                                "opponent_pace_10": "opp_pace_10"})
    return rows.sort_values(["tipoff", "player_id"]).reset_index(drop=True)


def eligible(rows):
    """Players a sportsbook would post a points line for: established rotation players."""
    return rows[(rows["games_prior"] >= 5) & (rows["min_10"] >= 15)]
