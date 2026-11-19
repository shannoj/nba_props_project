"""NBA player box scores from ESPN, via the sportsdataverse data releases.

These are plain parquet files on GitHub, refreshed daily, so they download fine
from GitHub Actions (stats.nba.com often blocks cloud servers). ESPN labels a
season by the year it ends: 2027 is the 2026-27 season.
"""
from datetime import date

import pandas as pd

URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
       "espn_nba_player_boxscores/player_box_{season}.parquet")
FIRST_SEASON = 2022
GAME_TYPES = {2, 3, 5}  # regular season, playoffs, play-in (1 = preseason)

# ESPN display name -> name The Odds API uses, where they differ.
ODDS_TEAM_NAME = {"LA Clippers": "Los Angeles Clippers"}

COLUMNS = [
    "game_id", "season", "season_type", "game_date", "game_date_time", "athlete_id",
    "athlete_display_name", "team_display_name", "opponent_team_display_name", "home_away",
    "minutes", "points", "field_goals_attempted", "three_point_field_goals_attempted",
    "free_throws_attempted", "rebounds", "assists", "turnovers", "starter", "did_not_play",
    "reason", "team_score", "opponent_team_score",
]


def current_season(today=None):
    today = today or date.today()
    return today.year + 1 if today.month >= 8 else today.year


def load_season(season):
    df = pd.read_parquet(URL.format(season=season), columns=COLUMNS)
    return df[df["season_type"].isin(GAME_TYPES)]


def load_box(through=None):
    """Every player-game from FIRST_SEASON through the current season.

    Columns are renamed to short names; `team` and `opponent` use The Odds API's
    team names so lines can be matched directly. Exhibition games (All-Star etc.)
    are dropped.
    """
    seasons = range(FIRST_SEASON, (through or current_season()) + 1)
    df = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    df = df.rename(columns={
        "athlete_id": "player_id", "athlete_display_name": "player", "team_display_name": "team",
        "opponent_team_display_name": "opponent", "field_goals_attempted": "fga",
        "three_point_field_goals_attempted": "tpa", "free_throws_attempted": "fta",
        "game_date_time": "tipoff",
    })
    df["team"] = df["team"].replace(ODDS_TEAM_NAME)
    df["opponent"] = df["opponent"].replace(ODDS_TEAM_NAME)
    teams = set(df.loc[df["season_type"] == 3, "team"])  # playoff teams are always real teams
    teams |= set(df["team"].value_counts().loc[lambda c: c > 500].index)
    df = df[df["team"].isin(teams) & df["opponent"].isin(teams)]
    df["player_id"] = df["player_id"].astype(str)
    df["game_id"] = df["game_id"].astype(str)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["played"] = ~df["did_not_play"].astype(bool) & (df["minutes"].fillna(0) > 0)
    return df.sort_values(["tipoff", "game_id", "player_id"]).reset_index(drop=True)
