"""Paper-trading experiment: does the model beat sportsbook NBA player points props?

Runs hourly through the NBA day (GitHub Action). For each game tipping off in
the next LOOKAHEAD_HOURS that hasn't been logged yet, it pulls every player's
points line from The Odds API, projects that player, and records the line, the
model's probability and a hypothetical STAKE bet whenever the better side's
expected value is at least MIN_EV. Each game is logged once, a few hours before
tip (after most injury news); nothing is edited afterwards, so the log is an
honest out-of-sample record. Every line is logged, bet or not, so controls and
other strategies can be judged on the same data.

The same run snapshots lines for logged games tipping off within the next
CLOSE_WINDOW_MINUTES, to measure closing line value (CLV).

Odds API cost: one credit per game to log + one for the closing snapshot.

    ODDS_API_KEY=... python autobet.py
"""
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest

import data
import features as F
import model as points_model

API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKET = "player_points"
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "betrivers", "fanatics"]  # first = primary
LOOKAHEAD_HOURS = 3
CLOSE_WINDOW_MINUTES = 60
STAKE = 100.0
MIN_EV = 0.03  # the pre-registered betting rule: bet only with >= +3% expected return
KELLY_CAP = 3.0  # shadow Kelly strategy: no bet larger than about 3x the average stake
MIN_BETS_FOR_VERDICT = 300

EXP = Path(__file__).parent / "experiment"
LOG_PATH, CLOSE_PATH = EXP / "lines.csv", EXP / "closing_lines.csv"
LOG_COLUMNS = [
    "logged_at", "event_id", "tipoff", "game_date", "player_id", "player", "team", "opponent",
    # Quote used for the model's side after line shopping (best expected value across books)
    "book", "line", "over_odds", "under_odds", "projection", "p_over", "ev_over", "ev_under",
    "side", "odds", "stake", "min_10", "model_trained_through",
    # Single-book reference (first of BOOKS quoting the player), to measure what shopping adds
    "primary_book", "primary_line", "primary_over_odds", "primary_under_odds", "primary_p_over",
    "consensus_line", "n_books", "quotes",
]
CLOSE_COLUMNS = ["captured_at", "event_id", "player_id", "player", "tipoff", "minutes_before",
                 "consensus_line", "n_books", "quotes"]


# ---------------------------------------------------------------- odds math

def implied_prob(odds):
    """Break-even win probability for American odds."""
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def payout_multiple(odds):
    """Profit per $1 staked on a win."""
    return 100 / -odds if odds < 0 else odds / 100


def expected_value(p, odds):
    return p * payout_multiple(odds) - (1 - p)


# ---------------------------------------------------------------- storage

def _read(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype={"event_id": str, "player_id": str, "quotes": str})


def load_log():
    return _read(LOG_PATH, LOG_COLUMNS)


def load_closing():
    return _read(CLOSE_PATH, CLOSE_COLUMNS)


def _append(path, columns, rows):
    if not rows:
        return
    EXP.mkdir(exist_ok=True)
    old = _read(path, columns)
    new = pd.DataFrame(rows, columns=columns)
    (new if old.empty else pd.concat([old, new], ignore_index=True)).to_csv(path, index=False)


def _quotes_str(quotes):
    return "|".join(f"{b}:{l:g}:{int(o)}:{int(u)}" for b, l, o, u in quotes)


def parse_quotes(text):
    """'draftkings:24.5:-110:-110|fanduel:...' -> {book: (line, over_odds, under_odds)}."""
    out = {}
    for part in str(text).split("|"):
        if part.count(":") == 3:
            b, l, o, u = part.split(":")
            out[b] = (float(l), int(o), int(u))
    return out


# ---------------------------------------------------------------- Odds API

class OddsAPI:
    def __init__(self, key):
        self.key, self.remaining = key, None

    def _get(self, url, **params):
        r = requests.get(url, params={"apiKey": self.key, **params}, timeout=20)
        r.raise_for_status()
        self.remaining = r.headers.get("x-requests-remaining", self.remaining)
        return r.json()

    def events(self):
        return self._get(f"{API}/events")  # free: costs no credits

    def props(self, event_id):
        # Up to 10 bookmakers cost the same single credit as one.
        return self._get(f"{API}/events/{event_id}/odds", regions="us", markets=MARKET,
                         oddsFormat="american", bookmakers=",".join(BOOKS))


def all_book_quotes(event_odds):
    """{normalized player name: [(book, line, over_odds, under_odds), ...]} in BOOKS order."""
    by_book = {b["key"]: b for b in event_odds.get("bookmakers", [])}
    found = {}
    for key in BOOKS:
        book = by_book.get(key)
        if not book:
            continue
        sides = {}
        for market in book.get("markets", []):
            if market["key"] != MARKET:
                continue
            for o in market["outcomes"]:
                sides.setdefault(norm(o["description"]), {})[o["name"].lower()] = (o["point"], o["price"])
        for name, s in sides.items():
            if "over" in s and "under" in s and s["over"][0] == s["under"][0]:
                found.setdefault(name, []).append((key, float(s["over"][0]), int(s["over"][1]), int(s["under"][1])))
    return found


def _tipoff(ev):
    return datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))


# ---------------------------------------------------------------- players

def norm(name):
    """'Jaren Jackson Jr.' -> 'jaren jackson'."""
    name = re.sub(r"[^a-z ]", "", str(name).lower().replace("-", " "))
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


def rosters(box):
    """Latest team for every player seen this season (traded players count for their new team)."""
    cur = box[box["season"] == box["season"].max()]
    latest = cur.sort_values("tipoff").groupby("player_id").tail(1)
    out = latest[["player_id", "player", "team"]].copy()
    out["key"] = out["player"].map(norm)
    return out


def match_player(name, roster):
    """Find the book's player name on the event's rosters: exact, else same last name + first initial."""
    key = norm(name)
    hit = roster[roster["key"] == key]
    if len(hit) == 1:
        return hit.iloc[0]
    parts = key.split()
    if len(parts) >= 2:
        last, first = parts[-1], parts[0][0]
        hit = roster[roster["key"].str.split().str[-1].eq(last) & roster["key"].str[0].eq(first)]
        if len(hit) == 1:
            return hit.iloc[0]
    return None


def find_quotes(quotes_by_player, player):
    """Quotes for a known player name (same matching rules as match_player)."""
    one = pd.DataFrame([{"key": norm(player)}])
    return next((q for n, q in quotes_by_player.items() if match_player(n, one) is not None), None)


# ---------------------------------------------------------------- daily run

def decide(meta, projection, line, over_odds, under_odds):
    p_over = points_model.prob_over(meta, projection, line)
    ev_over, ev_under = expected_value(p_over, over_odds), expected_value(1 - p_over, under_odds)
    side, odds = "none", np.nan
    if max(ev_over, ev_under) >= MIN_EV:
        side, odds = ("over", over_odds) if ev_over >= ev_under else ("under", under_odds)
    return p_over, ev_over, ev_under, side, odds


def shop(meta, projection, quotes):
    """The quote with the best expected value for whichever side the model prefers."""
    best = {}
    for q in quotes:
        _, ev_o, ev_u, _, _ = decide(meta, projection, *q[1:])
        for side, ev in (("over", ev_o), ("under", ev_u)):
            if side not in best or ev > best[side][0]:
                best[side] = (ev, q)
    side = "over" if best["over"][0] >= best["under"][0] else "under"
    q = best[side][1]
    return q, decide(meta, projection, *q[1:])


def log_lines(api, events, box, model, meta, now):
    log = load_log()
    done = set(log["event_id"])
    todo = [ev for ev in events if ev["id"] not in done
            and now < _tipoff(ev) <= now + timedelta(hours=LOOKAHEAD_HOURS)]
    if not todo:
        print("No new games in the window.")
        return
    roster = rosters(box)
    season = int(box["season"].max())
    rows = []
    for ev in todo:
        quotes_by_player = all_book_quotes(api.props(ev["id"]))
        if not quotes_by_player:
            print(f"  {ev['away_team']} @ {ev['home_team']}: no lines posted yet, will retry next run")
            continue
        teams = {ev["home_team"], ev["away_team"]}
        game_roster = roster[roster["team"].isin(teams)]
        up = []
        for name, quotes in quotes_by_player.items():
            p = match_player(name, game_roster)
            if p is None:
                print(f"  unmatched player: {name} ({ev['away_team']} @ {ev['home_team']})")
                continue
            opp = (teams - {p["team"]}).pop()
            up.append({"player_id": p["player_id"], "player": p["player"], "team": p["team"], "opponent": opp,
                       "home_away": "home" if p["team"] == ev["home_team"] else "away",
                       "tipoff": pd.Timestamp(_tipoff(ev)), "season": season, "game_id": ev["id"],
                       "quotes": quotes})
        if not up:
            continue
        up = pd.DataFrame(up)
        feats = F.build(box, up.drop(columns="quotes"))
        feats = feats[feats["upcoming"]].set_index("player_id")
        feats["projection"] = model.predict(feats)
        for _, u in up.iterrows():
            f = feats.loc[u["player_id"]]
            quotes = u["quotes"]
            (book, line, oo, uo), (p_over, ev_o, ev_u, side, odds) = shop(meta, f["projection"], quotes)
            primary = quotes[0]
            rows.append({
                "logged_at": now.isoformat(timespec="seconds"), "event_id": ev["id"],
                "tipoff": ev["commence_time"], "game_date": u["tipoff"].tz_convert("America/New_York").strftime("%Y-%m-%d"),
                "player_id": u["player_id"], "player": u["player"], "team": u["team"], "opponent": u["opponent"],
                "book": book, "line": line, "over_odds": oo, "under_odds": uo,
                "projection": round(float(f["projection"]), 2), "p_over": round(p_over, 4),
                "ev_over": round(ev_o, 4), "ev_under": round(ev_u, 4), "side": side, "odds": odds,
                "stake": STAKE if side != "none" else 0.0, "min_10": round(float(f["min_10"]), 1),
                "model_trained_through": meta["trained_through"],
                "primary_book": primary[0], "primary_line": primary[1], "primary_over_odds": primary[2],
                "primary_under_odds": primary[3],
                "primary_p_over": round(points_model.prob_over(meta, f["projection"], primary[1]), 4),
                "consensus_line": float(np.median([q[1] for q in quotes])), "n_books": len(quotes),
                "quotes": _quotes_str(quotes),
            })
            print(f"  {u['player']:24s} {book:10s} line {line:5.1f}  proj {f['projection']:5.1f}  "
                  f"P(over) {p_over:.0%}  -> {side}")
    _append(LOG_PATH, LOG_COLUMNS, rows)
    print(f"Logged {len(rows)} lines ({sum(r['side'] != 'none' for r in rows)} bets).")


def log_closing(api, events, now):
    """Snapshot lines for logged games tipping off soon (once per game)."""
    log, closing = load_log(), load_closing()
    done = set(closing["event_id"])
    logged = set(log["event_id"])
    due = [ev for ev in events if ev["id"] in logged and ev["id"] not in done
           and now < _tipoff(ev) <= now + timedelta(minutes=CLOSE_WINDOW_MINUTES)]
    rows = []
    for ev in due:
        quotes_by_player = all_book_quotes(api.props(ev["id"]))
        minutes = round((_tipoff(ev) - now).total_seconds() / 60)
        for _, r in log[log["event_id"] == ev["id"]].iterrows():
            quotes = find_quotes(quotes_by_player, r["player"])
            if not quotes:
                continue
            rows.append({"captured_at": now.isoformat(timespec="seconds"), "event_id": ev["id"],
                         "player_id": r["player_id"], "player": r["player"], "tipoff": ev["commence_time"],
                         "minutes_before": minutes, "consensus_line": float(np.median([q[1] for q in quotes])),
                         "n_books": len(quotes), "quotes": _quotes_str(quotes)})
    _append(CLOSE_PATH, CLOSE_COLUMNS, rows)
    if due:
        print(f"Captured {len(rows)} closing lines for {len(due)} games.")


def run(api_key, now=None):
    now = now or datetime.now(timezone.utc)
    api = OddsAPI(api_key)
    events = api.events()
    log_closing(api, events, now)
    soon = [ev for ev in events if now < _tipoff(ev) <= now + timedelta(hours=LOOKAHEAD_HOURS)]
    if soon and not set(ev["id"] for ev in soon) <= set(load_log()["event_id"]):
        box = data.load_box()
        model, meta = points_model.load()
        log_lines(api, events, box, model, meta, now)
    else:
        print("No new games in the window.")
    print(f"Odds API credits remaining: {api.remaining}")


# ---------------------------------------------------------------- evaluation

def grade_lines(log, box):
    """Add `actual` points, `settled` and `void` flags.

    A line settles once the player's team has a box score for that date. If the
    player has no minutes (DNP, inactive), sportsbooks void the bet: it's excluded.
    """
    out = log.copy()
    if out.empty:
        return out.assign(actual=[], settled=[], void=[])
    box = box.assign(day=box["game_date"].dt.strftime("%Y-%m-%d"))
    team_days = set(zip(box["team"], box["day"]))
    played = box[box["played"]].set_index(["player_id", "day"])["points"]
    played = played[~played.index.duplicated()]
    out["actual"] = [played.get(k) for k in zip(out["player_id"], out["game_date"])]
    final = np.array([k in team_days for k in zip(out["team"], out["game_date"])], dtype=bool)
    out["void"] = final & out["actual"].isna()
    out["settled"] = final & out["actual"].notna()
    return out


def kelly_stakes(p, odds, cap=KELLY_CAP):
    """Stakes proportional to the Kelly fraction (EV / payout), scaled to average STAKE.

    Scaling uses only probabilities and odds, never outcomes, so total money staked
    matches the flat rule and profit is directly comparable.
    """
    b = odds.map(payout_multiple)
    f = ((p * b - (1 - p)) / b).clip(lower=0)
    if len(f) == 0 or f.sum() == 0:
        return pd.Series(STAKE, index=p.index)
    stakes = (f / f.mean()).clip(upper=cap)
    return STAKE * stakes / stakes.mean()


def _strategy(rows, line, sides, odds, stakes=None):
    """Results of betting `sides` at `line`/`odds`; pushes excluded from W/L."""
    stakes = pd.Series(STAKE, index=rows.index) if stakes is None else stakes
    over_hit = rows["actual"] > line
    push = rows["actual"] == line
    win = (over_hit == (sides == "over")) & ~push
    profit = np.where(push, 0.0, np.where(win, stakes * odds.map(payout_multiple), -stakes))
    n, wins = int((~push).sum()), int(win.sum())
    breakeven = float(odds.map(implied_prob).mean()) if len(odds) else np.nan
    staked = float(stakes.sum())
    return {
        "bets": len(rows), "wins": wins, "losses": n - wins, "pushes": int(push.sum()),
        "win_rate": wins / n if n else np.nan, "breakeven": breakeven,
        "profit": float(profit.sum()), "roi": float(profit.sum() / staked) if staked else np.nan,
        # Chance of doing at least this well with no real edge (coin flips at break-even odds)
        "p_luck": binomtest(wins, n, breakeven, alternative="greater").pvalue if n else np.nan,
    }


def _ev(p, odds):
    return p * odds.map(payout_multiple) - (1 - p)


def no_vig_p_over(over_odds, under_odds):
    po, pu = over_odds.map(implied_prob), under_odds.map(implied_prob)
    return po / (po + pu)


def probability_scores(s):
    """Brier score and log loss of the model's P(over) vs. the no-vig market's (primary book).

    Lower is better; `prob_skill` > 0 means the model's probabilities beat the market's.
    """
    s = s[s["actual"] != s["primary_line"]]
    if s.empty:
        return {"prob_n": 0}
    y = (s["actual"] > s["primary_line"]).astype(float).to_numpy()
    model = s["primary_p_over"].astype(float).clip(0.01, 0.99).to_numpy()
    market = no_vig_p_over(s["primary_over_odds"], s["primary_under_odds"]).to_numpy()

    def brier(p):
        return float(np.mean((p - y) ** 2))

    def log_loss(p):
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return {"prob_n": int(len(y)), "model_brier": brier(model), "market_brier": brier(market),
            "model_log_loss": log_loss(model), "market_log_loss": log_loss(market),
            "prob_skill": 1 - brier(model) / brier(market)}


def evaluate(graded):
    """Strategy comparison, model-vs-line accuracy, probability scores and calibration."""
    s = graded[graded["settled"]].copy() if len(graded) else graded
    if s.empty:
        return {"strategies": pd.DataFrame(), "accuracy": {}, "calibration": pd.DataFrame(), "settled": s}
    for c in ("primary_over_odds", "primary_under_odds", "odds", "over_odds", "under_odds"):
        s[c] = s[c].astype(float)

    line1, over1, under1 = s["primary_line"], s["primary_over_odds"], s["primary_under_odds"]
    ev_o1, ev_u1 = _ev(s["primary_p_over"], over1), _ev(1 - s["primary_p_over"], under1)
    side1 = pd.Series(np.where(ev_o1 >= ev_u1, "over", "under"), index=s.index)
    odds1 = pd.Series(np.where(side1 == "over", over1, under1), index=s.index)
    best1 = np.maximum(ev_o1, ev_u1)

    rows = {}
    rule = s[s["side"] != "none"]
    rows[f"Model rule (EV ≥ {MIN_EV:.0%}), best price — the real test"] = _strategy(
        rule, rule["line"], rule["side"], rule["odds"])
    p_rule = pd.Series(np.where(rule["side"] == "over", rule["p_over"], 1 - rule["p_over"]), index=rule.index)
    kelly = _strategy(rule, rule["line"], rule["side"], rule["odds"], kelly_stakes(p_rule, rule["odds"]))
    kelly["p_luck"] = np.nan  # same bets as the rule: the luck test is identical
    rows["Shadow: model rule, Kelly-sized (same total stake)"] = kelly
    r1 = best1 >= MIN_EV
    rows["Model rule, one book (no line shopping)"] = _strategy(s[r1], line1[r1], side1[r1], odds1[r1])
    rows["Model's side on every line (one book)"] = _strategy(s, line1, side1, odds1)
    rows["Control: always over"] = _strategy(s, line1, pd.Series("over", index=s.index), over1)
    rows["Control: always under"] = _strategy(s, line1, pd.Series("under", index=s.index), under1)
    strategies = pd.DataFrame(rows).T

    accuracy = {
        "n": len(s),
        "model_mae": float((s["actual"] - s["projection"]).abs().mean()),
        "line_mae": float((s["actual"] - s["consensus_line"]).abs().mean()),
        "model_closer_pct": float(((s["actual"] - s["projection"]).abs()
                                   < (s["actual"] - s["consensus_line"]).abs()).mean()),
        **probability_scores(s),
    }

    model_side = np.where(s["ev_over"] >= s["ev_under"], "over", "under")
    p_side = np.where(model_side == "over", s["p_over"], 1 - s["p_over"])
    hit = np.where(model_side == "over", s["actual"] > s["line"], s["actual"] < s["line"])
    cal = pd.DataFrame({"p": p_side, "hit": hit})
    cal["bucket"] = pd.cut(cal["p"], [0.5, 0.53, 0.56, 0.6, 0.65, 1.0], include_lowest=True)
    calibration = (cal.groupby("bucket", observed=True)
                   .agg(lines=("hit", "size"), model_said=("p", "mean"), actually_hit=("hit", "mean"))
                   .reset_index())
    calibration["bucket"] = calibration["bucket"].astype(str)
    return {"strategies": strategies, "accuracy": accuracy, "calibration": calibration, "settled": s}


def closing_value(log, closing):
    """Per-bet closing line value: points the line moved toward the bet after it was logged."""
    bets = log[log["side"] != "none"]
    if bets.empty or closing.empty:
        return pd.DataFrame()
    c = closing[["event_id", "player_id", "consensus_line", "quotes"]].rename(
        columns={"consensus_line": "close_consensus", "quotes": "close_quotes"})
    m = bets.merge(c, on=["event_id", "player_id"], how="inner")
    same = [parse_quotes(q).get(b) for q, b in zip(m["close_quotes"], m["book"])]
    m["close_line"] = [s[0] if s else cc for s, cc in zip(same, m["close_consensus"])]
    m["clv"] = np.where(m["side"] == "over", m["close_line"] - m["line"], m["line"] - m["close_line"])
    return m


def clv_summary(clv):
    if clv.empty:
        return {"n": 0}
    x = clv["clv"]
    return {"n": int(len(x)), "mean": float(x.mean()),
            "ci95": float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            "beat_pct": float((x > 0).mean()), "same_pct": float((x == 0).mean()),
            "worse_pct": float((x < 0).mean())}


def verdict(res):
    strat, s = res["strategies"], res["settled"]
    n = int((s["side"] != "none").sum()) if len(s) else 0
    if strat.empty or n < MIN_BETS_FOR_VERDICT:
        return (f"**Too early to tell.** {n} settled bets so far (verdict starts at {MIN_BETS_FOR_VERDICT}). "
                "With 300 bets the win rate is known to about ±6%.")
    rule, acc = strat.iloc[0], res["accuracy"]
    if rule["roi"] > 0 and rule["p_luck"] < 0.05 and acc.get("prob_skill", 0) > 0:
        return (f"**Promising.** The model rule is up {rule['roi']:+.1%} over {n} bets, only a {rule['p_luck']:.1%} "
                "chance of that by luck, and its probabilities beat the market's.")
    if rule["roi"] > 0:
        return (f"**Profitable but not proven.** Up {rule['roi']:+.1%} over {n} bets, with a {rule['p_luck']:.0%} "
                "chance of doing this well with no real edge.")
    return f"**No edge so far.** The model rule is {rule['roi']:+.1%} over {n} bets. Don't bet real money on it."


if __name__ == "__main__":
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise SystemExit("ODDS_API_KEY not set. Get a free key at https://the-odds-api.com")
    run(key)
