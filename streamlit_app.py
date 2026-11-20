"""NBA points experiment dashboard. Built for a phone: one column, key numbers first.

    streamlit run streamlit_app.py
"""
import io
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

import autobet
import data

st.set_page_config(page_title="NBA Points Experiment", page_icon="🏀", layout="centered")

RAW = "https://raw.githubusercontent.com/shannoj/nba_props_project/main/experiment/{}"
ET = ZoneInfo("America/New_York")


@st.cache_data(ttl=600, show_spinner=False)
def load_csv(name, columns):
    """Latest experiment file straight from GitHub (the Action commits there), else the local copy."""
    try:
        r = requests.get(RAW.format(name), timeout=10)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text), dtype={"event_id": str, "player_id": str, "quotes": str})
    except requests.RequestException:
        pass
    return autobet._read(autobet.EXP / name, columns)


@st.cache_data(ttl=3 * 3600, show_spinner="Loading box scores...")
def load_box():
    return data.load_box()


def result(row):
    if row["side"] == "none" or not row["settled"]:
        return ""
    if row["actual"] == row["line"]:
        return "push"
    return "✅" if (row["actual"] > row["line"]) == (row["side"] == "over") else "❌"


def bet_profit(rows):
    odds = rows["odds"].astype(float)
    won = (rows["actual"] > rows["line"]) == (rows["side"] == "over")
    push = rows["actual"] == rows["line"]
    return np.where(push, 0.0, np.where(won, autobet.STAKE * odds.map(autobet.payout_multiple), -autobet.STAKE))


log = load_csv("lines.csv", autobet.LOG_COLUMNS)
closing = load_csv("closing_lines.csv", autobet.CLOSE_COLUMNS)

st.title("🏀 NBA Points Experiment")
st.caption("Paper bets on sportsbook player points lines. $100 per bet when the model sees ≥ +3% expected value.")

if log.empty:
    st.info("No lines logged yet.")
    st.stop()

graded = autobet.grade_lines(log, load_box())
graded["result"] = graded.apply(result, axis=1)
res = autobet.evaluate(graded)
strat = res["strategies"]
bets = graded[graded["side"] != "none"]
settled = bets[bets["settled"]].copy()

# ---------------------------------------------------------------- headline
if not strat.empty:
    rule = strat.iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Profit", f"${rule['profit']:+,.0f}", f"{rule['roi']:+.1%} ROI")
    c2.metric("Record", f"{int(rule['wins'])}-{int(rule['losses'])}-{int(rule['pushes'])}",
              f"{rule['win_rate']:.1%} (need {rule['breakeven']:.1%})", delta_color="off")
st.markdown(autobet.verdict(res))
pending = int((~bets["settled"] & ~bets["void"]).sum())
st.caption(f"{len(graded)} lines logged · {len(settled)} bets settled · {pending} pending · "
           f"{int(graded['void'].sum())} void")

# ---------------------------------------------------------------- tonight
today = datetime.now(ET).strftime("%Y-%m-%d")
tonight = graded[(graded["game_date"] == today) & (graded["side"] != "none")]
st.subheader("Tonight's bets")
if tonight.empty:
    st.caption("Nothing logged for tonight yet. Games are logged up to 6 hours before tip-off.")
else:
    t = tonight.assign(
        bet=lambda d: d["side"].str.upper() + " " + d["line"].map("{:g}".format),
        edge=lambda d: np.maximum(d["ev_over"], d["ev_under"]),
    )[["player", "bet", "projection", "edge", "book"]]
    st.dataframe(t.sort_values("edge", ascending=False), hide_index=True, width="stretch", column_config={
        "player": "Player", "bet": "Bet", "projection": st.column_config.NumberColumn("Proj", format="%.1f"),
        "edge": st.column_config.NumberColumn("EV", format="percent"), "book": "Book"})

# ---------------------------------------------------------------- recent results
st.subheader("Recent results")
if settled.empty:
    st.caption("No settled bets yet. Results come in the morning after each game.")
else:
    days = sorted(settled["game_date"].unique(), reverse=True)
    day = st.selectbox("Night", days, label_visibility="collapsed",
                       format_func=lambda d: pd.Timestamp(d).strftime("%a %b %-d"))
    d = settled[settled["game_date"] == day].copy()
    d["profit"] = bet_profit(d)
    w, l = (d["result"] == "✅").sum(), (d["result"] == "❌").sum()
    st.markdown(f"**{w}-{l}, ${d['profit'].sum():+,.0f}**")
    d["bet"] = d["side"].str.upper() + " " + d["line"].map("{:g}".format)
    st.dataframe(d.sort_values(["result", "player"])[["result", "player", "bet", "projection", "actual"]],
                 hide_index=True, width="stretch", column_config={
                     "result": st.column_config.TextColumn("", width="small"), "player": "Player", "bet": "Bet",
                     "projection": st.column_config.NumberColumn("Proj", format="%.1f"),
                     "actual": st.column_config.NumberColumn("Actual", format="%d")})

    # Cumulative profit, flat vs. Kelly-sized
    s = settled.sort_values(["tipoff", "player"]).copy()
    s["Flat $100"] = bet_profit(s).cumsum()
    p = pd.Series(np.where(s["side"] == "over", s["p_over"], 1 - s["p_over"]), index=s.index)
    kelly = autobet.kelly_stakes(p, s["odds"].astype(float))
    s["Kelly-sized"] = (bet_profit(s) / autobet.STAKE * kelly).cumsum()
    st.subheader("Profit over time")
    st.line_chart(s[["Flat $100", "Kelly-sized"]].reset_index(drop=True), height=220)

# ---------------------------------------------------------------- details
if not strat.empty:
    acc = res["accuracy"]
    with st.expander("Strategies and controls"):
        st.caption("The model rule is the real test. The controls show what blind betting would have done.")
        sd = strat.reset_index(names="Strategy")
        sd["Record"] = sd["wins"].astype(int).astype(str) + "-" + sd["losses"].astype(int).astype(str)
        st.dataframe(sd[["Strategy", "Record", "profit", "roi"]], hide_index=True, width="stretch", column_config={
            "profit": st.column_config.NumberColumn("Profit", format="$%+.0f"),
            "roi": st.column_config.NumberColumn("ROI", format="percent")})

    with st.expander("Model vs. the market"):
        a, b = st.columns(2)
        a.metric("Model avg miss", f"{acc['model_mae']:.2f} pts")
        b.metric("Line avg miss", f"{acc['line_mae']:.2f} pts")
        if acc.get("prob_n"):
            a.metric("Brier (model)", f"{acc['model_brier']:.3f}")
            b.metric("Brier (market)", f"{acc['market_brier']:.3f}")
            st.metric("Probability skill vs. market", f"{acc['prob_skill']:+.1%}",
                      help="Positive means the model's over/under probabilities beat the sportsbook's (vig removed).")
        cs = autobet.clv_summary(autobet.closing_value(log, closing))
        if cs["n"]:
            st.metric("Closing line value", f"{cs['mean']:+.2f} pts", f"beat the close {cs['beat_pct']:.0%}",
                      delta_color="off", help="How far the line moved toward the bets before tip-off.")
        else:
            st.caption("No closing lines captured yet.")

    with st.expander("Calibration"):
        st.caption("When the model says a side hits X%, how often does it?")
        st.dataframe(res["calibration"], hide_index=True, width="stretch", column_config={
            "bucket": "Model chance", "lines": "Lines",
            "model_said": st.column_config.NumberColumn("Said", format="percent"),
            "actually_hit": st.column_config.NumberColumn("Hit", format="percent")})

with st.expander("All logged lines"):
    cols = ["game_date", "player", "team", "line", "projection", "p_over", "side", "actual", "result"]
    st.dataframe(graded.sort_values(["tipoff", "player"], ascending=[False, True])[cols], hide_index=True,
                 width="stretch", column_config={"p_over": st.column_config.NumberColumn("P(over)", format="percent")})
