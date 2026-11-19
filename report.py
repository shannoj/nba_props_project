"""Weekly experiment report as Markdown (posted as a GitHub issue by the workflow).

    python report.py report.md     # writes the report, prints the issue title last
"""
import sys
from datetime import date

import pandas as pd

import autobet
import data


def _table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt[c](r[c]) if c in fmt and pd.notna(r[c]) else ("—" if pd.isna(r[c]) else str(r[c]))
                                     for c in cols) + " |")
    return "\n".join(out)


def build():
    log = autobet.load_log()
    if log.empty:
        return None, None
    graded = autobet.grade_lines(log, data.load_box())
    res = autobet.evaluate(graded)
    title = f"NBA points experiment report: {date.today():%Y-%m-%d}"
    lines = [f"# {title}", "", autobet.verdict(res), "",
             f"{len(graded)} lines logged, {int(graded['settled'].sum())} settled, "
             f"{int(graded['void'].sum())} void (player didn't play).", ""]

    strat = res["strategies"]
    if not strat.empty:
        s = strat.reset_index(names="Strategy")
        s["Record"] = (s["wins"].astype(int).astype(str) + "-" + s["losses"].astype(int).astype(str)
                       + "-" + s["pushes"].astype(int).astype(str))
        lines += ["## Strategies on the same lines", "", _table(
            s[["Strategy", "Record", "win_rate", "breakeven", "profit", "roi", "p_luck"]].rename(columns={
                "win_rate": "Win rate", "breakeven": "Break-even", "profit": "Profit", "roi": "ROI",
                "p_luck": "Chance it's luck"}),
            {"Win rate": "{:.1%}".format, "Break-even": "{:.1%}".format, "Profit": "${:+,.0f}".format,
             "ROI": "{:+.1%}".format, "Chance it's luck": "{:.0%}".format}), ""]

        acc = res["accuracy"]
        lines += [f"**Model vs. line** ({acc['n']} settled lines): the projection missed by {acc['model_mae']:.2f} pts "
                  f"on average vs. {acc['line_mae']:.2f} for the line, and was closer {acc['model_closer_pct']:.0%} "
                  "of the time.", ""]
        if acc.get("prob_n"):
            lines += [f"**Probability accuracy** (lower is better): Brier {acc['model_brier']:.3f} model vs. "
                      f"{acc['market_brier']:.3f} market (no-vig); log loss {acc['model_log_loss']:.3f} vs. "
                      f"{acc['market_log_loss']:.3f}. Skill vs. market: **{acc['prob_skill']:+.1%}**.", ""]
        cs = autobet.clv_summary(autobet.closing_value(log, autobet.load_closing()))
        if cs["n"]:
            lines += [f"**Closing line value** ({cs['n']} bets): the line moved **{cs['mean']:+.2f} pts** toward the "
                      f"bets on average (± {cs['ci95']:.2f}), beating the close {cs['beat_pct']:.0%} of the time.", ""]
        lines += ["**Calibration** (when the model says a side hits X%, how often it did):", "",
                  _table(res["calibration"].rename(columns={"bucket": "Model chance", "lines": "Lines",
                                                            "model_said": "Model said", "actually_hit": "Actually hit"}),
                         {"Model said": "{:.0%}".format, "Actually hit": "{:.0%}".format}), ""]
    return title, "\n".join(lines)


if __name__ == "__main__":
    title, body = build()
    if body is None:
        print("Nothing to report yet.")
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else "report.md"
        with open(path, "w") as f:
            f.write(body)
        print(title)
