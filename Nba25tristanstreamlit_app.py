# 24/25 NBA Playoff Predictor (CSV-only, playoff teams restricted)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import difflib
import plotly.express as px
import glob, os
from collections import deque

# ── PLAYOFF TEAMS & SEEDS ----------------------------------------------------
team_seeds = {
    # East
    "Cleveland Cavaliers": 1, "Boston Celtics": 2, "New York Knicks": 3,
    "Indiana Pacers": 4, "Milwaukee Bucks": 5, "Detroit Pistons": 6,
    "Orlando Magic": 7, "Atlanta Hawks": 8, "Chicago Bulls": 9,
    "Miami Heat": 10,
    # West
    "Oklahoma City Thunder": 1, "Houston Rockets": 2, "Los Angeles Lakers": 3,
    "Denver Nuggets": 4, "Los Angeles Clippers": 5,
    "Minnesota Timberwolves": 6, "Golden State Warriors": 7,
    "Memphis Grizzlies": 8, "Sacramento Kings": 9,
    "Dallas Mavericks": 10
}
PLAYOFF_TEAMS = list(team_seeds.keys())

# ── ALL CURRENT TEAMS (for fuzzy mapping) -----------------------------------
CURRENT_TEAMS = [
    "Atlanta Hawks","Boston Celtics","Brooklyn Nets","Charlotte Hornets",
    "Chicago Bulls","Cleveland Cavaliers","Dallas Mavericks","Denver Nuggets",
    "Detroit Pistons","Golden State Warriors","Houston Rockets","Indiana Pacers",
    "Los Angeles Clippers","Los Angeles Lakers","Memphis Grizzlies","Miami Heat",
    "Milwaukee Bucks","Minnesota Timberwolves","New Orleans Pelicans",
    "New York Knicks","Oklahoma City Thunder","Orlando Magic",
    "Philadelphia 76ers","Phoenix Suns","Portland Trail Blazers",
    "Sacramento Kings","San Antonio Spurs","Toronto Raptors",
    "Utah Jazz","Washington Wizards"
]

# ── LEGACY NAME MAP ---------------------------------------------------------
NAME_MAP = {
    "Seattle SuperSonics":"Oklahoma City Thunder",
    "New Jersey Nets":"Brooklyn Nets",
    "Charlotte Bobcats":"Charlotte Hornets",
    "New Orleans Hornets":"New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets":"New Orleans Pelicans",
    "NO/Oklahoma City Hornets":"New Orleans Pelicans",
    "LA Clippers":"Los Angeles Clippers"
}

# ── HELPERS ------------------------------------------------------------------
def fuzzy(name: str):
    if name in PLAYOFF_TEAMS or name in CURRENT_TEAMS:
        return name
    if name in NAME_MAP:
        return NAME_MAP[name]
    m = difflib.get_close_matches(name, PLAYOFF_TEAMS + CURRENT_TEAMS, 1, 0.7)
    return m[0] if m else None

prob_to_odds = lambda p: round(1/p, 2) if p > 0 else None
higher_lower = lambda a, b: (
    (a, b) if team_seeds[a] < team_seeds[b]
    else (b, a) if team_seeds[b] < team_seeds[a]
    else ((a, b) if np.random.rand() < .5 else (b, a))
)

def best_of_7_from_csv(row):
    w4, w5 = row["Prob_HigherSeed_WinIn4"], row["Prob_HigherSeed_WinIn5"]
    w6, w7 = row["Prob_HigherSeed_WinIn6"], row["Prob_HigherSeed_WinIn7"]
    return row["Prob_HigherSeed_SeriesWin"], (w4, w5, w6, w7)

# ── 1 · LOAD GAMES -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_games():
    folder = os.path.join(os.getcwd(), "AllNBAdata")
    files = sorted(glob.glob(os.path.join(folder, "NBA*-*.csv")),
                   key=lambda f: int(os.path.basename(f)[3:7]))
    if not files:
        return pd.DataFrame()
    df = pd.concat([
        pd.read_csv(f).rename(columns={
            "Visitor/Neutral":"Visitor", "Home/Neutral":"Home",
            "PTS":"VPTS", "PTS.1":"HPTS"
        })[["Date","Visitor","Home","VPTS","HPTS"]]
        for f in files
    ])
    df["Date"] = pd.to_datetime(df["Date"], format="%a %b %d %Y", errors="coerce")
    df = df[df["HPTS"].notna()].sort_values("Date")
    df["Season"] = df["Date"].apply(lambda d: d.year if d.month >= 10 else d.year - 1)
    df = df[df["Season"] != 2019]
    df["Home"] = df["Home"].map(fuzzy)
    df["Visitor"] = df["Visitor"].map(fuzzy)
    return df.dropna(subset=["Home","Visitor"]).reset_index(drop=True)

# ── 2 · BUILD ELO RATINGS ----------------------------------------------------
@st.cache_data(show_spinner=False)
def build_elos(raw: pd.DataFrame):
    teams = pd.unique(pd.concat([raw["Home"], raw["Visitor"]]))
    elo = {t: 1000 for t in teams}
    played = {t: 0 for t in teams}
    raw["home_elo_post"] = np.nan
    raw["visitor_elo_post"] = np.nan

    for i, r in raw.iterrows():
        h, v = r["Home"], r["Visitor"]
        pre_h, pre_v = elo[h], elo[v]
        exp_h = 1 / (1 + 10 ** ((pre_v - pre_h) / 400))
        home_win = r["HPTS"] > r["VPTS"]
        K_h = 20 if played[h] < 5 else 15 if played[h] < 10 else 12
        K_v = 20 if played[v] < 5 else 15 if played[v] < 10 else 12
        post_h = pre_h + K_h * ((1 if home_win else 0) - exp_h)
        post_v = pre_v + K_v * ((0 if home_win else 1) - (1 - exp_h))
        elo[h], elo[v] = post_h, post_v
        played[h] += 1
        played[v] += 1
        raw.at[i, "home_elo_post"] = post_h
        raw.at[i, "visitor_elo_post"] = post_v

    ts = []
    for _, r in raw.iterrows():
        ts.append({"Team": r["Home"],    "Date": r["Date"], "Elo": r["home_elo_post"]})
        ts.append({"Team": r["Visitor"], "Date": r["Date"], "Elo": r["visitor_elo_post"]})
    return raw, pd.DataFrame(ts)

# ── 3 · LOAD PRECOMPUTED CSVs ------------------------------------------------
@st.cache_data(show_spinner=False)
def load_matchups():
    df = pd.read_csv(os.path.join(os.getcwd(), "individual_matchups.csv"))
    df["Home"] = df["Home"].map(fuzzy)
    df["Away"] = df["Away"].map(fuzzy)
    return df.dropna(subset=["Home","Away"])

@st.cache_data(show_spinner=False)
def load_playoffs():
    df = pd.read_csv(os.path.join(os.getcwd(), "playoff_series_predictions.csv"))
    df["Higher_Seed"] = df["Higher_Seed"].map(fuzzy)
    df["Lower_Seed"]  = df["Lower_Seed"].map(fuzzy)
    return df.dropna(subset=["Higher_Seed","Lower_Seed"])

# ── 4 · STREAMLIT APP --------------------------------------------------------
def main():
    st.set_page_config(page_title="24/25 NBA Playoff Predictor", layout="wide")
    st.title("24/25 NBA Playoff Predictor")
    st.caption("Built by **Tristan Poul**")

    with st.spinner("⏳ Loading data, computing Elo, and loading CSVs…"):
        raw, elo_ts  = build_elos(load_games())
        if raw.empty:
            st.error("No game data found.")
            return
        matchups     = load_matchups()
        playoffs     = load_playoffs()
        cv_acc, hold_acc = 0.64, 0.64  # precomputed

    latest = elo_ts.sort_values("Date").groupby("Team")["Elo"].last()
    matchups["diff_elo"] = matchups.apply(
        lambda r: latest[r["Home"]] - latest[r["Away"]], axis=1
    )

    tab_play, tab_game, tab_model, tab_data, tab_info = st.tabs([
        "Playoffs Series Predictor",
        "Individual Game Predictions",
        "Model",
        "Data",
        "Info"
    ])

    # ── PLAYOFFS TAB ---------------------------------------------------------
    with tab_play:
        st.header("Playoffs Series Predictor")
        t1 = st.selectbox("Higher seed?", PLAYOFF_TEAMS, key="ps1")
        t2 = st.selectbox("Opponent", [t for t in PLAYOFF_TEAMS if t!=t1], key="ps2")
        hs, ls = higher_lower(t1, t2)
        dfp = playoffs[(playoffs["Higher_Seed"]==hs)&(playoffs["Lower_Seed"]==ls)]
        if dfp.empty:
            st.warning("No series prediction for that pairing.")
        else:
            row = dfp.iloc[0]
            total, (w4, w5, w6, w7) = best_of_7_from_csv(row)
            opp = 1 - total
            st.metric(f"{hs} Series Win %", f"{total:.1%}")
            st.metric(f"{ls} Series Win %", f"{opp:.1%}")

            probs = pd.DataFrame({
                hs: [row["Prob_HigherSeed_WinIn4"],
                     row["Prob_HigherSeed_WinIn5"],
                     row["Prob_HigherSeed_WinIn6"],
                     row["Prob_HigherSeed_WinIn7"]],
                ls: [row["Prob_LowerSeed_WinIn4"],
                     row["Prob_LowerSeed_WinIn5"],
                     row["Prob_LowerSeed_WinIn6"],
                     row["Prob_LowerSeed_WinIn7"]],
            }, index=["Win in 4","Win in 5","Win in 6","Win in 7"])
            st.subheader("Win-in-X distribution (%)")
            st.table(probs.style.format("{:.1%}"))
            st.subheader("Win-in-X no-vig decimal odds")
            st.table(probs.applymap(lambda x: prob_to_odds(x) if x>0 else None))

    # ── GAME TAB -------------------------------------------------------------
    with tab_game:
        st.header("Individual Game Predictions")
        home = st.selectbox("Home", CURRENT_TEAMS, key="g1")
        away = st.selectbox("Away", [t for t in CURRENT_TEAMS if t != home], key="g2")
        gm = matchups[(matchups["Home"] == home) & (matchups["Away"] == away)]
        if gm.empty:
            st.warning("No game prediction for that matchup.")
        else:
            row = gm.iloc[0]
            p_home = row["PredWinProb"]
            p_away = 1 - p_home
            if p_home >= p_away:
                winner, win_prob = home, p_home
            else:
                winner, win_prob = away, p_away
            st.subheader("Predicted winner")
            st.metric(label="Predicted winner", value=f"{winner} ({win_prob:.1%})")
            st.metric(label="No-vig odds",       value=f"{1/win_prob:.2f}")


    # ── MODEL TAB ------------------------------------------------------------
    with tab_model:
        st.header("Model Metrics & Interpretation")
        metrics = pd.DataFrame({
            "Metric": ["5-fold CV (train)", "Hold-out (test)"],
            "Accuracy": [f"{cv_acc:.2%}", f"{hold_acc:.2%}"]
        })
        st.table(metrics)
        st.markdown("""
I use a 300 tree `RandomForestClassifier` to predict a confidence level for the Home team to win.
The model is trained on a rolling window of games, excluding the first 5 games of each season.
The model is pre-processed with `StandardScaler` for robustness of each feature.
There is lots of room for improvement, off the top of my head I think I want to try `xgBoost`, as
some features just aren't giving the model much information. 
                    
The **5-fold cross-validation accuracy** of **64%** shows moderate generalization,  
and the **64% hold-out accuracy** confirms consistent performance on unseen data.  
These results indicate the Random-Forest captures key signals but has room for improvement  
through feature engineering or model tuning. I used a 95-5 train-test split, so the test data is about 
1350 data points. 
                    
The model uses 5 features:
                    
| Feature             | Definition                          | Rationale               |
|---------------------|-------------------------------------|-------------------------|
| diff_elo            | Home Elo − Away Elo                 | Raw skill gap           |
| home_recent_margin  | 5-game rolling avg scoring margin   | Momentum & fitness      |
| home_recent_wins    | Wins in last 5                      | Hot vs cold streaks     |
| win_pct_diff        | Season W% diff                      | Macro strength          |
| home_days_since_last| Rest days (capped at 30)            | Fatigue                 |
""")

    # ── DATA TAB -------------------------------------------------------------
    with tab_data:
        st.header("Elo Ratings Over Time")
        default = ["Minnesota Timberwolves","Oklahoma City Thunder","Washington Wizards"]
        picks = st.multiselect("Teams", CURRENT_TEAMS, default=default)
        if picks:
            df_ts = elo_ts[elo_ts["Team"].isin(picks)]
            st.plotly_chart(
                px.line(df_ts, x="Date", y="Elo", color="Team", template="plotly_white"),
                use_container_width=True
            )
        st.markdown("Most recent game-by-game Elo for all teams:")
        st.dataframe(raw.sort_values("Date", ascending=False)[
            ["Date","Home","Visitor","HPTS","VPTS","home_elo_post","visitor_elo_post"]
        ])

    # ── INFO TAB -------------------------------------------------------------
    with tab_info:
        st.header("Extra Info & Methodology")
        with st.expander("About the author"):
            st.markdown("""
**Tristan Poul** – Econ and Data Analysis Major at Santa Clara University.  
Feel free to reach me at tpoul@scu.edu or on [LinkedIn](https://www.linkedin.com/in/tristan-poul/)!
""")
        with st.expander("📚 1 · Data collection & cleaning"):
            st.markdown("""
* **Source:** Basketball-Reference, 2004-11-2 → 2025-4-30 (20 seasons).  
* **Exclusions:** 2019-20 “bubble” games (Season == 2019).  
* **Mapping:** legacy + fuzzy team-name mapping.
""")
        with st.expander("📈 2 · Dynamic Elo ratings"):
            st.markdown(r"""
* Teams start in the 04-05 season at Elo = 1000.  
* Off-season “compression”: ±3 around 1000, so they reset each new season at:
  * 1000 + (median rank - (last season rank)) * 3
  * This prevents drift from season to season, and keeps the elo somewhat centered, to account for offseason changes.
* K-factor: games 1–5→20; 6–10→15; 11+→12.
""")
        with st.expander("🛠️ 3 · Engineered features"):
            st.markdown("""
| Feature             | Definition                          | Rationale               |
|---------------------|-------------------------------------|-------------------------|
| diff_elo            | Home Elo − Away Elo                 | Raw skill gap           |
| home_recent_margin  | 5-game rolling avg scoring margin   | Momentum & fitness      |
| home_recent_wins    | Wins in last 5                      | Hot vs cold streaks     |
| win_pct_diff        | Season W% diff                      | Macro strength          |
| home_days_since_last| Rest days (capped at 30)            | Fatigue                 |
| *visitor analogues* | … analogous for visitor             |                         |
""")
        with st.expander("🤖 4 · Random-Forest winner model"):
            st.markdown(f"""
* **Model:** `RandomForestClassifier` with 300 trees.  
* **Pre-processing:** `StandardScaler`.  
* **Training window:** excludes first 5 games.  
* **Validation:** 5-fold CV ≈ **{cv_acc:.1%}**, hold-out ≈ **{hold_acc:.1%}**.  
* **Label:** `home_win` (1 = home victory).
* **Train-Test Split:** 95% train, 5% test.
""")
        with st.expander("🆚 5 · Match-up grid & no-vig odds"):
            st.markdown("""
* All ordered pairs precomputed in CSV.  
* WinProb ≥ .5 ⇒ favourite.  
* No-vig odds = 1 / WinProb.
""")
        with st.expander("🏆 6 · 2-2-1-1-1 series simulator"):
            st.markdown(r"""
* Higher seed hosts G1-2,5,7.  
* Series probabilities precomputed via N-choose-K.
""")
        with st.expander("🚧 7 · Limitations / future work"):
            st.markdown("""
* No injuries/lineups/travel data.  
* Elo ignores off-season churn.  
* Probabilities not market-calibrated.  
* Future: Bayesian Elo, gradient boosting, injury scraping.
""")

if __name__ == "__main__":
    main()
