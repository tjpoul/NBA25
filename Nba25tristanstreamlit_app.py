import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from collections import deque
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

@st.cache_data
def load_data():
    # Load and concatenate season CSV files
    file_pattern = os.path.join(os.getcwd(), "NBA*-*.csv")
    file_list = glob.glob(file_pattern)
    file_list.sort(key=lambda f: int(os.path.basename(f).replace("NBA", "").replace(".csv", "")[:4]))
    df_list = []
    for f in file_list:
        df = pd.read_csv(f)
        # Rename columns to standard names
        df = df.rename(columns={"Visitor/Neutral": "Visitor",
                                 "Home/Neutral": "Home",
                                 "PTS": "VPTS",
                                 "PTS.1": "HPTS"})
        df = df[["Date", "Visitor", "Home", "VPTS", "HPTS"]]
        df_list.append(df)
    full25 = pd.concat(df_list, ignore_index=True)
    full25['Date_clean'] = pd.to_datetime(full25['Date'], format='%a %b %d %Y', errors='coerce')
    full25 = full25.sort_values('Date_clean')
    full25 = full25[full25['HPTS'].notna()]
    full25['Season'] = full25['Date_clean'].apply(lambda d: d.year if d.month >= 10 else d.year - 1)
    # Remove lockout-shortened 2019-20 season
    full25 = full25[full25['Season'] != 2019]
    # Initialize columns
    cols = ['visitor_elo_pre','home_elo_pre','visitor_elo_post','home_elo_post',
            'visitor_win_pct','home_win_pct','visitor_game_num','home_game_num',
            'home_days_since_last','visitor_days_since_last','home_recent_margin',
            'visitor_recent_margin','home_recent_wins','visitor_recent_wins','home_win']
    for c in cols:
        full25[c] = np.nan
    return full25

@st.cache_data
def compute_elo(full25):
    teams = pd.unique(pd.concat([full25['Home'], full25['Visitor']]))
    # Starting Elo
    elo_vec = {t: 1000 for t in teams}
    games_played = {t: 0 for t in teams}
    season_game_count = {}
    season_wins = {}
    last_game_date = {t: None for t in teams}
    recent_margins = {t: deque(maxlen=5) for t in teams}
    recent_wins = {t: deque(maxlen=5) for t in teams}
    current_season = full25.iloc[0]['Season']
    def get_K(gp):
        if gp < 5:
            return 20
        elif gp < 10:
            return 15
        else:
            return 12

    for i, game in full25.iterrows():
        s = game['Season']; date = game['Date_clean']
        # Season reset
        if s != current_season:
            sorted_teams = sorted(elo_vec, key=lambda t: elo_vec[t], reverse=True)
            total = len(sorted_teams)
            for rank, t in enumerate(sorted_teams, start=1):
                elo_vec[t] = 1000 + ((total/2) - rank) * 3
            season_game_count = {}
            season_wins = {}
            current_season = s
        v, h = game['Visitor'], game['Home']
        vkey, hkey = (s, v), (s, h)
        pre_v = season_game_count.get(vkey, 0); pre_h = season_game_count.get(hkey, 0)
        winpct_v = season_wins.get(vkey, 0)/pre_v if pre_v > 0 else 0.5
        winpct_h = season_wins.get(hkey, 0)/pre_h if pre_h > 0 else 0.5
        full25.at[i, 'visitor_win_pct'] = winpct_v; full25.at[i, 'home_win_pct'] = winpct_h
        season_game_count[vkey] = pre_v + 1; season_game_count[hkey] = pre_h + 1
        full25.at[i, 'visitor_game_num'] = pre_v + 1; full25.at[i, 'home_game_num'] = pre_h + 1
        # Days since last game
        d_h = 7 if last_game_date[h] is None else (date - last_game_date[h]).days
        d_v = 7 if last_game_date[v] is None else (date - last_game_date[v]).days
        full25.at[i, 'home_days_since_last'] = min(d_h, 30)
        full25.at[i, 'visitor_days_since_last'] = min(d_v, 30)
        last_game_date[h], last_game_date[v] = date, date
        # Record pre-game Elo
        full25.at[i, 'visitor_elo_pre'] = elo_vec[v]
        full25.at[i, 'home_elo_pre'] = elo_vec[h]
        # Update Elo
        K_v, K_h = get_K(games_played[v]), get_K(games_played[h])
        exp_v = 1 / (1 + 10 ** ((elo_vec[h] - elo_vec[v]) / 400)); exp_h = 1 - exp_v
        act_v = 1 if game['VPTS'] > game['HPTS'] else 0; act_h = 1 - act_v
        elo_vec[v] += K_v * (act_v - exp_v); elo_vec[h] += K_h * (act_h - exp_h)
        full25.at[i, 'visitor_elo_post'] = elo_vec[v]
        full25.at[i, 'home_elo_post'] = elo_vec[h]
        games_played[v] += 1; games_played[h] += 1
        # Recent margins & wins
        m_h = np.mean(recent_margins[h]) if recent_margins[h] else 0
        m_v = np.mean(recent_margins[v]) if recent_margins[v] else 0
        full25.at[i, 'home_recent_margin'] = m_h
        full25.at[i, 'visitor_recent_margin'] = m_v
        recent_margins[h].append(game['HPTS'] - game['VPTS'])
        recent_margins[v].append(game['VPTS'] - game['HPTS'])
        w_h = sum(recent_wins[h]) if recent_wins[h] else 0
        w_v = sum(recent_wins[v]) if recent_wins[v] else 0
        full25.at[i, 'home_recent_wins'] = w_h
        full25.at[i, 'visitor_recent_wins'] = w_v
        recent_wins[h].append(act_h)
        recent_wins[v].append(act_v)
    full25['home_win'] = (full25['HPTS'] > full25['VPTS']).astype(int)
    # Build time series for plotting
    records = []
    for _, r in full25.iterrows():
        records.append({'Team': r['Home'],   'Date': r['Date_clean'], 'Elo': r['home_elo_post']})
        records.append({'Team': r['Visitor'],'Date': r['Date_clean'], 'Elo': r['visitor_elo_post']})
    elo_df = pd.DataFrame(records).sort_values(['Team','Date'])
    return full25, elo_df, elo_vec

@st.cache_data
def train_model(full25):
    df = full25[(full25['visitor_game_num'] > 5) & (full25['home_game_num'] > 5)].copy()
    df['diff_elo'] = df['home_elo_pre'] - df['visitor_elo_pre']
    df['win_pct_diff'] = df['home_win_pct'] - df['visitor_win_pct']
    features = ['diff_elo','home_recent_margin','visitor_recent_margin','win_pct_diff',
                'home_days_since_last','visitor_days_since_last','home_recent_wins','visitor_recent_wins']
    X = df[features]; y = df['home_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='accuracy').mean()
    model.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    test_acc = accuracy_score(y_test, model.predict(X_test_s))
    report = classification_report(y_test, model.predict(X_test_s), output_dict=True)
    return model, scaler, cv_score, train_acc, test_acc, report

@st.cache_data
def generate_matchups(full25, elo_vec, model, scaler):
    latest_h = full25.sort_values('Date_clean').groupby('Home').last().reset_index()
    latest_v = full25.sort_values('Date_clean').groupby('Visitor').last().reset_index()
    home_m = {r['Home']: r for _, r in latest_h.iterrows()}
    away_m = {r['Visitor']: r for _, r in latest_v.iterrows()}
    feats = ['diff_elo','home_recent_margin','visitor_recent_margin','win_pct_diff',
             'home_days_since_last','visitor_days_since_last','home_recent_wins','visitor_recent_wins']
    records = []
    teams = list(elo_vec.keys())
    for h in teams:
        for v in teams:
            if h == v: continue
            diff = elo_vec[h] - elo_vec[v]
            hm, am = home_m.get(h, {}), away_m.get(v, {})
            vals = [
                diff,
                hm.get('home_recent_margin', 0),
                am.get('visitor_recent_margin', 0),
                hm.get('home_win_pct', 0.5) - am.get('visitor_win_pct', 0.5),
                hm.get('home_days_since_last', 7),
                am.get('visitor_days_since_last', 7),
                hm.get('home_recent_wins', 0),
                am.get('visitor_recent_wins', 0)
            ]
            Xp = scaler.transform(pd.DataFrame([vals], columns=feats))
            p = model.predict_proba(Xp)[0,1]
            records.append({'Home': h, 'Away': v, 'PredWinProb': p, 'PredWinner': 'Home' if p>=0.5 else 'Away'})
    return pd.DataFrame(records)


def main():
    st.set_page_config(page_title='NBA 25 Elo & Predictions', layout='wide')
    st.title('NBA 2004–2025 Elo Ratings & Predictions')

    data = load_data()
    full25, elo_df, elo_vec = compute_elo(data)
    model, scaler, cv_score, train_acc, test_acc, report = train_model(full25)
    matchups_df = generate_matchups(full25, elo_vec, model, scaler)

    page = st.sidebar.radio('Page', ['Data','Elo Time Series','Model Performance','Matchup Predictions'])

    if page == 'Data':
        st.header('Season Data')
        seasons = sorted(full25['Season'].unique())
        sel = st.multiselect('Select Seasons', seasons, default=seasons)
        st.dataframe(full25[full25['Season'].isin(sel)])

    elif page == 'Elo Time Series':
        st.header('Elo Ratings Over Time')
        teams = sorted(elo_df['Team'].unique())
        sel_t = st.multiselect('Select Teams', teams[:5], default=teams[:5])
        fig, ax = plt.subplots(figsize=(10,5))
        for t in sel_t:
            df_t = elo_df[elo_df['Team'] == t]
            ax.plot(df_t['Date'], df_t['Elo'], label=t)
        ax.set_xlabel('Date'); ax.set_ylabel('Elo Rating'); ax.legend()
        st.pyplot(fig)

    elif page == 'Model Performance':
        st.header('Random Forest Model Performance')
        st.write(f"Cross-validated Accuracy: {cv_score:.2f}")
        st.write(f"Train Accuracy: {train_acc:.2f}")
        st.write(f"Test Accuracy: {test_acc:.2f}")
        st.subheader('Classification Report')
        rep_df = pd.DataFrame(report).transpose()
        st.dataframe(rep_df)

    else:
        st.header('Matchup Predictions')
        home_sel = st.selectbox('Home Team', sorted(elo_vec.keys()))
        away_sel = st.selectbox('Away Team', sorted(elo_vec.keys()))
        row = matchups_df[(matchups_df['Home']==home_sel) & (matchups_df['Away']==away_sel)]
        if not row.empty:
            p = row['PredWinProb'].iloc[0]
            st.metric(f"Win Probability: {home_sel} vs {away_sel}", f"{p:.1%}")
        st.subheader('All Matchups')
        st.dataframe(matchups_df)

if __name__=='__main__':
    main()
