# 24/25 NBA Playoff Predictor
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, difflib, plotly.express as px
import glob, os, itertools, textwrap
from collections import deque
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── CONSTANTS ----------------------------------------------------------------
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
NAME_MAP = {
    "Seattle SuperSonics":"Oklahoma City Thunder",
    "New Jersey Nets":"Brooklyn Nets",
    "Charlotte Bobcats":"Charlotte Hornets",
    "New Orleans Hornets":"New Orleans Pelicans",
    "New Orleans/Oklahoma City Hornets":"New Orleans Pelicans",
    "NO/Oklahoma City Hornets":"New Orleans Pelicans",
    "LA Clippers":"Los Angeles Clippers"
}
TEAM_SEEDS = {t:i+1 for i,t in enumerate(CURRENT_TEAMS)}

# ── HELPERS ------------------------------------------------------------------
def fuzzy(team:str):
    if team in CURRENT_TEAMS:      return team
    if team in NAME_MAP:           return NAME_MAP[team]
    m=difflib.get_close_matches(team,CURRENT_TEAMS,1,0.7)
    return m[0] if m else None

def dyn_K(gp): return 20 if gp<5 else 15 if gp<10 else 12
def prob_to_odds(p): return round(1/p,2) if p>0 else None

def higher_lower(a,b):
    sa,sb=TEAM_SEEDS[a], TEAM_SEEDS[b]
    return (a,b) if sa<sb else (b,a) if sb<sa else ((a,b) if np.random.rand()<.5 else (b,a))

def best_of_7(p):
    g4=p[0]*p[1]*p[2]*p[3]
    g5=sum(np.prod([p[i] if i in c else 1-p[i] for i in range(4)])
           for c in itertools.combinations(range(4),3))*p[4]
    g6=sum(np.prod([p[i] if i in c else 1-p[i] for i in range(5)])
           for c in itertools.combinations(range(5),3))*p[5]
    g7=sum(np.prod([p[i] if i in c else 1-p[i] for i in range(6)])
           for c in itertools.combinations(range(6),3))*p[6]
    return g4+g5+g6+g7,(g4,g5,g6,g7)

# ── 1 · LOAD -----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_games():
    folder=os.path.join(os.getcwd(),"AllNBAdata")
    files=sorted(glob.glob(os.path.join(folder,"NBA*-*.csv")),
                 key=lambda f:int(os.path.basename(f)[3:7]))
    if not files: return pd.DataFrame()
    df=pd.concat([
        pd.read_csv(f).rename(columns={
            "Visitor/Neutral":"Visitor","Home/Neutral":"Home",
            "PTS":"VPTS","PTS.1":"HPTS"})[["Date","Visitor","Home","VPTS","HPTS"]]
        for f in files], ignore_index=True)
    df['Date']=pd.to_datetime(df['Date'],format='%a %b %d %Y',errors='coerce')
    df=df[df['HPTS'].notna()].sort_values('Date')
    df['Season']=df['Date'].apply(lambda d:d.year if d.month>=10 else d.year-1)
    df=df[df['Season']!=2019]     # drop bubble artefact
    df['Home']=df['Home'].map(fuzzy); df['Visitor']=df['Visitor'].map(fuzzy)
    return df.dropna(subset=['Home','Visitor']).reset_index(drop=True)

# ── 2 · FEATURE PIPELINE (Elo etc.) -----------------------------------------
@st.cache_data(show_spinner=False)
def build_features(raw:pd.DataFrame):
    teams=pd.unique(pd.concat([raw['Home'],raw['Visitor']]))
    elo={t:1000 for t in teams}; played={t:0 for t in teams}
    game_ct,wins,last_dt={}, {}, {t:None for t in teams}
    marg={t:deque(maxlen=5) for t in teams}; rw={t:deque(maxlen=5) for t in teams}

    add=['home_elo_pre','visitor_elo_pre','home_elo_post','visitor_elo_post',
         'home_game_num','visitor_game_num','home_win_pct','visitor_win_pct',
         'home_days_since_last','visitor_days_since_last',
         'home_recent_margin','visitor_recent_margin',
         'home_recent_wins','visitor_recent_wins']
    for c in add: raw[c]=np.nan
    season=raw.iloc[0]['Season']

    for i,row in raw.iterrows():
        if row['Season']!=season:
            ordered=sorted(elo,key=elo.get,reverse=True)
            for r,t in enumerate(ordered,1): elo[t]=1000+((len(ordered)/2)-r)*3
            game_ct.clear(); wins.clear(); season=row['Season']

        h,v=row['Home'],row['Visitor']
        hk,vk=(season,h),(season,v)
        gh,gv=game_ct.get(hk,0),game_ct.get(vk,0)
        wph=wins.get(hk,0)/gh if gh else .5
        wpv=wins.get(vk,0)/gv if gv else .5
        raw.at[i,'home_win_pct']=wph; raw.at[i,'visitor_win_pct']=wpv
        game_ct[hk]=gh+1; game_ct[vk]=gv+1
        raw.at[i,'home_game_num']=gh+1; raw.at[i,'visitor_game_num']=gv+1

        dh=(row['Date']-last_dt[h]).days if last_dt[h] else 7
        dv=(row['Date']-last_dt[v]).days if last_dt[v] else 7
        raw.at[i,'home_days_since_last']=min(dh,30)
        raw.at[i,'visitor_days_since_last']=min(dv,30)
        last_dt[h]=last_dt[v]=row['Date']

        raw.at[i,'home_elo_pre']=elo[h]; raw.at[i,'visitor_elo_pre']=elo[v]

        home_win=1 if row['HPTS']>row['VPTS'] else 0
        exp_home=1/(1+10**((elo[v]-elo[h])/400))
        elo[h]+=dyn_K(played[h])*(home_win-exp_home)
        elo[v]+=dyn_K(played[v])*((1-home_win)-(1-exp_home))
        raw.at[i,'home_elo_post']=elo[h]; raw.at[i,'visitor_elo_post']=elo[v]
        played[h]+=1; played[v]+=1

        raw.at[i,'home_recent_margin']=np.mean(marg[h]) if marg[h] else 0
        raw.at[i,'visitor_recent_margin']=np.mean(marg[v]) if marg[v] else 0
        marg[h].append(row['HPTS']-row['VPTS']); marg[v].append(row['VPTS']-row['HPTS'])

        raw.at[i,'home_recent_wins']=sum(rw[h]); raw.at[i,'visitor_recent_wins']=sum(rw[v])
        rw[h].append(home_win); rw[v].append(1-home_win)

    raw['diff_elo']=raw['home_elo_pre']-raw['visitor_elo_pre']
    raw['win_pct_diff']=raw['home_win_pct']-raw['visitor_win_pct']
    raw['home_win']=(raw['HPTS']>raw['VPTS']).astype(int)
    return raw,elo

# ── 3 · MODEL (exact notebook steps) ----------------------------------------
@st.cache_data(show_spinner=False)
def train_model(df:pd.DataFrame):
    feats=['diff_elo','home_recent_margin','visitor_recent_margin','win_pct_diff',
           'home_days_since_last','visitor_days_since_last',
           'home_recent_wins','visitor_recent_wins']
    ml=df[(df['home_game_num']>5)&(df['visitor_game_num']>5)].copy()
    X,y=ml[feats],ml['home_win']
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=.05,random_state=42)
    sc=StandardScaler(); Xtr_s,Xte_s=sc.fit_transform(X_tr),sc.transform(X_te)
    rf=RandomForestClassifier(n_estimators=300,random_state=42)
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    cv_acc=cross_val_score(rf,Xtr_s,y_tr,cv=cv,scoring='accuracy').mean()
    rf.fit(Xtr_s,y_tr)
    hold_acc=accuracy_score(y_te,rf.predict(Xte_s))
    cls_report=pd.DataFrame(classification_report(y_te,rf.predict(Xte_s),output_dict=True)).T
    return rf,sc,cv_acc,hold_acc,cls_report

# ── 4 · MATCHUP GRID ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_grid(df, elo, _rf, _sc):
    rf,sc=_rf,_sc
    feats=['diff_elo','home_recent_margin','visitor_recent_margin','win_pct_diff',
           'home_days_since_last','visitor_days_since_last',
           'home_recent_wins','visitor_recent_wins']
    last_h=df.sort_values('Date').groupby('Home').last().reset_index()
    last_v=df.sort_values('Date').groupby('Visitor').last().reset_index()
    H={r['Home']:r for _,r in last_h.iterrows()}
    V={r['Visitor']:r for _,r in last_v.iterrows()}
    recs=[]
    for h in CURRENT_TEAMS:
        for a in CURRENT_TEAMS:
            if h==a: continue
            hm,vm=H[h],V[a]
            fv=[elo[h]-elo[a],
                hm['home_recent_margin'],vm['visitor_recent_margin'],
                hm['home_win_pct']-vm['visitor_win_pct'],
                hm['home_days_since_last'],vm['visitor_days_since_last'],
                hm['home_recent_wins'],vm['visitor_recent_wins']]
            p=rf.predict_proba(sc.transform([fv]))[0,1]
            recs.append({'Home':h,'Away':a,
                         'Winner':h if p>=.5 else a,
                         'WinProb':p if p>=.5 else 1-p})
    return pd.DataFrame(recs)

# ── 5 · APP ------------------------------------------------------------------
def main():
    st.set_page_config(page_title='24/25 NBA Playoff Predictor',layout='wide')
    st.title('24/25 NBA Playoff Predictor')

    raw=load_games()
    if raw.empty: st.error("No data found"); return
    data,elo=build_features(raw.copy())
    rf,sc,cv_acc,hold_acc,cls_df=train_model(data)
    grid=build_grid(data,elo,rf,sc)

    tab_play,tab_game,tab_data,tab_model,tab_info = st.tabs(
        ['Playoffs Series Predictor','Individual Game Predictions','Data',
         'Model','Info'])

    # Playoffs
    with tab_play:
        st.header('Playoffs Series Predictor')
        t1=st.selectbox('Higher seed?',CURRENT_TEAMS,key='ps1')
        t2=st.selectbox('Opponent',[t for t in CURRENT_TEAMS if t!=t1],key='ps2')
        hs,ls=higher_lower(t1,t2)
        ph=grid[(grid['Home']==hs)&(grid['Away']==ls)]['WinProb']
        pr=grid[(grid['Home']==ls)&(grid['Away']==hs)]['WinProb']
        if ph.empty or pr.empty:
            st.warning("Either team missing data."); return
        p_home,p_away=ph.iat[0],pr.iat[0]
        plist=[p_home,p_home,1-p_away,1-p_away,p_home,1-p_away,p_home]
        total,(w4,w5,w6,w7)=best_of_7(plist)
        opp=1-total
        st.metric(f'{hs} Series Win %',f'{total:.1%}')
        st.metric(f'{ls} Series Win %',f'{opp:.1%}')
        st.metric(f'{hs} No-Vig Odds',prob_to_odds(total))
        st.metric(f'{ls} No-Vig Odds',prob_to_odds(opp))

        # breakdown table
        qlist=[1-x for x in plist]; opp_tot,(q4,q5,q6,q7)=best_of_7(qlist)
        breakdown=pd.DataFrame({
            hs:[w4,w5,w6,w7],
            ls:[q4,q5,q6,q7]
        },index=['Win in 4','Win in 5','Win in 6','Win in 7']).style.format('{:.1%}')
        st.subheader('Win-in-X Distribution')
        st.table(breakdown)

    # Games
    with tab_game:
        st.header('Individual Game Predictions')
        home=st.selectbox('Home',CURRENT_TEAMS,key='gm1')
        away=st.selectbox('Away',[t for t in CURRENT_TEAMS if t!=home],key='gm2')
        row=grid[(grid['Home']==home)&(grid['Away']==away)]
        if not row.empty:
            win=row.iat[0,2]; prob=row.iat[0,3]
            st.metric('Predicted Winner',win)
            st.metric('Win Probability',f'{prob:.1%}')
            st.metric('No-Vig Decimal Odds',prob_to_odds(prob))

    # Data
    with tab_data:
        st.header('Elo – this & last season')
        cur=data['Season'].max()
        picks=st.multiselect('Teams',CURRENT_TEAMS,
                             default=['Minnesota Timberwolves','Los Angeles Lakers',
                                      'Oklahoma City Thunder','Washington Wizards',
                                      'Milwaukee Bucks'])
        if picks:
            recs=[]
            for tm in picks:
                s=data[(data['Season'].isin([cur,cur-1])) &
                       ((data['Home']==tm)|(data['Visitor']==tm))].copy()
                s['Elo']=np.where(s['Home']==tm,s['home_elo_post'],s['visitor_elo_post'])
                recs.append(s[['Date','Elo']].assign(Team=tm))
            st.plotly_chart(px.line(pd.concat(recs),x='Date',y='Elo',color='Team',
                                    template='plotly_white'),use_container_width=True)
        st.dataframe(data[['Date','Home','Visitor','HPTS','VPTS',
                           'home_elo_pre','visitor_elo_pre',
                           'home_elo_post','visitor_elo_post']])

    # Model results
    with tab_model:
        st.header('Random-Forest Model Performance')
        st.write(f'**Cross-validated accuracy:** {cv_acc:.2%}')
        st.write(f'**Hold-out accuracy:** {hold_acc:.2%}')
        st.subheader('Classification report (hold-out)')
        st.dataframe(cls_df.style.format('{:.2%}'))

    # Info tab – richer explanations
    with tab_info:
        st.header('How this app works')
        with st.expander('1 · Data Collection'):
            st.markdown("""
* All regular-season and playoff games from 2004-05 through the current day are
  scraped from **Basketball-Reference** (`NBA####-##.csv` files).
* Legacy franchise names are mapped to the current 30-team set  
  (e.g., *Seattle SuperSonics → Oklahoma City Thunder*).  
  Any leftover spelling quirks are resolved by fuzzy matching.
""")
        with st.expander('2 · Dynamic Elo Ratings'):
            st.markdown("""
* Every team starts each new season at **1000** Elo.  
  End-of-season balance is restored by compressing the distribution.
* **K-factor** depends on games played:  
  * first 5 g → **20**  
  * 6–10 g → **15**  
  * 11 g+ → **12**
* Rating update formula  

\\[
\\small\\;\\;\\Delta\\_{home}=K\\,(\\text{homeWin} - \\text{E}[home])\\qquad
\\Delta\\_{away}=K\\,(\\text{awayWin} - \\text{E}[away])
\\]
""")
        with st.expander('3 · Feature Engineering'):
            st.markdown("""
| Feature | Description |
|---------|-------------|
| **diff_elo** | Home pre-game Elo − Visitor pre-game Elo |
| **recent margin** | 5-game rolling average scoring margin |
| **recent wins** | Sum of wins in last 5 games |
| **win_pct_diff** | Home season W% − Visitor season W% |
| **days since last** | Rest days (capped at 30) for each team |
""")
        with st.expander('4 · Machine-Learning Model'):
            st.markdown(f"""
* **Random-Forest** (300 trees) on the eight engineered features.  
* Training set excludes the first five games each season for both teams.  
* 5-fold cross-val accuracy ≈ **{cv_acc:.1%}**; hold-out ≈ **{hold_acc:.1%}**.
""")
        with st.expander('5 · Per-Game Prediction Grid'):
            st.markdown("""
* For every ordered pair of current teams, we take their **latest** feature
  snapshot and feed it to the model.
* The higher of \\(P\\_\\text{home}\\) and \\(1-P\\_\\text{home}\\) becomes the
  win probability, which is then converted to **no-vig decimal odds** by
  \\(1/P\\).
""")
        with st.expander('6 · Playoff Series Simulator'):
            st.markdown(r"""
* 2-2-1-1-1 format: higher seed hosts Games 1-2, 5 & 7.  
* Game win probs:  
  * If higher seed hosts → model’s *home* win prob  
  * If lower seed hosts → \(1-\) model’s *home* win prob  
* Enumerate every path to 4 wins (in 4–7 games) to get series odds and
  score-line breakdowns.
""")
        with st.expander('7 · Limitations & Future Work'):
            st.markdown("""
* No injury / lineup context, back-to-backs, altitude or travel fatigue.  
* Elo reset rule is simple; offseason roster churn not captured.  
* A gradient-boosting model or calibrated probabilities might beat the forest.  
* UI could surface betting edge vs. market lines once those are added.
""")

# ── RUN ----------------------------------------------------------------------
if __name__=='__main__':
    main()
