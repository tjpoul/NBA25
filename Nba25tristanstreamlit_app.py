# 24/25 NBA Playoff Predictor
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, difflib
import plotly.express as px, plotly.graph_objects as go
import glob, os, itertools
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
def fuzzy(name:str):
    if name in CURRENT_TEAMS: return name
    if name in NAME_MAP:      return NAME_MAP[name]
    m=difflib.get_close_matches(name,CURRENT_TEAMS,1,0.7)
    return m[0] if m else None

dyn_K = lambda g: 20 if g<5 else 15 if g<10 else 12
prob_to_odds = lambda p: round(1/p,2) if p>0 else None
higher_lower = lambda a,b: (a,b) if TEAM_SEEDS[a]<TEAM_SEEDS[b] else (b,a) if TEAM_SEEDS[b]<TEAM_SEEDS[a] else ((a,b) if np.random.rand()<.5 else (b,a))

def best_of_7(p):
    g4=p[0]*p[1]*p[2]*p[3]
    g5=sum(np.prod([p[i] if i in c else 1-p[i] for i in range(4)])
           for c in itertools.combinations(range(4),3))*p[4]
    g6=sum(np.prod([p[i] if i in c else 1-p[i] for i in range(5)])
           for c in itertools.combinations(range(5),3))*p[5]
    g7=sum(np.prod([p[i] if i in c else 1-p[i] for i in range(6)])
           for c in itertools.combinations(range(6),3))*p[6]
    return g4+g5+g6+g7,(g4,g5,g6,g7)

# ── 1 · LOAD GAMES -----------------------------------------------------------
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
        for f in files])
    df['Date']=pd.to_datetime(df['Date'],format='%a %b %d %Y',errors='coerce')
    df=df[df['HPTS'].notna()].sort_values('Date')
    df['Season']=df['Date'].apply(lambda d:d.year if d.month>=10 else d.year-1)
    df=df[df['Season']!=2019]
    df['Home']=df['Home'].map(fuzzy); df['Visitor']=df['Visitor'].map(fuzzy)
    return df.dropna(subset=['Home','Visitor']).reset_index(drop=True)

# ── 2 · BUILD FEATURES & ELO -------------------------------------------------
@st.cache_data(show_spinner=False)
def build_features(raw:pd.DataFrame):
    teams=pd.unique(pd.concat([raw['Home'],raw['Visitor']]))
    elo={t:1000 for t in teams}; played={t:0 for t in teams}
    last_dt={t:None for t in teams}; game_ct,wins={},{}
    marg={t:deque(maxlen=5) for t in teams}; rw={t:deque(maxlen=5) for t in teams}

    cols=['home_elo_pre','visitor_elo_pre','home_elo_post','visitor_elo_post',
          'home_game_num','visitor_game_num','home_win_pct','visitor_win_pct',
          'home_days_since_last','visitor_days_since_last',
          'home_recent_margin','visitor_recent_margin',
          'home_recent_wins','visitor_recent_wins']
    for c in cols: raw[c]=np.nan
    season=raw.iloc[0]['Season']

    for i,row in raw.iterrows():
        if row['Season']!=season:
            for r,t in enumerate(sorted(elo,key=elo.get,reverse=True),1):
                elo[t]=1000+((len(elo)/2)-r)*3
            game_ct.clear(); wins.clear(); season=row['Season']

        h,v=row['Home'],row['Visitor']; hk,vk=(season,h),(season,v)
        gh,gv=game_ct.get(hk,0),game_ct.get(vk,0)
        wph=wins.get(hk,0)/gh if gh else .5; wpv=wins.get(vk,0)/gv if gv else .5
        raw.loc[i,['home_win_pct','visitor_win_pct']]=[wph,wpv]
        game_ct[hk]=gh+1; game_ct[vk]=gv+1
        raw.loc[i,['home_game_num','visitor_game_num']]=[gh+1,gv+1]

        dh=(row['Date']-last_dt[h]).days if last_dt[h] else 7
        dv=(row['Date']-last_dt[v]).days if last_dt[v] else 7
        raw.loc[i,['home_days_since_last','visitor_days_since_last']]=[min(dh,30),min(dv,30)]
        last_dt[h]=last_dt[v]=row['Date']

        raw.loc[i,['home_elo_pre','visitor_elo_pre']]=[elo[h],elo[v]]

        home_win=row['HPTS']>row['VPTS']; exp=1/(1+10**((elo[v]-elo[h])/400))
        elo[h]+=dyn_K(played[h])*((1 if home_win else 0)-exp)
        elo[v]+=dyn_K(played[v])*((0 if home_win else 1)-(1-exp))
        raw.loc[i,['home_elo_post','visitor_elo_post']]=[elo[h],elo[v]]
        played[h]+=1; played[v]+=1

        raw.loc[i,['home_recent_margin','visitor_recent_margin']]=[
            np.mean(marg[h]) if marg[h] else 0,
            np.mean(marg[v]) if marg[v] else 0]
        marg[h].append(row['HPTS']-row['VPTS']); marg[v].append(row['VPTS']-row['HPTS'])

        raw.loc[i,['home_recent_wins','visitor_recent_wins']]=[sum(rw[h]),sum(rw[v])]
        rw[h].append(1 if home_win else 0); rw[v].append(0 if home_win else 1)

    raw['diff_elo']=raw['home_elo_pre']-raw['visitor_elo_pre']
    raw['win_pct_diff']=raw['home_win_pct']-raw['visitor_win_pct']
    raw['home_win']=(raw['HPTS']>raw['VPTS']).astype(int)
    return raw,elo

# ── 3 · TRAIN MODEL ----------------------------------------------------------
@st.cache_data(show_spinner=False)
def train_model(df):
    feats=['diff_elo','home_recent_margin','visitor_recent_margin','win_pct_diff',
           'home_days_since_last','visitor_days_since_last',
           'home_recent_wins','visitor_recent_wins']
    ml=df[(df['home_game_num']>5)&(df['visitor_game_num']>5)]
    X,y=ml[feats],ml['home_win']
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=.05,random_state=42)
    scaler=StandardScaler(); Xtr_s,Xte_s=scaler.fit_transform(X_tr),scaler.transform(X_te)
    rf=RandomForestClassifier(n_estimators=300,random_state=42)
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    cv_acc=cross_val_score(rf,Xtr_s,y_tr,cv=cv,scoring='accuracy').mean()
    rf.fit(Xtr_s,y_tr)
    hold_acc=accuracy_score(y_te,rf.predict(Xte_s))
    cls=pd.DataFrame(classification_report(y_te,rf.predict(Xte_s),output_dict=True)).T.drop(columns='support')
    return rf,scaler,cv_acc,hold_acc,cls

# ── 4 · BUILD GRID -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_grid(df, elo, _rf, _sc):
    rf,_sc=_rf,_sc
    last_home=df.sort_values('Date').groupby('Home').last()
    last_away=df.sort_values('Date').groupby('Visitor').last()
    H={t:r for t,r in last_home.iterrows()}; V={t:r for t,r in last_away.iterrows()}
    recs=[]
    for h in CURRENT_TEAMS:
        for a in CURRENT_TEAMS:
            if h==a: continue
            hm,vm=H[h],V[a]
            fv=[elo[h]-elo[a], hm['home_recent_margin'], vm['visitor_recent_margin'],
                hm['home_win_pct']-vm['visitor_win_pct'],
                hm['home_days_since_last'], vm['visitor_days_since_last'],
                hm['home_recent_wins'], vm['visitor_recent_wins']]
            p=rf.predict_proba(_sc.transform([fv]))[0,1]
            recs.append({'Home':h,'Away':a,'WinProb':p if p>=.5 else 1-p,'Winner':h if p>=.5 else a})
    return pd.DataFrame(recs)

# ── 5 · STREAMLIT APP --------------------------------------------------------
def main():
    st.set_page_config(page_title='24/25 NBA Playoff Predictor',layout='wide')
    st.title('24/25 NBA Playoff Predictor')
    st.caption('Built by **Tristan Poul**')

    with st.spinner('⏳  Loading data, updating Elo ratings and training the model… '
                    'first launch can take up to **2 minutes**.'):
        raw = load_games()
        if raw.empty:
            st.error("No game data found.")
            return

        # heavy preprocessing & ML
        data, elo = build_features(raw.copy())
        rf, sc, cv_acc, hold_acc, cls = train_model(data)
        grid = build_grid(data, elo, rf, sc)
    raw=load_games()
    if raw.empty:
        st.error("No game data found."); return
    data,elo=build_features(raw.copy())
    rf,sc,cv_acc,hold_acc,cls=train_model(data)
    grid=build_grid(data,elo,rf,sc)
    feats=['diff_elo','home_recent_margin','visitor_recent_margin','win_pct_diff',
           'home_days_since_last','visitor_days_since_last',
           'home_recent_wins','visitor_recent_wins']

    tab_play,tab_game,tab_model,tab_data,tab_viz,tab_info=st.tabs(
        ['Playoffs Series Predictor','Individual Game Predictions','Model','Data','Visualizations','Info'])

    # ── PLAYOFFS TAB ---------------------------------------------------------
    with tab_play:
        st.header('Playoffs Series Predictor')
        t1=st.selectbox('Higher seed?',CURRENT_TEAMS,key='ps1')
        t2=st.selectbox('Opponent',[t for t in CURRENT_TEAMS if t!=t1],key='ps2')
        hs,ls=higher_lower(t1,t2)
        ph=grid[(grid['Home']==hs)&(grid['Away']==ls)]['WinProb'].iat[0]
        pr=grid[(grid['Home']==ls)&(grid['Away']==hs)]['WinProb'].iat[0]
        plist=[ph,ph,1-pr,1-pr,ph,1-pr,ph]
        total,(w4,w5,w6,w7)=best_of_7(plist); opp=1-total
        st.metric(f'{hs} Series Win %',f'{total:.1%}')
        st.metric(f'{ls} Series Win %',f'{opp:.1%}')
        st.metric(f'{hs} No-vig odds',prob_to_odds(total))
        st.metric(f'{ls} No-vig odds',prob_to_odds(opp))

        qlist=[1-x for x in plist]; _,(q4,q5,q6,q7)=best_of_7(qlist)
        probs=pd.DataFrame({hs:[w4,w5,w6,w7],ls:[q4,q5,q6,q7]},
                           index=['Win in 4','Win in 5','Win in 6','Win in 7'])
        odds=probs.applymap(lambda x:prob_to_odds(x) if x>0 else None)
        st.subheader('Win-in-X distribution (%)')
        st.table(probs.style.format('{:.1%}'))
        st.subheader('Win-in-X — no-vig decimal odds')
        st.table(odds)

    # ── GAME TAB -------------------------------------------------------------
    with tab_game:
        st.header('Individual Game Predictions')
        home=st.selectbox('Home',CURRENT_TEAMS,key='g1')
        away=st.selectbox('Away',[t for t in CURRENT_TEAMS if t!=home],key='g2')
        row=grid[(grid['Home']==home)&(grid['Away']==away)]
        if not row.empty:
            st.metric('Predicted winner',row.iat[0,3])
            st.metric('Win probability',f"{row.iat[0,2]*100:.1f}%")
            st.metric('No-vig odds',prob_to_odds(row.iat[0,2]))

    # ── MODEL TAB ------------------------------------------------------------
    with tab_model:
        st.header('Random-Forest Model Performance')
        st.write(f"**5-fold CV accuracy:** {cv_acc:.2%} &nbsp;|&nbsp; **Hold-out accuracy:** {hold_acc:.2%}")
        st.dataframe(cls.style.format('{:.2%}'))

    # ── DATA TAB -------------------------------------------------------------
    with tab_data:
        st.header('Elo ratings – this & last season')
        default=['Minnesota Timberwolves','Oklahoma City Thunder','Washington Wizards']
        picks=st.multiselect('Teams',CURRENT_TEAMS,default=default)
        cur=data['Season'].max()
        if picks:
            frames=[]
            for tm in picks:
                s=data[(data['Season'].isin([cur,cur-1])) & ((data['Home']==tm)|(data['Visitor']==tm))].copy()
                s['Elo']=np.where(s['Home']==tm,s['home_elo_post'],s['visitor_elo_post'])
                frames.append(s[['Date','Elo']].assign(Team=tm))
            st.plotly_chart(px.line(pd.concat(frames),x='Date',y='Elo',color='Team',
                                    template='plotly_white'),use_container_width=True)
        st.dataframe(data.sort_values('Date',ascending=False)[
            ['Date','Home','Visitor','HPTS','VPTS',
             'home_elo_pre','visitor_elo_pre','home_elo_post','visitor_elo_post']])

    # ── VISUALIZATIONS TAB ---------------------------------------------------
    with tab_viz:
        st.header('Visual Explorations')

        # 1) Elo diff vs win probability (scaled)
        st.subheader('Elo difference vs win probability (%)')
        diff=pd.Series([elo[r['Home']]-elo[r['Away']] for _,r in grid.iterrows()])
        scat=pd.DataFrame({'Elo':diff,'WinPct':grid['WinProb']*100})

        col1,col2=st.columns(2)
        for df,title,slot in [
            (scat[scat['Elo']>=0],'Home Elo ≥ Away',col1),
            (scat[scat['Elo']<=0],'Home Elo ≤ Away',col2)]:
            with slot:
                st.markdown(f'**{title}**')
                fig=px.scatter(df,x='Elo',y='WinPct',template='plotly_white',height=380,
                               labels={'Elo':'Elo diff','WinPct':'Win %'})
                if len(df)>3 and df['Elo'].std()>0:
                    try:
                        m,b=np.polyfit(df['Elo'],df['WinPct'],1)
                        xs=np.linspace(df['Elo'].min(),df['Elo'].max(),50)
                        fig.add_trace(go.Scatter(x=xs,y=m*xs+b,mode='lines',name='OLS'))
                        rel='positive' if m>0 else 'negative'
                        st.plotly_chart(fig,use_container_width=True)
                        st.info(f"Slope ≈ **{m:+.2f}** → {rel} relationship.")
                    except np.linalg.LinAlgError:
                        st.plotly_chart(fig,use_container_width=True)
                        st.warning("Regression failed.")
                else:
                    st.plotly_chart(fig,use_container_width=True)
                    st.caption("Not enough spread.")

        # 2) Feature vs win probability (%)
        st.subheader('Model feature vs win probability (%)')
        ft_cols=st.columns(4)
        for i,f in enumerate(feats):
            with ft_cols[i%4]:
                vals=[]
                for _,r in grid.iterrows():
                    h,a=r['Home'],r['Away']
                    row=data[((data['Home']==h)&(data['Visitor']==a))|
                             ((data['Home']==a)&(data['Visitor']==h))].iloc[-1]
                    vals.append(row[f])
                df_f=pd.DataFrame({f:vals,'WinPct':grid['WinProb']*100})
                fig=px.scatter(df_f,x=f,y='WinPct',template='plotly_white',height=260,
                               labels={'WinPct':'Win %'})
                try:
                    m,b=np.polyfit(df_f[f],df_f['WinPct'],1)
                    xs=np.linspace(df_f[f].min(),df_f[f].max(),40)
                    fig.add_trace(go.Scatter(x=xs,y=m*xs+b,mode='lines',name='OLS'))
                    slope=f"{m:+.2f}"
                except np.linalg.LinAlgError:
                    slope="n/a"
                fig.update_layout(showlegend=False,margin=dict(t=25,l=10,r=10,b=10))
                st.plotly_chart(fig,use_container_width=True)
                st.caption(f"Slope: **{slope}**")

    # ── INFO TAB -------------------------------------------------------------
    with tab_info:
        st.header('Everything under the hood  🔍')

    ## 0 · Author
        with st.expander('About the author'):
             st.markdown("""
    **Tristan Poul** – Econ and Data Analysis Major at Santa Clara University.
    Feel free to reach me at tpoul@scu.edu or on [LinkedIn](https://www.linkedin.com/in/tristan-poul/)!
    """)

        ## 1 · Data ingestion
        with st.expander('📚 1 · Data collection & cleaning'):
            st.markdown("""
    * **Source:** [Basketball-Reference](https://www.basketball-reference.com/)  
      Box-score tables for every regular-season & playoff game, **2004-05&nbsp;→ present**.
    * **Automation:** A one-shot Python scraper (not part of this app) stores each season as
      `NBAYYYY-YY.csv`.
    * **Legacy franchise mapping**  
      | Old name | Current franchise |
      |----------|-------------------|
      | Seattle SuperSonics | Oklahoma City Thunder |
      | New Jersey Nets     | Brooklyn Nets |
      | Charlotte Bobcats   | Charlotte Hornets |
      | New Orleans Hornets | New Orleans Pelicans |
      *Fuzzy match* cleans minor typos (e.g. “Portland Trailblazers” → “Portland Trail Blazers”).
    * **Exclusions:**  
      * 2019-20 “bubble” games (Season == 2019) are dropped—COVID hiatus breaks time-series logic.
    """)

        ## 2 · Dynamic Elo engine
        with st.expander('📈 2 · Dynamic Elo ratings'):
            st.markdown(r"""
    * **Initialisation** – every team starts each **season** at **Elo = 1000**.  
    * **Season reset** – off-season “compression”:  
      after the Finals, ratings are re-centred (1000 ± 5) to avoid drift.
    * **K-factor schedule**  

      | Games played | K |
      |--------------|---|
      | 1–5          | 20|
      | 6–10         | 15|
      | 11 +         | 12|
    """)

        ## 3 · Feature engineering
        with st.expander('🛠️ 3 · Engineered features'):
            st.markdown("""
    | Feature | Definition | Rationale |
    |---------|------------|-----------|
    | `diff_elo` | Home Elo − Away Elo | Raw skill gap |
    | `home_recent_margin` / `visitor_recent_margin` | 5-game rolling avg scoring margin | Momentum & strength |
    | `home_recent_wins` / `visitor_recent_wins` | Wins in last 5 | Hot vs cold |
    | `win_pct_diff` | Season W% diff | Macro strength |
    | `home_days_since_last` / `visitor_days_since_last` | Rest days (capped 30) | Fatigue |""")

        ## 4 · Machine-learning model
        with st.expander('🤖 4 · Random-Forest winner model'):
            st.markdown(f"""
    * **Model:** `sklearn.ensemble.RandomForestClassifier` with **300** trees, default depth.  
    * **Pre-processing:** features standardised via `StandardScaler`.  
    * **Training window:** excludes each team’s first **5** games of a season to let metrics stabilise.  
    * **Validation:**  
      * 5-fold cross-validation accuracy ≈ **{cv_acc:.1%}**  
      * 5 % hold-out accuracy ≈ **{hold_acc:.1%}**  
    * **Label:** `home_win` (1 = home victory).""")

        ## 5 · Match-up probability grid
        with st.expander('🆚 5 · Match-up grid & no-vig odds'):
            st.markdown("""
    * For every ordered pair of current teams we take **their latest feature snapshot** and run it through the forest.  
    * Probability > 0.5 ⇒ model says the home team is favourite; else the road team is.  
    * Decimal *no-vig* odds = `1 / probability` (removes sportsbook margin).""")

        ## 6 · Playoff series simulator
        with st.expander('🏆 6 · 2-2-1-1-1 series simulator'):
           st.markdown(r"""
    * Higher seed hosts **Games 1-2, 5 & 7**.  
    * Per-game win probs:  
      * higher-seed home = grid **home** prob  
      * lower-seed home = 1 – grid **home** prob  
    * Use an N choose K process to calculate the probability of winning a best-of-7 series at any given point:  
      * win-in-X odds (4-0, 4-1, 4-2, 4-3)   
    * Use sum of the N choose K process to output overall series odds.
      * Outright win probability (Eg. 54.6%).""")

    ## 7 · Known limitations & next steps
        with st.expander('🚧 7 · Limitations / future work'):
           st.markdown("""
    * No injury, lineup or travel data.  
    * Elo ignores roster churn in off-season.  
    * Probabilities not calibrated to market lines.  
    * Potential upgrades: gradient-boosting, Bayesian Elo, market-edge overlay, injury scraping.
    * The model is hovering between 64-66%, so further improvement of accuracy is probably necessary.""")



# ── RUN ----------------------------------------------------------------------
if __name__=='__main__':
    main()
