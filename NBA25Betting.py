# ------------------------
#       Setup & Imports
# ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# For rolling window operations
# (pandas’ groupby and rolling functions will be used)
pd.options.mode.chained_assignment = None  # suppress copy warnings

# ------------------------
#         Data
# ------------------------
full25season = pd.read_csv("/Users/tristanpoul/Desktop/Personal code/NBAMar25Season.csv")
oldseasonrankings = pd.read_csv("/Users/tristanpoul/Desktop/Personal code/2324SeasonRankings.csv")

# ------------------------
#    Clean & Sort
# ------------------------
# Parse the Date column (e.g., "Sat Mar 1 2025") into datetime.
# Note: The format used here assumes the day may not be zero-padded.
full25season['Date_clean'] = pd.to_datetime(full25season['Date'], format='%a %b %d %Y', errors='coerce')

# Sort by the cleaned date and remove rows without valid HPTS (e.g. future games)
full25season = full25season.sort_values('Date_clean')
full25season = full25season[full25season['HPTS'].notna()]

# ------------------------
#  Initialize Starting Elo Ratings
# ------------------------
# Calculate baseline Elo ratings from last season's rankings.
max_rank = oldseasonrankings['Rank'].max()
oldseasonrankings['Elo'] = 1000 + ((max_rank / 2) - oldseasonrankings['Rank']) * 15

# Create a lookup dictionary for team Elo ratings.
elo_vec = dict(zip(oldseasonrankings['Team'], oldseasonrankings['Elo']))

# ------------------------
#    Apply the Elo System
# ------------------------
# Add columns to store pre- and post-game Elo ratings.
for col in ['visitor_elo_pre', 'home_elo_pre', 'visitor_elo_post', 'home_elo_post']:
    full25season[col] = np.nan

# Track the number of games played for each team.
teams = list(elo_vec.keys())
games_played_vec = {team: 0 for team in teams}

# Define helper function for dynamic K
def get_dynamic_K(gp):
    # gp: games played so far (before this new game)
    if gp < 10:
        return 50  # High K for first 10 games
    elif gp < 20:
        return 30  # Medium K for games 11–20
    else:
        return 15  # Normal K for 21+

# Loop over each game in chronological order to update Elo ratings.
for i, game in full25season.iterrows():
    visitor_team = game['Visitor']
    home_team = game['Home']
    
    # Get current Elo ratings
    visitor_elo_pre = elo_vec[visitor_team]
    home_elo_pre = elo_vec[home_team]
    
    # Store pre-game Elo ratings
    full25season.at[i, 'visitor_elo_pre'] = visitor_elo_pre
    full25season.at[i, 'home_elo_pre'] = home_elo_pre
    
    # Determine dynamic K values
    K_visitor = get_dynamic_K(games_played_vec[visitor_team])
    K_home = get_dynamic_K(games_played_vec[home_team])
    
    # Calculate expected win probabilities using the Elo formula
    exp_visitor = 1 / (1 + 10 ** ((home_elo_pre - visitor_elo_pre) / 400))
    exp_home = 1 - exp_visitor
    
    # Determine actual outcomes (1 if win, 0 if loss)
    actual_visitor = 1 if game['VPTS'] > game['HPTS'] else 0
    actual_home = 1 - actual_visitor
    
    # Update Elo ratings
    visitor_elo_post = visitor_elo_pre + K_visitor * (actual_visitor - exp_visitor)
    home_elo_post = home_elo_pre + K_home * (actual_home - exp_home)
    
    # Save the new Elo ratings back to the lookup dictionary
    elo_vec[visitor_team] = visitor_elo_post
    elo_vec[home_team] = home_elo_post
    
    # Record post-game Elo ratings in the DataFrame
    full25season.at[i, 'visitor_elo_post'] = visitor_elo_post
    full25season.at[i, 'home_elo_post'] = home_elo_post
    
    # Increment each team's games played count
    games_played_vec[visitor_team] += 1
    games_played_vec[home_team] += 1

# Create and display final team Elo rankings in descending order.
final_rankings = pd.DataFrame({'Team': list(elo_vec.keys()), 'Elo': list(elo_vec.values())})
final_rankings = final_rankings.sort_values('Elo', ascending=False)
print("Final Elo Rankings:")
print(final_rankings)

# ------------------------
#  Plot Elo Ratings Over Time
# ------------------------
# Prepare visitor and home Elo rating data with date.
visitor_elo_time = full25season[['Date_clean', 'Visitor', 'visitor_elo_post']].rename(
    columns={'Visitor': 'Team', 'visitor_elo_post': 'Elo'})
home_elo_time = full25season[['Date_clean', 'Home', 'home_elo_post']].rename(
    columns={'Home': 'Team', 'home_elo_post': 'Elo'})

# Combine both and sort.
elo_time = pd.concat([visitor_elo_time, home_elo_time], ignore_index=True)
elo_time = elo_time.sort_values(by=['Team', 'Date_clean'])

# Plot the Elo ratings over time for each team.
plt.figure(figsize=(10, 6))
for team, group in elo_time.groupby('Team'):
    plt.plot(group['Date_clean'], group['Elo'], label=team, alpha=0.8)
plt.title("Elo Ratings Throughout the Season")
plt.xlabel("Date")
plt.ylabel("Elo Rating")
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
plt.tight_layout()
plt.show()

# ------------------------
#  Unique Matchups with Higher Win Probability First
# ------------------------
# Generate all unique matchup pairs (each combination only once)
teams_list = list(elo_vec.keys())
matchups_list = []
for team1, team2 in itertools.combinations(teams_list, 2):
    elo1_temp = elo_vec[team1]
    elo2_temp = elo_vec[team2]
    # Swap teams if necessary so that Team1 has the higher Elo
    if elo1_temp >= elo2_temp:
        Team1_final, Team2_final = team1, team2
        Elo1, Elo2 = elo1_temp, elo2_temp
    else:
        Team1_final, Team2_final = team2, team1
        Elo1, Elo2 = elo2_temp, elo1_temp
    # Calculate win probabilities with the higher-rated team first
    winProbTeam1 = 1 / (1 + 10 ** ((Elo2 - Elo1) / 400))
    winProbTeam2 = 1 - winProbTeam1
    matchups_list.append({
        'Team1': Team1_final,
        'Team2': Team2_final,
        'Elo1': Elo1,
        'Elo2': Elo2,
        'winProbTeam1': winProbTeam1,
        'winProbTeam2': winProbTeam2
    })

matchups = pd.DataFrame(matchups_list)
# Optionally, display the matchup DataFrame:
print("\nUnique Matchups with Win Probabilities:")
print(matchups)

##############################################################
##############################################################
##############################################################

# Create new features:
#   - sum_elo: Sum of (adjusted) home Elo and visitor Elo.
#   - diff_elo: Difference between (adjusted) home Elo and visitor Elo.
full25season['sum_elo'] = full25season['home_elo_pre'] + full25season['visitor_elo_pre']
full25season['diff_elo'] = full25season['home_elo_pre'] - full25season['visitor_elo_pre']


# Compute recent wins for the last 5 games for each team.
# We'll iterate over games in chronological order and use a deque (fixed-length list) to store each team's outcomes.
from collections import deque

# Initialize a deque for each team to store outcomes (1 for win, 0 for loss)
recent_results = {team: deque(maxlen=5) for team in teams}  # 'teams' was defined earlier

# Create columns to store recent wins counts
full25season['home_recent_wins'] = 0
full25season['visitor_recent_wins'] = 0

# Iterate through games in chronological order (data is already sorted by Date_clean)
for i, game in full25season.iterrows():
    home_team = game['Home']
    visitor_team = game['Visitor']
    
    # Count wins in the last 5 games for each team (if no previous games, count as 0)
    home_recent = sum(recent_results[home_team]) if home_team in recent_results else 0
    visitor_recent = sum(recent_results[visitor_team]) if visitor_team in recent_results else 0
    
    full25season.at[i, 'home_recent_wins'] = home_recent
    full25season.at[i, 'visitor_recent_wins'] = visitor_recent
    
    # Determine outcome for each team (for updating the rolling record)
    if game['HPTS'] > game['VPTS']:
        home_win = 1
        visitor_win = 0
    else:
        home_win = 0
        visitor_win = 1
        
    # Update the deques for each team
    recent_results[home_team].append(home_win)
    recent_results[visitor_team].append(visitor_win)

# Create the target variable: 1 if home wins, 0 otherwise.
full25season['home_win'] = (full25season['HPTS'] > full25season['VPTS']).astype(int)

# ------------------------
#       Machine Learning Model Training
# ------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix

# Define feature columns. We include:
#   - sum_elo, diff_elo, home_recent_wins, and visitor_recent_wins.
features = ['sum_elo', 'diff_elo', 'home_recent_wins', 'visitor_recent_wins']
X = full25season[features]
y = full25season['home_win']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities and classes on the test set
y_prob = model.predict_proba(X_test)[:, 1]  # probability of home win
y_pred = model.predict(X_test)

# Evaluate the model using the Brier score and a confusion matrix
brier = brier_score_loss(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Brier Score:", brier)
print("Confusion Matrix:")
print(cm)

##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
import pandas as pd
import numpy as np
import itertools
from collections import deque

# ---------------------------
# Assume the following are already defined:
# - elo_vec: dict mapping team -> final Elo rating (from your Elo updates)
# - full25season: your season DataFrame (with columns like 'home_recent_wins' and 'visitor_recent_wins')
# - scaler: the fitted StandardScaler from your training
# - model: the trained LogisticRegression model
# - teams: list(elo_vec.keys())
# ---------------------------

# 1. Get each team’s most recent "recent wins" in the appropriate role.
#    We group full25season by Home and Visitor and take the last recorded value.
last_home_recent = full25season.groupby('Home')['home_recent_wins'].last().to_dict()
last_visitor_recent = full25season.groupby('Visitor')['visitor_recent_wins'].last().to_dict()

# 2. Create DataFrame for every possible matchup (home team vs. away team, excluding self-matchups).
matchups_all = pd.DataFrame(
    [(home, away) for home in teams for away in teams if home != away],
    columns=['Home', 'Visitor']
)

# Map the final Elo ratings and recent wins for each team.
matchups_all['home_elo'] = matchups_all['Home'].map(elo_vec)
matchups_all['away_elo'] = matchups_all['Visitor'].map(elo_vec)
matchups_all['home_recent_wins'] = matchups_all['Home'].map(last_home_recent).fillna(0)
matchups_all['visitor_recent_wins'] = matchups_all['Visitor'].map(last_visitor_recent).fillna(0)

# Compute the features as used in your model:
matchups_all['sum_elo'] = matchups_all['home_elo'] + matchups_all['away_elo']
matchups_all['diff_elo'] = matchups_all['home_elo'] - matchups_all['away_elo']

# 3. Prepare the feature matrix in the same order as during training.
X_new = matchups_all[['sum_elo', 'diff_elo', 'home_recent_wins', 'visitor_recent_wins']].values

# Standardize the features using your previously fitted scaler.
X_new_scaled = scaler.transform(X_new)

# Predict probabilities using your logistic regression model.
# Note: model.predict_proba returns probabilities for [loss, win] so [:,1] is the home win probability.
matchups_all['model_home_prob'] = model.predict_proba(X_new_scaled)[:, 1]
matchups_all['model_away_prob'] = 1 - matchups_all['model_home_prob']

matchups_all
view(matchups_all)