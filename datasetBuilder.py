import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamelog

SEASONS = ['2021-22', '2022-23', '2023-24'] 
output_file = 'processed.csv'

def getData(seasons):
    all_games = []
    print(f"Seasons: {seasons}")
    
    for season in seasons:
        print(f" -> Current: {season}")
        # get Regular Season games
        log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
        df = log.get_data_frames()[0] # call to API, [0] grabs the main table we care about
        df['SEASON_ID'] = season # tag the season
        all_games.append(df)
        time.sleep(1) # give time
        
    return pd.concat(all_games, ignore_index=True)

def cleanAndGetStats(df):
    # clean data
    
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values(by='GAME_DATE') # sort the games by date
    
     # code win/loss
     # W = 1, L = 0
    df['HOME_WIN'] = df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    
    # stats from the LAST 5 games

    importantStats = ['PTS', 'FG_PCT', 'AST', 'REB', 'target_win']

    for col in importantStats:
        # Group by Team. 
        # Shift(1): Moves stats down 1 row (So Game 10 uses stats from Game 9)
        # Rolling(5): Averages the previous 5 shifted rows
        df[f'rolling_{col}'] = df.groupby('TEAM_ID')[col].transform(lambda x: x.shift(1).rolling(5).mean())

    # The API returns 2 rows per game. We only need 1 to predict the result.
    # We will look at games from the HOME team's perspective.
    df['is_home'] = df['MATCHUP'].str.contains('vs.')
    
    # Drop the rows where the team is Away (to remove duplicates)
    # For now, predicts if the home team wins
    df_home = df[df['is_home']].copy()
    
    # The first 5 games of every season will have NaN because there isn't enough history for rolling average.
    df_clean = df_home.dropna()
    
    return df_clean

if __name__ == "__main__":
    # Get Data
    raw_df = getData(SEASONS)
    
    # Process Data
    final_df = cleanAndGetStats(raw_df)
    
    # Save
    print(f"Saving {len(final_df)} rows to {output_file}...")
    
    # Select only columns useful for ML
    cols_to_keep = ['GAME_DATE', 'SEASON_ID', 'MATCHUP', 'TEAM_NAME', 'target_win', 'rolling_PTS', 'rolling_FG_PCT', 'rolling_AST', 'rolling_REB', 'rolling_target_win']
    final_df[cols_to_keep].to_csv(output_file, index=False)
    print("Open 'processed.csv'")