
import numpy as np
import pandas as pd

variables = ['kicks_s', 'marks_s', 'handballs_s', 'disposals_s', 'effective_disposals_s',  
             'goals_s', 'behinds_s', 'hitouts_s', 'tackles_s', 'rebounds_s', 'inside_fifties_s', 
             'clearances_s', 'clangers_s', 'free_kicks_for_s', 'free_kicks_against_s', 
             'contested_possessions_s', 'uncontested_possessions_s', 'contested_marks_s', 
             'marks_inside_fifty_s', 'one_percenters_s', 'bounces_s', 'goal_assists_s',
             'afl_fantasy_score_s', 'centre_clearances_s', 
             'stoppage_clearances_s', 'score_involvements_s', 'metres_gained_s', 'turnovers_s', 
             'intercepts_s', 'tackles_inside_fifty_s', 'contest_def_losses_s',
             'contest_def_one_on_ones_s', 'contest_off_one_on_ones_s', 'contest_off_wins_s',
             'def_half_pressure_acts_s', 'effective_kicks_s', 'f50_ground_ball_gets_s',
             'ground_ball_gets_s', 'hitouts_to_advantage_s',
             'intercept_marks_s', 'marks_on_lead_s', 'pressure_acts_s', 'rating_points_s',
             'ruck_contests_s', 'score_launches_s', 'shots_at_goal_s', 'spoils_s', 
             'match_margin', 'win', 'loss', 'draw', 'disposal_efficiency_percentage', 
             'hitout_win_percentage', 'time_on_ground_percentage']

labels = ['brownlow_votes']

extra_info = ['match_id', 'match_date', 'date', 'year', 'match_round', 'venue_name', 
              'player_id', 'player_first_name', 'player_last_name', 
              'player_position', 'player_team', 'match_home_team_goals', 
              'match_home_team_behinds', 'match_home_team_score', 
              'match_away_team_goals', 'match_away_team_behinds', 
              'match_away_team_score', 'match_winner']

def get_data_for_model(data):
    
    # only keep the variables we are interested in
    data = data[variables + labels + extra_info]
    
    # drop rows with invalid values
    data = data.dropna(axis=0, how='any')
    
    # get x dataset containing the variables to be trained on
    x = data[variables]
    
    # get the y dataset containing the number of brownlow votes awarded
    y = data[labels]
    
    # get a dataset to store game id's
    info = data[extra_info]
    
    # get training data
    x_tr = x[info.year < 2021]
    y_tr = y[info.year < 2021]
    y_oh_tr = pd.get_dummies(y_tr['brownlow_votes'])    
    info_tr = info[info.year < 2021]
    match_ids_tr = np.unique(info_tr.match_id)
    player_names_tr = np.unique(info_tr.player_id)
    training_data = [x_tr, y_tr, y_oh_tr, info_tr, match_ids_tr, player_names_tr]
    
    # get individual testing data
    x_ts = x[info.year == 2021]
    y_ts = y[info.year == 2021]
    y_oh_ts = pd.get_dummies(y_ts['brownlow_votes'])
    info_ts = info[info.year == 2021]
    match_ids_ts = np.unique(info_ts.match_id)
    player_names_ts = np.unique(info_ts.player_id)
    testing_data = [x_ts, y_ts, y_oh_ts, info_ts, match_ids_ts, player_names_ts]
    
    return training_data, testing_data
    
