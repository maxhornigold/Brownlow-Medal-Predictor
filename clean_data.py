
import numpy as np

def clean_data(data):
    
    data['match_round'] = data['match_round'].astype('str')
    
    
    # Remove games in finals series'. Brownlow votes are not awarded in final's series'.
    data = data[(data['match_round'] != 'Elimination Final') &
                (data['match_round'] != 'Finals Week 1') &
                (data['match_round'] != 'Qualifying Final') &
                (data['match_round'] != 'Semi Final') &
                (data['match_round'] != 'Semi Finals') &
                (data['match_round'] != 'Preliminary Final') &
                (data['match_round'] != 'Preliminary Finals') &
                (data['match_round'] != 'Grand Final')]
    
    data['year'] = data.apply(lambda row: row.match_date[0:4], axis=1)
    data['year'] = data['year'].astype(float)
    
    # Add columns to show whether the game was a win or a loss
    data['win'] = np.where((data['match_margin']) > 0, 1, 0)
    data['draw'] = np.where((data['match_margin']) == 0, 1, 0)
    data['loss'] = np.where((data['match_margin']) < 0, 1, 0)
    
    # drop the first unnecessary column
    data = data.drop('Unnamed: 0', axis=1)
    
    # normalize variables as a proportion of that game
    normalisable_variables = ['kicks', 'marks', 'handballs', 'disposals', 'effective_disposals', 'goals',  
                              'behinds', 'hitouts', 'tackles', 'rebounds', 'inside_fifties', 
                              'clearances', 'clangers', 'free_kicks_for', 'free_kicks_against', 
                              'contested_possessions', 'uncontested_possessions', 'contested_marks', 
                              'marks_inside_fifty', 'one_percenters', 'bounces', 'goal_assists', 
                              'afl_fantasy_score', 'centre_clearances', 
                              'stoppage_clearances', 'score_involvements', 'metres_gained',
                              'turnovers', 'intercepts', 'tackles_inside_fifty', 'contest_def_losses',
                              'contest_def_one_on_ones', 'contest_off_one_on_ones', 'contest_off_wins',
                              'def_half_pressure_acts', 'effective_kicks', 'f50_ground_ball_gets',
                              'ground_ball_gets', 'hitouts_to_advantage', 'intercept_marks', 
                              'marks_on_lead', 'pressure_acts', 'rating_points', 'ruck_contests', 
                              'score_launches', 'shots_at_goal', 'spoils']

    for variable in normalisable_variables:
        name = variable + "_s"
        data[name] = data.groupby('match_id', group_keys=False).apply(lambda g: g[variable] / g[variable].sum())
    
    return data