
import numpy as np

# a function which sorts a given array by game-id
def sort_by_game(x, info, array):
    
    match_ids = np.array(info.match_id)
    
    # create array to store predictions
    sorted_array = [[] for i in range(len(match_ids))]
    
    # for every entry
    for i in range(x.shape[0]):
        
        # get the game-id
        match_id = np.array(info.match_id)[i]
        
        # 
        
        # get the position of that game-id
        match_index = np.argwhere(match_ids == match_id)
        
        # add prediction to list
        sorted_array[match_index].append(array[i])
    
    # return sorted predictions
    return sorted_array

###############################################################################

# a function which computes a player's expected votes
# given the probability distribution of votes
def get_expected_votes(preds_sort):

    # array to store expected votes
    exp_votes = [[] for i in range(len(preds_sort))]
    
    # for every game
    for game in range(len(preds_sort)):
        
        # for every player in that game
        for player in range(len(preds_sort[0])):
            
            # compute expected votes
            exp_votes[game].append(preds_sort[game][player][0]*0 + 
                                   preds_sort[game][player][1]*1 + 
                                   preds_sort[game][player][2]*2 + 
                                   preds_sort[game][player][3]*3)
    
    # return expected votes
    return exp_votes

###############################################################################

# scale expected votes so they the sum of expected votes adds to 6 in each game
def scale_expected_votes(exp_votes_game):
    
    # array to store scaled expected votes
    exp_votes_game_scaled = [[] for i in range(len(exp_votes_game))]
    
    # for every game
    for i in range(len(exp_votes_game)):
        
        # compute total votes
        total = np.sum(exp_votes_game[i])
        
        # for every player in that game
        for j in range(len(exp_votes_game[i])):
            
            exp_votes_game_scaled[i].append(exp_votes_game[i][j]/total*6)
            
    return exp_votes_game_scaled

###############################################################################

def compute_predictions(exp_votes_sort):
    
    pred_votes_sort = [[0 for player in game] for game in exp_votes_sort]

    # for every game
    for game in range(len(exp_votes_sort)):

        # get index of highest vote-getters
        sorted_array = np.argsort(exp_votes_sort[game])
        index_3_vote = sorted_array[-1]
        index_2_vote = sorted_array[-2]
        index_1_vote = sorted_array[-3]

        pred_votes_sort[game][index_3_vote] = 3
        pred_votes_sort[game][index_2_vote] = 2
        pred_votes_sort[game][index_1_vote] = 1
    
    return pred_votes_sort

###############################################################################

def sort_votes_by_player(exp_votes_sort_scaled, info_sort, player_names):
    
    # create array to store player votes
    player_votes = [[] for i in range(len(player_names))]

    # for every game in the dataset
    for game in range(len(exp_votes_sort_scaled)):
    
        # for every player in that game
        for player in range(len(exp_votes_sort_scaled[0])):
        
            # retrieve player votes
            votes = exp_votes_sort_scaled[game][player]
        
            # player name
            player_name = info_sort[game][player][4]
        
            # get the position of that player
            player_index = np.argwhere(player_names == player_name)[0][0]
        
            # add votes to list
            player_votes[player_index].append(votes)
    
    # return player votes
    return player_votes

###############################################################################

def compute_total_votes(player_votes):
    player_vote_totals = [np.sum(player_votes[i]) for i in range(len(player_votes))]
    return player_vote_totals

###############################################################################

def see_confusion_matrix(preds_sort, y_sort):

    # make confusion matrix
    confusion_matrix = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

    # for every game
    for game in range(len(preds_sort)):

        # for every player in the game
        for player in range(len(preds_sort[game])):

            truth = y_sort[game][player]
            prediction = preds_sort[game][player]
            confusion_matrix[truth][prediction] += 1

    print(confusion_matrix)

###############################################################################

# show a prediction from the testing dataset
def show_game_prediction(probs_game, exp_votes_raw_game, exp_votes_game, pred_votes_game, y_game, info_game, game_num):

    # print the corresponding game
    print(info_game[game_num][0][1], info_game[game_num][0][2], info_game[game_num][0][3], '\n')
    
    # print each player's predicted votes + true votes
    for i in range(len(exp_votes_game[game_num])):
        print('[ %.2f %.2f %.2f %.2f ] %.2f %.2f  %d  |  %d  %s' % (probs_game[game_num][i][0],
                                                               probs_game[game_num][i][1],
                                                               probs_game[game_num][i][2],
                                                               probs_game[game_num][i][3],
                                                               exp_votes_raw_game[game_num][i],
                                                               exp_votes_game[game_num][i],
                                                               pred_votes_game[game_num][i],
                                                               y_game[game_num][i], 
                                                               info_game[game_num][i][4]))

###############################################################################

# show a prediction from the testing dataset
def show_game_prediction_2(exp_votes_raw_game, exp_votes_game, pred_votes_game, y_game, info_game, game_num):

    # print the corresponding game
    print(info_game[game_num][0][1], info_game[game_num][0][2], info_game[game_num][0][3], '\n')
    
    # print each player's predicted votes + true votes
    for i in range(len(exp_votes_game[game_num])):
        print('%.2f %.2f  %d  |  %d  %s' % (exp_votes_raw_game[game_num][i],
                                            exp_votes_game[game_num][i],
                                            pred_votes_game[game_num][i],
                                            y_game[game_num][i], 
                                            info_game[game_num][i][4]))

###############################################################################
        
# show n highest vote-getters
def show_most_votes(player_vote_totals, player_names, n):
    for player_num in np.argsort(player_vote_totals)[-n:]:
        print('%.2f  %s' % (player_vote_totals[player_num], player_names[player_num]))