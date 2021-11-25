
import pandas as pd

def retrieve_data():
    
    # read the data from the file
    data_2012 = pd.read_csv("../../../Resources/Datasets/player_stats_2012.csv")
    data_2013 = pd.read_csv("../../../Resources/Datasets/player_stats_2013.csv")
    data_2014 = pd.read_csv("../../../Resources/Datasets/player_stats_2014.csv")
    data_2015 = pd.read_csv("../../../Resources/Datasets/player_stats_2015.csv")
    data_2016 = pd.read_csv("../../../Resources/Datasets/player_stats_2016.csv")
    data_2017 = pd.read_csv("../../../Resources/Datasets/player_stats_2017.csv")
    data_2018 = pd.read_csv("../../../Resources/Datasets/player_stats_2018.csv")
    data_2019 = pd.read_csv("../../../Resources/Datasets/player_stats_2019.csv")
    data_2020 = pd.read_csv("../../../Resources/Datasets/player_stats_2020.csv")
    data_2021 = pd.read_csv("../../../Resources/Datasets/player_stats_2021.csv")
    
    # combine into a single dataset
    data = pd.concat([data_2012, data_2013, data_2014, data_2015, 
                      data_2016, data_2017, data_2018, data_2019, 
                      data_2020, data_2021])
    
    return data