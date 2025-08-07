'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv("data/prediction_model_03.csv")
    genres_df = pd.read_csv("data/genres.csv")
    # print(model_pred_df)
    # print(genres_df)
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''

    # Your code here
    genre_list = genres_df['genre'].tolist()
    # print(genre_list)

    genre_true_counts = model_pred_df['actual genres'].value_counts().to_dict()
    # print(genre_true_counts)

    genre_tp_counts = model_pred_df.loc[model_pred_df['correct?'] == 1, 'predicted'].value_counts().to_dict()
    print(f'Number of True Positives: {genre_tp_counts}')

    
    genre_fp_counts = model_pred_df.loc[model_pred_df['correct?'] == 0, 'predicted'].value_counts().to_dict()
    print(f'Number of False Positives: {genre_fp_counts}')

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts

