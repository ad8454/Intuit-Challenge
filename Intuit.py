"""
This program generates a a list of features
for each user

Author: Ajinkya Dhaigude (ad8454@rit.edu)
"""

import pandas as pd
import numpy as np

def main():

    # read in the data
    user_data = None
    file_id = 0
    while(file_id < 100):
        file_name = 'transaction-data/user-' + str(file_id) + '.csv'
        file_id += 1
        df = pd.read_csv(file_name)
        if user_data is None:
            user_data = df
        else:
            user_data = user_data.append(df, ignore_index=True)

    user_data.columns = ['auth_id', 'date', 'vendor', 'amount', 'location']
    user_data.drop('location', axis=1, inplace=True)
    user_data['date'] = pd.to_datetime(user_data['date'], format='%m/%d/%Y', errors='coerce')


    # create keywords for features
    transport = ['train', 'lyft', 'taxi', 'uber', 'bus']
    art = ['art', 'paint', 'craft', 'art museum']
    music = ['guitar', 'concert']
    food = ['pizza', 'burger', 'chinese', 'coffee', 'restaurant', 'grill']
    groceries = ['grocery', 'groceries', 'market', 'foods']
    living = ['rent', 'loan', 'water', 'gas', 'cable']
    movie = ['netflix', 'movie', 'dvd', 'on demand']
    baby = ['baby','babies', 'parental']
    moving = ['movers', 'move', 'furniture', 'home', 'refrigerator']
    wedding = ['wedding']
    outdoor = ['skating', 'bowling', 'tickets', 'concert', 'resort', 'nfl', 'nba', 'bike']
    pubs = ['club', 'bar', 'brewery', 'wine']
    income = ['paycheck']
    divorce = ['divorce']
    hotel = ['hotel', 'inn', 'storage']
    video_games = ['video game', 'playstation']
    scifi = ['star wars', 'star trek', 'geek']
    education = ['education', 'course', 'science', 'mathematics', 'biology']
    sports = ['sports', 'sporting', 'nfl', 'nba', 'athletic', 'gnc', 'bike', 'gym', 'vitamin']
    late = ['late', 'penalty', 'negative balance', 'overdraft']
    pets = ['pet', 'cat']
    indoor = ['delivery', 'rental', 'grubhub', 'library']

    # add numerical features
    user_data['year'], user_data['month'] = user_data['date'].dt.year, user_data['date'].dt.month
    user_info = add_feature_num(user_data, None, income, 'Yearly_Income')
    user_data = user_data.drop('year', 1)
    user_data = user_data.drop('month', 1)


    # add binary categorical features
    user_info = add_feature(user_data, user_info, education, 'Is_Student')
    user_info = add_feature(user_data, user_info, baby, 'Has_Baby')
    user_info = add_feature(user_data, user_info, pets, 'Has_Pet')
    user_info = add_feature(user_data, user_info, divorce, 'Is_Divorced')
    user_info = add_feature(user_data, user_info, wedding, 'Recently_Married')
    user_info = add_feature(user_data, user_info, late, 'Has_High_Debt', 4000)
    user_info = add_feature(user_data, user_info, pubs, 'Lifestyle_Nightlife', 8000)
    user_info = add_feature(user_data, user_info, outdoor, 'Lifestyle_Outdoor', 8000)
    user_info = add_feature(user_data, user_info, indoor, 'Lifestyle_Indoor', 8000)
    user_info = add_feature(user_data, user_info, art, 'Hobby_Art', 2000)
    user_info = add_feature(user_data, user_info, movie, 'Hobby_Movies', 2000)
    user_info = add_feature(user_data, user_info, music, 'Hobby_Music', 2000)
    user_info = add_feature(user_data, user_info, scifi, 'Hobby_SciFi', 2000)
    user_info = add_feature(user_data, user_info, sports, 'Hobby_Sports', 2000)
    user_info = add_feature(user_data, user_info, video_games, 'Hobby_Video_Games', 2000)
    user_info = add_feature(user_data, user_info, food, 'Dine_Eat_Out', 7000)
    user_info = add_feature(user_data, user_info, groceries, 'Dine_Cook_At_Home', 5000)

    # output to file
    user_info.to_csv('output/user_features.csv')


def add_feature(user_data, user_info, featureList, featureName, thresh=0):
    df = user_data[user_data['vendor'].str.contains('|'.join(featureList), case=False)]
    df = df.groupby(['auth_id'])[['amount']].sum().abs().reset_index()
    df[featureName] = np.where(df['amount'] >= thresh, 'Yes', 'No')
    df.drop('amount', 1, inplace=True)
    user_info = pd.merge(user_info, df, how='outer', on='auth_id')
    user_info[featureName].fillna('No', inplace=True)
    return user_info

def add_feature_num(user_data, user_info, featureList, featureName, thresh=0):
    df = user_data[user_data['vendor'].str.contains('|'.join(featureList), case=False)]
    df = df.groupby(['auth_id', 'year'])[['amount']].sum().abs().reset_index().groupby('auth_id') \
        .mean().reset_index().drop('year', 1)
    df[featureName] = df['amount']
    df.drop('amount', 1, inplace=True)
    if user_info is None:
        user_info = df
    else:
        user_info = pd.merge(user_info, df, how='outer', on='auth_id')
    user_info[featureName].fillna('No', inplace=True)
    return user_info

main()