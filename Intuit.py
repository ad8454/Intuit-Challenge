import pandas as pd
import numpy as np

def main():
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
    #user_data.set_index('date', inplace=True)

    transport = ['train', 'lyft', 'taxi', 'uber', 'bus']
    art = ['art', 'paint', 'craft', 'art museum']
    music = ['guitar', 'concert']
    food = ['pizza', 'burger', 'chinese', 'grubhub', 'coffee', 'restaurant', 'grill']
    supplies = ['grocery', 'groceries', 'market', 'foods']
    living = ['rent', 'loan', 'water', 'gas', 'cable']
    movie = ['netflix', 'movie', 'dvd', 'on demand']
    baby = ['baby','babies', 'parental']
    moving = ['movers', 'move', 'storage']
    furniture = ['furniture', 'home', 'refrigerator']
    wedding = ['wedding', ]
    outdoor = ['skating', 'bowling', 'tickets', 'club', 'bar']
    pubs = ['club', 'bar', 'brewery', 'wine']
    vacation = ['flight', 'resort']
    income = ['paycheck', 'credit card payment']
    divorce = ['divorce']
    hotel = ['hotel', 'inn']
    video_games = ['video game', 'playstation']
    books = ['library', 'book']
    misc = ['museum']
    scifi = ['star wars', 'star trek']
    science = ['science', 'education', 'geek', 'mathematics', 'biology', 'science museum'],
    education = ['education', 'course']
    sports = ['sports', 'sporting', 'nfl', 'nba', 'athletic', 'gnc', 'bike', 'gym', 'vitamin']
    late = ['late', 'penalty', 'negative balance', 'overdraft']
    pets = ['pet', 'cat']
    indoor = ['delivery']

    user_data['year'], user_data['month'] = user_data['date'].dt.year, user_data['date'].dt.month
    user_info = user_data.groupby(['auth_id', 'year'])[['amount']].sum().abs().reset_index().groupby('auth_id')\
        .mean().reset_index().drop('year', 1)
    user_info.columns = ['auth_id', 'Yearly_Spending']
    user_data = user_data.drop('year', 1)
    user_data = user_data.drop('month', 1)
    #print user_data

    user_info = add_feature(user_data, user_info, education, 'Is_Student')
    user_info = add_feature(user_data, user_info, baby, 'Has_Baby')
    user_info = add_feature(user_data, user_info, pets, 'Has_Pet')
    user_info = add_feature(user_data, user_info, divorce, 'Is_Divorced')

    print user_info

    #np.where(df['age'] >= 0, 'yes', 'no')

    #print user_data.groupby('auth_id')[['amount']].transform(sum)
    #print user_data.groupby('auth_id')[['amount']].sum()
    #print user_data

    new_columns = ['Monthly Savings', 'Transport', 'Art', 'Music', 'Food', 'House Supplies', 'Essential Living Cost', '']

    #print list(user_data)
    #print user_data['vendor'].unique()

    #df =  user_data[user_data['vendor'].str.contains("", case=False)]
    #print df.groupby(['auth_id'])[['amount']].sum()


def add_feature(user_data, user_info, featureList, featureName):
    df = user_data[user_data['vendor'].str.contains('|'.join(featureList), case=False)]
    df = df.groupby(['auth_id'])[['amount']].sum().reset_index()
    df[featureName] = 'Yes'
    df.drop('amount', 1, inplace=True)
    user_info = pd.merge(user_info, df, how='outer', on='auth_id')
    user_info[featureName].fillna('No', inplace=True)
    return user_info

main()