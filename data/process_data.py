import sys
import pandas as pd
from sqlalchemy import create_engine


"""
This script is used to Extract, Transform and Load (ETL) data from the
provided .csv file to the DB. The transformation involve data manipulation
with Pandas and some NLP
"""

def load_data(messages_filepath, categories_filepath):
    """
    This function extracts data from .csv into a pandas Dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')

    return df


def clean_data(df):
    """
    This function transforms extracted data in order to clean both features
    and labels
    """
    #creating "categories" columns
    categories = df['categories'].str.split(";", expand=True)

    #renaming "categories" columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames

    #cleaning "categories" columns
    for column in categories:
        # set each value to be the last character of the string and integer
        categories[column] = categories[column].apply(lambda x : x[-1])
        categories[column] = categories[column].apply(lambda x : int(x))
    df.drop('categories', axis=1, inplace = True)

    #concatenating feature columns and categories columns
    df = pd.concat([df, categories], axis=1)

    #removing duplicates and rows with not binary values
    df.drop_duplicates(inplace=True)
    dfDupsMessage = df[df.duplicated(['message'],keep=False)]
    df.drop(dfDupsMessage.index, inplace=True)
    df.drop('original', axis=1, inplace = True)
    df.drop(df[df.related == 2].index, inplace= True)

    return df
    
def save_data(df, database_filename):
    """
    This function loads cleaned data into a Database
    """
    #connecting to DATABASE and saving data to table DisasterResponseTable
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('DisasterResponseTable', engine, if_exists='replace', index=False)  


def main():
    """
    Main function used to execute the other functions and manage errors
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponseDB.db')


if __name__ == '__main__':
    main()
