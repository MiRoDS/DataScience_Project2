# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# Extract step: Reads in a message and a categories dataset and merges them into dataframe
def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - Path to the message dataset
    categories_filepath - Path to the categories dataset

    OUTPUT:
    df - Dataframe with the combined dataset

    This function reads a message and a categories dataset and merges them into dataframe.
    '''
    
    # Read in messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Read in categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merge messages and categories datasets    
    df=messages.merge(categories, on='id')
    
    return df


# Transform step: Creates 36 category columns describing whether a message belongs to a category or not 
def clean_data(df):
    '''
    INPUT:
    df - Dataframe with the dataset to be cleaned

    OUTPUT:
    b - Dataframe with the cleaned dataset 

    This function creates 36 category columns describing whether a message belongs to a category or not. 
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.map(lambda i: (i[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda i: (i[-1:]))

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    del df['categories']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


# Load step: Stores that transformed dataset in a SQLite database
def save_data(df, database_filename):
    '''
    INPUT:
    df - Dataframe with the dataset to be saved
    database_filename - Filename for the dataset to be saved

    OUTPUT:
    n/a

    This function stores a transformed dataset in a SQLite database.
    '''
    
    # Create a database
    engine = create_engine("sqlite:///"+database_filename)
    
    #Store the dataframe as a database table
    df.to_sql('OneAndOnlyTable', engine, index=False)  


# Processes an ETL pipeline to prepare disaster messages data for a ML pipeline    
def main():
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
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
