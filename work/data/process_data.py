import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """load messages and categories dataset
    return: a merge dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id",how="inner")
    return df

def clean_data(df):
    """clean categories info
    return: a cleande dataframe
    """
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    category_colnames = list(row.apply(lambda x:x.split("-")[0]))
    categories.columns = category_colnames
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    #Id = pd.DataFrame(df["id"])
    #categories =categories.reset_index()
    #categories = Id.join(categories,how="inner")
    #categories = pd.melt(categories, id_vars="id",value_vars=categories.columns[1:])
    #categories =categories[categories["value"]==1]
    
    del df["categories"]
    df = df.join(categories,how="inner")
    #df = df.loc[:,["id","message","variable"]]
    df = df.drop_duplicates()
    
    #group = list(df["variable"].unique())
    #dic  ={}
    #for i in group:
        #dic[i]=group.index(i)
    #df["categories"]= df["variable"].apply(lambda x:dic[x])
    
    return df


def save_data(df, database_filename):
    """save to a database
    return: none
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False)


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