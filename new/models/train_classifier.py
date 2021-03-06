import sys
# import libraries
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# NLP packages
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Pipeline packages
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

# ML packagess
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Evaluating packages
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report





def load_data(database_filepath):
    """Load data from database_filepath
    return : a dataframe
    """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df["message"]
    Y = df[df.columns[4:]]
    category_names = list(df.columns[4:])
    return X, Y,category_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text =text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """ build a multiOutputClassifier model
    return a GridSearchCV model
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC(C=0.1)))
    ])    
    parameters = {
    'vect__max_df': (0.5, 0.75), #0.5,  
    # 'vect__max_features': (None, 5000, 10000),
    # 'tfidf__use_idf': (True, False),
    #'clf__penalty': ['l1', 'l2'],
    #'clf__C':[0.01,0.1]
    }
    model = GridSearchCV(model, param_grid=parameters)
    return model 


def evaluate_model(model, X_test, Y_test,category_names):
    """ Here I caculated the accuracy and F1_scores."""
    y_pred = model.predict(X_test)    
    for i in range(len(category_names)):
        print("Accuracy score for "+Y_test.columns[i], accuracy_score(Y_test.values[:,i],y_pred[:,i]))
        print("classification_report",classification_report(Y_test.values[:,i],y_pred[:,i],target_names=["0","1"]))
    
def save_model(model, model_filepath):
    """save model to pickle file
    """
    file = open(model_filepath, "wb")
    pickle.dump(model, file)
    file.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
