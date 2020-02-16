import sys

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle


# Loads data prepared by an ETL pipeline from a SQLite database
def load_data(database_filepath):
    # load data from database
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('OneAndOnlyTable', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = pd.Series(df.columns)
    category_names = category_names.loc[4:]
    return X, Y, category_names


# Tokenizes text
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Builds a model
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])    
    
    parameters = {
        'clf__estimator__n_estimators': [25, 50],
        'clf__estimator__criterion': ["gini", "entropy"],
        'clf__estimator__min_samples_split': [2, 3]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


# Evaluates the model
def evaluate_model(model, X_test, Y_test, category_names):
    # Predict categories for test data
    Y_pred = model.predict(X_test)
    
    # Statistics
    
    # Shows the overall mean accuracy of all categories
    print("Overall mean accuracy:", (Y_pred == Y_test).mean().mean())
    print("")

    # Loop over all categories for more statistics
    for cat in range(len(category_names)):
        print("Category:", category_names[cat]) # Print the category name
        print("Accuracy:", (Y_test.loc[:, category_names[cat]] == Y_pred[:, cat]).mean()) # Print the category accuracy
        print(classification_report(Y_test.loc[:, category_names[cat]], Y_pred[:, cat])) # Print the corresponding classification report
    

# Saves model as pickle file
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f: 
        pickle.dump(model, f)


# Processes a ML pipeline for the classification of disaster messages           
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
