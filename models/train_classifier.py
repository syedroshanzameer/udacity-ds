import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

import pickle

#Downloading required nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath): 
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('clean_table', con = engine)
    X = df['message']
    Y = df.iloc[:,4:]
    return X,Y


def tokenize(text):
    tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in filtered_sentence:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer = tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
])
#     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 1)
    
    parameters = {
        'clf__estimator__n_estimators': [10,20],
        # 'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__n_jobs': [-1]
    }

    cv = GridSearchCV(pipeline,param_grid=parameters)
#     cv.fit(X_train,y_train)
    return cv


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], Y_pred[:, index]))


def save_model(model, model_filepath):
#     filename = 'model.pkl'
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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