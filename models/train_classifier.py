import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
	'''
	Load sql table DR from database_filepath
	return messages, encoded_categories, categories names 
	'''
	engine = create_engine('sqlite:///' + database_filepath)
	df = pd.read_sql_table('DR', engine)
	X = df['message']
	Y = df.drop(columns=['id', 'message', 'original', 'genre'])	
	return X, Y, Y.columns

def tokenize(text):
	'''
	remove special characters.
	tokenize text and lemmatize
	'''
	text = re.sub("^a-z0-9A-Z", " ", text)
	tokenized_words = word_tokenize(text)
	word_lemmatizer = WordNetLemmatizer()
	lemmatized_tokens = list()
	for token, postag in pos_tag(tokenized_words):
		lemmatized_tokens.append(word_lemmatizer.lemmatize(token))
	return lemmatized_tokens	
	
def build_model():
	'''
	create ML Pipeline.
	CountVectorizer -> TfIdfTransformer -> MultiOutPutClassifier with LinerSVC
	'''
	pipeline = Pipeline([
		('cvt', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', MultiOutputClassifier(estimator=LinearSVC(verbose=0)))
	])
	#perform grid search
	parameters = {'clf__estimator__loss':['hinge', 'squared_hinge'],
             'clf__estimator__multi_class': ['ovr', 'crammer_singer'],
             'clf__estimator__max_iter' : [10000, 20000, 25000]}
	
	cv = GridSearchCV(pipeline, parameters)
	return cv


def evaluate_model(model, X_test, Y_test, category_names):
	'''
	prints Accuracy
	'''
	score = model.score(X_test, Y_test)
	print("Accuracy : ", score)

def save_model(model, model_filepath):
	'''
	save train model
	'''
	pickle.dump(model, open(model_filepath, 'wb'))


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

