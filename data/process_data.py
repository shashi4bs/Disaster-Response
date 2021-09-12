import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	'''
	Loads data from categories_filepath and messages_filepath as dataframe
	MEGRE both on id
	return merged df
	'''
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv(categories_filepath)
	df = pd.merge(messages,categories,how="inner",on='id')
	return df

def clean_data(df):
	'''
	clean df.
	encode categories and drop duplicate rows
	return df -> original_data + encoded_categories - categories(text)
	'''
	categories = df.categories.str.split(';', expand=True)
	row = categories.loc[0]
	category_colnames = row.apply(lambda x: x[:-2])
	categories.columns = category_colnames
	for column in categories:
		categories[column] = categories[column].str[-1]
		# convert column from string to numeric
		categories[column] = categories[column].apply(int)
	
	#drop related column
	categories.drop(columns="related", inplace=True)

	df.drop(columns='categories', inplace=True)
	df = pd.concat((df, categories), axis=1)
	df.drop_duplicates(inplace=True)
	df.drop(columns='child_alone', inplace=True)
	return df


def save_data(df, database_filename):
	'''
	dump df as sql table
	Table Name : DR
	'''
	engine = create_engine('sqlite:///'+ database_filename)
	df.to_sql('DR', engine, index=False, if_exists="replace") 

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
