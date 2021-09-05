# Disaster Response Pipeline Project with Figure Eight
Udacity Data Scientist Nanodegree Program Project
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#description)
4. [How To Run](#run)


## Installation<a name="installation"></a>
To run the code use python>=3.8, to install dependecies run pip install -r requirements.txt

## Project Motivation<a name="motivation"></a>
Figure Eigth has collected and provided a pre-labeled tweets and text messages from real life disasters. This project tries to utilize the data and ML pipelines to build a Supervised Learning model which can be used to categorize the type of emergency or messgae received during a disater.

## File Descriptions<a name="description"></a>
1. data/process_data.py does the data preparation using DATA for ML model, it <b>load_data</b>,<b>clean_data</b> and <b>save_data</b>
2. model/train_classifier preapares the model using the data present in the DB. it creates ML Pipeline for data preprocessing(text vectorization) and training a Supervised Learning Model.

## How To Run<a name="run"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


