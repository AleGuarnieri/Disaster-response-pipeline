# Disaster Response Pipeline Project

##  Installation and Prerequisites
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseDB.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseDB.db models/RandomForestClassifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to localhost:3001

## Motivation
This project was implemented as part of a Data Science course to put into practice ETL and ML pipelines.
Implementing it allowed me to understand and practice Data engineering skills, the basics of NLP and their application
to the implementation of an ML model through pipelines. 

## File Description
Data directory: contains data from Figure8 and the script process_data.py containing the ETL pipeline to clean data. 
Models directory: contains the script train_classifier.py containing ML pipeline based on NLP to train and evaluate ML model.
App directory: contains the script run.py used to run the flask web-app which visualize the data and deploy the classifier so
it can be used.

## SUmmary and Details
The purpose of the project is to classify disaster messages. A user can input a message into the web-app, which highlights
the categories to which the messsage corresponds. The app runs an ML model in order to classify new messages.
This model was trained on data provided by Udacity and manipulated by an ETL pipeline.

## Acknowledgements
Udacity provided the course material (both data and the skeleton of the web app)




