# DataScience Project 2: Disaster Response Pipeline Project

This is the second project of the Data Science Nanodegree program.

##1. Installations
The 3 folders app, data, and model must be copied to a server which runs Flask, Python, and the following Python packages: Pandas, Matplotlib, and scikit-learn.

##2. Project Motivation
Goal was to find out whethers renters can benefit from the findings if it is possible to influence the review score rating.

##3. File Description
The project has the following structure:
app
|-templates
| |-go.html
| |-master.html
|-run.py
data
|-disaster_categories.csv
|-disaster_messages.csv
|-process_data.py
models
|-train_classifier

##4. How to execute the project:

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory (Very important!) to run your web app.
	- Enter the command `env | grep WORK` to find your workspace variables
	- Enter the command `python run.py`
    
3. Open a new web browser window and go to the web address: 
	`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

##5. Licensing, Authors, Acknowledgements
Thanks to Udacity for the great lessons regarding ETL and Machine Learning Pipelines.
