# DataScience Project 2: Disaster Response Pipeline Project
This is the second project of the Data Science Nanodegree program: A web app is developed in which disaster messages are classified according to 36 different categories.

## 1. Installations
The 3 folders app, data, and model must be copied to a server which runs Flask and Python with the following Python packages: NLTK, RE, SQLAlchemy, Numpy, Pandas, and scikit-learn.

## 2. Project Motivation
The project shows how typical data science pipelines can be applied using the example of disaster message classification. At first, an ETL pipeline is used to gather data from two sources (disaster messages and corresponding category classifications), combines them, and stores them in a database. Subsequently, the data is loaded into a second pipeline which performs some natural language processing on disaster messages before a Machine Learning model is created and trained. The stored model is then incorporated into a Flask web app. Here, users can enter their own messages which are then classified according to the 36 categories.

## 3. File Description
The project has the following structure:
```bash
app
├── templates
│   ├── go.html             # Renders classification results
│   └── master.html         # Renders the main view for the web browser including database statistics
└── run.py                  # Script that starts the web app
data
├── disaster_categories.csv # csv file containing the categories corresponding to the messages in disaster_messages.csv
├── disaster_messages.csv   # csv file containing the disaster messages
└── process_data.py         # ETL pipeline to transform the two csv files into one SQLite database
models
└── train_classifier        # Machine Learning pipeline which creates a multi-categorial classifier and stores it as a pickle-file
```

## 4. How to execute the project:
1. Run the following commands in the root directory to set up the database and model.
    - At first run the ETL pipeline that cleans data and stores in a database. The first two parameters are the paths of the two input files. The third parameter is path to the output file. Of course, the paths can be adapted. To process the pipeline, enter:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - Subsequently, run the ML pipeline that trains the classifier and saves it. The first parameter is the path to the previously created database. Do not change the second parameter which specifies the output file path. The web app requires that the file is names "classifier.pkl". Please note, that the training process takes a long time (up to an hour depending on your system). To process the pipeline, enter:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command from the app directory to run the web app.
	- Enter the command `env | grep WORK` to find your workspace variables
	- Enter the command `python run.py`
    
3. Open a new web browser window and go to the following web address: 
	`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values. Note: This works only in the Udacity Project Workspace IDE. If you run the application somewhere else, check the corresponding instructions. 

## 5. Licensing, Authors, Acknowledgements
Thanks to Udacity for the great lessons regarding ETL and Machine Learning Pipelines.
