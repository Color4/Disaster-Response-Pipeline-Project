# Disaster-Response-Pipeline-Project

## Overview #

Disaster Response Pipeline Project is intended to be a library and webapp that can be used to quickly generate high-quality information classfication to support disaster response.  

The basic archictecture here is:

1.  Data Process
2.  Classfication Model Build
3.  Web App

## The Need For High Quality Disaster information classification

Effective disaster recovery is aided by quick and accurate damage assements so that volunteers and material can be efficiently
put to use in a timely manner.  However, building special purpose software for this task to work 
across handsets of all different types is a pretty herculean task - much more ideal to come up a with a machine learning protocol that will
work with pretty much any combination of hardware and operating system.

## Constraints On Such A System

* This system will require working wireless data infrastructure.  While this is often the case, very serious damage could make this effort useless
until the data networks are repaired in disaster areas.

## What The Code Does

The stated goal of this project is to two things to the data in a disaster response project:

1. Trim out all information that's unecessary for disaster recovery operations
2. Transform the data into simpler data structures or data visualization

The ultimate goal being the creation of clean and useful data for ingestion into various disaster management systems.

# Technical Info

## Depends #
1. Pandas for data process.
2. NLTK  and sklearn for modeling.
3. plotly and flask for web app.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
