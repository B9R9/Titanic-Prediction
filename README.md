# Titanic-Prediction
***
## Overview
This project aims to predict survival on the Titanic using machine learning techniques. We analyze the Titanic dataset and develop predictive models to determine whether a passenger survived or not.

## Project Context
This project is part of the [Kaggle](#https://www.kaggle.com/) [Titanic: Machine Learning from Disaster](#https://www.kaggle.com/competitions/titanic) competition, which challenges participants to predict the survival of passengers aboard the RMS Titanic based on various features such as age, gender, ticket class, etc.

### Problem Statement
The sinking of the Titanic on April 15, 1912, is one of the most infamous shipwrecks in history. Despite being equipped with advanced safety features for its time, a combination of factors such as insufficient lifeboats and the mismanagement of the evacuation resulted in a significant loss of life.

The goal of this project is to develop a machine learning model that can accurately predict whether a passenger survived the Titanic disaster based on the available data. By analyzing the factors that contributed to survival, we aim to gain insights into the demographics of the passengers and the effectiveness of different strategies for evacuation.

### Objectives
- Build a predictive model that can classify passengers as survivors or non-survivors.
- Evaluate the performance of the model using appropriate metrics such as accuracy, precision, recall, and ROC AUC score.
- Explore the relationship between various features and the likelihood of survival.
- Use the insights gained to improve the model and potentially inform future disaster preparedness efforts.

### Why This project
I embarked on this project with the goal of delving into the realms of machine learning and data analysis. With little prior knowledge in statistics except for my logical intuition, I found myself more captivated by the process than the ultimate outcome, which still leaves something to be desired, but that wasn't the primary objective.

From data preparation to uncovering the interconnections within the dataset, leveraging the data to its fullest, and grasping the event's context, proved to be both challenging and incredibly instructive. Experimenting with various configurations to find the optimal setup was a stimulating endeavor.

I endeavored to construct a database that provided binary responses to enhance data exploitation, recognizing that there's significant room for improvement in data exploitation.

Exploring the structure of the code was equally fascinating, especially coming from a background of coding in C. Creating and utilizing classes stood out as a key aspect for me.

Looking ahead, I envision creating a Jupyter notebook to meticulously outline each step of the process and visually showcase the data I'm working with, aiming for a more comprehensive and illustrative presentation of my work.


## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-strucuture)
***

## Installation
1. Clone the repository 
```bash
git clone https://github.com/B9R9/Titanic-Prediction.git
```
2. Create an virtual environment (Python 3.10.10 used) 
```bash
python3 -m venv venv
```
3. Activate the virtual environement.
  - On Windows:
```bash
venv\Scripts\activate
```
  - On Unix or MacOS:
```bash
source venv/bin/activate
```
4. Install requirements:
```bash
pip install -r requirements.txt
```
To desactivate the virtual environement:
  - On Window:
  ```bash
  path_to_env\Scripts\deactivate
  ```
  - On Unix or MacOS:
```bash
deactivate
```

## Project-Strucuture
- data/: Contains the dataset files.
- modules/: Contains custom Python modules for data preprocessing and modeling.
- preprocessing.py: Script for preprocessing the data.
- Transformers.py: Script for preprocessing the data.
- TitanicModel.py: Script for training the predictive model and for making predictions on new data.
- requirements.txt: List of dependencies required for the project.

## Contact Me
Feel free to reach out if you have any questions, suggestions, or just want to chat about data science, machine learning, or any related topics. Your feedback and ideas are always welcome and appreciated. You can contact me via Discord at [baptiste_b9r9](https://discord.com/). Let's collaborate and explore the fascinating world of data together!
