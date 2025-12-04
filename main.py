'''
Goal: I want to analyse youtube data and predict how future videos will perform based off of previous results.
Need to answer questions: 
1. On average at what point do most people exit my video? --> how long should my videos be?
2. At what time in the day do I get the most interactions? --> What time should I typically post
3. What is the duration of my videos vs the average view duration --> Again how long should my videos be?
4. How often should I post?
5. How does average view duration compare to total video length? --> Are your videos too long or too short?
6. Which geography delivers the highest watch time? (Think I should start here, easiest)
'''
# Second goal: I want to generate recommendations/advise for future channel performance
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Data frame for geography
geography_data_frame = pd.read_csv("analytics/geography/table.csv")
geography_data_frame.info()

content_data_frame = pd.read_csv("analytics/content/table.csv")
content_data_frame.info()

def convert_duration_to_seconds(x):
    if pd.isna(x):
        return None
    h, m, s = x.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

def preprocess_geography(data):
    # Converting data types for minutes and then for Geography
    # data = pd.get_dummies(geography_data_frame, columns=["Geography"], drop_first=True)
    geography_data_frame["Average view duration (sec)"] = geography_data_frame["Average view duration"].apply(convert_duration_to_seconds)
    # data.groupby("Geography")["Views"].sum().sort_values(ascending=False)
    return geography_data_frame

def preprocess_content(data):
    data.drop(columns=["Content"], inplace=True)
    data.fillna(0, inplace=True)
    data["Estimated clicks"] = data["Impressions"] * (data["Impressions click-through rate (%)"] / 100)
    data["Estimated clicks"] = data["Estimated clicks"].round().astype(int)
    data["Video publish time"] = pd.to_datetime(data["Video publish time"], errors="coerce")
    data["Average view duration (sec)"] = data["Average view duration"].apply(convert_duration_to_seconds) 

    # Defining params for ctr
    # Define threshold for high CTR (e.g., top 20%)
    threshold = data["Impressions click-through rate (%)"].quantile(0.8)  # top 20%
    data["High_CTR"] = (data["Impressions click-through rate (%)"] >= threshold).astype(int)
    return data

cleaned_content_data = preprocess_content(content_data_frame)

'''
Question: Will this video have a high click through rate based off of the metrics providded?
'''
x = cleaned_content_data[["Video title", "Duration", "Estimated clicks", "Average view duration (sec)"]]
y = cleaned_content_data["High_CTR"]

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

scaler = MinMaxScaler()

# This takes the training data
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Tuning the model
def tune_model(x_train, y_train):

    param_grid = {
        "n_neighbors" : range(1,10),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    return grid_search.best_estimator_

best_model = tune_model(x_train, y_train)

# Predictions + evaluation
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)

    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_train, y_train)


print(geography_data_frame.isnull().sum())
# print(preprocess_geography(geography_data_frame))
print(preprocess_content(content_data_frame))

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)