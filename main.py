import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
geography_data_frame = pd.read_csv("analytics/geography/table.csv")
content_data_frame = pd.read_csv("analytics/content/table.csv")

def convert_duration_to_seconds(x):
    if pd.isna(x):
        return None
    h, m, s = x.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)

def preprocess_content(data):
    data.fillna(0, inplace=True)
    data["Estimated clicks"] = (data["Impressions"] * (data["Impressions click-through rate (%)"] / 100)).round().astype(int)
    data["Video publish time"] = pd.to_datetime(data["Video publish time"], errors="coerce")
    data["Average view duration (sec)"] = data["Average view duration"].apply(convert_duration_to_seconds)
    
    # Define threshold for high CTR (top 20%)
    threshold = data["Impressions click-through rate (%)"].quantile(0.8)
    data["High_CTR"] = (data["Impressions click-through rate (%)"] >= threshold).astype(int)
    
    return data

# Preprocess content
cleaned_content_data = preprocess_content(content_data_frame)

# Features and target
x = cleaned_content_data[["Duration", "Estimated clicks", "Average view duration (sec)"]]
y = cleaned_content_data["High_CTR"]

# Split and keep a DataFrame copy of x_test for later
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
x_test_df = x_test.copy()  # Keep as DataFrame for attaching predictions later

# Scale numeric features
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Tune KNN model
def tune_model(x_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 10),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(x_train_scaled, y_train)

# Evaluate on training data
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_train_scaled, y_train)

# Predict on test set
y_pred = best_model.predict(x_test_scaled)

# Attach predictions and actual labels to test DataFrame
x_test_df = x_test_df.copy()
x_test_df['Actual_High_CTR'] = y_test.values  # align index
x_test_df['Predicted_High_CTR'] = y_pred

# Merge Video title and publish time for context
x_test_df = x_test_df.merge(
    cleaned_content_data[["Video title", "Video publish time"]],
    left_index=True, right_index=True
)

# Format durations for readability
def format_duration(seconds):
    if pd.isna(seconds):
        return None
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes}m {sec}s"

x_test_df["Duration_fmt"] = x_test_df["Duration"].apply(format_duration)
x_test_df["Avg_view_duration_fmt"] = x_test_df["Average view duration (sec)"].apply(format_duration)

# Filter predicted high CTR videos
predicted_high_ctr_videos = x_test_df[x_test_df["Predicted_High_CTR"] == 1]

# Select columns for display
display_cols = ["Video title", "Video publish time", "Duration_fmt", "Avg_view_duration_fmt",
                "Estimated clicks", "Actual_High_CTR", "Predicted_High_CTR"]

print("Predicted High CTR Videos:")
print(predicted_high_ctr_videos[display_cols])

# Output model evaluation
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(matrix)
