from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import pandas as pd
import os


# Load the uploaded CSV file
file_folder = "/media/max/OS/Users/maxlg/Desktop/==RISELABS/Pusher Data Batch 2/"
data_files = {
    "jenga": "jenga/csv_output/merged_pushers_with_pose.csv",
    "allen_key": "allen_key/csv_output/merged_pushers_with_pose.csv",
    "wrench": "wrench/csv_output/merged_pushers_with_pose.csv",
}

# Input feature columns
input_cols = ['pose_x', 'pose_y', 'ori_y']

# Output model store
models = {}

for object_name, file_name in data_files.items():
    path = os.path.join(file_folder, file_name)
    df = pd.read_csv(path)

    # Create 'pusher_class': 1 if both present, 0 if only one
    df['pusher_class'] = ((df['green_index'].notna()) & (df['yellow_index'].notna())).astype(int)

    # Train classifier
    X = df[input_cols]
    y_class = df['pusher_class']
    clf = RandomForestClassifier(random_state=42)
    # clf = MLPClassifier(random_state=42)
    clf.fit(X, y_class)

    # One-pusher regressor
    one_pusher = df[df['pusher_class'] == 0].copy()
    one_pusher['index_value'] = one_pusher['green_index'].combine_first(one_pusher['yellow_index'])
    reg_one = RandomForestRegressor(random_state=42)
    # reg_one = MLPRegressor(random_state=42)
    reg_one.fit(one_pusher[input_cols], one_pusher['index_value'])

    # Two-pusher regressor
    two_pushers = df[df['pusher_class'] == 1]
    reg_two = RandomForestRegressor(random_state=42)
    # reg_two = MLPRegressor(random_state=42)
    reg_two.fit(two_pushers[input_cols], two_pushers[['green_index', 'yellow_index']])

    # Store models for object
    models[object_name] = {
        'classifier': clf,
        'reg_one': reg_one,
        'reg_two': reg_two,
    }

# Function to make predictions on new input samples
def predict_pusher_outputs(name, pose_x, pose_y, ori_y):

    model = models[name]
    sample = pd.DataFrame([[pose_x, pose_y, ori_y]], columns=input_cols)

    pusher_type = model['classifier'].predict(sample)[0]
    
    if pusher_type == 0:
        index_value = model['reg_one'].predict(sample)[0]
        return {
            'object_id': name,
            'pusher_class': 'one_pusher',
            'predicted_index': [int(index_value)]
        }
    else:
        green_index, yellow_index = model['reg_two'].predict(sample)[0]
        return {
            'object_id': name,
            'pusher_class': 'two_pusher',
            'predicted_index': [int(green_index), int(yellow_index)]
        }
