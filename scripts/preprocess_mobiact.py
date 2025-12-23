import os
import pandas as pd
import numpy as np
from glob import glob

DATA_ROOT = '/Users/rehmatsinghchawla/Desktop/Project/Annotated Data'

FALL_CLASSES = ['FOL', 'FKL', 'BSC', 'SDL']

def load_all_data():
    data, labels = [], []
    for activity_folder in os.listdir(DATA_ROOT):
        activity_path = os.path.join(DATA_ROOT, activity_folder)
        if not os.path.isdir(activity_path):
            continue

        # Label as fall (1) or non-fall (0)
        label = 1 if activity_folder in FALL_CLASSES else 0

        for csv_file in glob(os.path.join(activity_path, '*_annotated.csv')):
            df = pd.read_csv(csv_file)
            # Standardize column names (check in one file first)
            # Assume columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
            df = df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]

            data.append(df.values)
            labels.append(label)

    return data, labels