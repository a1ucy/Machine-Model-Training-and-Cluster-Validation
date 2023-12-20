import pandas as pd
import pickle
import numpy as np
from numpy.fft import fft
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

#reads the Insulin and CGM datasets from the given file paths and merges the 'Date' and 'Time' columns into a single 'DateTime' column for easier processing.
def get_datasets(insulin_data_file_path, cgm_data_file_path):
    insulin_data = pd.read_csv(insulin_data_file_path, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'], low_memory=False)
    cgm_data = pd.read_csv(cgm_data_file_path, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'], low_memory=False).dropna()
    
    for data in [insulin_data, cgm_data]:
        data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
        data.drop(columns=['Date', 'Time'], inplace=True)

    return insulin_data, cgm_data

insulin_dataset, cgm_dataset = get_datasets('InsulinData.csv', 'CGMData.csv')

#  extracts the meal start times from the insulin dataset based on non-zero carb input values.
def get_meal_start_times(insulin_dataset):
    insulin_data_carb = insulin_dataset.loc[insulin_dataset['BWZ Carb Input (grams)'].notna() & (insulin_dataset['BWZ Carb Input (grams)'] != 0), 'DateTime']
    meal_start_times = insulin_data_carb.sort_values().tolist()
    return meal_start_times

meal_start_times = get_meal_start_times(insulin_dataset)

#extracts meal-related CGM data based on the valid meal start times obtained in the previous step.
def extract_meal_data(cgm_dataset, meal_start_times):
    valid_meal_start_times = []
    
    for i, timestamp in enumerate(meal_start_times):
        if i > 0 and meal_start_times[i-1] > timestamp - timedelta(hours=0.5):
            continue
        if i < len(meal_start_times) - 1 and meal_start_times[i+1] < timestamp + timedelta(hours=2):
            continue
        valid_meal_start_times.append(timestamp)
    
    meal_data = []
    for meal_time in valid_meal_start_times:
        start_time = meal_time - timedelta(minutes=30)
        end_time = meal_time + timedelta(hours=2)
        filtered_data = cgm_dataset[(cgm_dataset['DateTime'] >= start_time) & (cgm_dataset['DateTime'] <= end_time)]
        
        if not filtered_data.empty:
            meal_data.append(filtered_data['Sensor Glucose (mg/dL)'].tolist())
    
    return meal_data

meal_data = extract_meal_data(cgm_dataset, meal_start_times)

#extracts CGM data that is not meal-related, based on the valid no-meal start times calculated using the insulin dataset and meal_start_times.
def extract_no_meal_data(insulin_dataset, cgm_dataset, meal_start_times):
    start_times_to_consider = [min(insulin_dataset['DateTime']) - timedelta(hours=2), *meal_start_times, max(insulin_dataset['DateTime'])]

    valid_no_meal_start_times = []
    for prev_time, curr_time in zip(start_times_to_consider[:-1], start_times_to_consider[1:]):
        start = prev_time + timedelta(hours=2)
        while start + timedelta(hours=2) <= curr_time:
            valid_no_meal_start_times.append(start)
            start += timedelta(hours=2)

    no_meal_data = []
    for start_time in valid_no_meal_start_times:
        end_time = start_time + timedelta(hours=2)
        filtered_data = cgm_dataset[(cgm_dataset['DateTime'] >= start_time) & (cgm_dataset['DateTime'] <= end_time)]
        
        if not filtered_data.empty:
            no_meal_data.append(filtered_data['Sensor Glucose (mg/dL)'].tolist())

    return no_meal_data

no_meal_data = extract_no_meal_data(insulin_dataset, cgm_dataset, meal_start_times)

#calculates slope features from a given data row.
def compute_slope_features(datarow):
    slopes = [(datarow[i] + datarow[i+2] - 2 * datarow[i+1]) / ((i+2-i) * 5.0) for i in range(len(datarow)-2)]
    zero_crossing_indices = np.where(np.diff(np.sign(slopes)))[0]
    zero_crossing_delta = sorted([(index, abs(slopes[index+1] - slopes[index])) for index in zero_crossing_indices], key=lambda x: x[1], reverse=True)
    return zero_crossing_delta[:3]

#calculates frequency domain features using the Fast Fourier Transform (FFT).
def frequency_domain_features(datarow):
    frequencies = fft(datarow)
    top_frequency_indices = np.argsort(frequencies)[::-1][1:4]
    return top_frequency_indices.tolist()

#extracts all the required features from the CGM data rows and returns a DataFrame with the feature values.
def extract_features(dataset):
    feature_data = []

    for datarow in dataset:
        max_val, min_val = max(datarow), min(datarow)
        max_min_diff = max_val - min_val
        max_min_time_diff = (datarow.index(max_val) - datarow.index(min_val)) * 5
        
        slope_feature_tuples = compute_slope_features(datarow)
        slope_features = [tuple_data[1] if i < len(slope_feature_tuples) else None for i, tuple_data in enumerate(slope_feature_tuples)]
        slope_loc_features = [tuple_data[0] if i < len(slope_feature_tuples) else None for i, tuple_data in enumerate(slope_feature_tuples)]

        top_frequencies = frequency_domain_features(datarow)
        fft_features = [top_frequencies[i] if i < len(top_frequencies) else None for i in range(3)]

        feature_data.append([max_min_diff, *slope_features, *slope_loc_features, max_min_time_diff, *fft_features])

    result_df = pd.DataFrame(feature_data, columns=['CGM_Max_Min_Diff', 'slope_delta_1', 'slope_delta_2', 'slope_delta_3','slope_delta_1_loc', 'slope_delta_2_loc', 'slope_delta_3_loc','CGM_Max_Min_Time_Diff', 'fft_2', 'fft_3', 'fft_4'])
    return result_df

F_meal_data_df = extract_features(meal_data)

#normalizes a DataFrame to have values between 0 and 1.
def normalize(df):
    return (df - df.min())/((df.max() - df.min()) * 1.0)

F_no_meal_data_df = extract_features(no_meal_data)
F_meal_data_df.dropna(inplace = True)
F_no_meal_data_df.dropna(inplace = True)

pca = PCA(n_components = 8)
def get_PCA():
    return pca

#performs PCA on the given dataset and returns the transformed dataset.
def perform_PCA(dataset):
    pca = get_PCA()
    pca.fit(dataset)
    transformed_dataset = pca.transform(dataset)
    return pd.DataFrame(transformed_dataset)

#trains an SVM model on the provided training data and returns the trained model.
def svm_classifier(X_train, y_train):
    svm_model = SVC(gamma = 'scale')
    svm_model.fit(X_train, y_train)
    return svm_model

F_meal_df = F_meal_data_df
F_no_meal_df = F_no_meal_data_df
F_meal_df['Class'] = 1
F_no_meal_df['Class'] = 0
F_data_df = pd.concat([F_meal_df, F_no_meal_df])
F_data_df = F_data_df.reset_index().drop(columns = 'index')
class_labels = F_data_df['Class']
F_data_df.drop(columns = 'Class', inplace = True)
F_data_df_normalized = normalize(F_data_df)
X_train = perform_PCA(F_data_df_normalized)
y_train = class_labels
model = svm_classifier(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))