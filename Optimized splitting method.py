# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset from a CSV file. Make sure to replace 'your_dataset_path.csv' with the actual path.
file_path = 'your_dataset_path.csv'
data = pd.read_csv(file_path)

# Function to extract the stage from the 'Diagnosis' column in the dataset
# Clinical stages:
# M1: CRC Stages 0-1, M2: CRC Stage 2, M3: CRC Stage 3, M4: CRC Stage 4, B: Healthy controls
def extract_stage(diagnosis):
    if diagnosis.startswith('M4'):
        return 'M4'  # CRC Stage 4
    elif diagnosis.startswith('M3'):
        return 'M3'  # CRC Stage 3
    elif diagnosis.startswith('M2'):
        return 'M2'  # CRC Stage 2
    elif diagnosis.startswith('M1'):
        return 'M1'  # CRC Stages 0-1
    else:
        return 'B'  # Healthy control

# Apply the extract_stage function to the 'Diagnosis' column to create a new 'Stage' column
data['Stage'] = data['Diagnosis'].apply(extract_stage)

# Create a binary label where 1 represents CRC stages ('M') and 0 represents healthy control ('B')
data['Label'] = data['Diagnosis'].apply(lambda x: 1 if 'M' in x else 0)

# Define the feature groups to use in the model (gene expression data and biomarker levels)
feature_groups = {
    'miR-23a-3p': ['miR-23a-3p'],
    'miR-92a-3p': ['miR-92a-3p'],
    'miR-125a-3p': ['miR-125a-3p'],
    'miR-150-5p': ['miR-150-5p'],
    'CEA': ['CEA']  # Carcinoembryonic antigen (CEA)
}

# Combine all the features into a single list
all_features = sum(feature_groups.values(), [])

# Function to find the optimal split for train and test sets based on mean and standard deviation
# The goal is to find a split where both the train and test sets have similar distributions of features
def optimal_split(group_data, train_size=14, test_size=6, iterations=1000):
    best_train, best_test = None, None
    best_train_stats, best_test_stats = None, None
    best_score = float('inf')  # Start with a very high score, aiming to minimize it

    # Perform random splits for a specified number of iterations
    for i in range(iterations):
        random_state = np.random.randint(0, 10000)
        train, test = train_test_split(group_data, train_size=train_size, test_size=test_size, random_state=random_state)
        
        # Calculate the mean and standard deviation of features in both train and test sets
        train_mean = train[all_features].mean()
        test_mean = test[all_features].mean()
        train_std = train[all_features].std()
        test_std = test[all_features].std()

        # Calculate mean squared error for the means and std deviations
        mean_score = mean_squared_error(train_mean, test_mean)
        std_score = mean_squared_error(train_std, test_std)
        
        # Total score gives more weight to the mean difference but includes std dev. difference
        total_score = mean_score + 0.1 * std_score

        # If the current split has a better score, update the best split
        if total_score < best_score:
            best_train, best_test = train, test
            best_train_stats = (train_mean, train_std)
            best_test_stats = (test_mean, test_std)
            best_score = total_score

    return best_train, best_test, best_train_stats, best_test_stats

# Split the healthy control group ('B') into train and test sets
B_data = data[data['Stage'] == 'B']
B_train, B_test, B_train_stats, B_test_stats = optimal_split(B_data, train_size=14, test_size=6)

# Split each CRC stage ('M1', 'M2', 'M3', 'M4') into train and test sets
individual_splits = {}
for group in ['M1', 'M2', 'M3', 'M4']:
    group_data = data[data['Stage'] == group]
    train, test, train_stats, test_stats = optimal_split(group_data)
    individual_splits[group] = {
        'train': train,
        'test': test,
        'train_stats': train_stats,
        'test_stats': test_stats
    }

# Combine stages for more robust groupings
# M1 + M2 group (CRC Stage 0-1 and 2)
M1_M2_train = pd.concat([individual_splits['M1']['train'], individual_splits['M2']['train']])
M1_M2_test = pd.concat([individual_splits['M1']['test'], individual_splits['M2']['test']])
M1_M2_train_stats = (M1_M2_train[all_features].mean(), M1_M2_train[all_features].std())
M1_M2_test_stats = (M1_M2_test[all_features].mean(), M1_M2_test[all_features].std())

# M3 + M4 group (CRC Stage 3 and 4)
M3_M4_train = pd.concat([individual_splits['M3']['train'], individual_splits['M4']['train']])
M3_M4_test = pd.concat([individual_splits['M3']['test'], individual_splits['M4']['test']])
M3_M4_train_stats = (M3_M4_train[all_features].mean(), M3_M4_train[all_features].std())
M3_M4_test_stats = (M3_M4_test[all_features].mean(), M3_M4_test[all_features].std())

# M1 + M2 + M3 + M4 group (combining all CRC stages)
M1_M2_M3_M4_train = pd.concat([individual_splits['M1']['train'], individual_splits['M2']['train'], individual_splits['M3']['train'], individual_splits['M4']['train']])
M1_M2_M3_M4_test = pd.concat([individual_splits['M1']['test'], individual_splits['M2']['test'], individual_splits['M3']['test'], individual_splits['M4']['test']])
M1_M2_M3_M4_train_stats = (M1_M2_M3_M4_train[all_features].mean(), M1_M2_M3_M4_train[all_features].std())
M1_M2_M3_M4_test_stats = (M1_M2_M3_M4_test[all_features].mean(), M1_M2_M3_M4_test[all_features].std())

# Function to validate splits by comparing means and standard deviations of training and test sets
# This helps to ensure the feature distributions are similar across splits
def validate_splits(train_stats, test_stats, description):
    train_mean, train_std = train_stats
    test_mean, test_std = test_stats
    
    print(f"{description} Training Set Mean:")
    print(train_mean)
    print(f"{description} Training Set Std Dev:")
    print(train_std)
    print(f"{description} Test Set Mean:")
    print(test_mean)
    print(f"{description} Test Set Std Dev:")
    print(test_std)
    print("\n")

# Validate the splits for each group
validate_splits(B_train_stats, B_test_stats, "B group (Healthy Controls)")
validate_splits(individual_splits['M1']['train_stats'], individual_splits['M1']['test_stats'], "M1 group (CRC Stage 2, Low-Risk)")
validate_splits(individual_splits['M2']['train_stats'], individual_splits['M2']['test_stats'], "M2 group (CRC Stage 2)")
validate_splits(individual_splits['M3']['train_stats'], individual_splits['M3']['test_stats'], "M3 group (CRC Stage 3)")
validate_splits(individual_splits['M4']['train_stats'], individual_splits['M4']['test_stats'], "M4 group (CRC Stage 4)")
validate_splits(M1_M2_train_stats, M1_M2_test_stats, "M1 + M2 group (CRC Stage 2)")
validate_splits(M3_M4_train_stats, M3_M4_test_stats, "M3 + M4 group (CRC Stage 3 and 4)")
validate_splits(M1_M2_M3_M4_train_stats, M1_M2_M3_M4_test_stats, "M1 + M2 + M3 + M4 group (All CRC Stages)")

# Combine the train and test sets and their statistics for each group
# This function labels the data as 'Train' or 'Test' and returns combined sets and statistics
def combine_sets_and_stats(train, test, train_stats, test_stats):
    train['Set'] = 'Train'
    test['Set'] = 'Test'
    combined_set = pd.concat([train, test])
    combined_stats = pd.DataFrame({
        'Train Mean': train_stats[0],
        'Train Std Dev': train_stats[1],
        'Test Mean': test_stats[0],
        'Test Std Dev': test_stats[1]
    })
    return combined_set, combined_stats

# Combine and save data for each group
B_combined_set, B_combined_stats = combine_sets_and_stats(B_train, B_test, B_train_stats, B_test_stats)
M1_combined_set, M1_combined_stats = combine_sets_and_stats(individual_splits['M1']['train'], individual_splits['M1']['test'], individual_splits['M1']['train_stats'], individual_splits['M1']['test_stats'])
M2_combined_set, M2_combined_stats = combine_sets_and_stats(individual_splits['M2']['train'], individual_splits['M2']['test'], individual_splits['M2']['train_stats'], individual_splits['M2']['test_stats'])
M3_combined_set, M3_combined_stats = combine_sets_and_stats(individual_splits['M3']['train'], individual_splits['M3']['test'], individual_splits['M3']['train_stats'], individual_splits['M3']['test_stats'])
M4_combined_set, M4_combined_stats = combine_sets_and_stats(individual_splits['M4']['train'], individual_splits['M4']['test'], individual_splits['M4']['train_stats'], individual_splits['M4']['test_stats'])
M1_M2_combined_set, M1_M2_combined_stats = combine_sets_and_stats(M1_M2_train, M1_M2_test, M1_M2_train_stats, M1_M2_test_stats)
M3_M4_combined_set, M3_M4_combined_stats = combine_sets_and_stats(M3_M4_train, M3_M4_test, M3_M4_train_stats, M3_M4_test_stats)
M1_M2_M3_M4_combined_set, M1_M2_M3_M4_combined_stats = combine_sets_and_stats(M1_M2_M3_M4_train, M1_M2_M3_M4_test, M1_M2_M3_M4_train_stats, M1_M2_M3_M4_test_stats)

# Save all combined train, test sets, and statistics to an Excel file
with pd.ExcelWriter('training_test_sets_with_stats.xlsx') as writer:
    # Save the B group (Healthy Controls) data
    B_combined_set.to_excel(writer, sheet_name='B_combined_set', index=False)
    B_combined_stats.to_excel(writer, sheet_name='B_combined_stats')
    
    # Save individual CRC stage group data
    M1_combined_set.to_excel(writer, sheet_name='M1_combined_set', index=False)
    M1_combined_stats.to_excel(writer, sheet_name='M1_combined_stats')
    
    M2_combined_set.to_excel(writer, sheet_name='M2_combined_set', index=False)
    M2_combined_stats.to_excel(writer, sheet_name='M2_combined_stats')
    
    M3_combined_set.to_excel(writer, sheet_name='M3_combined_set', index=False)
    M3_combined_stats.to_excel(writer, sheet_name='M3_combined_stats')
    
    M4_combined_set.to_excel(writer, sheet_name='M4_combined_set', index=False)
    M4_combined_stats.to_excel(writer, sheet_name='M4_combined_stats')
    
    # Save combined CRC Stage 0-1 + 2 (M1 + M2 group) data
    M1_M2_combined_set.to_excel(writer, sheet_name='M1_M2_combined_set', index=False)
    M1_M2_combined_stats.to_excel(writer, sheet_name='M1_M2_combined_stats')
    
    # Save combined CRC Stage 3 + 4 (M3 + M4 group) data
    M3_M4_combined_set.to_excel(writer, sheet_name='M3_M4_combined_set', index=False)
    M3_M4_combined_stats.to_excel(writer, sheet_name='M3_M4_combined_stats')
    
    # Save combined CRC Stage 0-1 + 2 + 3 + 4 (M1 + M2 + M3 + M4 group) data
    M1_M2_M3_M4_combined_set.to_excel(writer, sheet_name='M1_M2_M3_M4_combined_set', index=False)
    M1_M2_M3_M4_combined_stats.to_excel(writer, sheet_name='M1_M2_M3_M4_combined_stats')

# Indicate that the process has completed
print("Training and test sets with statistics have been saved to 'training_test_sets_with_stats.xlsx'.")
