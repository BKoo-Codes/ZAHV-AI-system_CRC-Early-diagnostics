# Import necessary libraries
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from sklearn.utils.class_weight import compute_class_weight
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import os

# Load the correct font for Matplotlib
# Ensure the font file exists in the specified path
font_path = r'font_path.ttf'
assert os.path.exists(font_path), f"Font file not found: {font_path}"

# Add the font to Matplotlib's font manager
fm.fontManager.addfont(font_path)

# Create a FontProperties object to use this font
font_prop = fm.FontProperties(fname=font_path)

# Set the global font family to the manually added font
plt.rcParams['font.family'] = font_prop.get_name()

# Fixing random seeds to ensure reproducibility of results
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

# Load the dataset from a CSV file. Make sure to replace 'your_dataset_path.csv' with the actual path.
file_path = 'your_dataset_path.csv'  # Replace with the correct dataset path
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

# Apply the function to extract stages and create a 'Stage' column
data['Stage'] = data['Diagnosis'].apply(extract_stage)

# Create a binary 'Label' column where 1 represents CRC stages (M) and 0 represents healthy controls (B)
data['Label'] = data['Diagnosis'].apply(lambda x: 1 if 'M' in x else 0)

# Define the feature groups used in the model, focusing on gene expression data and biomarkers
feature_groups = {
    'miR-23a-3p': ['miR-23a-3p'],
    'miR-92a-3p': ['miR-92a-3p'],
    'miR-125a-3p': ['miR-125a-3p'],
    'miR-150-5p': ['miR-150-5p'],
    'CEA': ['CEA']  # Carcinoembryonic antigen (CEA)
}

# Combine all features into a single list
all_features = sum(feature_groups.values(), [])

# Initialize a StandardScaler for Z-score normalization of the features
scaler = StandardScaler()

# Simplify marker names for visualization purposes
simplified_names = {
    'miR-23a-3p': '23a',
    'miR-92a-3p': '92a',
    'miR-125a-3p': '125a',
    'miR-150-5p': '150',
    'CEA': 'CEA'
}

# Display the stages in the dataset and their counts
print("Stages:", data['Stage'])
data_counts = data['Stage'].value_counts()
print("Stages counts:\n", data_counts)

# Function to compute 95% confidence interval for AUC using bootstrap
def bootstrap_auc(y_test, y_pred_prob, n_bootstrap=1000, alpha=0.95):
    bootstrapped_aucs = []
    for i in range(n_bootstrap):
        # Resample with replacement from the test set
        indices = np.random.randint(0, len(y_test), len(y_test))
        if len(np.unique(y_test[indices])) < 2:
            continue  # Skip if we don't have both classes in the resampled set
        auc = roc_auc_score(y_test[indices], y_pred_prob[indices])
        bootstrapped_aucs.append(auc)
    
    sorted_aucs = np.sort(bootstrapped_aucs)
    
    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(sorted_aucs, ((1 - alpha) / 2) * 100)
    upper_bound = np.percentile(sorted_aucs, (alpha + (1 - alpha) / 2) * 100)
    
    return lower_bound, upper_bound

# Updated evaluate_performance function to include 95% confidence interval for AUC
def evaluate_performance(y_test, y_pred_prob, n_bootstrap=1000):
    # Calculate AUC score and ROC curve
    auc_score = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    
    # Determine the Youden's index threshold
    youden_index = np.argmax(tpr - fpr)
    youden_threshold = thresholds[youden_index]

    # Convert probabilities to binary predictions using Youden's index
    y_pred_youden = (y_pred_prob >= youden_threshold).astype(int)

    # Calculate confusion matrix for Youden's threshold
    tn_youden, fp_youden, fn_youden, tp_youden = confusion_matrix(y_test, y_pred_youden).ravel()

    # Compute the 95% confidence interval for AUC using bootstrap
    lower_ci, upper_ci = bootstrap_auc(y_test, y_pred_prob, n_bootstrap=n_bootstrap)

    # Store performance metrics
    metrics = {
        'youden_threshold': youden_threshold,
        'sensitivity_youden': tp_youden / (tp_youden + fn_youden) if (tp_youden + fn_youden) > 0 else 0,
        'specificity_youden': tn_youden / (tn_youden + fp_youden) if (tn_youden + fp_youden) > 0 else 0,
        'accuracy_youden': accuracy_score(y_test, y_pred_youden),
        'f1_youden': f1_score(y_test, y_pred_youden),
        'auc_score': auc_score,
        'auc_95ci': (lower_ci, upper_ci)  # 95% confidence interval for AUC
    }

    return metrics, fpr, tpr, auc_score

# Define the neural network model
# Hyperparameters such as layer sizes (128, 64 units), dropout rates (0.1), and learning rate (0.1) 
# were determined through previous K-fold cross-validation
train_indices = [0,1,2,4,7,8,9,10,11,12,13,14,16,19,20,21,22,24,25,26,28,29,31,32,33,34,38,39,41,43,44,45,47,48,50,51,52,53,54,55,56,59,61,62,64,65,66,67,68,69,70,71,72,73,74,78,80,82,83,86,87,89,90,91,92,94,95,96,97,99]
test_indices = [3,5,6,15,17,18,23,27,30,35,36,37,40,42,46,49,57,58,60,63,75,76,77,79,81,84,85,88,93,98]

# Extract features (X) and labels (y) for modeling
X = data[all_features]
y = data['Label']

# Split data into training and testing sets using predefined indices
X_train = X.loc[train_indices].reset_index(drop=True)
X_test = X.loc[test_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)
y_test = y.loc[test_indices].reset_index(drop=True)

# Display training and testing set details
print("Train set indices:", train_indices)
print("Test set indices:", test_indices)

# Display label counts in the train and test sets
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()

print("Train set label counts:\n", train_counts)
print("Test set label counts:\n", test_counts)

# Normalize the training and testing sets using Z-score normalization
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights to handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Loop through each combination of feature groups and evaluate performance
auc_values_by_combination = {}
performance_metrics_by_combination = {}
auc_values_by_combination_train = {}
performance_metrics_by_combination_train = {}
auc_values_by_combination_test = {}
performance_metrics_by_combination_test = {}
predictions_by_combination = {}

for r in range(1, len(feature_groups) + 1):
    for group_combination in combinations(feature_groups.keys(), r):
        # Select features based on the current combination
        selected_features = sum([feature_groups[group] for group in group_combination], [])
        X_train_comb = X_train[selected_features]
        X_test_comb = X_test[selected_features]
        
        # Normalize the selected features
        X_train_comb_scaled = scaler.fit_transform(X_train_comb)
        X_test_comb_scaled = scaler.transform(X_test_comb)
        
        # Define the neural network model
        # The architecture and hyperparameters such as layer sizes (128, 64 units), dropout rates (0.1), and activation functions 
        # were determined based on prior K-fold cross-validation hyperparameter tuning.
        # The key hyperparameters are:
        # - Learning Rate: 0.1
        # - Batch Size: 16
        # - Dense Units: 128 (first layer), 64 (second layer)
        # - Dropout Rate: 0.1
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train_comb_scaled.shape[1]),  # First hidden layer with 128 units
            Dropout(0.1),  # Dropout rate of 0.1
            Dense(64, activation='relu'),  # Second hidden layer with 64 units
            Dropout(0.1),  # Dropout rate of 0.1
            Dense(1, activation='sigmoid')  # Output layer for binary classification
        ])
        
        # Set the learning rate to 0.1 (determined from hyperparameter tuning)
        optimizer = Adam(learning_rate=0.1)

        # Compile the model and train without validation data
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X_train_comb_scaled, y_train, epochs=100, batch_size=16, callbacks=[early_stopping], verbose=0, class_weight=class_weight_dict)

        
        # Predict on test data and evaluate
        y_pred_prob = model.predict(X_test_comb_scaled).ravel()
        y_pred_prob_train = model.predict(X_train_comb_scaled).ravel()
        
        # Evaluate performance and store results
        metrics, fpr, tpr, auc_score = evaluate_performance(y_test, y_pred_prob)
        metrics_train, fpr_train, tpr_train, auc_score_train = evaluate_performance(y_train, y_pred_prob_train)
        
        performance_metrics_by_combination_test[group_combination] = (fpr, tpr, auc_score)
        auc_values_by_combination[group_combination] = auc_score
        performance_metrics_by_combination[group_combination] = metrics  
        performance_metrics_by_combination_train[group_combination] = (fpr_train, tpr_train, auc_score_train)
             
        # Save predictions for each combination
        y_pred_youden = (y_pred_prob >= metrics['youden_threshold']).astype(int)
        predictions_by_combination[group_combination] = {'y_test': y_test, 'y_pred_youden': y_pred_youden}

# Identify the best combination based on AUC score
best_combination = max(auc_values_by_combination, key=auc_values_by_combination.get)
best_auc = auc_values_by_combination[best_combination]

# Print the best combination with 95% CI for AUC
best_metrics = performance_metrics_by_combination[best_combination]
lower_ci, upper_ci = best_metrics['auc_95ci']
print(f"\nBest Combination: {', '.join(best_combination)}, AUC: {best_auc:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")

# Optionally, print all combinations with AUC and 95% CI
sorted_auc_values = sorted(auc_values_by_combination.items(), key=lambda item: item[1], reverse=True)
print("\nAll Combinations AUC Values with 95% Confidence Interval:")
for combination, auc_value in sorted_auc_values:
    metrics = performance_metrics_by_combination[combination]
    lower_ci, upper_ci = metrics['auc_95ci']
    print(f"Combination: {', '.join(combination)}, AUC: {auc_value:.4f}, 95% CI: ({lower_ci:.4f}, {upper_ci:.4f})")

# Print all combinations performance metrics (Youden's Index metrics)
print("All Combinations Performance Metrics:")
for combination, metrics in performance_metrics_by_combination.items():
    print(f"\nCombination: {', '.join(combination)}")
    print(f"Youden's Index - Sensitivity, Specificity, Accuracy, F1 Score")
    print(f"Youden's Index - {metrics['sensitivity_youden'] * 100:.2f}, {metrics['specificity_youden'] * 100:.2f}, {metrics['accuracy_youden'] * 100:.2f}, {metrics['f1_youden'] * 100:.2f}")

# Print train and test set diagnoses
train_diagnosis = data.loc[train_indices, 'Diagnosis']
test_diagnosis = data.loc[test_indices, 'Diagnosis']

print("Train set diagnoses:", train_diagnosis.values)
print("Test set diagnoses:", test_diagnosis.values)

# Print performance metrics and predictions by combination
print("\nPredictions by Combinations:")
for combination, preds in predictions_by_combination.items():
    combined_results = pd.DataFrame({
        'Sample Index': test_indices,
        'Youden\'s Index Prediction': preds['y_pred_youden'],
        'True Label': preds['y_test'].values
    })
    print(f"\nCombination: {', '.join(combination)}")
    print(combined_results)

# Plot ROC curve for each combination on test data
for r in range(1, len(feature_groups) + 1):
    plt.figure(figsize=(6.5, 5.5), dpi=500)
    plt.plot([0, 1], [0, 1], '--', color='gray', zorder=0)
    rect = patches.Rectangle((0,0), 1, 1, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    plt.xlabel('1-Specificity', fontproperties=FontProperties(size=18, weight='bold'))
    plt.ylabel('Sensitivity', fontproperties=FontProperties(size=18, weight='bold'))

    for group_combination in combinations(feature_groups.keys(), r):
        if group_combination in performance_metrics_by_combination_test:
            fpr, tpr, auc = performance_metrics_by_combination_test[group_combination]
            simplified_labels = [simplified_names[group] for group in group_combination]
            label_str = ' + '.join(simplified_labels)
            plt.plot(fpr, tpr, label=f'{label_str}', zorder=1, linewidth=3)

    plt.legend(loc='lower right', prop={'size': 14})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.show()

# Plot ROC curve for each combination on train data
for r in range(1, len(feature_groups) + 1):
    plt.figure(figsize=(6.5, 5.5), dpi=500)
    plt.plot([0, 1], [0, 1], '--', color='gray', zorder=0)
    rect = patches.Rectangle((0,0), 1, 1, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    plt.gca().add_patch(rect)
    plt.xlabel('1-Specificity', fontproperties=FontProperties(size=18, weight='bold'))
    plt.ylabel('Sensitivity', fontproperties=FontProperties(size=18, weight='bold'))

    for group_combination in combinations(feature_groups.keys(), r):
        if group_combination in performance_metrics_by_combination_train:
            fpr, tpr, auc = performance_metrics_by_combination_train[group_combination]
            simplified_labels = [simplified_names[group] for group in group_combination]
            label_str = ' + '.join(simplified_labels)
            plt.plot(fpr, tpr, linestyle='dashdot', label=f'{label_str} (AUC = {auc:.2f})', zorder=1, linewidth=3)

    plt.legend(loc='lower right', prop={'size': 13})
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.show()
