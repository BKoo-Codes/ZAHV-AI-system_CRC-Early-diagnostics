# Import necessary libraries
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Fixing random seeds to ensure reproducibility of results
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

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

# Combining all selected features into a single list for use in modeling
all_features = sum(feature_groups.values(), [])

# Initializing a StandardScaler for normalizing the data (Z-score normalization)
scaler = StandardScaler()

# Printing the stages and their counts to get an overview of the dataset
print("Stages:", data['Stage'])
data_counts = data['Stage'].value_counts()
print("Stages counts:\n", data_counts)

# Function to optimize hyperparameters of a neural network using Stratified K-Fold Cross Validation
def optimize_hyperparameters(X, y):
    # Defining a grid of hyperparameters to test
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    batch_sizes = [4, 8, 16, 32]
    dense_units = [16, 32, 64, 128]
    dropout_rates = [0.1, 0.2, 0.3, 0.4]

    performance_metrics = []

    # StratifiedKFold ensures that each fold has a similar proportion of classes
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    # Computing class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))

    # Number of epochs for training
    epochs = 100

    # Loop through each hyperparameter combination
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dense in dense_units:
                for dropout in dropout_rates:
                    fold_val_losses = []
                    fold_aucs = []
                    
                    # Perform K-fold cross-validation
                    for train_index, val_index in skf.split(X, y):
                        X_fold_train, X_fold_val = X[train_index], X[val_index]
                        y_fold_train, y_fold_val = y.iloc[train_index], y.iloc[val_index]

                        # Define the model architecture
                        model = Sequential([
                            Dense(dense, activation='relu', input_dim=X_fold_train.shape[1]),
                            Dropout(dropout),
                            Dense(int(dense/2), activation='relu'),
                            Dropout(dropout),
                            Dense(1, activation='sigmoid')
                        ])

                        # Compile the model with the specified learning rate and Adam optimizer
                        optimizer = Adam(learning_rate=lr)
                        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                        
                        # Use EarlyStopping to prevent overfitting
                        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                        # Train the model and evaluate on the validation fold
                        history = model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, 
                                            validation_data=(X_fold_val, y_fold_val), 
                                            callbacks=[early_stopping], verbose=0, class_weight=class_weight_dict)

                        # Calculate validation loss and AUC score
                        val_loss = min(history.history['val_loss'])
                        y_pred_prob = model.predict(X_fold_val).ravel()
                        
                        # Ensure that the AUC score can be calculated (i.e., validation set has both classes)
                        if len(np.unique(y_fold_val)) == 2:
                            auc_score = roc_auc_score(y_fold_val, y_pred_prob)
                            fold_aucs.append(auc_score)
                        
                        fold_val_losses.append(val_loss)
                    
                    # Store the average validation loss and AUC score for the current hyperparameter combination
                    avg_val_loss = np.mean(fold_val_losses)
                    avg_auc = np.mean(fold_aucs) if fold_aucs else None
                    performance_metrics.append({
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dense_units': dense,
                        'dropout_rate': dropout,
                        'avg_val_loss': avg_val_loss,
                        'avg_auc': avg_auc
                    })

    # Sort the performance metrics by validation loss (ascending) and AUC score (descending)
    performance_metrics.sort(key=lambda x: (x['avg_val_loss'], -x['avg_auc'] if x['avg_auc'] is not None else float('inf')))

    return performance_metrics

# Define the train and test indices
# These indices were derived using an optimized splitting method, ensuring balanced representation
# of different stages in both training and test sets.
train_indices = [0,1,2,4,7,8,9,10,11,12,13,14,16,19,20,21,22,24,25,26,28,29,31,32,33,34,38,39,41,43,44,45,47,48,50,51,52,53,54,55,56,59,61,62,64,65,66,67,68,69,70,71,72,73,74,78,80,82,83,86,87,89,90,91,92,94,95,96,97,99]
test_indices = [3,5,6,15,17,18,23,27,30,35,36,37,40,42,46,49,57,58,60,63,75,76,77,79,81,84,85,88,93,98]

# Extracting features (X) and labels (y)
X = data[all_features]
y = data['Label']

# Creating the training and testing sets using the predefined indices
X_train = X.loc[train_indices].reset_index(drop=True)
X_test = X.loc[test_indices].reset_index(drop=True)
y_train = y.loc[train_indices].reset_index(drop=True)
y_test = y.loc[test_indices].reset_index(drop=True)

# Normalizing the training and testing sets using Z-score normalization
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Printing the indices for the Stratified K-Fold splits for hyperparameter optimization
print("\nStratifiedKFold splits (for hyperparameter optimization):")
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
for fold, (train_index, val_index) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"Fold {fold+1}:")
    print("Train indices:", [train_indices[i] for i in train_index])
    print("Validation indices:", [train_indices[i] for i in val_index])

# Printing the final train/test split for confirmation
print("\nFinal Train/Test split for evaluation:")
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Optimizing hyperparameters using the training set
performance_metrics = optimize_hyperparameters(X_train_scaled, y_train)

# Printing the train/test set indices for further evaluation
print("Train set indices (for optimization and evaluation):", train_indices)
print("Test set indices (for evaluation):", test_indices)

# Calculating the score for each hyperparameter combination (higher score is better)
for metrics in performance_metrics:
    if metrics['avg_auc'] is not None:
        metrics['score'] = 1 - metrics['avg_val_loss'] + metrics['avg_auc']
    else:
        metrics['score'] = 1 - metrics['avg_val_loss']

# Sorting the performance metrics by the calculated score in descending order
performance_metrics.sort(key=lambda x: x['score'], reverse=True)

# Outputting the results of all hyperparameter combinations, validation loss, AUC, and calculated scores
print("All Hyperparameter Combinations and their Validation Loss, AUC Scores, and Calculated Scores:")
for metrics in performance_metrics:
    if metrics['avg_auc'] is not None:
        print(f"Learning Rate: {metrics['learning_rate']}, Batch Size: {metrics['batch_size']}, "
              f"Dense Units: {metrics['dense_units']}, Dropout Rate: {metrics['dropout_rate']}, "
              f"Avg Validation Loss: {metrics['avg_val_loss']:.4f}, Avg AUC: {metrics['avg_auc']:.4f}, "
              f"Score: {metrics['score']:.4f}")
    else:
        print(f"Learning Rate: {metrics['learning_rate']}, Batch Size: {metrics['batch_size']}, "
              f"Dense Units: {metrics['dense_units']}, Dropout Rate: {metrics['dropout_rate']}, "
              f"Avg Validation Loss: {metrics['avg_val_loss']:.4f}, Avg AUC: N/A, "
              f"Score: {metrics['score']:.4f}")

# Output the best hyperparameters based on the calculated score
best_hyperparameters = performance_metrics[0]
print("\nBest Hyperparameters based on the new score:")
if best_hyperparameters['avg_auc'] is not None:
    print(f"Learning Rate: {best_hyperparameters['learning_rate']}, Batch Size: {best_hyperparameters['batch_size']}, "
          f"Dense Units: {best_hyperparameters['dense_units']}, Dropout Rate: {best_hyperparameters['dropout_rate']}, "
          f"Avg Validation Loss: {best_hyperparameters['avg_val_loss']:.4f}, Avg AUC: {best_hyperparameters['avg_auc']:.4f}, "
          f"Score: {best_hyperparameters['score']:.4f}")
else:
    print(f"Learning Rate: {best_hyperparameters['learning_rate']}, Batch Size: {best_hyperparameters['batch_size']}, "
          f"Dense Units: {best_hyperparameters['dense_units']}, Dropout Rate: {best_hyperparameters['dropout_rate']}, "
          f"Avg Validation Loss: {best_hyperparameters['avg_val_loss']:.4f}, Avg AUC: N/A, "
          f"Score: {best_hyperparameters['score']:.4f}")
