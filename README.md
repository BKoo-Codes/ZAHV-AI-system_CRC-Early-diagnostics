# ZAHV-AI-System_CRC-Early-Diagnostics

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Usage](#usage)
4. [Modules](#modules)
    - [Optimized Splitting Method](#optimized-splitting-method)
    - [K-fold Cross-Validation](#k-fold-cross-validation)
    - [AI-Driven Analysis](#ai-driven-analysis)
5. [File Structure](#file-structure)
6. [Contributing](#contributing)
7. [License](#license)

## Project Overview
The **ZAHV-AI-System_CRC-Early-Diagnostics** project is designed to enhance early detection of colorectal cancer (CRC) by combining extracellular vesicle (EV)-derived biomarkers with conventional markers, particularly carcinoembryonic antigen (CEA). The ZAHV-AI system utilizes the ZAHVIS platform for efficient EV isolation and deep learning-based AI analysis, evaluating multiple biomarker combinations to identify the most effective set for CRC diagnosis, providing a more accurate and accessible tool for early detection.

Key features include:
- **Optimized Data Splitting**: Ensures balanced training and test sets through 1,000 iterations, minimizing feature differences across CRC stages and healthy controls (HC) for improved model reliability.
- **K-fold Cross-Validation**: Applies 5-fold cross-validation to fine-tune hyperparameters and enhance model accuracy across all data subsets.
- **AI-Driven Biomarker Analysis**: Uses deep learning to identify optimal biomarker combinations, enhancing diagnostic performance for CRC detection.

The **ZAHV-AI_CRC_Diagnostic_Data.csv** file contains data for the overall CRC analysis. We conducted analyses for multiple CRC stages, which include:
- **Overall CRC Analysis**: Includes all CRC stages (0-1, 2, 3, 4).
- **Early-Stage Analysis**: Focuses on CRC stages 0-1 and 2.
- **Advanced-Stage Analysis**: Focuses on CRC stages 3 and 4.
- **Individual Stage Analysis**: Includes separate analyses for CRC stages 0-1, 2, 3, and 4.

To perform analyses for early-stage, advanced-stage, and individual stages, make sure to select appropriate samples from the `ZAHV-AI_CRC_Diagnostic_Data.csv` file that correspond to the desired stage group.

## Installation and Setup

### Prerequisites
- **Python 3.7+**
- Required Python libraries (listed in `requirements.txt`):
    ```plaintext
    numpy==1.24.3
    pandas==2.1.1
    scikit-learn==1.3.0
    tensorflow==2.14.0
    matplotlib==3.7.1
    # Additional libraries can be found in the full requirements.txt
    ```

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/ZAHV-AI-System_CRC-Early-Diagnostics.git
    cd ZAHV-AI-System_CRC-Early-Diagnostics
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Dataset Setup**:
   - Make sure the dataset (`ZAHV-AI_CRC_Diagnostic_Data.csv`) is available in the project root directory. Replace `'your_dataset_path.csv'` in the scripts with the path to your dataset if necessary.

## Usage

### Optimized Splitting Method
This module optimizes the splitting of the dataset into training and test sets, ensuring balanced feature distributions between CRC stages and HC.

To run the script:
```bash
python optimized_splitting_method.py
```
This will use the dataset `ZAHV-AI_CRC_Diagnostic_Data.csv` to generate optimized splits.

### K-fold Cross-Validation
The `k_fold_cross_validation.py` script conducts k-fold cross-validation to fine-tune hyperparameters for the deep learning model.

To run the script:
```bash
python k_fold_cross_validation.py
```

### AI-Driven Analysis
The `ai_driven_analysis.py` script identifies the optimal biomarker combinations using deep learning-based analysis to maximize diagnostic accuracy.

To run the script:
```bash
python ai_driven_analysis.py
```

### Data Files
- **ZAHV-AI_CRC_Diagnostic_Data.csv**: Contains the dataset used for training, testing, and analysis.

## Modules

### Optimized Splitting Method
This module focuses on splitting the dataset into training and test sets with a balanced distribution of CRC stages and HC, ensuring unbiased model performance.

### K-fold Cross-Validation
This module performs 5-fold cross-validation, optimizing model hyperparameters (such as learning rate, batch size, and layer sizes). The model is evaluated based on metrics such as validation loss and Area Under the Curve (AUC).

### AI-Driven Analysis
This module uses deep learning to evaluate different biomarker combinations, aiming to identify the most effective set for CRC diagnosis. The performance metrics include Receiver Operating Characteristic (ROC) curves, AUC values, sensitivity, specificity, accuracy, and F1 scores.

## File Structure
```
├── optimized_splitting_method.py  # Script for optimized train-test splitting
├── k_fold_cross_validation.py     # Script for k-fold cross-validation
├── ai_driven_analysis.py          # Script for AI-driven biomarker analysis
├── requirements.txt               # Python dependencies with versioning
├── ZAHV-AI_CRC_Diagnostic_Data.csv # Input dataset for CRC diagnostics
└── README.md                      # Project documentation
```

## Contributing
Contributions are highly encouraged! If you would like to contribute:
- **Fork the repository** and create a feature branch.
- **Submit a pull request** with a detailed explanation of your changes.
- For significant changes, consider opening an issue to discuss your ideas first.

All contributions should follow best practices for coding and documentation to ensure quality and consistency.

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software, provided that proper credit is given to the original authors.
