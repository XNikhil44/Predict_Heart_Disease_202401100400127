# Predict_Heart_Disease_202401100400127

🫀 Heart Disease Prediction Using Machine Learning
📌 Overview
This project aims to predict the presence of heart disease in patients using machine learning techniques. By analyzing various medical parameters, the model assists in early detection, facilitating timely medical interventions.

📁 Dataset
Filename: 4. Predict Heart Disease.csv

Description: Contains patient medical records with features such as age, blood pressure, cholesterol levels, and more.

Target Variable: target (1 indicates presence of heart disease, 0 indicates absence)

🧪 Approach
Data Upload: Utilize Google Colab's file upload feature to import the dataset.

Exploratory Data Analysis (EDA):

Assess dataset shape and check for missing values.

Display initial records for a preliminary overview.

Data Preprocessing:

Separate features (X) and target (y).

Encode categorical variables using one-hot encoding.

Scale features using StandardScaler to normalize the data.

Model Training:

Split data into training and testing sets (80% train, 20% test).

Train a RandomForestClassifier on the training data.

Model Evaluation:

Predict on the test set.

Evaluate using accuracy score, confusion matrix, and classification report.

Visualization:

Correlation Heatmap: Visualize feature correlations.

Target Distribution: Display the distribution of the target variable.

Feature Importance: Highlight the most influential features in prediction.

🛠️ Setup & Installation
Environment: Google Colab (no local setup required)

Libraries Used:

pandas

numpy

seaborn

matplotlib

scikit-learn

Ensure all libraries are installed. In Google Colab, most are pre-installed. If not, install using pip:

python
Copy
Edit
!pip install pandas numpy seaborn matplotlib scikit-learn
🚀 Usage
Upload Dataset:

python
Copy
Edit
from google.colab import files
uploaded = files.upload()
Run the Complete Code:

Execute each code cell sequentially in the provided notebook to perform data analysis, model training, evaluation, and visualization.

📊 Results
Accuracy: Achieved using Random Forest Classifier.

Visualizations:

Correlation heatmap to understand feature relationships.

Distribution plot of the target variable.

Bar chart showcasing feature importances.

📌 Notes
Ensure the dataset file is named exactly as 4. Predict Heart Disease.csv when uploading.

The code automatically handles categorical variables and scales numerical features.

Modify the RandomForestClassifier parameters to experiment with model performance.

📷 Sample Output
Note: As Google Colab does not support direct screenshot capture via code, please manually take a screenshot of the output cells after execution for documentation purposes.

📚 References
Heart Disease Prediction using Machine Learning - Kaggle

Heart Disease Prediction Using Machine Learning and Deep Learning - GitHub

