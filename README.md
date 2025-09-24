BAYMAX-Heart-Disease-Prediction-App
BAYMAX is a Streamlit app that predicts heart disease risk using patient data. It leverages multiple ML models, provides evaluation metrics, ROC curves, and visualizations, offering an intuitive interface for both healthcare enthusiasts and data science users.


-Project Structure
├── app.py                       # Streamlit web application
├── heart_disease_model.pkl       # Trained ML model
├── heart_disease_clean.csv       # Preprocessed dataset
├── results/                      # Folder for evaluation metrics & saved figures
│   └── evaluation_metrics.txt
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore 

-Installation

1.Clone the repository:

git clone <https://github.com/mohamed15adel9-maker/BAYMAX-Heart-Disease-Prediction-App>
cd <BAYMAX Heart Disease Prediction>

2.Install the required packages:

pip install -r requirements.txt


-Usage

Run the Streamlit application from the project root:

streamlit run app.py

-Data

heart_disease_clean.csv is a cleaned and preprocessed version of the dataset.

Continuous features are standardized, categorical features are encoded, and the target variable is num (0 = no disease, 1 = disease).

-Models & Evaluation

The following models are used for prediction:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

Metrics are automatically calculated for each model:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Evaluation metrics are saved in the results/evaluation_metrics.txt file. ROC-AUC curves and other visualizations can be generated in Jupyter notebooks.
[Author]

Mohamed Adel

Email: [mohamed.saber@student.guc.edu.eg]

GitHub: [mohamed15adel9-maker]