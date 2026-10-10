A machine learning web app that estimates the likelihood of heart disease from just four patient attributes. It uses a Support Vector Classifier (SVC) trained on the classic UCI Heart Disease (Cleveland) dataset and is deployed live on Streamlit Community Cloud.
	
Task	Binary classification: heart disease present (1) or absent (0)
Dataset	UCI Heart Disease (Cleveland), 303 patients
Input features	4 (age, sex, chest pain type, maximum heart rate)
Model	SVC (RBF kernel) with StandardScaler, wrapped in a scikit-learn Pipeline
Output	Prediction plus disease probability
Deployment	Streamlit Community Cloud

This project was built for the course AIC354 at COMSATS University Islamabad, Lahore Campus. The assignment constraint was to select only 4 input attributes from the dataset.

🧠 How it works
Data: the Cleveland subset of the UCI Heart Disease data (303 rows: 164 without disease, 139 with disease). The original target num (0 to 4) is converted to binary: num > 0 means disease present.
Feature selection: four attributes are used.
Split: 80% training and 20% testing, stratified so both sets keep the same class ratio (random_state=42).
Model comparison: SVC, Logistic Regression and Decision Tree are compared with 5-fold cross-validation.
Final model: an SVC pipeline (StandardScaler followed by SVC) is trained and saved as svc_trained_model.pkl.
Evaluation: accuracy, precision, recall, F1-score and a confusion matrix on the held-out test set.
App: a Streamlit interface loads the saved model and predicts for new input.
🧾 Input features
Feature	Description	Values
Age	Age in years	Number
Sex	Sex of the patient	1 = Male, 0 = Female
ChestPain	Chest pain type	1 Typical Angina, 2 Atypical Angina, 3 Non-anginal Pain, 4 Asymptomatic
MaxHeartRate	Maximum heart rate achieved	Number (bpm)

Target: HeartDisease (0 = no disease, 1 = disease present)

In the web app, Sex and Chest pain are chosen from dropdowns with readable labels. The numeric codes are handled behind the scenes.

📊 Results

Evaluated on the 20% held-out test set.

Metric	Score
Accuracy	fill in your value
Precision (disease)	fill in your value
Recall (disease)	fill in your value
F1-score (disease)	fill in your value

With only four input features, moderate accuracy is expected. Important clinical signals such as cholesterol, resting ECG and exercise-induced angina are intentionally left out because of the assignment constraint. The full classification report and confusion matrix are in the notebook.

🗂️ Repository structure
heart-disease-app/
├── app.py                              # Streamlit web app
├── heart.csv                           # Dataset (4 features + target)
├── svc_trained_model.pkl               # Trained SVC pipeline
├── requirements.txt                    # Dependencies for deployment
└── heart_disease_prediction (2).ipynb  # Full workflow: EDA, training, evaluation, prediction
💻 Run it locally
bash
# 1. Clone the repository
git clone https://github.com/emantair3912-lab/heart-disease-app.git
cd heart-disease-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
streamlit run app.py

To run the notebook as well, install the extra notebook libraries:

bash
pip install jupyter matplotlib seaborn prettytable
jupyter notebook

Use the same scikit-learn version as in requirements.txt (1.9.0). A model pickled with one scikit-learn version may not load correctly in another.

⚠️ Limitations
Educational project only. This is not a medical tool and must not be used for diagnosis or treatment decisions. Always consult a qualified doctor.
Small dataset (303 patients) from a single source (Cleveland).
Only four features are used, so the model ignores many important risk factors.
ChestPain is a nominal category but is fed to the model as numbers 1 to 4, which implies an order that does not really exist.
🔮 Future improvements
Use more attributes (cholesterol, resting blood pressure, exercise-induced angina, ST depression).
Combine the Cleveland data with the Hungarian, Switzerland and VA Long Beach sets for more training data.
One-hot encode chest pain type.
Tune SVC hyperparameters (C, gamma, kernel) with GridSearchCV.
Compare Random Forest and Gradient Boosting, and prefer high recall on the disease class.
🛠️ Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, Jupyter

👤 Author

Your Name (FA24-BSE-034) COMSATS University Islamabad, Lahore Campus, Course AIC354

📚 Data source

Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. (1988). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X (licensed CC BY 4.0)
