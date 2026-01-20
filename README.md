# CKD-Prediction-Project
https://kidney-disease-predict.streamlit.app/   
A machine learning model for predicting chronic kidney disease.     
This project aims to predict Chronic Kidney Disease (CKD) using the powerful XGBoost machine learning algorithm, with its performance further optimized through hyperparameter tuning with Optuna.        
1. Introduction
Chronic Kidney Disease (CKD) is a global health concern affecting millions worldwide. Early and accurate detection is crucial for timely intervention and treatment, which can help slow down disease progression and improve patient outcomes. Machine learning offers a promising approach for early detection by analyzing patient data and identifying patterns indicative of CKD.
2. Motivation
Traditional diagnostic methods can be time consuming and expensive. This project utilizes machine learning, specifically XGBoost, known for its accuracy and efficiency in handling complex datasets, to develop a predictive model for CKD. Hyperparameter tuning with Optuna helps in achieving optimal performance and generalization of the XGBoost model.
3. Dataset
The dataset used in this project is sourced from UCI Machine Learning Repository. It contains various clinical parameters and patient characteristics known to be associated with CKD development. 






📝 Overview
This project is a machine learning application designed to predict the likelihood of   Chronic Kidney Disease (CKD)   in patients. By analyzing clinical parameters such as age, blood pressure, and specific blood chemistry values, the model provides an early risk assessment to assist in timely medical intervention.
The core predictive model is built using (XGBoost), a powerful gradient boosting framework, and is fine-tuned for optimal performance using (Optuna) for hyperparameter optimization. The application is deployed as a user-friendly web interface using (Streamlit).

🚀 Features
  High-Accuracy Prediction: Utilizes the XGBoost algorithm, known for its efficiency and performance on structured data.
  Hyperparameter Tuning: Model performance is maximized through automated hyperparameter optimization using Optuna.
  Interactive Web Interface: A simple, intuitive Streamlit dashboard allows users to input medical data and get instant predictions.
  Data Preprocessing: Includes robust handling of missing values and feature scaling (using stored median values).

---

## 📂 Project Structure

Here is an overview of the key files in the repository:

| File Name | Description |
| :--- | :--- |
| `app.py` | The main Streamlit application file that runs the web interface. |
| `Revised.ipynb` | Jupyter Notebook containing the data analysis, preprocessing, model training, and Optuna optimization logic. |
| `xgb_ckd_model` | The trained and saved XGBoost model file. |
| `features.pkl` | Pickle file containing the list of feature names used by the model. |
| `median_values.pkl` | Pickle file storing median values for imputing missing data in user inputs. |
| `final_ckd_data.csv` | The clean dataset used for training the model. |
| `requirements.txt` | List of Python dependencies required to run the project. |
| `kidney.jpeg` | Image used for the project banner/display. |

---

## 📊 Dataset

The model is trained on the   Chronic Kidney Disease dataset   sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease).

  Key Features Considered:  
    Age, Blood Pressure  
    Specific Gravity, Albumin, Sugar  
    Red Blood Cells, Pus Cells  
    Blood Glucose Random, Blood Urea, Serum Creatinine  
    Hemoglobin, Packed Cell Volume  
    Hypertension, Diabetes Mellitus, Coronary Artery Disease  
   (And other clinical indicators) 

---

## 🛠️ Installation & Usage

To run this project locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Md-Inam/CKD-Prediction-Project.git](https://github.com/Md-Inam/CKD-Prediction-Project.git)
cd CKD-Prediction-Project

```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Application

```bash
streamlit run app.py

```

The app will launch in your default web browser at `http://localhost:8501`.

---

## 🧠 Model Development

The `Revised.ipynb` notebook outlines the full lifecycle of the model development:

1.   Data Cleaning:   Handling null values and encoding categorical variables.
2.   Exploratory Data Analysis (EDA):   Understanding feature distributions and correlations.
3.   Feature Selection:   Identifying the most significant predictors for CKD.
4.   Model Training:   Training the XGBoost Classifier.
5.   Optimization:   Using Optuna to find the best hyperparameters (e.g., learning rate, max depth, n_estimators).
6.   Evaluation:   Assessing performance using metrics like Accuracy, Precision, Recall, and F1-Score.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model or the UI, please feel free to:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

---

## 👤 Author

  Md. Inam  

  GitHub: [@Md-Inam](https://www.google.com/search?q=https://github.com/Md-Inam)

---

 Disclaimer: This project is for educational and informational purposes only. It should not be used as a substitute for professional medical diagnosis or advice. 

```

```
