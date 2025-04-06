# 🫀 Heart Attack Risk Predictor

A simple Streamlit web application that uses machine learning to predict whether a person is at risk of a heart attack based on clinical data.

---

## 🚀 Features

- Predicts heart attack risk using clinical inputs like age, blood pressure, glucose levels, and more.
- Uses a **Decision Tree Classifier** for prediction.
- Displays **model accuracy**.
- Allows user interaction via dropdowns and number inputs.
- Logs "At Risk" predictions to a CSV file (`prediction_log.csv`).

---

## 🧠 How it Works

1. The model is trained on a labeled dataset (`Heart Attack.csv`).
2. The target variable (`class`) is encoded.
3. The user inputs clinical values via the Streamlit UI.
4. The trained model predicts the risk and displays the result.
5. If the result is "At Risk", it is logged for future review.

---

## 🛠️ Setup Instructions

### 🔹 Prerequisites
- Python 3.x
- `pandas`
- `scikit-learn`
- `streamlit`

### 🔹 Installation

```bash
# Clone the repository (or just download the files)
git clone https://github.com/yourusername/heart-attack-predictor.git
cd heart-attack-predictor

# Install dependencies
pip install -r requirements.txt

