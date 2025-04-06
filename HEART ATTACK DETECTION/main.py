import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Attack Predictor", page_icon="🫀")

st.title("🫀 Heart Attack Risk Predictor")
st.markdown("""
Welcome to the **Heart Attack Risk Predictor** app!  
This tool uses machine learning to predict whether a person may be at risk of a heart attack based on clinical features.
👉 Fill out the form below and click **Check** to get a prediction.  
⚠️ **Disclaimer:** This is for educational purposes only. Always consult a medical professional.
""")


df = pd.read_csv(r"Heart Attack.csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])

x = df.iloc[:,:-1]
y = df["class"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Evaluation")
st.write(f"✅ Model Accuracy on Test Set: **{accuracy * 100:.2f}%**")

col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.selectbox("Age", sorted(df["age"].unique()))
with col2:
    gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
with col3:
    impluse = st.selectbox("Heart Rate (Impulse)", sorted(df["impluse"].unique()))
with col4:
    systolic = st.selectbox("Systolic Pressure", sorted(df["pressurehight"].unique()))

col5, col6 = st.columns(2)
with col5:
    diastolic = st.selectbox("Diastolic Pressure", sorted(df["pressurelow"].unique()))
with col6:
    glucose = st.selectbox("Glucose Level", sorted(df["glucose"].unique()))

col7, col8 = st.columns(2)
with col7:
    kcm = st.number_input("KCM (Creatine Kinase)", min_value=0.0, max_value=5000.0, step=0.1)
with col8:
    troponin = st.number_input("Troponin Level", min_value=0.0, max_value=50.0, step=0.01)


if st.button("Check"):
    inp = [[age, gender, impluse, systolic, diastolic, glucose, kcm, troponin]]
    r = dt.predict(inp)

    if r == 0:
        st.header(":green[Positive]")
        result = "Not at Risk"
    else:
        st.header(":red[Negative]")
        result = "At Risk"
    new_row = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'impluse': [impluse],
        'systolic': [systolic],
        'diastolic': [diastolic],
        'glucose': [glucose],
        'kcm': [kcm],
        'troponin': [troponin],
        'prediction': [result]
    })

    try:
        log_df = pd.read_csv("prediction_log.csv")
        log_df = pd.concat([log_df, new_row], ignore_index=True)
    except FileNotFoundError:
        log_df = new_row

    log_df.to_csv("prediction_log.csv", index=False)


