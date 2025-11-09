import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib
import io
import os

st.set_page_config(page_title="🌦️ Weather Predictor", page_icon="🌈")

st.title("🌦️ Weather Prediction App (Random Forest)")
st.write("อัปโหลดไฟล์ข้อมูลอากาศ (CSV) เพื่อสร้างโมเดล หรือกรอกข้อมูลเพื่อพยากรณ์สภาพอากาศ")

# ======================================
# ส่วนที่ 1: อัปโหลดและฝึกโมเดล
# ======================================
uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ weather_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("🧾 ตัวอย่างข้อมูล")
    st.dataframe(data.head())

    required_cols = ['temp_max', 'temp_min', 'wind', 'precipitation', 'weather']
    if not all(col in data.columns for col in required_cols):
        st.error(f"❌ ไฟล์ต้องมีคอลัมน์: {required_cols}")
    else:
        X = data[['temp_max', 'temp_min', 'wind', 'precipitation']]
        y = data['weather']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        if st.button("🚀 Train Model"):
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

            st.success(f"✅ Accuracy: {acc:.4f}")

            st.subheader("📊 Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Plot True vs Predicted
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.set_xlabel("True (encoded)")
            ax.set_ylabel("Predicted (encoded)")
            ax.set_title("True vs Predicted (encoded labels)")
            st.pyplot(fig)

            # Save model + label encoder
            joblib.dump(model, "weather_model.pkl")
            joblib.dump(le, "label_encoder.pkl")

            buf = io.BytesIO()
            joblib.dump(model, buf)
            buf.seek(0)
            st.download_button(
                label="💾 ดาวน์โหลดโมเดล (weather_model.pkl)",
                data=buf,
                file_name="weather_model.pkl"
            )

            st.success("🎉 โมเดลถูกบันทึกเรียบร้อยแล้ว!")

# ======================================
# ส่วนที่ 2: พยากรณ์จากการกรอกข้อมูล
# ======================================
st.subheader("🌤️ ทำนายสภาพอากาศจากข้อมูลที่กรอก")

if os.path.exists("weather_model.pkl") and os.path.exists("label_encoder.pkl"):
    model = joblib.load("weather_model.pkl")
    le = joblib.load("label_encoder.pkl")

    col1, col2 = st.columns(2)
    with col1:
        temp_max = st.number_input("🌡️ อุณหภูมิสูงสุด (°C)", min_value=-20.0, max_value=60.0, value=35.0)
        wind = st.number_input("💨 ความเร็วลม (km/h)", min_value=0.0, max_value=200.0, value=10.0)
    with col2:
        temp_min = st.number_input("🌡️ อุณหภูมิต่ำสุด (°C)", min_value=-20.0, max_value=60.0, value=25.0)
        precipitation = st.number_input("☔ ปริมาณฝน (mm)", min_value=0.0, max_value=500.0, value=5.0)

    if st.button("🔍 พยากรณ์"):
        input_data = pd.DataFrame([[temp_max, temp_min, wind, precipitation]],
                                  columns=['temp_max', 'temp_min', 'wind', 'precipitation'])
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = le.inverse_transform([prediction_encoded])[0]

        st.success(f"🌈 ผลการพยากรณ์: **{prediction_label}**")
else:
    st.warning("⚠️ ยังไม่มีโมเดลที่ถูกฝึก กรุณาอัปโหลดและฝึกโมเดลก่อน")
