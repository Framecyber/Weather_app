import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# โหลดข้อมูลอากาศ
data = pd.read_csv(r'Data\weather_data.csv')

# สำรวจข้อมูล
print(data.head())

# เลือก feature ข้อมูลที่ใช้พยากรณ์ และ target ข้อมูลที่ใช้ทำนาย
X = data[['temp_max', 'temp_min', 'wind', 'precipitation']]
y = data['weather']  # สมมติว่า 'weather' เป็นตัวแปรเป้าหมายที่เป็น categorical เช่น 'drizzle'

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ตรวจสอบชนิดของ y: หากเป็นสตริง ให้เข้ารหัสเป็นตัวเลขสำหรับตัวจำแนก
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# สร้างโมเดลการจำแนก (classification)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train_enc)

# ทำนายผลการทำนาย (ได้เป็นค่าเข้ารหัส)
y_pred_enc = model.predict(X_test)

joblib.dump(model, 'weather_model.pkl')

# ประเมินผลโมเดล
acc = accuracy_score(y_test_enc, y_pred_enc)
print(f"Accuracy: {acc:.4f}")
print("Classification report:\n", classification_report(y_test_enc, y_pred_enc, target_names=le.classes_))

# แปลงผลทำนายกลับเป็นป้ายกำกับข้อความ (ถ้าต้องการแสดง)
y_pred = le.inverse_transform(y_pred_enc)

# แสดงผลการทำนาย: plot ระหว่างค่าเข้ารหัสของจริงและทำนาย
plt.scatter(y_test_enc, y_pred_enc, alpha=0.6)
plt.xlabel('True (encoded)')
plt.ylabel('Predicted (encoded)')
plt.title('True vs Predicted (encoded labels)')
plt.show()



