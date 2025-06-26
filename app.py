import streamlit as st
import pandas as pd
import joblib

# === Load mô hình và các file hỗ trợ ===
model = joblib.load('logistic_model_tuned.pkl')
scaler = joblib.load('scaler.pkl')
input_features = joblib.load('input_features.pkl')

# === Các biến cần đảo chiều (Likert 1-5) ===
reverse_cols = [
    'al_learn_boring', 'al_learn_useless', 'al_learn_waste',
    'al_teach_nervous', 'al_teach_comfort', 'al_teach_respect',
    'al_teach_under', 'al_teach_care', 'al_teach_feeling', 'al_teach_trust',
    'al_class_nervous', 'al_class_fit', 'al_class_part',
    'al_class_nice', 'al_class_care', 'al_class_trust'
]

# === Giao diện Streamlit ===
st.title("🎓 Dự đoán Mức độ Xa lánh Học đường")

st.markdown("Vui lòng điền thông tin theo thang điểm 1-5 với mỗi câu hỏi:")

# === Tạo form nhập liệu động từ danh sách đặc trưng ===
user_input = {}
for feature in input_features:
    user_input[feature] = st.slider(feature, min_value=1, max_value=5, value=3)

if st.button("📊 Dự đoán"):
    # Chuyển thành DataFrame
    df_input = pd.DataFrame([user_input])

    # Đảo chiều các câu hỏi cần thiết
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x)

    # Chuẩn hóa dữ liệu
    df_scaled = scaler.transform(df_input[input_features])

    # Dự đoán
    result = model.predict(df_scaled)[0]

    # Hiển thị kết quả
    ket_qua = {1: "🔵 Mức độ THẤP", 2: "🟡 Mức độ VỪA", 3: "🔴 Mức độ CAO"}
    st.success(f"✅ Kết quả dự đoán: **{ket_qua[result]}**")