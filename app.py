import streamlit as st
import pandas as pd
import joblib

# ===== Load model and scaler =====
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ===== Page config =====
st.set_page_config(page_title="Dự đoán xa lánh học đường", layout="wide")

# ===== Header and Introduction =====
st.markdown("""
    <div style="background-color:#125f50;padding:2rem;border-radius:10px">
        <h1 style="color:white;font-size:2.8rem">📚 Khảo Sát Về Trải Nghiệm Học Đường</h1>
        <p style="color:white;font-size:1.2rem">Chào em,</p>
        <p style="color:white;font-size:1.2rem">Anh chị đang thực hiện một nghiên cứu về trải nghiệm học đường của học sinh THPT. Mong em dành chút thời gian trả lời bảng khảo sát dưới đây. Những câu trả lời của em rất quan trọng và sẽ giúp anh chị hiểu hơn về những khó khăn trong môi trường học đường mà em đang gặp phải.</p>
        <p style="color:white;font-size:1.2rem">Em cứ thoải mái chia sẻ mọi suy nghĩ thật lòng nhé!</p>
        <p style="color:white;font-size:1.2rem">Cảm ơn em rất nhiều. Chúc em một ngày vui vẻ!</p>
    </div>
""", unsafe_allow_html=True)

# ===== Demographic Info Section =====
st.markdown("""
    <h2 style='font-size:1.5rem'>🎓 Thông tin khái quát:</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("1. Họ và tên")
    gender = st.radio("2. Giới tính", ["Nam", "Nữ", "Khác"])
    grade = st.selectbox("3. Bạn đang học lớp mấy?", ["Lớp 10", "Lớp 11", "Lớp 12"])
with col2:
    birth_year = st.text_input("4. Năm sinh của bạn")
    school = st.text_input("5. Trường bạn đang học")
    gpa = st.selectbox("6. Điểm trung bình học kỳ trước", ["Dưới 3.5", "3.5 - 5.0", "5.0 - 6.5", "6.5 - 8.0", "> 8.0"])

# ===== Survey Questions by Topic with numbering and model prediction =====
st.markdown("""
    <h2 style='font-size:1.5rem;margin-top:2rem'>📘 Khảo sát cảm nhận</h2>
    <p>Vui lòng trả lời theo thang điểm từ 1 (Rất không đồng ý) đến 5 (Rất đồng ý).</p>
""", unsafe_allow_html=True)

question_blocks = {
    "14. Bố mẹ (tích cực)": [
        ("q141", "14.1. Mình cảm thấy nếu mình học chăm chỉ thì ba mẹ sẽ quý trọng mình hơn"),
        ("q142", "14.2. Mình cảm thấy nếu mình làm bài kiểm tra tốt thì ba mẹ sẽ chấp nhận mình hơn"),
        ("q143", "14.3. Mình cảm thấy nếu mình học giỏi thì ba mẹ sẽ quan tâm đến mình hơn"),
        ("q144", "14.4. Mình cảm thấy nếu mình học tốt thì ba mẹ sẽ yêu thương mình hơn"),
        ("q145", "14.5. Mình cảm thấy nếu mình đạt kết quả học tập tốt thì ba mẹ sẽ dịu dàng và ấm áp với mình hơn")
    ],
    "15. Bố mẹ (tiêu cực)": [
        ("q151", "15.1. Nếu mình không học chăm chỉ thì ba mẹ sẽ quý trọng mình ít hơn"),
        ("q152", "15.2. Nếu mình làm bài kiểm tra không tốt thì ba mẹ sẽ ít chấp nhận mình"),
        ("q153", "15.3. Nếu mình học không tốt thì ba mẹ sẽ ít quan tâm đến mình"),
        ("q154", "15.4. Nếu mình không học tốt thì ba mẹ sẽ ít yêu thương mình"),
        ("q155", "15.5. Nếu mình học không tốt thì ba mẹ sẽ bớt dịu dàng và ấm áp với mình")
    ],
    "16. Thầy cô (tích cực)": [
        ("q161", "16.1. Nếu mình học chăm chỉ thì thầy cô sẽ quý trọng mình hơn"),
        ("q162", "16.2. Nếu mình làm bài kiểm tra tốt thì thầy cô sẽ chấp nhận mình hơn"),
        ("q163", "16.3. Nếu mình học giỏi thì thầy cô sẽ quan tâm đến mình hơn"),
        ("q164", "16.4. Nếu mình học tốt thì thầy cô sẽ thích mình hơn"),
        ("q165", "16.5. Nếu mình học giỏi thì thầy cô sẽ thân thiện với mình hơn")
    ],
    "17. Thầy cô (tiêu cực)": [
        ("q171", "17.1. Nếu mình không học chăm chỉ thì thầy cô sẽ ít quý trọng mình"),
        ("q172", "17.2. Nếu mình làm bài kiểm tra không tốt thì thầy cô sẽ ít chấp nhận mình"),
        ("q173", "17.3. Nếu mình học không tốt thì thầy cô sẽ ít quan tâm đến mình"),
        ("q174", "17.4. Nếu mình học không tốt thì thầy cô sẽ ít thích mình"),
        ("q175", "17.5. Nếu mình học không tốt thì thầy cô sẽ bớt thân thiện với mình")
    ]
}

user_input = {}
for section, questions in question_blocks.items():
    st.subheader(section)
    for key, text in questions:
        with st.container():
            st.markdown(f"<div style='background-color:#f2f2f2;padding:1rem;border-radius:8px;margin-bottom:1rem'><strong>{text}</strong></div>", unsafe_allow_html=True)
            user_input[key] = st.radio("", [1, 2, 3, 4, 5], index=2, horizontal=True, key=key)

# Mapping keys for model (ensure you map exactly to input_features expected by the model)
model_input_keys = input_features

# Dummy mapping: you will need to replace below with correct mapping to match model features
model_input = {k: user_input.get(k, 3) for k in model_input_keys}

# ===== Predict Button =====
if st.button("🔍 Dự đoán"):
    df_input = pd.DataFrame([model_input])

    # Đảo chiều nếu cần
    reverse_cols = []  # Add if necessary
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x)

    df_scaled = scaler.transform(df_input[model_input_keys])
    result = model.predict(df_scaled)[0]

    ket_qua = {
        1: "🟢 Mức độ THẤP",
        2: "🟡 Mức độ VỪA",
        3: "🔴 Mức độ CAO"
    }
    st.success(f"✅ Kết quả dự đoán: **{ket_qua[result]}**")
