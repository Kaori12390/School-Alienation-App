import streamlit as st
import pandas as pd
import joblib

# ===== Load dữ liệu và mô hình =====
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ===== Cấu hình giao diện =====
st.set_page_config(page_title="Dự đoán xa lánh học đường", layout="wide")

# ===== CSS tuỳ chỉnh =====
st.markdown("""
    <style>
    .main {
        background-color: #f2f2f2;
    }
    h1 {
        color: white;
        background-color: #146356;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .stRadio > div {
        background-color: #ffffff;
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Tiêu đề & giới thiệu =====
st.title("🎓 Dự đoán Mức độ Xa lánh Học đường")
st.markdown("""
Vui lòng điền thông tin theo thang điểm 1 (**rất không đồng ý**) đến 5 (**rất đồng ý**) với mỗi câu hỏi:
""")

# ===== Câu hỏi tiếng Việt (rút gọn ví dụ) =====
question_texts = {
    "alien_learn_score": "Bạn cảm thấy hứng thú với việc học ở trường",
    "alien_teacher_score": "Bạn cảm thấy được giáo viên chấp nhận",
    "alien_peer_score": "Bạn cảm thấy được bạn bè chấp nhận",
    "achv_value": "Bạn cảm thấy mình có giá trị hơn khi học tốt",
    "achv_bad_feel": "Bạn cảm thấy tệ hơn nếu kết quả học tập kém",
    "achv_worth": "Bạn cảm thấy tự ti nếu điểm thấp",
    "teach_respect": "Bạn cảm thấy được thầy cô coi trọng",
    "teach_care": "Bạn cảm thấy thầy cô quan tâm đến mình",
    "class_part": "Bạn cảm thấy hạnh phúc khi là một phần của lớp học",
    "class_trust": "Bạn nghĩ mình có thể tin tưởng bạn bè trong lớp",
    "class_fit": "Bạn cảm thấy mình không phù hợp với lớp học",
    "learn_useful": "Những điều học ở trường hữu ích cho cuộc sống"
}

# ===== Tạo form chia làm 2 cột =====
col1, col2 = st.columns(2)
user_input = {}

for i, feature in enumerate(input_features):
    label = question_texts.get(feature, feature)
    with col1 if i % 2 == 0 else col2:
        user_input[feature] = st.radio(label, [1, 2, 3, 4, 5], index=2)

# ===== Dự đoán khi người dùng nhấn nút =====
if st.button("📊 Dự đoán"):
    df_input = pd.DataFrame([user_input])
    df_scaled = scaler.transform(df_input[input_features])
    result = model.predict(df_scaled)[0]

    # ===== Hiển thị kết quả =====
    ket_qua = {
        1: "🟢 Mức độ THẤP",
        2: "🟡 Mức độ VỪA",
        3: "🔴 Mức độ CAO"
    }
    st.success(f"✅ Kết quả dự đoán: **{ket_qua[result]}**")
