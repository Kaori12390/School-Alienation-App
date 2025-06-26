import streamlit as st
import pandas as pd
import joblib

# ===== Load dữ liệu và mô hình =====
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ===== Giao diện người dùng =====
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

# ===== Banner & giới thiệu =====
st.image("image_banner.png", use_column_width=True)
st.title("Dự Đoán Sớm Mức Độ Xa Lánh Học Đường")
st.markdown("""
### Ở Học Sinh Trung Học Việt Nam Bằng Mô Hình Học Máy Có Giám Sát  
Vui lòng trả lời các câu hỏi dưới đây theo thang điểm 1 (**rất không đồng ý**) đến 5 (**rất đồng ý**).
""")

# ===== Câu hỏi tiếng Việt =====
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
    # 👉 Thêm các biến khác tại đây nếu cần
}

# ===== Thu thập phản hồi =====
user_input = {}
for feature in input_features:
    label = question_texts.get(feature, feature)
    user_input[feature] = st.radio(label, [1, 2, 3, 4, 5], index=2)

# ===== Khi nhấn nút Dự đoán =====
if st.button("📊 Dự đoán"):
    df_input = pd.DataFrame([user_input])

    # Đảo chiều nếu cần (nếu có reverse_cols thì thêm vào)
    # reverse_cols = ['class_fit', ...]
    # for col in reverse_cols:
    #     if col in df_input.columns:
    #         df_input[col] = df_input[col].apply(lambda x: 6 - x)

    # Chuẩn hóa và dự đoán
    df_scaled = scaler.transform(df_input[input_features])
    result = model.predict(df_scaled)[0]

    # Hiển thị kết quả
    ket_qua = {
        1: "🟢 Mức độ THẤP",
        2: "🟡 Mức độ VỪA",
        3: "🔴 Mức độ CAO"
    }
    st.success(f"✅ Kết quả dự đoán: **{ket_qua[result]}**")
