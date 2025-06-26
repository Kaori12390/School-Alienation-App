import streamlit as st
import pandas as pd
import joblib

# ======================== Load model & preprocessing ========================
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ======================== Page configuration ========================
st.set_page_config(page_title="Dự đoán xa lánh học đường", layout="wide")

# ======================== Custom CSS ========================
st.markdown("""
<style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .question-box {
        background-color: #f2f2f2;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .intro-box {
        background-color: #146356;
        padding: 25px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ======================== Introduction ========================
st.markdown("""
<div class="intro-box">
    <h2>📖 Khảo Sát Về Trải Nghiệm Học Đường</h2>
    <p>Chào em,
    <br>Chị/một anh đang thực hiện một nghiên cứu về trải nghiệm học đường của học sinh THPT.
    Mong em dành chút thời gian trả lời bảng khảo sát dưới đây. Những câu trả lời của em rất quan trọng và sẽ giúc chị/anh hiểu hơn về những khó khăn mà học sinh đang gặp phải.
    <br>Không có đáp án đúng hay sai. Em hãy trả lời thật thể nhé!
    <br>Chân thành cảm ơn em rất nhiều 🙏🏻</p>
</div>
""", unsafe_allow_html=True)

# ======================== Survey Form ========================
st.markdown("""
### 🎓 Thông tin khái quát:
""")

col1, col2 = st.columns(2)
with col1:
    ho_ten = st.text_input("Họ và tên")
    gioi_tinh = st.radio("Giới tính", ["Nam", "Nữ", "Khác"])
with col2:
    lop = st.text_input("Lớp")
    truong = st.text_input("Trường")

if ho_ten and lop and truong:
    if st.button("Bắt đầu khảo sát"):
        st.session_state.start_survey = True

# ======================== Questions ========================
if st.session_state.get("start_survey"):
    st.markdown("""
    <h3>🎮 Khảo sát trải nghiệm học đường</h3>
    <p>Vui lòng trả lời các câu hỏi theo thang điểm 1 (<strong>rất không đồng ý</strong>) đến 5 (<strong>rất đồng ý</strong>).</p>
    """, unsafe_allow_html=True)

    # Define all questions based on the survey document
    question_texts = {
        "al_learn_expect": "1. Bạn mong đợi được học ở trường",
        "al_learn_like": "2. Bạn thích nội dung học ở trường",
        "al_learn_exciting": "3. Việc học ở trường rất thú vị",
        "al_learn_pleasure": "4. Bạn cảm thấy vui khi học ở trường",
        "al_learn_useless": "5. Bạn cảm thấy những điều học ở trường là vô ích",
        "al_learn_boring": "6. Bạn thấy việc học ở trường thật nhàm chán",
        "al_learn_waste": "7. Việc đi học là lãng phí thời gian",
        "al_learn_useful": "8. Những điều học ở trường hữu ích cho cuộc sống",
        # ... Thêm tiếp các câu hỏi tiếp theo ...
    }

    user_input = {}
    q_keys = list(question_texts.keys())
    for i in range(0, len(q_keys), 2):
        col1, col2 = st.columns(2)
        with col1:
            key1 = q_keys[i]
            user_input[key1] = st.radio(
                question_texts[key1], [1, 2, 3, 4, 5], key=key1, horizontal=True
            )
        if i + 1 < len(q_keys):
            with col2:
                key2 = q_keys[i + 1]
                user_input[key2] = st.radio(
                    question_texts[key2], [1, 2, 3, 4, 5], key=key2, horizontal=True
                )

    if st.button("🔢 Dự đoán"):
        df_input = pd.DataFrame([user_input])

        # Đảo chiều
        reverse_cols = ['al_learn_boring', 'al_learn_useless', 'al_learn_waste']
        for col in reverse_cols:
            if col in df_input.columns:
                df_input[col] = df_input[col].apply(lambda x: 6 - x)

        # Chuẩn hóa
        df_scaled = scaler.transform(df_input[input_features])
        result = model.predict(df_scaled)[0]

        # Kết quả
        ket_qua = {
            1: "🔴 Mức ĐỘ THẤP",
            2: "🟡 Mức ĐỘ Vừa",
            3: "🔵 Mức ĐỘ CAO"
        }
        st.success(f"Kết quả dự đoán: **{ket_qua[result]}**")


