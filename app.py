import streamlit as st
import pandas as pd
import joblib

# Load mô hình và scaler
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

st.set_page_config(page_title="Dự đoán xa lánh học đường", layout="wide")

# ===== Giới thiệu =====
st.markdown("""
    <div style="background-color:#17665D;padding:2rem;border-radius:10px">
        <h1 style="color:white">📘 Khảo Sát Về Trải Nghiệm Học Đường</h1>
        <p style="color:white;font-size:1.1rem">Chào em,<br>
        Anh/chị đang thực hiện một khảo sát để tìm hiểu trải nghiệm học đường của học sinh THPT. Những chia sẻ chân thật của em sẽ giúp bọn anh/chị hiểu hơn về cảm nhận và suy nghĩ của học sinh như em trong quá trình học tập tại trường.<br>
        Không có câu trả lời đúng hay sai đâu, nên em hãy trả lời theo đúng cảm nhận của mình nhé. Cảm ơn em rất nhiều 💚</p>
    </div>
""", unsafe_allow_html=True)

# ===== Thông tin cá nhân =====
st.header("1. Thông tin cá nhân")
info = {}
col1, col2 = st.columns(2)
with col1:
    info["1. Năm sinh"] = st.text_input("1. Năm sinh")
    info["2. Lớp"] = st.selectbox("2. Bạn đang học lớp mấy?", ["10", "11", "12"])
    info["3. Giới tính"] = st.radio("3. Giới tính", ["Nam", "Nữ", "Khác"])
    info["4. Trường đang học"] = st.text_input("4. Trường bạn đang học")
    info["5. Điểm TB học kỳ trước"] = st.selectbox("5. Điểm trung bình học kỳ trước", ["<3.5", "3.5–5.0", "5.0–6.5", "6.5–8.0", ">8.0"])
    info["6. Xếp loại học lực"] = st.selectbox("6. Xếp loại học lực học kỳ trước", ["Yếu", "Trung bình", "Khá", "Giỏi"]) 
with col2:
    info["7. Sống với ai"] = st.selectbox("7. Bạn đang sống với ai?", ["Bố mẹ", "Chỉ bố", "Chỉ mẹ", "Người thân khác"])
    info["8. Học vấn của mẹ"] = st.selectbox("8. Trình độ học vấn của mẹ", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
    info["9. Nghề nghiệp của mẹ"] = st.text_input("9. Nghề nghiệp của mẹ")
    info["10. Thu nhập mẹ"] = st.selectbox("10. Thu nhập trung bình tháng của mẹ", ["<5 triệu", "5–10 triệu", "10–15 triệu", ">15 triệu"])
    info["11. Học vấn của bố"] = st.selectbox("11. Trình độ học vấn của bố", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
    info["12. Nghề nghiệp của bố"] = st.text_input("12. Nghề nghiệp của bố")
    info["13. Thu nhập bố"] = st.selectbox("13. Thu nhập trung bình tháng của bố", ["<5 triệu", "5–10 triệu", "10–15 triệu", ">15 triệu"])

# ===== Bộ câu hỏi khảo sát =====
st.header("2. Bộ câu hỏi khảo sát")

# Câu hỏi nhóm Học tập
hoc_tap = {
    "18.1": "Mình cảm thấy mình có giá trị hơn khi học tốt",
    "18.2": "Mình cảm thấy tệ hơn nếu kết quả học tập kém",
    "18.3": "Việc học tốt khiến mình thấy bản thân đáng tự hào",
    "18.4": "Mình cảm thấy mình có giá trị hơn khi đạt thành tích tốt",
    "18.5": "Mình cảm thấy tự ti hơn khi kết quả học tập không tốt",
    "19.1": "Mình mong đợi được học ở trường",
    "19.2": "Mình thích những gì được học ở trường",
    "19.3": "Những gì học ở trường rất nhàm chán",
    "19.4": "Việc học ở trường rất thú vị",
    "19.5": "Mình không thấy hứng thú với việc học ở trường",
    "19.6": "Những điều học ở trường không hữu ích trong cuộc sống",
    "19.7": "Mình thấy những thứ phải học ở trường thật vô dụng",
    "19.8": "Học ở trường là lãng phí thời gian"
}

# Câu hỏi nhóm Giáo viên
giao_vien = {
    "16.1": "Nếu mình học chăm chỉ thì thầy cô sẽ quý trọng mình hơn",
    "16.2": "Nếu mình làm bài kiểm tra tốt thì thầy cô sẽ chấp nhận mình hơn",
    "16.3": "Nếu mình học giỏi thì thầy cô sẽ quan tâm đến mình hơn",
    "16.4": "Nếu mình học tốt thì thầy cô sẽ thích mình hơn",
    "16.5": "Nếu mình học giỏi thì thầy cô sẽ thân thiện với mình hơn",
    "17.1": "Nếu mình không học chăm chỉ thì thầy cô sẽ ít quý trọng mình",
    "17.2": "Nếu mình làm bài kiểm tra không tốt thì thầy cô sẽ ít chấp nhận mình",
    "17.3": "Nếu mình học không tốt thì thầy cô sẽ ít quan tâm đến mình",
    "17.4": "Nếu mình học không tốt thì thầy cô sẽ ít thích mình",
    "17.5": "Nếu mình học không tốt thì thầy cô sẽ bớt thân thiện với mình",
    "20.1": "Thầy cô làm mình bực bội",
    "20.2": "Mình cảm thấy được thầy cô chấp nhận",
    "20.3": "Mình không thấy thoải mái khi thầy cô ở gần",
    "20.4": "Mình không cảm thấy được thầy cô coi trọng",
    "20.5": "Mình nghĩ thầy cô không hiểu mình",
    "20.6": "Mình nghĩ thầy cô không quan tâm đến mình",
    "20.7": "Mình nghĩ thầy cô không quan tâm đến cảm xúc của mình",
    "20.8": "Mình có thể tin tưởng thầy cô"
}

# Câu hỏi nhóm Bạn bè
ban_be = {
    "21.1": "Bạn bè làm mình bực bội",
    "21.2": "Mình cảm thấy được bạn bè chấp nhận",
    "21.3": "Mình cảm thấy mình không phù hợp với lớp",
    "21.4": "Mình vui khi được là một phần của lớp",
    "21.5": "Mình thấy trường học là nơi tuyệt vời vì có nhiều bạn bè",
    "21.6": "Mình không quan tâm đến bạn học",
    "21.7": "Mình nghĩ mình có thể tin tưởng bạn học",
    "21.8": "Lớp học của mình rất tuyệt"
}

def render_block(title, questions):
    st.subheader(title)
    answers = {}
    for q_num, q_text in questions.items():
        with st.container():
            st.markdown(f"<div style='background-color:#f0f0f0;padding:1rem;border-radius:10px;margin-bottom:1rem'><strong>{q_num}. {q_text}</strong></div>", unsafe_allow_html=True)
            answers[q_num] = st.radio("", [1, 2, 3, 4, 5], horizontal=True, key=q_num)
    return answers

res_ht = render_block("📘 Học tập", hoc_tap)
res_gv = render_block("👨‍🏫 Giáo viên", giao_vien)
res_bb = render_block("🧑‍🤝‍🧑 Bạn bè", ban_be)

# ===== Dự đoán =====
if st.button("Dự đoán"):
    all_responses = {**res_ht, **res_gv, **res_bb}
    model_input = {k: all_responses.get(k, 3) for k in input_features}
    df_input = pd.DataFrame([model_input])

    reverse_cols = []  # thêm nếu cần
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x)

    df_scaled = scaler.transform(df_input[input_features])
    result = model.predict(df_scaled)[0]

    labels = {
        1: "🟢 Mức độ THẤP",
        2: "🟡 Mức độ VỪA",
        3: "🔴 Mức độ CAO"
    }
    st.success(f"✅ Kết quả dự đoán: **{labels[result]}**")

