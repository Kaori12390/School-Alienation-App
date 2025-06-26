import streamlit as st
import pandas as pd
import joblib

# ===== Load dữ liệu và mô hình =====
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ===== Danh sách các câu hỏi khảo sát theo thứ tự gốc =====
questions = [
    "1. Năm sinh của bạn?",
    "2. Bạn đang học lớp mấy?",
    "3. Giới tính của bạn là gì?",
    "4. Trường bạn đang học tên gì?",
    "5. Điểm trung bình học kỳ trước của bạn là?",
    "6. Xếp loại học lực học kỳ trước của bạn là gì?",
    "7. Mô tả đúng nhất về gia đình bạn đang sống cùng?",
    "8. Trình độ học vấn của mẹ (hoặc mẹ kế)?",
    "9. Nghề nghiệp của mẹ (hoặc mẹ kế)?",
    "10. Thu nhập trung bình hằng tháng của mẹ (hoặc mẹ kế)?",
    "11. Trình độ học vấn của bố (hoặc bố dượng)?",
    "12. Nghề nghiệp của bố (hoặc bố dượng)?",
    "13. Thu nhập trung bình hằng tháng của bố (hoặc bố dượng)?",
]

# ===== Câu hỏi thang Likert 1-5 cần dùng mô hình dự đoán =====
likert_questions = {
    "al_learn_enjoy": "19.1. Mình mong đợi được học ở trường",
    "al_learn_like": "19.2. Mình thích những gì được học ở trường",
    "al_learn_boring": "19.3. Những gì học ở trường rất nhàm chán",
    "al_learn_excite": "19.4. Việc học ở trường rất thú vị",
    "al_learn_interest": "19.5. Mình không thấy hứng thú với việc học ở trường",
    "al_learn_useful": "19.6. Những điều học ở trường không hữu ích trong cuộc sống",
    "al_learn_useless": "19.7. Mình thấy những thứ phải học ở trường thật vô dụng",
    "al_learn_waste": "19.8. Học ở trường là lãng phí thời gian",
    "al_teacher_angry": "20.1. Thầy cô làm mình bực bội",
    "al_teacher_accept": "20.2. Mình cảm thấy được thầy cô chấp nhận",
    "al_teacher_uncomfort": "20.3. Mình không thấy thoải mái khi thầy cô ở gần",
    "al_teacher_disrespect": "20.4. Mình không cảm thấy được thầy cô coi trọng",
    "al_teacher_understand": "20.5. Mình nghĩ thầy cô không hiểu mình",
    "al_teacher_care": "20.6. Mình nghĩ thầy cô không quan tâm đến mình",
    "al_teacher_emotion": "20.7. Mình nghĩ thầy cô không quan tâm đến cảm xúc của mình",
    "al_teacher_trust": "20.8. Mình có thể tin tưởng thầy cô",
    "al_peer_angry": "21.1. Bạn bè làm mình bực bội",
    "al_peer_accept": "21.2. Mình cảm thấy được bạn bè chấp nhận",
    "al_peer_fit": "21.3. Mình cảm thấy mình không phù hợp với lớp",
    "al_peer_part": "21.4. Mình vui khi được là một phần của lớp",
    "al_peer_fun": "21.5. Mình thấy trường học là nơi tuyệt vời vì có nhiều bạn bè",
    "al_peer_ignore": "21.6. Mình không quan tâm đến bạn học",
    "al_peer_trust": "21.7. Mình nghĩ mình có thể tin tưởng bạn học",
    "al_peer_like": "21.8. Lớp học của mình rất tuyệt",
}

reverse_cols = [
    "al_learn_boring", "al_learn_useless", "al_learn_waste",
    "al_teacher_angry", "al_teacher_uncomfort", "al_teacher_disrespect",
    "al_teacher_understand", "al_teacher_care", "al_teacher_emotion",
    "al_peer_angry", "al_peer_ignore", "al_peer_fit",
]

# ===== Giao diện Streamlit =====
st.set_page_config(page_title="Dự đoán xa lánh học đường", layout="wide")

st.markdown("""
    <style>
    .question-block {
        background-color: #f2f2f2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 Dự đoán Mức độ Xa lánh Học đường")

# ===== Giới thiệu =====
st.markdown("""
    <div style="background-color:#17665D;padding:2rem;border-radius:10px">
        <h1 style="color:white">📘 Khảo Sát Về Trải Nghiệm Học Đường</h1>
        <p style="color:white;font-size:1.1rem">Chào em,<br>
        Anh/chị đang thực hiện một khảo sát để tìm hiểu trải nghiệm học đường của học sinh THPT. Những chia sẻ chân thật của em sẽ giúp bọn anh/chị hiểu hơn về cảm nhận và suy nghĩ của học sinh như em trong quá trình học tập tại trường.<br>
        Không có câu trả lời đúng hay sai đâu, nên em hãy trả lời theo đúng cảm nhận của mình nhé. Cảm ơn em rất nhiều!</p>
    </div>
""", unsafe_allow_html=True)

# ===== Thông tin nền tảng =====
with st.form("info_form"):
    st.subheader("📝 Thông tin cá nhân")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("1. Năm sinh của bạn?")
        st.selectbox("2. Bạn đang học lớp mấy?", ["Lớp 10", "Lớp 11", "Lớp 12"])
        st.selectbox("3. Giới tính của bạn là gì?", ["Nam", "Nữ", "Không muốn tiết lộ"])
        st.text_input("4. Trường bạn đang học tên gì?")
        st.selectbox("5. Điểm trung bình học kỳ trước", ["Dưới 3.5", "3.5 - 5.0", "5.0 - 6.5", "6.5 - 8.0", "> 8.0"])
        st.selectbox("6. Xếp loại học lực học kỳ trước", ["Yếu", "Kém", "Trung bình", "Khá", "Giỏi"])
    with col2:
        st.selectbox("7. Mô tả gia đình bạn đang sống cùng", ["Cả bố và mẹ", "Bố không sống cùng", "Mẹ không sống cùng", "Không sống cùng bố mẹ"])
        st.selectbox("8. Trình độ học vấn của mẹ", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
        st.selectbox("9. Nghề nghiệp của mẹ", ["Quản lý", "Chuyên gia", "Công nhân", "Giáo viên", "Lao động khác"])
        st.selectbox("10. Thu nhập mẹ", ["< 5tr", "5-10tr", "10-15tr", "15-20tr", "> 20tr"])
        st.selectbox("11. Trình độ học vấn của bố", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
        st.selectbox("12. Nghề nghiệp của bố", ["Quản lý", "Chuyên gia", "Công nhân", "Giáo viên", "Lao động khác"])
        st.selectbox("13. Thu nhập bố", ["< 5tr", "5-10tr", "10-15tr", "15-20tr", "> 20tr"])
    st.form_submit_button("Lưu thông tin")

# ===== Thang đo dự đoán (Likert) =====
st.subheader("📊 Trả lời các câu hỏi khảo sát")
st.markdown("Vui lòng trả lời các câu hỏi theo thang điểm 1 (_rất không đồng ý_) đến 5 (_rất đồng ý_).")
user_input = {}
cols = st.columns(2)
for i, (var, question) in enumerate(likert_questions.items(), start=14):
    col = cols[i % 2]
    with col:
        with st.container():
            st.markdown(f"<div class='question-block'><strong>{question}</strong>", unsafe_allow_html=True)
            user_input[var] = st.radio("", [1, 2, 3, 4, 5], key=var, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ===== Dự đoán mô hình =====
if st.button("Dự đoán"):
    # Đảm bảo tạo DataFrame với đầy đủ các cột như mô hình yêu cầu
    full_input = {feature: user_input.get(feature, 0) for feature in input_features}
    df_input = pd.DataFrame([full_input])

    # Đảo chiều nếu có biến cần đảo
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x if pd.notnull(x) else x)

    # Chuẩn hóa và dự đoán
    df_scaled = scaler.transform(df_input)
    result = model.predict(df_scaled)[0]

    ket_qua = {
        1: "🟢 Mức độ THẤP",
        2: "🟡 Mức độ VỪA",
        3: "🔴 Mức độ CAO"
    }

    st.success(f"✅ Kết quả dự đoán: **{ket_qua[result]}**")
