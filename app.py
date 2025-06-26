import streamlit as st
import pandas as pd
import joblib

# ===== Load dữ liệu và mô hình =====
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ===== Các biến Likert cần đảo ngược thang điểm =====
reverse_cols = [
    "al_learn_boring", "al_learn_useless", "al_learn_waste",
    "al_teacher_angry", "al_teacher_uncomfort", "al_teacher_disrespect",
    "al_teacher_understand", "al_teacher_care", "al_teacher_emotion",
    "al_peer_angry", "al_peer_ignore", "al_peer_fit"
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

# ===== Giới thiệu =====
st.title("📚 Dự đoán Mức độ Xa lánh Học đường")

st.markdown("""
<div style="background-color:#17665D;padding:2rem;border-radius:10px">
    <h1 style="color:white">📘 Khảo Sát Về Trải Nghiệm Học Đường</h1>
    <p style="color:white;font-size:1.1rem">Chào em,<br>
    Anh/chị đang thực hiện một khảo sát để tìm hiểu trải nghiệm học đường của học sinh THPT.
    Những chia sẻ chân thật của em sẽ giúp bọn anh/chị hiểu hơn về cảm nhận và sự khó khăn mà em đang gặp phải.<br>
    Không có câu trả lời đúng hay sai đâu, nên em hãy trả lời theo đúng cảm nhận của mình nhé. Cảm ơn em rất nhiều!</p>
</div>
""", unsafe_allow_html=True)

# ===== Form thông tin cá nhân =====
st.subheader("🎓 Thông tin cá nhân")
user_input = {}

col1, col2 = st.columns(2)
with col1:
    user_input["birth"] = st.text_input("1. Năm sinh của bạn?")
    user_input["grade"] = st.selectbox("2. Bạn đang học lớp mấy?", ["10", "11", "12"])
    user_input["gender"] = st.selectbox("3. Giới tính của bạn là gì?", ["Nam", "Nữ", "Khác"])
    user_input["school"] = st.text_input("4. Trường bạn đang học tên gì?")
    user_input["score"] = st.selectbox("5. Điểm trung bình học kỳ trước?", ["<3.5", "3.5-5", "5-6.5", "6.5-8", ">8"])
    user_input["rank"] = st.selectbox("6. Xếp loại học lực kỳ trước?", ["Yếu", "Kém", "Trung bình", "Khá", "Giỏi", "Xuất sắc"])
with col2:
    user_input["live"] = st.selectbox("7. Mô tả gia đình bạn đang sống cùng?", ["Cả bố và mẹ", "Chỉ bố", "Chỉ mẹ", "Người thân khác"])
    user_input["mom_edu"] = st.selectbox("8. Trình độ học vấn của mẹ?", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
    user_input["mom_occ"] = st.selectbox("9. Nghề nghiệp của mẹ?", ["Quản lý", "Chuyên gia", "Nhân viên văn phòng", "Dịch vụ, bán hàng", "Nông lâm ngư nghiệp", "Thủ công", "Công nhân", "Lao động giản đơn", "Quân nhân"])
    user_input["mom_inc"] = st.selectbox("10. Thu nhập mẹ?", ["< 5tr", "5-10tr", "10-15tr", "15-20tr", "> 20tr"])
    user_input["dad_edu"] = st.selectbox("11. Trình độ học vấn của bố?", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
    user_input["dad_occ"] = st.selectbox("12. Nghề nghiệp của bố?", ["Quản lý", "Chuyên gia", "Nhân viên văn phòng", "Dịch vụ, bán hàng", "Nông lâm ngư nghiệp", "Thủ công", "Công nhân", "Lao động giản đơn", "Quân nhân"])
    user_input["dad_inc"] = st.selectbox("13. Thu nhập bố?", ["< 5tr", "5-10tr", "10-15tr", "15-20tr", "> 20tr"])

# ===== Thang đo Likert =====
st.subheader("📊 Trả lời các câu hỏi khảo sát")
st.markdown("Vui lòng trả lời các câu hỏi theo thang điểm 1 (_rất không đồng ý_) đến 5 (_rất đồng ý_).")

likert_questions = joblib.load("likert_questions.pkl")  # chứa các câu hỏi đã đặt tên biến khớp với mô hình
cols = st.columns(2)
for i, (var, question) in enumerate(likert_questions.items(), start=14):
    with cols[i % 2]:
        with st.container():
            st.markdown(f"<div class='question-block'><strong>{i}. {question}</strong>", unsafe_allow_html=True)
            user_input[var] = st.radio("", [1, 2, 3, 4, 5], key=var, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ===== Dự đoán =====
if st.button("Dự đoán"):
    df_input = pd.DataFrame([user_input])

    # Điền giá trị 0 cho các feature thiếu
    for col in input_features:
        if col not in df_input.columns:
            df_input[col] = 0

    # Đảo ngược Likert nếu cần
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x)

    # Chuẩn hóa và dự đoán
    df_scaled = scaler.transform(df_input[input_features])
    result = model.predict(df_scaled)[0]

    ket_qua = {
        1: "🟢 Mức độ THẤP",
        2: "🟡 Mức độ VỪA",
        3: "🔴 Mức độ CAO"
    }
    st.success(f"✅ Kết quả dự đoán: **{ket_qua[result]}**")



