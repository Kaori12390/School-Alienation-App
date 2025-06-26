import streamlit as st
import pandas as pd
import joblib

# Load mô hình và scaler
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# Câu hỏi nền tảng (1-13)
def thong_tin_nen_tang():
    st.header("1. Thông tin chung")
    birth = st.text_input("1. Năm sinh của bạn")
    grade = st.selectbox("2. Bạn đang học lớp mấy", ["Lớp 10", "Lớp 11", "Lớp 12"])
    gender = st.radio("3. Giới tính", ["Nam", "Nữ", "Không muốn tiết lộ"])
    school = st.text_input("4. Trường bạn đang học")
    gpa = st.selectbox("5. Điểm trung bình học kì trước", ["Dưới 3.5", "3.5 - 5.0", "5.0 - 6.5", "6.5 - 8.0", "Trên 8.0"])
    gpa_des = st.selectbox("6. Xếp loại học lực", ["Yếu", "Kém", "Trung bình", "Khá", "Giỏi"])
    livewith = st.selectbox("7. Mô tả về gia đình bạn sống cùng", ["Sống với bố và mẹ", "Chỉ có mẹ", "Chỉ có bố", "Không sống cùng bố mẹ"])
    mom_edu = st.selectbox("8. Trình độ học vấn của mẹ", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
    mom_occ = st.text_input("9. Nghề nghiệp của mẹ")
    mom_inc = st.selectbox("10. Thu nhập trung bình/tháng của mẹ", ["< 5tr", "5-10tr", "10-15tr", "15-20tr", ">20tr"])
    dad_edu = st.selectbox("11. Trình độ học vấn của bố", ["Tiểu học", "THCS", "THPT", "Đại học", "Sau đại học"])
    dad_occ = st.text_input("12. Nghề nghiệp của bố")
    dad_inc = st.selectbox("13. Thu nhập trung bình/tháng của bố", ["< 5tr", "5-10tr", "10-15tr", "15-20tr", ">20tr"])

# Biến đảo chiều
reverse_cols = [
    'al_learn_boring', 'al_learn_useless', 'al_learn_waste',
    'al_teach_nervous', 'al_teach_comfort', 'al_teach_respect',
    'al_teach_under', 'al_teach_care', 'al_teach_feeling', 'al_teach_trust',
    'al_class_nervous', 'al_class_fit', 'al_class_part',
    'al_class_nice', 'al_class_care', 'al_class_trust'
]

# Nhóm câu hỏi và nhãn
groups = {
    "📖 Việc học": ['al_learn_like', 'al_learn_enjoy', 'al_learn_boring', 'al_learn_exciting', 'al_learn_useless', 'al_learn_waste'],
    "👩‍🏫 Giáo viên": ['al_teach_nervous', 'al_teach_accept', 'al_teach_comfort', 'al_teach_respect', 'al_teach_under', 'al_teach_care', 'al_teach_feeling', 'al_teach_trust'],
    "🤝 Bạn bè": ['al_class_nervous', 'al_class_accept', 'al_class_fit', 'al_class_part', 'al_class_nice', 'al_class_care', 'al_class_trust', 'al_class_cool'],
    "🧠 Tự đánh giá bản thân": ['achv_value', 'achv_bad_feel', 'achv_worth']
}

feature_labels = {
    'al_learn_like': "19.1. Mình mong đợi được học ở trường",
    'al_learn_enjoy': "19.2. Mình thích những gì được học ở trường",
    'al_learn_boring': "19.3. Những gì học ở trường rất nhàm chán",
    'al_learn_exciting': "19.4. Việc học ở trường rất thú vị",
    'al_learn_useless': "19.7. Mình thấy những thứ phải học ở trường thật vô dụng",
    'al_learn_waste': "19.8. Học ở trường là lãng phí thời gian",
    'achv_value': "18.4. Mình cảm thấy mình có giá trị hơn khi đạt thành tích tốt",
    'achv_bad_feel': "18.2. Mình cảm thấy tệ về bản thân nếu học không tốt",
    'achv_worth': "18.5. Mình cảm thấy tự ti hơn khi kết quả học tập không tốt"
}

# CSS xám nền
st.markdown("""
    <style>
    .question-block {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề
st.title("Khảo sát cảm nhận về việc học, giáo viên và gia đình")
st.markdown("""
Chúng mình là nhóm nghiên cứu thuộc EdLab Asia đang thực hiện khảo sát nhằm tìm hiểu cảm nhận của học sinh về sự quan tâm học tập từ phía gia đình và nhà trường.
""")

# Thông tin chung
thong_tin_nen_tang()

# Thu thập dữ liệu cho các nhóm câu hỏi
user_input = {}
for group_name, features in groups.items():
    st.subheader(group_name)
    for feat in features:
        label = feature_labels.get(feat, feat)
        with st.container():
            st.markdown(f'<div class="question-block">**{label}**</div>', unsafe_allow_html=True)
            user_input[feat] = st.radio("", [1, 2, 3, 4, 5], index=2, key=feat)

# Khi nhấn dự đoán
if st.button("Dự đoán"): 
    df_input = pd.DataFrame([user_input])
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x)
    df_scaled = scaler.transform(df_input[input_features])
    result = model.predict(df_scaled)[0]
    st.success(f"Kết quả: {'🟢 Thấp' if result==1 else '🟡 Trung bình' if result==2 else '🔴 Cao'}")

