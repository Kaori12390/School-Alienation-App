import streamlit as st
import pandas as pd
import joblib

# ===== Load mô hình & scaler & feature =====
model = joblib.load("logistic_model_tuned.pkl")
scaler = joblib.load("scaler.pkl")
input_features = joblib.load("input_features.pkl")

# ===== Biến cần đảo chiều =====
reverse_cols = [
    'al_learn_boring', 'al_learn_useless', 'al_learn_waste',
    'al_teach_nervous', 'al_teach_comfort', 'al_teach_respect',
    'al_teach_under', 'al_teach_care', 'al_teach_feeling', 'al_teach_trust',
    'al_class_nervous', 'al_class_fit', 'al_class_part',
    'al_class_nice', 'al_class_care', 'al_class_trust'
]

# ===== Nhóm biến theo chủ đề =====
grouped_features = {
    "🎓 Học tập": [
        'al_learn_like', 'al_learn_enjoy', 'al_learn_exciting',
        'al_learn_pleasure', 'al_learn_useful', 'al_learn_boring',
        'al_learn_useless', 'al_learn_waste'
    ],
    "👩‍🏫 Giáo viên": [
        'al_teach_nervous', 'al_teach_accept', 'al_teach_comfort',
        'al_teach_respect', 'al_teach_under', 'al_teach_care',
        'al_teach_feeling', 'al_teach_trust'
    ],
    "👫 Bạn bè": [
        'al_class_nervous', 'al_class_accept', 'al_class_fit',
        'al_class_part', 'al_class_nice', 'al_class_care',
        'al_class_trust', 'al_class_cool'
    ]
}

# ===== Tên tiếng Việt (ví dụ đầy đủ nên bạn cần thêm vào nếu có biến mới) =====
feature_labels = {
    'al_learn_like': "Bạn mong đợi được học ở trường",
    'al_learn_enjoy': "Bạn thích nội dung học ở trường",
    'al_learn_exciting': "Việc học ở trường rất thú vị",
    'al_learn_pleasure': "Bạn cảm thấy vui khi học ở trường",
    'al_learn_useful': "Những điều học ở trường hữu ích",
    'al_learn_boring': "Những nội dung học ở trường rất nhàm chán",
    'al_learn_useless': "Bạn thấy kiến thức học là vô dụng",
    'al_learn_waste': "Học ở trường là lãng phí thời gian",
    
    'al_teach_nervous': "Thầy cô làm bạn cảm thấy căng thẳng",
    'al_teach_accept': "Bạn cảm thấy được thầy cô chấp nhận",
    'al_teach_comfort': "Bạn không thoải mái khi thầy cô ở gần",
    'al_teach_respect': "Bạn không được thầy cô coi trọng",
    'al_teach_under': "Bạn nghĩ thầy cô không hiểu mình",
    'al_teach_care': "Bạn nghĩ thầy cô không quan tâm bạn",
    'al_teach_feeling': "Thầy cô không quan tâm đến cảm xúc của bạn",
    'al_teach_trust': "Bạn có thể tin tưởng thầy cô",

    'al_class_nervous': "Bạn cảm thấy bạn bè làm bạn bực bội",
    'al_class_accept': "Bạn cảm thấy được bạn bè chấp nhận",
    'al_class_fit': "Bạn thấy mình không phù hợp với lớp",
    'al_class_part': "Bạn thấy vui khi là một phần của lớp",
    'al_class_nice': "Bạn thấy trường học tuyệt vì có bạn bè",
    'al_class_care': "Bạn không quan tâm đến bạn học",
    'al_class_trust': "Bạn tin tưởng bạn học",
    'al_class_cool': "Lớp học của bạn thật tuyệt"
}

# ===== CSS tùy chỉnh =====
st.set_page_config(page_title="Dự đoán xa lánh học đường", layout="wide")
st.markdown("""
    <style>
    .stRadio > div {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #146356;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Tiêu đề =====
st.title("📚 Dự đoán Mức độ Xa lánh Học đường")
st.markdown("Vui lòng trả lời các câu hỏi theo thang điểm 1 (**rất không đồng ý**) đến 5 (**rất đồng ý**).")

# ===== Giao diện theo nhóm =====
user_input = {}
for group_name, features in grouped_features.items():
    st.header(group_name)
    col1, col2 = st.columns(2)
    for i, feat in enumerate(features):
        label = feature_labels.get(feat, feat)
        with col1 if i % 2 == 0 else col2:
            user_input[feat] = st.radio(label, [1, 2, 3, 4, 5], index=2)

# ===== Dự đoán khi người dùng nhấn nút =====
if st.button("📊 Dự đoán"):
    df_input = pd.DataFrame([user_input])

    # Đảo chiều Likert cho các biến cần thiết
    for col in reverse_cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].apply(lambda x: 6 - x)

    # Chuẩn hoá
    df_scaled = scaler.transform(df_input[input_features])

    # Dự đoán
    result = model.predict(df_scaled)[0]

    # Kết quả
    ket_qua = {
        1: "🟢 Xa lánh học đường THẤP",
        2: "🟡 Xa lánh học đường VỪA",
        3: "🔴 Xa lánh học đường CAO"
    }
    st.success(f"🎯 **Kết quả dự đoán:** {ket_qua[result]}")
