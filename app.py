import streamlit as st
import numpy as np
import joblib
import json

st.set_page_config(page_title="ระบบทำนายความเสี่ยงนิสิตลาออก", page_icon="🎓")

import os


@st.cache_resource
def load_model():
    """โหลด pipeline และ metadata — ทำครั้งเดียวตอนเริ่ม app"""
    pipeline = joblib.load("model_artifacts/dropout_model.pkl")
    with open("model_artifacts/model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata

# โหลดโมเดล — Streamlit จะแสดง spinner ระหว่างรอ
with st.spinner("กำลังโหลดโมเดล..."):
    pipeline, metadata = load_model()

# ===== Sidebar: ข้อมูลเกี่ยวกับโมเดล =====
# Sidebar เหมาะสำหรับข้อมูลเสริมที่ไม่ใช่ส่วนหลักของ app
with st.sidebar:
    st.header("ℹ️ เกี่ยวกับโมเดลนี้")
    st.write(f"**ประเภทโมเดล:** {metadata['model_type']}")
    st.write(f"**ความแม่นยำ:** {metadata['accuracy']*100:.1f}%")
    st.write(f"**ข้อมูล train:** {metadata['training_samples']:,} ราย")

    st.divider()  # เส้นคั่น

    st.subheader("⚠️ ข้อควรระวัง")
    st.warning(
        "ผลลัพธ์นี้เป็นการประเมินเบื้องต้นจาก AI เท่านั้น "
        "ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้ "
        "กรุณาปรึกษาแพทย์หากมีข้อสงสัย"
    )


pipeline, metadata = load_model()

st.title("🔍 Student Dropout Prediction")
st.write("กรุณากรอกข้อมูลนิสิตเพื่อประเมินความเสี่ยงในการลาออก")

# สร้าง Form รับข้อมูลตาม Features ในไฟล์ CSV ของคุณ
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("อายุ (Age)", min_value=15.0, max_value=40.0, value=20.0)
        gender_str = st.selectbox("เพศ (Gender)", ["Male", "Female"])
        income = st.number_input("รายได้ครอบครัว (Family Income)", min_value=0, value=25000, step=1000)
        internet = st.selectbox("มีอินเทอร์เน็ตที่บ้านไหม (Internet)", ["Yes", "No"])
        study_hours = st.number_input("ชั่วโมงอ่านหนังสือ/วัน (Study Hours)", min_value=0.0, max_value=24.0, value=3.0)
        attendance = st.slider("อัตราเข้าเรียน % (Attendance)", 0, 100, 80)

    with col2:
        delay_days = st.number_input("ส่งงานเลทเฉลี่ย/วัน (Delay Days)", min_value=0, value=2)
        travel_time = st.number_input("เวลาเดินทาง/นาที (Travel Time)", min_value=0.0, max_value=200.0, value=30.0)
        part_time = st.selectbox("ทำงานพาร์ทไทม์ไหม (Part Time Job)", ["Yes", "No"])
        scholarship = st.selectbox("ได้รับทุนไหม (Scholarship)", ["Yes", "No"])
        stress = st.slider("ระดับความเครียด (Stress Index) 1-10", 1.0, 10.0, 5.0)
        semester = st.selectbox("ชั้นปี (Semester)", ["Year 1", "Year 2", "Year 3", "Year 4"])

    with col3:
        department = st.selectbox("คณะ (Department)", ["Arts", "Business", "CS", "Engineering", "Science"])
        parent_edu = st.selectbox("การศึกษาผู้ปกครอง (Parental Edu)", ["None", "High School", "Bachelor", "Master", "PhD"])
        gpa = st.number_input("GPA ปัจจุบัน", min_value=0.0, max_value=4.0, value=2.5)
        semester_gpa = st.number_input("GPA เทอมล่าสุด", min_value=0.0, max_value=4.0, value=2.5)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=4.0, value=2.5)

    submit = st.form_submit_button("ประเมินผล")

if submit:
    # แปลง Category เป็นตัวเลข (Label Encoding ตามลำดับตัวอักษร)
    gender_map = {"Female": 0, "Male": 1}
    yes_no_map = {"No": 0, "Yes": 1}
    semester_map = {"Year 1": 0, "Year 2": 1, "Year 3": 2, "Year 4": 3}
    dept_map = {"Arts": 0, "Business": 1, "CS": 2, "Engineering": 3, "Science": 4}
    parent_map = {"Bachelor": 0, "High School": 1, "Master": 2, "None": 3, "PhD": 4}

    # สร้าง Dictionary ข้อมูลนำเข้า ให้ลำดับคอลัมน์ตรงกับ feature_names ที่เซฟไว้
    input_data = {
        "Student_ID": 999999, # Dummy ID (โมเดลเก่าติดฟีเจอร์นี้มาด้วย)
        "Age": age,
        "Gender": gender_map[gender_str],
        "Family_Income": income,
        "Internet_Access": yes_no_map[internet],
        "Study_Hours_per_Day": study_hours,
        "Attendance_Rate": attendance,
        "Assignment_Delay_Days": delay_days,
        "Travel_Time_Minutes": travel_time,
        "Part_Time_Job": yes_no_map[part_time],
        "Scholarship": yes_no_map[scholarship],
        "Stress_Index": stress,
        "GPA": gpa,
        "Semester_GPA": semester_gpa,
        "CGPA": cgpa,
        "Semester": semester_map[semester],
        "Department": dept_map[department],
        "Parental_Education": parent_map[parent_edu]
    }

    import pandas as pd
    X_input = pd.DataFrame([input_data])

    prediction = pipeline.predict(X_input)[0]
    prob = pipeline.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ มีความเสี่ยงในการลาออกสูง ({prob * 100:.1f}%)")
    else:
        st.success(f"✅ นิสิตมีแนวโน้มเรียนต่อปกติ (โอกาสลาออกเพียง {prob * 100:.1f}%)")