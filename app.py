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
    col1, col2 = st.columns(2)

    with col1:
        gpa = st.number_input("GPA ปัจจุบัน", min_value=0.0, max_value=4.0, value=2.5)
        attendance = st.slider("อัตราการเข้าเรียน (%)", 0, 100, 80)
        stress = st.slider("ระดับความเครียด (1-10)", 1, 10, 5)

    with col2:
        delay_days = st.number_input("จำนวนวันที่ส่งงานเลท (เฉลี่ย)", min_value=0, value=2)
        income = st.number_input("รายได้ครอบครัว", min_value=0, value=25000)
        gender = st.selectbox("เพศ", ["Male", "Female"])

    submit = st.form_submit_button("ประเมินผล")

if submit:
    # สร้าง Dictionary ข้อมูลนำเข้า (ต้องชื่อตรงกับ Column ใน CSV)
    input_data = {
        "GPA": gpa,
        "Attendance_Rate": attendance,
        "Stress_Index": stress,
        "Assignment_Delay_Days": delay_days,
        "Family_Income": income,
        "Gender": gender,
        # เพิ่มตัวแปรอื่นๆ ให้ครบตามที่ใช้ใน Model
    }

    # แปลงเป็น DataFrame หรือ Array ตามที่ Pipeline ต้องการ
    import pandas as pd

    X_input = pd.DataFrame([input_data])

    prediction = pipeline.predict(X_input)[0]
    prob = pipeline.predict_proba(X_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ มีความเสี่ยงในการลาออกสูง ({prob * 100:.1f}%)")
    else:
        st.success(f"✅ นิสิตมีแนวโน้มเรียนต่อปกติ (โอกาสลาออกเพียง {prob * 100:.1f}%)")