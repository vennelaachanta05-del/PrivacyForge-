
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PrivacyForge",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 PrivacyForge")
st.subheader("Synthetic Data Generation Platform")
st.write(
    "Generate artificial student data for safe "
    "experimentation and machine learning practice."
)

st.sidebar.header("Dataset Settings")

records = st.sidebar.slider(
    "Number of records",
    min_value=100,
    max_value=5000,
    value=500,
    step=100
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=999999,
    value=42
)

if st.sidebar.button("Generate Dataset"):
    rng = np.random.default_rng(int(seed))

    age = rng.integers(18, 25, records)
    attendance = rng.integers(50, 101, records)
    study_hours = rng.integers(1, 9, records)
    marks = np.clip(
        30 + study_hours * 5
        + (attendance - 75) * 0.3
        + rng.normal(0, 8, records),
        0, 100
    ).round(1)

    data = pd.DataFrame({
        "Student_ID": [
            f"STU{i:04d}" for i in range(1, records + 1)
        ],
        "Age": age,
        "Attendance_Percent": attendance,
        "Study_Hours_Per_Day": study_hours,
        "Marks": marks
    })

    data["Grade"] = pd.cut(
        data["Marks"],
        bins=[-1, 39, 59, 74, 89, 100],
        labels=["F", "D", "C", "B", "A"]
    ).astype(str)

    st.session_state["data"] = data

if "data" in st.session_state:
    data = st.session_state["data"]

    st.success("Synthetic dataset generated!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(data))
    c2.metric("Average Marks", f"{data['Marks'].mean():.1f}")
    c3.metric(
        "Average Attendance",
        f"{data['Attendance_Percent'].mean():.1f}%"
    )

    st.subheader("Generated Dataset")
    st.dataframe(data, use_container_width=True)

    st.subheader("Marks Distribution")
    fig, ax = plt.subplots()
    ax.hist(data["Marks"], bins=10, edgecolor="black")
    ax.set_xlabel("Marks")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Synthetic Student Marks")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Attendance vs Marks")
    fig2, ax2 = plt.subplots()
    ax2.scatter(
        data["Attendance_Percent"],
        data["Marks"],
        alpha=0.6
    )
    ax2.set_xlabel("Attendance (%)")
    ax2.set_ylabel("Marks")
    ax2.set_title("Attendance vs Marks")
    st.pyplot(fig2)
    plt.close(fig2)

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Dataset as CSV",
        data=csv,
        file_name="privacyforge_student_data.csv",
        mime="text/csv"
    )

    st.info(
        "These records are generated artificially. "
        "They are not real student records or proof of privacy."
    )
else:
    st.info("Click 'Generate Dataset' in the sidebar to begin.")
