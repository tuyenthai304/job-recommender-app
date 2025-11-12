import numpy as np
import pandas as pd
import streamlit as st
import time
import os

#--------------------------------#
# Load dữ liệu
new_user = pd.read_csv('users.csv')
new_job = pd.read_csv('jobs.csv')

new_user.fillna('không', inplace=True)
new_job.fillna('không', inplace=True)

new_user.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
new_job.rename(columns={"Unnamed: 0": "JobID"}, inplace=True)

# Giả lập BERT similarity matrix
bert_similarity = np.load('sim_industry.npy')

#--------------------------------#
# Hàm gợi ý dùng BERT (mock)
def get_recommendation_bert(user_id, location, top_n=5):
    # Lọc job theo location
    jobs_in_location = new_job[new_job['Job Address'] == location].reset_index(drop=True)
    if jobs_in_location.empty:
        return pd.DataFrame()

    # Lấy index người dùng
    user_idx = new_user[new_user['UserID'] == user_id].index.values[0]


    # Tính similarity (chỉ lấy subset theo job location)
    job_indices = jobs_in_location.index.tolist()
    sim_scores = bert_similarity[user_idx, job_indices]

    jobs_in_location['similarity'] = sim_scores
    top_jobs = jobs_in_location.sort_values('similarity', ascending=False).head(top_n)

    return top_jobs

#--------------------------------#
# Giao diện Streamlit
st.set_page_config(page_title="BERT Job Recommender")

with st.sidebar:
    st.header("Login")
    add_userID = st.number_input('Enter User ID:', min_value=1, step=1)
    add_password = st.text_input('Enter password:', type="password")
    submit = st.button('Login')

if submit:
    if add_userID not in new_user['UserID'].values:
        st.error("Invalid User ID. Please try again.")
        st.session_state["logged_in"] = False
    else:
        st.session_state["logged_in"] = True
        st.success(f"Welcome, User {add_userID}!")



st.title("Job Recommendation System using BERT")
st.subheader("Welcome to BERT-powered Recommendation Demo")

location = st.text_input("Enter job location:")
if location:
    with st.spinner('Finding jobs for you...'):
        time.sleep(2)
        recommendations = get_recommendation_bert(add_userID, location, top_n=5)

    if recommendations.empty:
        st.warning("No matching jobs found.")
    else:
        st.success(f"Top {len(recommendations)} jobs recommended for you:")
        for i, row in recommendations.iterrows():
            # Chọn ảnh theo URL
            job_url = str(row.get("URL Job", "")).lower()

            if "timviec365" in job_url:
                image_file = "hinh1.png"
            else:
                image_file = "hinh2.png"  # ảnh mặc định nếu không khớp

            # Ghép đường dẫn đến ảnh trong thư mục images
            image_path = os.path.join("images", image_file)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_path, caption="Job Preview")

            with col2:
                st.markdown(f"**Job Title:** {row['Job Title']}")
                st.markdown(f"**Industry:** {row['Industry']}")
                st.markdown(f"**Salary:** {row['Salary']}")
                st.markdown(f"**Address:** {row['Job Address']}")
                st.markdown(f"**Description:** {row['Job Description'][:100]}...")
                st.markdown(f"[View Details]({row['URL Job']})")
