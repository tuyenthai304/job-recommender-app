# Job Recommender System (Streamlit App)

This Streamlit web app recommends jobs for users using **TF-IDF**, **Word2Vec**, and **BERT** models.

## Features
- Job–candidate semantic similarity scoring.
- Top-K recommendations with accuracy and precision evaluation.
- Cosine similarity, Accuracy@K, and Precision@K metrics.
- Built with **Python**, **Streamlit**, **Pandas**, **Scikit-learn**, and **Sentence-Transformers**.

## Data
Do giới hạn dung lượng GitHub, các file dữ liệu (jobs.csv, users.csv, sim_industry.npy)
được lưu tại Google Drive:
 [Download tại đây](https://drive.google.com/drive/folders/1SInPBfdK0h-PfHgadTOzaDCSgfLP-anV)

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
