# 📚 Book Recommendation System — Web-mining Project

**Short:** Teaching/demo repository implementing simple recommender baselines and experiments using a Books / Ratings dataset. Includes data cleaning, EDA, a user-average baseline, cosine similarities, and an NCF prototype.

---

## 🔍 Overview
This repo demonstrates core recommender system concepts and baseline comparisons:
- **Baseline:** User-average predictor + popularity-based ranking  
- **Comparisons:** Cosine similarity (user/item), and a simple NCF prototype

---

## 📁 Project Structure
- `train.csv`, `test.csv` — train/test splits used in experiments  
- `book_dataset/Books.csv`, `Ratings.csv`, `Users.csv` — original dataset files  
- `baseline.ipynb` — user-average baseline and evaluation (`predict_rating_user_avg`, `recommend_user_avg`, `get_model_scorecard`)  
- `cosine-sim.ipynb` — similarity-based recommenders (user/item cosine)  
- `ncf-model.ipynb` — neural collaborative filtering prototype  
- `data_cleaning.ipynb`, `eda.ipynb` — preprocessing and exploratory analysis  
- `split_data.py` — utility to generate train/test splits

---

