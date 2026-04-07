# TMDB Movie Revenue Prediction (Educational Project)

This project is part of an educational exercise focused on applying regression techniques to real-world data.  
The goal is to explore how different features influence movie revenue and to build progressively improved models.

---

## 📂 Project Structure

- `exercise.ipynb` → Main notebook with full workflow (EDA → modeling → feature engineering)
- `pyproject.toml` / `uv.lock` → Dependency management (uv)
- `tmdb_ejercicio_regresion.ipynb` → Original reference notebook

---

## 📊 Dataset

Dataset from Kaggle:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## 🚀 Workflow

1. Data Cleaning  
   - Removed invalid values (budget/revenue = 0)  
   - Parsed JSON columns (genres, cast, crew)  

2. Feature Engineering  
   - Extracted main genre  
   - Created genre dummies  
   - Computed director average revenue  

3. Exploratory Data Analysis  
   - Distribution analysis (log transformation)  
   - Correlation analysis  
   - Scatter plots (budget vs revenue)  

4. Modeling  
   - Simple Linear Regression (baseline)  
   - Multiple Regression  
   - Feature-enhanced model  

5. Model Evaluation  
   - R² and RMSE comparison  
   - Residual diagnostics  

---

## 📈 Results

| Model | R² | RMSE |
|------|----|------|
| Simple Model | ~0.40 | ~0.0377 |
| Multiple Model | ~0.41 | ~0.0374 |
| Feature Engineered Model | **~0.76** | **~0.0240** |

---

## 💡 Key Insights

- **Budget is the strongest predictor** of revenue  
- Runtime and release year have minimal impact  
- Genre significantly influences revenue patterns  
- Director-based features greatly improve performance (though they may introduce leakage if not handled carefully)  

---

## ⚠️ Notes

- This is an **educational project**, not a production-ready model  
- Some engineered features (e.g., director averages) may introduce data leakage if computed on the full dataset  

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## 📌 Goal

The main objective of this project is to understand:

- How to prepare real-world data  
- How to build baseline vs improved models  
- How feature engineering impacts performance  