## Netflix Recommendation System 

This project builds a data-driven movie recommendation system using the [Netflix Prize dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data). It focuses on end-to-end data processing, exploratory data analysis, and predictive modeling using machine learning techniques.

### Goals:

- Preprocess and clean large-scale user-movie rating data
- Analyze viewing patterns and ratings trends
- Build recommendation models using:
  - Matrix factorization (Truncated SVD)
  - Cosine similarity
  - XGBoost regression
- Evaluate model performance on validation and test datasets

### Technologies Used:

- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- XGBoost
- Truncated SVD (matrix decomposition)
- Cosine similarity (collaborative filtering)
- Sparse matrix optimization
- Jupyter Notebook

### Project Structure:
```
Netflix-Data-Processing/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
├── Netflex_data_analysis.ipynb # Main notebook with full pipeline
└── README.md
```

### Key Steps:

1. **Data Preprocessing**:
   - Parsed txt data
   - Cleaned missing and duplicate entries
   - Converted dates, sorted by time

2. **Exploratory Analysis**:
   - User/movie statistics
   - Rating distributions over time

3. **Modeling**:
   - **Collaborative Filtering**: Truncated SVD + Cosine Similarity
   - **XGBoost Regression**: Supervised learning to predict user ratings

4. **Evaluation**:
   - RMSE on validation/test sets
   - Comparison of different algorithms

### How to Run:

1. Clone the repository.

2. Download dataset from [Netflix Prize dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data).

3. Install dependencies:

Run **pip install -r requirements.txt** to download all required packages.

4. Run the notebook:

Open Netflex_data_analysis.ipynb in Jupyter Notebook
