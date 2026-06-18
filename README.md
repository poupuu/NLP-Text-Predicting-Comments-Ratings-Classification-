# NLP-Text-Predicting-Comments-Ratings-Classification

![Ratings 5 Review](assets/Modified-Model2-Results.png)

## Summary
This project develops a Natural Language Processing (NLP) machine learning pipeline to predict numerical user ratings based on unstructured text reviews. Using data scraped from the Google Play Store for the game "Genshin Impact," the project evaluates and compares different text representation methods (TF-IDF vs. Word2Vec) alongside various classification algorithms (Logistic Regression, SVM, LightGBM) to discover the most effective predictive model.

## Problem
1. **Unstructured Data Interpretation:** Transforming raw, noisy user reviews into meaningful features that can accurately predict a user's satisfaction level (rating).
2. **Transferable Sentiment Analysis:** Building a robust rating prediction prototype that can eventually be scaled and applied to predict review ratings for other applications or content types.

## Methodology
1. **Data Crawling:** Extracted 1,000 English reviews for the highly discussed game "Genshin Impact" directly from the Google Play Store, ensuring a varied distribution of sentiment.
2. **Exploratory Data Analysis (EDA):** Conducted thorough checks on rating distributions, review lengths, and text noise (HTML, URLs, mentions, non-ASCII characters). Extracted non-text sentiment signals including emojis, punctuation counts, and ALL CAPS usage.
3. **Text Preprocessing:** Cleaned the text by removing formatting and converting emojis to text. Normalized slang and typos, applied case folding, tokenized the strings, removed custom stopwords, and performed lemmatization using NLTK.
4. **Text Representation:** Transformed the cleaned text into numerical features by experimenting with two contrasting methods: TF-IDF (Term Frequency-Inverse Document Frequency) to evaluate word importance, and Word2Vec (Skip-gram) to capture contextual word embeddings.
5. **Modeling & Tuning:** Trained Logistic Regression (as a baseline), Support Vector Machines (SVM), and LightGBM models. Applied Hyperparameter Tuning via GridSearchCV to optimize model parameters.
6. **Imbalance Handling:** Evaluated the performance impact of handling imbalanced rating distributions using synthetic data generation (SMOTE).

## Skills
1. **Python:** Pandas, Numpy, Matplotlib, Seaborn, WordCloud
2. **Data Collection:** Web Scraping (google-play-scraper)
3. **NLP Preprocessing:** NLTK, Regex, Tokenization, Lemmatization, Stopword Removal
4. **Feature Engineering:** TF-IDF (TfidfVectorizer), Word Embeddings (Word2Vec)
5. **Machine Learning:** Scikit-learn, LightGBM, Logistic Regression, SVM, GridSearchCV
6. **Imbalance Handling:** Imbalanced-learn (SMOTE)

## Results
1. **Word2Vec + LightGBM Outperformed:** The combination of Word2Vec embeddings and the LightGBM classifier successfully discovered complex, non-linear patterns in the text and emerged as the best-performing model.
2. **SMOTE Limitations Identified:** The experiments revealed that utilizing SMOTE for this specific NLP task generated noise and caused overfitting during parameter tuning. LightGBM's internal mechanisms for handling imbalanced data proved superior to synthetic oversampling.
3. **Small Data Viability:** The prototype successfully demonstrated that by carefully adjusting configuration parameters (e.g., specific min_count and vector_size in Word2Vec), it is highly possible to build a functional NLP model even with a limited dataset of 1,000 reviews.

## Next Steps
1. **Scale for Production:** Implement the pipeline on a much larger volume of review data to improve model robustness and generalize the predictions.
2. **Alternative Imbalance Techniques:** Move away from SMOTE and experiment with other advanced techniques to handle class imbalances more effectively without introducing noise.
3. **Deepen Evaluation Metrics:** Shift evaluation focus away from simple accuracy and prioritize metrics like precision, recall, and F1-score to get a truer picture of model performance across all rating classes.
4. **Hyperparameter Explainability:** Further document and justify the reasoning behind specific hyperparameter choices (such as specific learning rates) to improve model transparency.
