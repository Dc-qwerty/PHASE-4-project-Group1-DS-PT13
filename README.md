# # A MODEL THAT CATEGORIZES A TWEET SENTIMENT AS POSITIVE, NEGATIVE OR NEUTRAL

## Ds-Pt13(Group 1)
![performing-twitter-sentiment-analysis1](https://github.com/user-attachments/assets/0525b5ae-1e11-42c4-85f0-3d4289ecf01c)

## PROJECT OVERVIEW
This project develops a natural language processing (NLP) model to automatically classify tweet sentiment as positive, negative, or neutral for Apple and Google products. It uses a labeled dataset of over 9,000 tweets to provide a scalable, real-time sentiment analysis tool, eliminating the need for manual review.

## BUSINESS PROBLEM
On social media, consumer perceptions of technology products change quickly and openly. As two of the most talked-about tech companies in the world, Apple and Google receive thousands of mentions every day on Twitter. However, neither company nor its partners have a scalable method to track the real-time reception of product launches, updates, or service outages. This amount of tweets cannot be manually reviewed, and traditional survey-based feedback is too slow and too limited to capture natural public opinion.

## PROBLEM STATEMENT
Create an NLP-based sentiment classifier as a proof-of-concept to help stakeholders understand public opinion at scale.

## STAKEHOLDERS
1. Product Teams -
identify user pain and points and feature resonance quickly.
2. Marketing -
assess campaign success and adjust messaging in real time.
3. Corporate Communications -
monitor brand reputation and detect emerging issues early.
4. Investors - 
track sentiment trends for informed investment decisions.

## Team members
* Grace Wangui
* Salma Jediel
* David Clement
* Eddy Wiwatsu
* Charity Kilonzo

## Data Understanding

### Data Source
The dataset used in this project is sourced from `judge.csv`, containing over 9,000 labeled tweets related to Apple and Google products.

### Data Description
- Columns:
  - `tweet_text`: The text content of the tweet.
  - `emotion_in_tweet_is_directed_at`: The brand or product the emotion is directed at (e.g., iPhone, Android, Apple, Google).
  - `is_there_an_emotion_directed_at_a_brand_or_product`: The sentiment label (Positive, Negative, Neutral, or No emotion toward brand or product).
- Sentiment Classes: The target variable has four classes: Positive, Negative, Neutral, and No emotion toward brand or product. Tweets labeled as "I can't tell" were filtered out during preprocessing.
- Brands/Products: Tweets mention various Apple products (e.g., iPhone, iPad, iOS), Google products (e.g., Android, Gmail), or general brand references.

### Data Exploration and Preprocessing
- Missing Values: Approximately 5,551 rows had missing values in the `emotion_in_tweet_is_directed_at` column. These were filled using keyword-based inference (e.g., matching words like "iPhone" to "iPhone").
- Cleaning: Rows with null `tweet_text` were dropped. The dataset was filtered to exclude ambiguous sentiments.
- Output: The cleaned dataset was saved as `tweets_cleaned.csv` for modeling.

Key Insights: Sentiment distribution shows a mix of positive, negative, and neutral opinions, with specific products like iPhone and Android frequently mentioned. This highlights the need for product-level sentiment analysis.

## Data Analysis

### Exploratory Data Analysis (EDA)
- Sentiment Distribution: The dataset shows an imbalance in sentiment classes. After cleaning, the distribution is approximately:
  - Neutral:  ~60%
  - Positive: ~25%
  - Negative: ~15%
- Brand/Product Distribution: Common mentions include iPhone, iPad, Google, Android, and Apple products. A 'brand' column was created to categorize tweets as Apple, Google, Both, or Other.

Key Insights: Neutral sentiments dominate, indicating many tweets do not express strong emotions toward brands. Apple products (e.g., iPhone) are more frequently mentioned than Google products.

### Data Preprocessing
- Text Cleaning: Raw tweets were preprocessed to remove URLs, mentions, hashtags, special characters, and extra whitespace. Text was lowercased, tokenized, and lemmatized, with stopwords removed.
- Feature Engineering: TF-IDF vectorization was applied with a maximum of 8,000 features, using unigrams and bigrams, to convert text into numerical features.
- Handling Imbalance: SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the classes and prevent model bias toward the majority class (Neutral).
- Train-Test Split: The data was split into 80% training and 20% testing sets, stratified by sentiment to maintain class proportions.

### Analytical Techniques
- Preprocessing Pipeline: A custom `TextPreprocessor` class was implemented for consistent text cleaning.
- Vectorization: TF-IDF ensured that important terms (e.g., product names) were weighted appropriately without overfitting to common words.
- Class Balancing: SMOTE addressed the imbalance, improving model performance on minority classes (Positive and Negative sentiments).
![performing-twitter-sentiment-analysis1](https://github.com/user-attachments/assets/1cba7fb2-096c-4c22-a043-8be57848a05d)


## Modeling

### Model Selection
The project evaluated multiple machine learning models for multi-class sentiment classification:
- XGBoost: Chosen as the primary model due to its robustness with sparse data, ability to handle class imbalance, and strong performance in text classification tasks.
- Logistic Regression: Used as a baseline model for comparison, with class weighting to address imbalance.

### Model Training
- Label Encoding: Sentiment classes were encoded numerically (Negative: 0, Neutral: 1, Positive: 2) for model compatibility.
- XGBoost Parameters: n_estimators=200, max_depth=8, learning_rate=0.1, eval_metric='mlogloss'.
- Training Data: Models were trained on SMOTE-resampled data to balance classes, with sample weights applied for further emphasis on minority classes.
- Pipeline: Integrated preprocessing, vectorization, and classification into a cohesive workflow.

## Model Evaluation

### Evaluation Metrics
Models were evaluated using:
- Accuracy: Overall correct predictions.
- Precision, Recall, F1-Score: Per-class and weighted averages to assess performance across imbalanced classes.
- Confusion Matrix: Visualized prediction errors and patterns.

### Results
- XGBoost Performance (after hyperparameter tuning):
  - Accuracy: ~0.70
  - Weighted F1-Score: ~0.68
  - Best performance on Neutral class (high recall), challenges with Positive/Negative distinction.
- Logistic Regression: Served as a baseline with lower performance compared to XGBoost.
- Confusion Matrix Insights: Models tended to misclassify Positive and Negative sentiments as Neutral, highlighting the difficulty in distinguishing nuanced emotions.

### Hyperparameter Tuning
- Grid Search: Optimized XGBoost parameters (n_estimators, max_depth, learning_rate, etc.) using 3-fold cross-validation.
- Scoring: Focused on macro F1-score to balance performance across classes.
- Outcome: Tuned model showed improved generalization on test data.

### Key Insights
- XGBoost outperformed baseline models, demonstrating the value of ensemble methods for text classification.
- Class imbalance remained a challenge; SMOTE and weighting helped but did not fully resolve misclassifications.
- The model provides a solid proof-of-concept for real-time sentiment analysis, with room for improvement through advanced NLP techniques (e.g., transformers).

## Conclusion

### Project Summary
This project successfully developed an NLP-based sentiment classifier using XGBoost and TF-IDF features to categorize tweets about Apple and Google products as Positive, Negative, or Neutral. The model achieved reasonable performance (~70% accuracy) on a dataset of over 9,000 tweets, demonstrating the feasibility of automated sentiment analysis for real-time public opinion tracking.

### Key Achievements
- **Scalable Solution**: Eliminated the need for manual tweet review, enabling stakeholders to monitor sentiment at scale.
- **Robust Preprocessing**: Implemented text cleaning, lemmatization, and SMOTE to handle noisy data and class imbalance.
- **Model Performance**: XGBoost provided reliable classification, with strong results on the Neutral class and acceptable performance on Positive/Negative distinctions.
- **Business Impact**: Offers actionable insights for product teams, marketing, corporate communications, and investors to respond quickly to public sentiment shifts.

### Limitations
- Class Imbalance: Neutral sentiments dominate, leading to challenges in accurately classifying Positive and Negative tweets.
- Text Nuances: Sarcasm, context, and emojis can affect sentiment detection; the model may misclassify subtle expressions.
- Data Scope: Limited to Apple and Google products; may not generalize to other domains without retraining.
- Real-Time Deployment: The proof-of-concept requires further engineering for production-scale, real-time monitoring.

### Recommendations
For Stakeholders:
  - Establish sentiment baselines and track trends over time, especially around product launches.
  - Develop brand-specific dashboards for targeted insights (e.g., iPhone vs. Android).
  - Integrate real-time monitoring to respond to emerging negative trends promptly.
Model Improvements:
  - Explore advanced models like BERT or RoBERTa for better handling of context and nuances.
  - Incorporate additional features (e.g., user metadata, temporal analysis).
  - Expand dataset with more diverse sentiments to reduce imbalance.
Future Work: Deploy the model in a production pipeline with APIs for continuous social media monitoring, enabling proactive brand management.

This project lays a strong foundation for sentiment analysis in tech industries, empowering organizations to leverage social media data for strategic decision-making.
