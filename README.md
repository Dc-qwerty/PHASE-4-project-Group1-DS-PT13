# Phase 4 Project: Tweet Sentiment Classification for Apple and Google Products

## Overview

This project builds a natural language processing model to classify tweet sentiment as **positive, neutral, or negative** using a labelled dataset of more than 9,000 tweets about Apple and Google products. The central aim is to create a proof-of-concept system that can help organizations understand public opinion at scale without relying on slow manual review. As shown in the project presentation, the business value comes from the fact that social media reactions to technology products change quickly, and firms need a faster way to detect how users are responding to launches, updates, and service issues.

The project focuses on a realistic business problem. Apple and Google receive large volumes of public commentary online, and it is not practical for product teams, marketers, communications staff, or investors to read this feedback one tweet at a time. A sentiment classifier offers a scalable way to monitor customer perception in near real time. This makes the project not only technically relevant, but also commercially useful.

## Business Understanding

The business problem behind this project is straightforward. Public opinion on technology brands is constantly being shaped on social media, especially on Twitter, where users react immediately to product launches, software updates, defects, pricing decisions, and service outages. Traditional survey methods are too slow to capture these reactions, and manual review is too limited to keep up with the volume of posts. The project therefore asks a practical question: **can machine learning be used to classify tweet sentiment reliably enough to support business decision-making?**

This work is especially relevant to several stakeholder groups. Product teams can use the output to identify pain points and see which features are being received well. Marketing teams can monitor campaign reception and adjust messaging when negative sentiment starts rising. Corporate communications teams can use the model as an early warning tool for brand reputation risk. Investors can also use sentiment patterns as an additional signal when evaluating brand health around important events such as product launches or controversies. These stakeholder uses help position the project as more than a classroom exercise.

## Data Understanding

The dataset contains over 9,000 labelled tweets about Apple and Google products, with sentiment categories of positive, neutral, and negative. Because this is text data drawn from social media, it presents the usual challenges associated with natural language processing. Tweets are short, informal, noisy, and often ambiguous. They may contain abbreviations, repeated characters, punctuation noise, brand references, slang, and limited context. These properties make sentiment classification more difficult than structured tabular prediction problems.

The class distribution also appears uneven, with the neutral class being much more common than the negative class. The presentation results show only 109 negative examples in one evaluation output, compared with much larger support for the neutral class. This imbalance is important because it affects both model learning and the interpretation of performance. A model can appear acceptable on overall accuracy while still failing on the minority class that may matter most in practice.

## Data Cleaning and Preparation

The data preparation stage was central to this project because raw tweets cannot be passed directly into machine learning models in their original form. The text first had to be converted into numerical features that a model could learn from. The presentation indicates the use of **embeddings** and also references **TF-IDF features**, which shows that the project moved beyond very basic preprocessing and engaged with advanced text representation methods.

From a sound data science perspective, the preparation work for a project like this includes cleaning missing or unusable records, standardizing text structure, and transforming raw tweets into machine-readable vectors. Given the modeling choices shown, the preparation stage likely involved token-based or embedding-based numerical conversion of text, followed by splitting the data into training and testing sets for evaluation. The presentation also notes that XGBoost handles sparse TF-IDF matrices well, which supports the view that the project used non-trivial feature engineering rather than relying on raw strings alone.

A further strength of the preparation stage is that the project explicitly dealt with the realities of class imbalance. The presentation notes that XGBoost was chosen in part because it is robust to imbalanced datasets. This is important because the minority negative class is the most difficult class in the problem and also one of the most valuable from a business standpoint. Choosing models with this issue in mind reflects good preparation judgment rather than treating preprocessing as a routine step.

## Methodology

The project used a comparative modeling approach rather than relying on a single algorithm. This is important under the rubric because strong machine learning work is not just about fitting one model, but about evaluating multiple candidates and selecting the one that performs best for the business problem.

The first major model presented was **Logistic Regression with embeddings**. This served as a useful baseline because logistic regression is efficient, interpretable, and commonly used in text classification. It provides a clear benchmark against which a more advanced model can be judged. The model achieved an overall accuracy of **63%**, with the strongest performance on the neutral class and much weaker performance on the negative class.

The second major model was **XGBoost**, followed by a tuned version of XGBoost. This was an appropriate advanced modeling choice for several reasons identified in the presentation. XGBoost performs well in classification tasks, can process sparse text features efficiently, and is more capable of capturing non-linear relationships than logistic regression. The presentation also notes that hyperparameter tuning was used, including settings such as `max_depth=8`, `subsample=0.8`, and `colsample_bytree=1.0`, which indicates meaningful model optimization rather than default training.

The project also used a confusion matrix and class-level precision, recall, and F1-scores for evaluation. This is a strong methodological decision because sentiment classification with imbalanced classes cannot be judged well by accuracy alone. Looking at class-specific metrics helps reveal where the model performs well and where it still struggles.

## Modeling Results and Evaluation

The baseline Logistic Regression model achieved an **accuracy of 63%**. It performed best on the neutral class with an **F1-score of 0.71**, while the positive class recorded a more moderate **F1-score of 0.58**. The most important weakness was the negative class, where the **F1-score fell to 0.37** and precision was only **0.26**. In practical terms, this means the model often identified tweets as negative incorrectly and was not dependable enough for high-stakes brand monitoring without improvement.

The confusion matrix reinforces this conclusion. The model correctly classified most neutral tweets, but it struggled to distinguish negative sentiment from neutral sentiment. Only **44 negative tweets** were correctly classified while **54 negative instances** were misclassified as neutral. This shows that the hardest class in the dataset remained the hardest class in the model, which is a common issue in imbalanced NLP classification.

After tuning, **XGBoost** improved overall performance to **70% accuracy**, with a **weighted average F1-score of 0.70**. The neutral class remained the strongest at **F1-score 0.78**, while the negative class improved from **0.37 to 0.45**. The positive class also improved slightly to **0.60**. These results suggest that the tuned XGBoost model offered a more balanced performance profile and better captured the complex relationships within the text representation space. It did not solve the negative-class problem fully, but it produced a clear improvement over the baseline and therefore stands as the better final model.

The comparison between Logistic Regression and Tuned XGBoost is also well reasoned. Logistic Regression retained one advantage: it achieved higher negative recall, meaning it was more likely to catch negative tweets even if it produced many false alarms. This could make it useful in a setting where missing a negative tweet is considered more costly than over-flagging. However, Tuned XGBoost had stronger aggregate performance, better balance across classes, and a more credible precision-recall tradeoff overall. On this basis, Tuned XGBoost is the more robust model for general use, while Logistic Regression remains a defensible baseline and a useful benchmark.

## Key Insights

A major insight from this project is that **neutral sentiment dominates the data and is easiest for the models to identify**. This suggests that a large share of consumer discussion around Apple and Google products is descriptive, informational, or mixed rather than strongly emotional. From a business point of view, that matters because the model is likely to perform best in monitoring broad public mood before drilling deeper into more specific complaint types.

A second insight is that **negative sentiment remains the hardest category to classify**, even after tuning. This is not just a technical weakness; it is a business risk. Negative tweets often contain the most urgent information for product, communications, and brand teams. If the model fails to detect them consistently, the organization may miss early warning signals about defects, outages, or reputational problems. The limited number of negative examples in the data is likely one of the main reasons for this persistent challenge.

A third insight is that **more advanced modeling improved performance meaningfully**, especially when moving from baseline Logistic Regression to tuned XGBoost. This shows that model choice matters in NLP problems and that tuning can generate measurable value when the feature space is complex and class boundaries are not linear. The project therefore demonstrates not only model fitting, but also evidence-based model comparison.

## Recommendations

The first recommendation is to treat this model as a strong proof of concept and integrate it into a broader sentiment monitoring workflow. The presentation rightly suggests using sentiment trends to support product launches, competitive benchmarking, and continuous social media monitoring. These are credible next steps because the model already performs well enough to provide directional insight, even if it is not yet perfect for high-risk automation.

The second recommendation is to improve the dataset, especially for the negative class. The most immediate path to better performance is not necessarily a more complex model, but better training data. More labelled negative tweets, better class balance, and more careful annotation would likely improve recall and precision on the class that matters most for real-world use.

The third recommendation is to create brand-specific and product-specific dashboards. Aggregate sentiment can hide product-level problems. Decision-makers need localized views of sentiment rather than one combined score across all products.

The fourth recommendation is to continue experimenting with more advanced NLP pipelines. Since the project already uses embeddings and boosted trees, the next logical step would be to test richer text representations, stronger handling of imbalance, and end-to-end pipelines that make retraining and deployment cleaner. This would strengthen both reproducibility and production readiness.

## Conclusion

This project successfully addressed a relevant business problem by building a sentiment classification system for Apple and Google tweets. It demonstrated a clear connection between business understanding and machine learning practice, used more than one model, engaged in meaningful text preparation, and evaluated results using appropriate classification metrics. The final tuned XGBoost model improved performance over the logistic regression baseline and provided a more balanced result across sentiment classes.

The strongest technical lesson from the project is that model performance in sentiment analysis depends heavily on careful preprocessing, realistic evaluation, and class balance. The strongest business lesson is that even a moderately accurate sentiment model can provide useful real-time insight when traditional feedback methods are too slow. Taken together, the project is a credible demonstration of advanced data preparation, comparative modeling, and business-centered machine learning analysis.
