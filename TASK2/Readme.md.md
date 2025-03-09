# Subreddit-Based Topic Classification

## Model Architecture and Methodology
This project employs **BERT (Bidirectional Encoder Representations from Transformers)** with **Word2Vec embeddings** for classifying text into subreddit-based topics. The model is fine-tuned on a labeled dataset of subreddit comments. The pipeline consists of:
- **Data Preprocessing**: Removing null values, encoding labels, tokenizing text using `BertTokenizer`, and training a Word2Vec embedding model.
- **Model Training**: Fine-tuning `BertForSequenceClassification` with concatenated Word2Vec embeddings.
- **Evaluation**: Using classification metrics such as precision, recall, F1-score, and confusion matrix analysis.
- **Visualization**: Confusion matrix and F1-score distribution across classes.

## Why BERT and Word2Vec?
- **BERT** captures contextual meaning, making it ideal for text classification tasks.
- **Word2Vec** enhances feature representation by capturing semantic relationships between words.
- The combination improves classification accuracy and robustness.

## Evaluation Metrics
- **Macro Precision**: 0.9268
- **Macro Recall**: 0.9193
- **Macro F1 Score**: 0.9201

## Challenges and Solutions
- **Handling Long Texts**: Used truncation while preserving meaningful parts of the text.
- **Class Imbalance**: Applied oversampling techniques to balance dataset distribution.
- **Feature Fusion**: Combined BERT embeddings with Word2Vec for richer text representation.
- **Optimizing Training**: Utilized weighted loss functions to mitigate misclassification.

## Required Dependencies
```bash
pip install pandas numpy torch transformers scikit-learn imbalanced-learn gensim
```

## How to Run
1. **Upload Dataset**: First, upload the dataset to Kaggle and name it `new-hsioa`.
2. **Execute Notebook**: Run the Jupyter Notebook (`.ipynb` file) in Kaggle.
3. **Training and Evaluation**: The model will be trained, evaluated, and results will be visualized.
4. **Visualization**: The confusion matrix and F1-score distribution will be displayed.

## Visualization
- **Confusion Matrix**: Visual representation of classification performance.
- **F1-Score Distribution**: Bar chart showing F1-scores across subreddit categories.

All visualizations are generated within the notebook (`.ipynb`) after evaluation.