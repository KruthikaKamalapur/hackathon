# Detecting Toxic or Harmful Comments

## Model Architecture and Methodology
This project employs **BERT (Bidirectional Encoder Representations from Transformers)** for classifying text as toxic or non-toxic. The model is fine-tuned on a labeled dataset of comments. The pipeline consists of:
- **Data Preprocessing**: Removing null values, encoding labels, and tokenizing text using `BertTokenizer`.
- **Model Training**: Fine-tuning `BertForSequenceClassification` from the `transformers` library.
- **Evaluation**: Using classification metrics such as precision, recall, F1-score, and AUC-ROC.
- **Text Rewriting**: A `BART` model generates alternative, non-toxic phrasing for detected toxic comments.
- **Visualization**: Confusion matrix, precision-recall curve, and ROC curve.

## Why BERT for this Task?
BERT is chosen because:
- It captures **contextual meaning** in text, making it ideal for NLP tasks.
- Pre-trained transformers significantly **improve classification accuracy**.
- Fine-tuning BERT requires minimal labeled data compared to training from scratch.

Additionally, **BART** is used for text rewriting due to its capability in sequence-to-sequence transformations.

## Evaluation Metrics
- **Micro Precision**: 0.9347
- **Recall**: 0.9347
- **F1 Score**: 0.9347
- **AUC-ROC**: 0.6981
- **False Positive Rate (FPR)**: 0.0068
- **False Negative Rate (FNR)**: 0.5969

## Challenges and Solutions
- **Handling Long Texts**: Limited input size (512 tokens) in BERT. Used truncation while preserving meaningful parts of the text.
- **Class Imbalance**: Applied balanced sampling and weighted loss functions to improve model fairness.
- **Mitigating False Negatives**: Adjusted decision thresholds and optimized the loss function to reduce under-detection of toxic content.
- **Ensuring Context-Aware Rewriting**: Used `BART` paraphrasing to generate meaningful and relevant rewrites.

## Required Dependencies
```bash
pip install torch torchvision torchaudio transformers scikit-learn pandas numpy matplotlib seaborn
```

## How to Run
1. **Upload Dataset**: First, upload the dataset to Kaggle and name it `new-hsioa`.
2. **Execute Notebook**: Run the Jupyter Notebook (`.ipynb` file) in Kaggle.
3. **Training and Evaluation**: The model will be trained, evaluated, and results will be visualized.
4. **Toxic Comment Rewriting**: The `BART` model will suggest alternative phrasings for toxic comments.

## Visualization
- **Confusion Matrix**: Visual representation of classification performance.
- **Precision-Recall Curve**: Displays precision-recall trade-offs.
- **ROC Curve**: Illustrates classifier performance across various thresholds.

All visualizations are generated within the notebook (`.ipynb`) after evaluation.

