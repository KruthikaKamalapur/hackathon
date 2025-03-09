# Detecting Misogynistic Language in Online Conversations

## Model Architecture and Methodology
This project employs **RoBERTa (Robustly Optimized BERT Pretraining Approach)** for detecting misogynistic language while considering context. The pipeline consists of:
- **Data Preprocessing**: Removing null values, encoding labels, and tokenizing text using `RobertaTokenizer`.
- **Model Training**: Fine-tuning `RobertaModel` for classification.
- **Evaluation**: Using classification metrics such as accuracy, precision, recall, and F1-score.
- **Text Highlighting and Rewriting**: 
  - **Detection**: The model flags misogynistic language.
  - **Highlighting**: Problematic words or phrases are marked for transparency.
  - **Rewriting**: A `BART` model generates alternative, non-toxic phrasing.
- **Visualization**: Confusion matrix and classification reports for performance analysis.

## Why RoBERTa for this Task?
RoBERTa is chosen because:
- It **captures nuanced context** in online conversations.
- Pre-trained transformers significantly **improve classification accuracy**.
- Fine-tuning RoBERTa allows for **better generalization** on misogyny detection.

Additionally, **BART** is used for text rewriting due to its strong **sequence-to-sequence** capabilities.

## Evaluation Metrics
- **Accuracy**: 0.9393
- **Precision**: 0.9342
- **Recall**: 0.9393
- **F1 Score**: 0.9328
## Challenges and Solutions
- **Distinguishing Between Jokes and Harmful Intent**: Leveraged context-aware training to minimize false positives.
- **Handling Long Texts**: Used truncation while preserving meaningful parts of the text.
- **Class Imbalance**: Applied balanced sampling and weighted loss functions to improve model fairness.
- **Ensuring Context-Aware Rewriting**: Used `BART` paraphrasing to generate meaningful and relevant rewrites.

## Required Dependencies
```bash
pip install torch torchvision torchaudio transformers scikit-learn pandas numpy matplotlib seaborn
```

## How to Run
1. **Upload Dataset**: First, upload the dataset to Kaggle and name it `new-hsioa`.
2. **Execute Notebook**: Run the Jupyter Notebook (`.ipynb` file) in Kaggle.
3. **Training and Evaluation**: The model will be trained, evaluated, and results will be visualized.
4. **Misogynistic Comment Detection & Rewriting**: The `BART` model will suggest alternative phrasings for detected misogynistic content.

## Visualization
- **Confusion Matrix**: Displays correct vs incorrect predictions.
All visualizations are generated within the notebook (`.ipynb`) after evaluation.

