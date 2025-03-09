# Parent-Child Conversation Reconstruction

## Model Architecture and Methodology

1. **Data Preprocessing**: 
   - Load conversation data from `final_labels.csv`.
   - Resolve duplicate entries based on strength.
   - Remove missing values and clean text by filtering out non-ASCII characters.

2. **Conversation Thread Reconstruction**:
   - Construct parent-child relationships from `entry_id` and `parent_id`.
   - Sort replies chronologically and rebuild full conversation threads.
   - Store reconstructed conversations in `reconstructed_threads.json`.

3. **Text Summarization**:
   - Use `facebook/bart-large-cnn` for conversation summarization.
   - Filter conversations with sufficient length before summarization.
   - Store summaries in `conversation_summaries.json`.

4. **Evaluation Metrics**:
   - **BLEU Score**: Measures word overlap between summary and original conversation.
   - **ROUGE Score**: Compares n-grams and sentence structure.
   - **Perplexity**: Evaluates fluency using GPT-2.
   - **Semantic Similarity**: Measures vector similarity using `all-MiniLM-L6-v2`.
   - Store results in `evaluation_results.json`.

5. **Result Aggregation**:
   - Compute average values for all metrics.
   - Store aggregated results in `summary_matrix.json`.

## Justification for Model Selection
- **BART for Summarization**: Pretrained on document-level summarization tasks, ensuring high-quality summaries.
- **GPT-2 for Perplexity**: Provides a reliable measure of summary fluency.
- **SentenceTransformer for Similarity**: Captures semantic consistency effectively.

## Evaluation Metrics
- **BLEU**: Average BLEU score across summaries.
- **ROUGE-1, ROUGE-2, ROUGE-L**: Assess summary recall and precision.
- **Perplexity**: Measures model fluency.
- **Semantic Similarity**: Evaluates meaning retention.

## Challenges and Solutions
### 1. Handling Incomplete Conversations
   **Problem**: Missing parent or child comments disrupt flow.
   **Solution**: Assign orphaned comments to independent threads.

### 2. Summarization Length Constraints
   **Problem**: Some summaries were too short to be meaningful.
   **Solution**: Enforced minimum summary length during generation.

### 3. Computational Constraints
   **Problem**: Processing long conversations was slow.
   **Solution**: Used batched summarization and optimized memory usage.

## Required Dependencies
```bash
pip install transformers rouge-score nltk sentence-transformers torch pandas numpy
```

## How to Run
1. **Upload Dataset**:
   - Place `final_labels.csv` in the collab.

2. **Execute Script**:
   run the cells
   
3. **Download Results**:
   - Outputs are stored as JSON files:
     - `reconstructed_threads.json`
     - `conversation_summaries.json`
     - `evaluation_results.json`
     - `summary_matrix.json`

## Visualization Details
- Generate histograms and bar plots for evaluation metrics.
- Display confusion matrices for summary quality assessment.

