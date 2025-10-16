# AI Essay Rater

> **Automated Essay Scoring using Machine Learning and Transformer Models**

A comprehensive comparative analysis leveraging Natural Language Processing to provide accurate, consistent, and scalable evaluation of student writing, progressing from classical regression baselines to a state-of-the-art BERT-based architecture.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance Results](#performance-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [License](#license)

---

## 🎯 Project Overview

The manual grading of student essays is time-consuming, costly, and susceptible to inter-rater variability and subjectivity. This project addresses these challenges through Automated Essay Scoring (AES) systems that employ machine learning and natural language processing.

This project undertakes a systematic exploration of AES by:
- Developing and training models with increasing complexity
- Comparing traditional machine learning baselines with state-of-the-art transformers
- Providing quantifiable performance metrics across different approaches

---

## 🛠️ Technology Stack

This project is built upon a robust, modern, and open-source technology stack:

- **Deep Learning:** TensorFlow, Transformers (Hugging Face)
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Lab/Notebook

---

## 📊 Dataset

### ASAP-AES Corpus

The project uses the Hewlett Foundation's **Automated Student Assessment Prize (ASAP)** dataset (`training_set_rel3.tsv`), a widely recognized benchmark for AES research.

**Key Characteristics:**
- **Student Population:** Essays from Grade 7 to Grade 10
- **Essay Prompts:** 8 distinct essay sets with unique writing prompts
- **Structural Diversity:** Essays range from 150 to 550 words on average
- **Reliable Scoring:** Each essay scored by at least two human raters
- **Privacy:** Pre-anonymized with PII replaced by generic placeholders

### Data Preprocessing Pipeline

1. **Text Cleaning:** Lowercase conversion and whitespace normalization
2. **PII Anonymization:** Already handled in the dataset
3. **Tokenization:** Word-level for baselines, subword (WordPiece) for BERT
4. **Noise Reduction:** Stop word and punctuation removal for classical models
5. **Lemmatization:** Reducing words to their base forms

---

## 🧠 Model Architecture

### Baseline Models

#### Ridge Regression
- Linear model with L2 regularization
- Features: TF-IDF vectors
- Purpose: Computational efficiency and robust linear baseline

#### Random Forest Regression
- Ensemble of decision trees
- Captures non-linear relationships
- Features: TF-IDF vectors

### Advanced Model: BERT Transformer

**Architecture:**
- Base: Pre-trained BERT (trained on Wikipedia and BookCorpus)
- Custom: Regression head (dense neural network layer)
- Training: Fine-tuned on ASAP-AES dataset

**Key Advantages:**
- Dynamic, contextual embeddings
- Bidirectional context understanding
- Captures syntax, semantics, and nuanced text relationships

---

## 📈 Performance Results

### Comparative Model Performance

| Model | Quadratic Weighted Kappa | Pearson r | MAE | RMSE |
|-------|-------------------------|-----------|-----|------|
| **Ridge Regression** | 0.9240 | 0.7327 | 0.1262 | 0.1629 |
| **Random Forest** | 0.9336 | 0.8021 | 0.1115 | 0.1476 |
| **BERT-based Model** | 0.9867 | 0.8345 | 0.1013 | 0.1356 |

### Evaluation Metrics

- **Pearson Correlation (r):** Linear relationship strength between predicted and actual scores
- **Mean Absolute Error (MAE):** Average magnitude of prediction errors
- **Root Mean Squared Error (RMSE):** Sensitivity to large errors
- **Quadratic Weighted Kappa (QWK):** Primary AES metric; measures agreement on ordinal scale with penalty for large deviations

### Key Observations

- Strong positive linear relationship in BERT predictions
- Slight heteroscedasticity: tendency to regress toward the mean
- Model under-predicts high scores and over-predicts low scores
- Smoother prediction distribution compared to actual score distribution

---

## 📁 Repository Structure

```
AI_ESSAY_GRADER/
├── essay_scoring_model/       # Saved BERT model files
├── .gitattributes             # Git attributes configuration
├── .gitignore                 # Git ignore rules
├── 01_Ridge_Regression.ipynb  # Ridge Regression baseline
├── 02_Random_Forest.ipynb     # Random Forest baseline
├── 03_BERT_vs_Regression.ipynb # BERT model development & comparison
├── all_comparison.png         # Comparative performance plots
├── essays.txt                 # Sample essays for testing
├── LICENSE                    # MIT License
├── model_eval_metrics.csv     # Performance metrics for all models
├── README.md                  # Project documentation
├── rf_vals.png                # Random Forest visualizations
├── ridge_vals.png             # Ridge Regression visualizations
├── test.ipynb                 # Model inference notebook
└── training_set_rel3.tsv      # ASAP-AES dataset
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package installer
- (Optional) NVIDIA GPU with CUDA and cuDNN for accelerated BERT training

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RawEgg6/AI-essay-rater.git
   ```

2. **Navigate to the project directory:**
   ```bash
   cd AI-essay-rater
   ```

3. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install tensorflow transformers scikit-learn pandas numpy matplotlib seaborn jupyterlab
   ```

### Running the Notebooks

1. **Launch Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Execute notebooks in order:**
   - `Ridge_Regression.ipynb` - Ridge Regression baseline
   - `Random_Forest.ipynb` - Random Forest baseline
   - `main.ipynb` - BERT model fine-tuning and evaluation

---

## 🔮 Future Work

### Potential Enhancements

1. **Advanced Transformer Architectures**
   - Experiment with RoBERTa, DeBERTa, or Longformer
   - Better handling of long essay sequences

2. **Prompt-Specific Modeling**
   - Train separate models for each of the 8 essay sets
   - Incorporate essay_set ID as a categorical feature

3. **Error Analysis & Bias Mitigation**
   - Address systematic bias in residual patterns
   - Implement targeted data augmentation
   - Design custom loss functions to reduce mean regression

4. **Enhanced Feature Engineering**
   - Add readability scores (Flesch-Kincaid)
   - Include syntactic complexity metrics
   - Incorporate essay length and grammatical error counts

5. **Web Application Deployment**
   - Package model as REST API (Flask/FastAPI)
   - Create user-friendly interface for instant essay scoring

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Hewlett Foundation for the ASAP-AES dataset
- Hugging Face for the Transformers library
- The open-source community for TensorFlow and scikit-learn
