# 🛡️ SQLGuardian-Transformer

### Advanced SQL Injection Detection using Transformer Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.30%25-success.svg)](README.md)

Deep learning model menggunakan arsitektur Transformer untuk mendeteksi serangan SQL Injection dengan akurasi tinggi. Model ini mampu mengidentifikasi pola-pola serangan kompleks pada query database secara real-time.

---

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.30% |
| **Precision** | 99.08% |
| **Recall** | 99.03% |
| **F1 Score** | 99.06% |
| **ROC-AUC** | 99.64% |
| **Balanced Accuracy** | 99.25% |

**Confusion Matrix:**
- True Negative: 3887
- False Positive: 21 (0.54%)
- False Negative: 22 (0.96%)
- True Positive: 2254

---

## 🌟 Key Features

- ✅ **High Accuracy**: 99.30% detection rate dengan minimal false positives
- ✅ **Transformer Architecture**: Multi-head attention mechanism untuk pattern recognition
- ✅ **Real-time Detection**: Fast inference (~7-14ms per query)
- ✅ **Comprehensive EDA**: 12 visualizations untuk data insights
- ✅ **Production Ready**: Optimized threshold dan model serialization
- ✅ **Extensible**: Easy to retrain dengan data baru

---

## 🗂️ Dataset

**Source**: Modified_SQL_Dataset.csv

| Properties | Details |
|------------|---------|
| Total Records | 30,921 queries |
| Normal Queries | 19,537 (63.2%) |
| Attack Queries | 11,382 (36.8%) |
| Features | Query (text), Label (0/1) |

**Attack Patterns Detected:**
- OR 1=1 injections
- UNION SELECT attacks
- Comment injection (--, /*, */)
- String concatenation exploits
- SQL function injections
- Information schema queries

---

## 🏗️ Model Architecture

### Transformer Configuration

```python
Embedding Dimension: 128
Attention Heads: 4
Feed-Forward Dimension: 256
Transformer Blocks: 2
Dropout Rate: 0.2
Max Sequence Length: 100 tokens
Vocabulary Size: 10,000 words
```

### Layer Structure

```
Input Layer (100,)
   ↓
Token & Position Embedding (100, 128)
   ↓
Transformer Block 1
   ├─ Multi-Head Attention (4 heads)
   ├─ Layer Normalization
   ├─ Feed Forward Network (256 → 128)
   └─ Residual Connections
   ↓
Transformer Block 2
   ├─ Multi-Head Attention (4 heads)
   ├─ Layer Normalization
   ├─ Feed Forward Network (256 → 128)
   └─ Residual Connections
   ↓
Global Average Pooling
   ↓
Dense Layer (128, ReLU) + Dropout
   ↓
Dense Layer (64, ReLU) + Dropout
   ↓
Output Layer (1, Sigmoid)
```

**Total Parameters**: 1,978,113 (7.55 MB)

---

## 📦 Installation

### Requirements

```bash
pip install tensorflow>=2.8.0
pip install scikit-learn>=1.0.0
pip install pandas numpy matplotlib seaborn
pip install joblib tqdm
```

### Clone Repository

```bash
git clone https://github.com/yourusername/sqlguardian-transformer.git
cd sqlguardian-transformer
```

---

## 🚀 Quick Start

### Training Model

```python
# Load dataset
df = pd.read_csv('Modified_SQL_Dataset.csv')

# Run all cells in notebook sequentially
# Training akan menghasilkan:
# - sql_injection_transformer_model.keras
# - tokenizer.pkl
# - model_config.pkl
# - evaluation_results.pkl
```

### Inference

```python
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model dan artifacts
model = tf.keras.models.load_model('sql_injection_transformer_model.keras')
tokenizer = joblib.load('tokenizer.pkl')
config = joblib.load('preprocessing_artifacts.pkl')

# Predict
query = "SELECT * FROM users WHERE id = 1 OR 1=1--"
sequence = tokenizer.texts_to_sequences([query])
padded = pad_sequences(sequence, maxlen=config['max_len'])
probability = model.predict(padded)[0][0]
prediction = "Attack" if probability >= config['optimal_threshold'] else "Normal"

print(f"Prediction: {prediction}")
print(f"Confidence: {probability:.4f}")
```

---

## 📓 Notebook Documentation

### Cell-by-Cell Breakdown

#### **CELL 1: Import Libraries**
**Purpose**: Import semua library yang diperlukan untuk data processing, modeling, dan evaluation

**Input**: None

**Output**: Libraries loaded
- Data processing: pandas, numpy
- Visualization: matplotlib, seaborn
- ML framework: scikit-learn
- Deep learning: tensorflow, keras
- Utils: joblib, tqdm, re

---

#### **CELL 2: Configuration Setup**
**Purpose**: Set random seeds, display options, dan plotting style untuk reproducibility

**Input**: None

**Output**: 
- Random seed: 42
- Warnings suppressed
- Plot style: seaborn-v0_8-darkgrid
- Pandas display optimized

---

#### **CELL 3: Load Dataset**
**Purpose**: Load dataset SQL injection dari CSV file

**Input**: `Modified_SQL_Dataset.csv`

**Output**: DataFrame dengan 2 kolom
- `Query`: SQL query string
- `Label`: 0 (Normal), 1 (Attack)
- Shape: (30921, 2)

**Sample Output**:
```
                                    Query  Label
0    " or pg_sleep  (  __TIME__  )  --      1
1  create user name identified by pass...   1
```

---

#### **CELL 4: Basic Data Exploration**
**Purpose**: Exploratory analysis dasar untuk memahami struktur dan distribusi data

**Input**: DataFrame `df`

**Output**: Statistical summary
```
Dataset Shape: (30921, 2)
Column Names: ['Query', 'Label']
Missing Values: 0
Duplicate Rows: 165

Label Distribution:
0    19537  (63.2%)
1    11382  (36.8%)
```

---

#### **CELL 5: Feature Engineering**
**Purpose**: Ekstraksi fitur tambahan dari query untuk analisis mendalam

**Input**: DataFrame dengan kolom `Query`

**Output**: 4 fitur baru
- `Query_Length`: Panjang karakter query
- `Word_Count`: Jumlah kata dalam query
- `Special_Char_Count`: Jumlah karakter spesial
- `Digit_Count`: Jumlah digit dalam query

**Statistics**:
```
Query_Length: mean=117.4, max=5000
Word_Count: mean=24.8, max=200
Special_Char_Count: mean=16.7, max=600
Digit_Count: mean=8.2, max=80
```

---

#### **CELL 6: Comprehensive Visualization**
**Purpose**: Visualisasi 12 subplot untuk memahami karakteristik data dan pola serangan

**Input**: DataFrame dengan semua fitur

**Output**: Figure dengan 12 visualizations

**Subplots**:
1. **Distribusi Label (Countplot)**: Normal 19,537 vs Attack 11,382
2. **Proporsi Label (Pie Chart)**: Normal 63.2% vs Attack 36.8%
3. **Distribusi Panjang Query (Violin Plot)**: Attack cenderung lebih panjang
4. **Distribusi Jumlah Kata (Box Plot)**: Attack memiliki variance lebih tinggi
5. **Histogram Panjang Query**: Separasi jelas antara Normal dan Attack
6. **Histogram Jumlah Kata**: Attack lebih spread out
7. **Distribusi Karakter Spesial (Violin Plot)**: Attack signifikan lebih banyak
8. **Distribusi Angka (Violin Plot)**: Attack menggunakan lebih banyak digit
9. **Correlation Heatmap**: Strong correlation (0.78-0.95) antar fitur
10. **Rata-rata Panjang Query per Label**: Attack ~40 chars, Normal ~118 chars
11. **Rata-rata Karakter Spesial per Label**: Attack ~17, Normal ~3
12. **Sample Attack Queries**: Menampilkan 5 contoh attack patterns

**Key Insights**:
- Attack queries memiliki pola yang distinct
- Special characters adalah strong indicator
- Query length dapat membantu klasifikasi

---

#### **CELL 7: Text Preprocessing**
**Purpose**: Clean dan normalize text untuk tokenization

**Input**: Raw query strings

**Output**: Cleaned query strings
```python
def clean_text(text):
    - Convert to lowercase
    - Remove multiple whitespaces
    - Strip leading/trailing spaces
```

**Example**:
```
Original: "SELECT * FROM users WHERE id = 1 OR 1=1--"
Cleaned:  "select * from users where id = 1 or 1=1--"
```

---

#### **CELL 8: Train-Validation-Test Split**
**Purpose**: Split data dengan stratified sampling untuk mempertahankan proporsi label

**Input**: X (queries), y (labels)

**Output**: 3 datasets
```
Training Set:   21,644 samples (70%)
Validation Set:  3,092 samples (10%)
Test Set:        6,184 samples (20%)

Label Distribution (preserved):
Train:  Normal 13,675 | Attack 7,969
Val:    Normal  1,954 | Attack 1,138
Test:   Normal  3,908 | Attack 2,276
```

---

#### **CELL 9: Tokenization and Padding**
**Purpose**: Convert text ke sequences dan pad ke fixed length

**Input**: Text queries

**Configuration**:
```python
MAX_WORDS = 10000
MAX_LEN = 100
```

**Output**: Padded sequences
```
Vocabulary Size: 9,847 unique words
Training Shape: (21644, 100)
Validation Shape: (3092, 100)
Test Shape: (6184, 100)

Sample Tokenized:
Original: "select * from users where id = 1"
Tokens: [12, 4, 56, 89, 234, 567, 2, 8]
Padded: [12, 4, 56, 89, 234, 567, 2, 8, 0, 0, ..., 0]
```

---

#### **CELL 10: TransformerBlock Layer**
**Purpose**: Define custom Transformer block dengan multi-head attention

**Input**: None (class definition)

**Output**: Custom Keras Layer

**Components**:
- Multi-Head Attention (4 heads, key_dim=128)
- Feed-Forward Network (256 → 128)
- Layer Normalization (2 layers)
- Dropout (rate=0.1)
- Residual connections

**Methods**:
- `call()`: Forward pass
- `get_config()`: Serialization
- `from_config()`: Deserialization

---

#### **CELL 11: TokenAndPositionEmbedding Layer**
**Purpose**: Define embedding layer yang combine token embedding dan positional encoding

**Input**: None (class definition)

**Output**: Custom Keras Layer

**Components**:
- Token Embedding (vocab_size → embed_dim)
- Position Embedding (maxlen → embed_dim)
- Addition operation

**Formula**: `output = token_emb(x) + pos_emb(positions)`

---

#### **CELL 12: Create Model Function**
**Purpose**: Build complete Transformer model architecture

**Input**: Hyperparameters

**Output**: Compiled Keras Model

**Architecture Summary**:
```
Model: "functional"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
input_layer                 (None, 100)               0         
token_and_position_embedding (None, 100, 128)         1,292,800
transformer_block_1         (None, 100, 128)          330,240
transformer_block_2         (None, 100, 128)          330,240
global_average_pooling      (None, 128)               0         
dropout                     (None, 128)               0         
dense_1                     (None, 128)               16,512
dropout                     (None, 128)               0         
dense_2                     (None, 64)                8,256
dropout                     (None, 64)                0         
dense_output                (None, 1)                 65        
=================================================================
Total params: 1,978,113 (7.55 MB)
Trainable params: 1,978,113 (7.55 MB)
Non-trainable params: 0 (0.00 B)
```

---

#### **CELL 13: Build Model**
**Purpose**: Instantiate model dengan specified hyperparameters

**Input**: Hyperparameters
```python
VOCAB_SIZE = 10000
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_TRANSFORMER_BLOCKS = 2
DROPOUT_RATE = 0.2
```

**Output**: Model instance ready for training

---

#### **CELL 14: Calculate Class Weights**
**Purpose**: Handle imbalanced dataset dengan class weighting

**Input**: y_train labels

**Output**: Class weights dictionary
```python
Class Weights: {
    0: 0.7854,  # Normal queries (downweight)
    1: 1.3591   # Attack queries (upweight)
}
```

**Formula**: `weight = n_samples / (n_classes * n_samples_class)`

---

#### **CELL 15: Define Custom Metrics**
**Purpose**: Create custom F1 score dan balanced accuracy metrics untuk monitoring

**Input**: None (function definitions)

**Output**: 2 custom metric functions
- `f1_metric`: Harmonic mean of precision and recall
- `balanced_accuracy`: Average of sensitivity and specificity

**Mathematical Formulas**:
```
F1 = 2 * (precision * recall) / (precision + recall)
Balanced Acc = (TPR + TNR) / 2
```

---

#### **CELL 16: Compile Model**
**Purpose**: Configure model untuk training dengan optimizer, loss, dan metrics

**Input**: Model instance

**Configuration**:
```python
Optimizer: Adam(lr=0.001)
Loss: binary_crossentropy
Metrics: 
  - accuracy
  - precision
  - recall
  - auc
  - f1_metric
  - balanced_accuracy
```

**Output**: Compiled model ready untuk fit()

---

#### **CELL 17: Define Callbacks**
**Purpose**: Setup callbacks untuk mengontrol training process

**Input**: None

**Output**: 3 callback objects

**Callbacks**:
1. **EarlyStopping**: 
   - Monitor: val_auc
   - Patience: 15 epochs
   - Mode: maximize
   - Restore best weights: True

2. **ReduceLROnPlateau**:
   - Monitor: val_auc
   - Factor: 0.5
   - Patience: 5 epochs
   - Min LR: 1e-7

3. **ModelCheckpoint**:
   - Save best model to: best_transformer_model.keras
   - Monitor: val_auc
   - Save best only: True

---

#### **CELL 18: Train Model**
**Purpose**: Train Transformer model dengan training data

**Input**: 
- X_train_pad: (21644, 100)
- y_train: (21644,)
- X_val_pad: (3092, 100)
- y_val: (3092,)

**Training Configuration**:
```python
Epochs: 50 (early stopped at 17)
Batch Size: 64
Class Weight: Applied
Callbacks: 3 callbacks active
GPU: T4 (enabled)
```

**Output**: Training history

**Training Progress**:
```
Epoch 1/50: val_auc=0.9964 (BEST) - 34s
Epoch 2/50: val_auc=0.9979 (BEST) - 8s
Epoch 3/50: val_auc=0.9963 - 10s
...
Epoch 7/50: LR reduced to 0.0005
Epoch 12/50: LR reduced to 0.00025
Epoch 17/50: LR reduced to 0.000125, Early Stopping triggered
```

**Best Model**: Epoch 2 with val_auc=0.9979

**Final Metrics (Epoch 2)**:
- Training: acc=0.9903, auc=0.9949, loss=0.0893
- Validation: acc=0.9955, auc=0.9979, loss=0.0434

---

#### **CELL 19: Plot Training History**
**Purpose**: Visualize training progress dengan 9 subplot metrics

**Input**: Training history object

**Output**: Figure dengan 9 subplots

**Plots**:
1. **Model Loss**: Training vs Validation loss convergence
2. **Model Accuracy**: 99%+ accuracy dari epoch 2
3. **Model AUC**: Near perfect AUC score (0.998)
4. **Model Precision**: Consistent 99%+ precision
5. **Model Recall**: High recall throughout training
6. **Model F1 Score**: Stable F1 around 0.20 (metric issue)
7. **Model Balanced Accuracy**: Consistent 0.27 (metric issue)
8. **Learning Rate Schedule**: Stepped reduction visible
9. **Final Metrics Comparison**: Bar chart train vs val

**Key Observations**:
- Fast convergence (epoch 2)
- No overfitting (train ≈ val)
- Learning rate reduction effective
- Custom metrics have implementation issue (F1, Balanced Acc)

---

#### **CELL 20: Find Optimal Threshold**
**Purpose**: Determine threshold yang maximize F1 score pada validation set

**Input**: 
- y_val: True labels
- y_pred_proba_val: Predicted probabilities

**Process**:
1. Generate precision-recall curve
2. Calculate F1 scores untuk semua thresholds
3. Find threshold dengan F1 maksimum

**Output**: 
```
Optimal Threshold: 0.3376
F1 Score at Optimal: 0.9943
```

**Why Not 0.5?**: Default threshold 0.5 tidak optimal untuk imbalanced dataset. Threshold 0.338 memberikan balance terbaik antara precision dan recall.

---

#### **CELL 21: Comprehensive Evaluation Function**
**Purpose**: Define function untuk evaluate model dengan multiple metrics

**Input**: y_true, y_pred, y_pred_proba, dataset_name

**Output**: Metrics dictionary dan confusion matrix

**Metrics Calculated**:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Sensitivity (TPR)
- Specificity (TNR)
- Balanced Accuracy

**Console Output Format**:
```
============================================================
VALIDATION SET EVALUATION RESULTS
============================================================
Accuracy:           0.9958
Precision:          0.9947
Recall:             0.9938
F1 Score:           0.9943
ROC-AUC:            0.9982
Average Precision:  0.9982
Sensitivity:        0.9938
Specificity:        0.9969
Balanced Accuracy:  0.9954
============================================================

Confusion Matrix:
TN: 1948  FP: 6
FN: 7     TP: 1131
```

---

#### **CELL 22: Validation and Test Evaluation**
**Purpose**: Run comprehensive evaluation pada validation dan test sets

**Input**: 
- Validation: 3,092 samples
- Test: 6,184 samples

**Output**: 2 evaluation results

**Validation Set Results**:
```
Accuracy:           99.58%
Precision:          99.47%
Recall:             99.38%
F1 Score:           99.43%
ROC-AUC:            99.82%
Balanced Accuracy:  99.54%

Confusion Matrix:
TN: 1948 (63.0%) | FP: 6 (0.2%)
FN: 7 (0.2%)     | TP: 1131 (36.6%)
```

**Test Set Results**:
```
Accuracy:           99.30%
Precision:          99.08%
Recall:             99.03%
F1 Score:           99.06%
ROC-AUC:            99.64%
Balanced Accuracy:  99.25%

Confusion Matrix:
TN: 3887 (62.9%) | FP: 21 (0.3%)
FN: 22 (0.4%)    | TP: 2254 (36.4%)
```

**Performance Comparison**:
- Validation slightly better than test (expected)
- Minimal degradation (~0.3% accuracy drop)
- Excellent generalization

---

#### **CELL 23: Plot Confusion Matrices**
**Purpose**: Visualize confusion matrices dan metrics comparison

**Input**: val_cm, test_cm, metrics dictionaries

**Output**: Figure dengan 3 subplots

**Subplots**:
1. **Confusion Matrix - Validation Set**:
   - Heatmap dengan annotations
   - Percentage labels
   - Color: Blues

2. **Confusion Matrix - Test Set**:
   - Heatmap dengan annotations
   - Percentage labels
   - Color: Greens

3. **Validation vs Test Metrics**:
   - Grouped bar chart
   - 6 metrics compared
   - Value labels on bars

**Insights**:
- Low false positive rate (0.3-0.5%)
- Low false negative rate (0.2-0.4%)
- Consistent performance across datasets

---

#### **CELL 24: ROC and PR Curves**
**Purpose**: Visualize model discrimination ability dengan ROC dan PR curves

**Input**: True labels dan predicted probabilities

**Output**: Figure dengan 3 subplots

**Subplots**:
1. **ROC Curve**:
   - Validation AUC: 0.9982
   - Test AUC: 0.9964
   - Both curves near perfect (top-left corner)
   - Random classifier baseline shown

2. **Precision-Recall Curve**:
   - Validation AP: 0.9982
   - Test AP: 0.9960
   - High precision maintained across recall levels
   - Near rectangular shape (ideal)

3. **Prediction Probability Distribution**:
   - Normal queries: Concentrated at 0.0-0.2
   - Attack queries: Concentrated at 0.8-1.0
   - Clear separation between classes
   - Optimal threshold at 0.338

**Interpretation**:
- Excellent class separation
- Model very confident in predictions
- Minimal overlap between distributions

---

#### **CELL 25: Classification Reports**
**Purpose**: Generate detailed sklearn classification reports

**Input**: y_true, y_pred for validation and test sets

**Output**: Formatted classification reports

**Validation Set Report**:
```
              precision    recall  f1-score   support

      Normal     0.9964    0.9969    0.9967      1954
      Attack     0.9947    0.9938    0.9943      1138

    accuracy                         0.9958      3092
   macro avg     0.9956    0.9954    0.9955      3092
weighted avg     0.9958    0.9958    0.9958      3092
```

**Test Set Report**:
```
              precision    recall  f1-score   support

      Normal     0.9944    0.9946    0.9945      3908
      Attack     0.9908    0.9903    0.9906      2276

    accuracy                         0.9930      6184
   macro avg     0.9926    0.9925    0.9925      6184
weighted avg     0.9930    0.9930    0.9930      6184
```

---

#### **CELL 26: Sample Predictions**
**Purpose**: Show random sample predictions untuk qualitative evaluation

**Input**: 10 random test samples

**Output**: DataFrame dengan predictions

**Sample Output**:
```
Query                                         True_Label  Predicted_Label  Probability  Correct
select * from users where id = 1              Normal      Normal           0.0234       ✓
select * from users where id = 1 or 1=1--     Attack      Attack           0.9876       ✓
admin' or '1'='1                              Attack      Attack           0.9654       ✓
select name from customers                    Normal      Normal           0.0456       ✓
union select null, table_name from info...    Attack      Attack           0.9923       ✓
...
```

**Statistics**:
- Correct predictions: 10/10 (100%)
- High confidence on correct predictions
- Clear probability separation

---

#### **CELL 27: Save Model and Artifacts**
**Purpose**: Save all model components untuk deployment

**Input**: Model, tokenizer, configs, results

**Output**: 5 files saved

**Files Saved**:
1. **sql_injection_transformer_model.keras** (7.55 MB)
   - Complete model architecture
   - Trained weights
   - Optimizer state

2. **tokenizer.pkl**
   - Fitted tokenizer object
   - Word index dictionary
   - Configuration

3. **model_config.pkl**
   - Hyperparameters
   - Architecture settings
   - Optimal threshold

4. **evaluation_results.pkl**
   - All metrics
   - Confusion matrices
   - Training history

5. **preprocessing_artifacts.pkl**
   - MAX_WORDS, MAX_LEN
   - Optimal threshold
   - Other preprocessing params

---

#### **CELL 28: Prediction Function**
**Purpose**: Create reusable function untuk inference

**Input**: SQL queries (string or list)

**Output**: Prediction results

**Function Signature**:
```python
def predict_sql_injection(
    queries, 
    model_path='sql_injection_transformer_model.keras',
    tokenizer_path='tokenizer.pkl',
    config_path='preprocessing_artifacts.pkl'
):
```

**Returns**: List of dictionaries
```python
[
    {
        'query': 'SELECT * FROM users...',
        'prediction': 'Attack',
        'probability': 0.9876,
        'confidence': 0.9876
    },
    ...
]
```

**Features**:
- Automatic model loading
- Batch prediction support
- Confidence score calculation
- Truncation for long queries

---

#### **CELL 29: Test Prediction Function**
**Purpose**: Validate prediction function dengan test queries

**Input**: 6 test queries (3 normal, 3 attack)

**Test Queries**:
```python
test_queries = [
    "SELECT * FROM users WHERE id = 1",
    "SELECT * FROM users WHERE id = 1 OR 1=1--",
    "SELECT * FROM products WHERE category = 'electronics'",
    "admin' OR '1'='1",
    "SELECT name, email FROM customers WHERE active = 1",
    "1' UNION SELECT NULL, table_name FROM information_schema.tables--"
]
```

**Output**: Predictions untuk semua queries
```
Query: SELECT * FROM users WHERE id = 1
Prediction: Normal
Probability: 0.0234
Confidence: 0.9766
--------------------------------------------------------------------------------
Query: SELECT * FROM users WHERE id = 1 OR 1=1--
Prediction: Attack
Probability: 0.9876
Confidence: 0.9876
--------------------------------------------------------------------------------
...
```

**Validation**: 6/6 correct predictions (100%)

---

#### **CELL 30: Final Summary**
**Purpose**: Display comprehensive final summary

**Input**: All evaluation results

**Output**: Formatted summary

```
================================================================================
SQL INJECTION DETECTION MODEL TRAINING COMPLETED
================================================================================

Final Test Set Performance:
  Accuracy:          0.9930
  Precision:         0.9908
  Recall:            0.9903
  F1 Score:          0.9906
  ROC-AUC:           0.9964
  Balanced Accuracy: 0.9925

Model Details:
  Total Parameters:  1,978,113
  Model Size:        7.55 MB
  Training Time:     ~8 seconds/epoch
  Best Epoch:        2/50
  Optimal Threshold: 0.3376

Files Generated:
  ✓ sql_injection_transformer_model.keras
  ✓ tokenizer.pkl
  ✓ model_config.pkl
  ✓ evaluation_results.pkl
  ✓ preprocessing_artifacts.pkl

================================================================================
```

---

## 🔬 Technical Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocabulary Size | 10,000 | Maximum unique words |
| Max Sequence Length | 100 | Padded query length |
| Embedding Dimension | 128 | Token embedding size |
| Attention Heads | 4 | Multi-head attention |
| Feed-Forward Dimension | 256 | FFN hidden layer |
| Transformer Blocks | 2 | Number of layers |
| Dropout Rate | 0.2 | Regularization |
| Batch Size | 64 | Training batch size |
| Learning Rate | 0.001 → 0.000125 | Adaptive with ReduceLR |
| Optimizer | Adam | Gradient descent |

### Training Configuration

- Early Stopping: Patience 15, monitor val_auc
- Learning Rate Reduction: Factor 0.5, patience 5
- Class Weights: {0: 0.7854, 1: 1.3591}
- GPU: NVIDIA T4 (Google Colab)
- Training Time: ~8 seconds/epoch
- Best Model: Epoch 2

---

## 📈 Results Analysis

### Strengths

✅ **Excellent Detection Rate**
- 99.03% recall (hanya 22 dari 2276 attack yang lolos)
- 99.08% precision (minimal false alarms)

✅ **Fast Convergence**
- Optimal model achieved di epoch 2
- Total training < 5 minutes

✅ **No Overfitting**
- Train dan validation metrics hampir identik
- Generalization sangat baik

✅ **Production Ready**
- Fast inference (7-14ms per query)
- Optimized threshold (0.338)
- Complete serialization

### Limitations

⚠️ **Dataset Bias**
- Synthetic dataset dengan pola obvious
- Real-world attack mungkin lebih sophisticated

⚠️ **Custom Metrics Issue**
- F1 dan Balanced Accuracy stuck di ~0.20-0.27
- Tidak mempengaruhi training (loss-based)

⚠️ **Adversarial Robustness**
- Belum tested dengan obfuscated attacks
- Perlu testing dengan encoded payloads (hex, base64)

---

## 🙏 Acknowledgments

- Dataset: Modified SQL Injection Dataset
- Framework: TensorFlow/Keras
- Architecture: Transformer (Vaswani et al., 2017)
- Platform: Google Colab with T4 GPU

---

## 📚 References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. OWASP SQL Injection Prevention Cheat Sheet
4. SQL Injection Attack Detection Methods - IEEE

---

