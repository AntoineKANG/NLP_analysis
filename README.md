# Restaurant Reviews
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Machine Learning](#machine-learning)
- [Deep Learning](#deep-learning)
- [Final Results](#final-results)
- [Conclusion](#conclusion)
- [Author](#author)

## About Dataset
link to kaggle: https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews/data

Dataset of restaurant reviews with 10000 rows and 8 columns
- Try to perform NLP by using the "Review" and "Rating columns" 
- Try sentiment analysis by considering rating above 3 as "Positive" and below 3 as "Negative"

Source: https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis/tree/master
https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews/


The project explores restaurant reviews, with a focus on using machine learning (ML) and deep learning (DL) techniques to predict ratings from review text.
By examining popular themes in reviews, the aim is to give restaurants actionable insights to improve to attract customers to come and dine.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AntoineKANG/NLP_analysis.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd NLP_analysis
   ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Evironment**:
Execute the following cells at the beginning of the notebook to ensure the correct directory structure is linked: (for mac)

```python
import sys
sys.path.append('../../NLP_analysis/')
```

**Accessing cleaned data**:
The cleaned data is available in the `data/complete_cleaned_spellings_Restaurant_reviews` file. It is not recommended to run the preprocessor again because the Spellings run for a long time.

## Code Structure

This section provides an overview of the project's code structure and organization.
```plaintext
NLP_analysis/
├── data/
│   ├── cleaned_spellings_Restaurant_reviews.csv
│   ├── complete_cleaned_spellings_Restaurant_reviews.csv
│   └── Restaurant reviews.csv
├── Notebooks/
│   ├── Baseline_model.ipynb
│   ├── Deep_learning.ipynb
│   ├── Exploratory_data.ipynb
│   ├── Improve_baseline.ipynb
│   └── model1.h5
├── Scripts/
│   ├── Preprocesing.py
│   ├── Preprocessing.ipynb
│   ├── textclassfier.py
│   └── utils.py
├── README.md
└── requirements.txt
```

### data
1. `complete_cleaned_spellings_Restaurant_reviews.csv`
   - Preprocessed data
2. `Restaurant reviews.csv`:
   - Raw data

### Scripts
1. `preprocesing.py`&`preprocessing.ipynb`
   - The preprocessing pipeline
2. `textclassfier.py`:
   - The `TextClassifier` class

### Notebooks
1. `Exploratory_data.ipynb`:
   - Data exploration
2. `Baseline_model.ipynb`:
   - Creation and evaluation of the baseline model
3. `Improve_baseline.ipynb`:
   - Improve the baseline model
4. `Deep_Learning.ipynb`:
   - Use deep learning for better results

## Results

Due to the uneven distribution of data between categories, the main metric is not known to be accuracy, but also the value of the recall rate of category 2 and the overall f1 in the results need to be considered.

## Machine Learning

### Experiment and result
- **Techniques Employed**: To counter the class imbalance, techniques like `class_weight`.
- **Model Employed**: The model that yielded the best results is a text classifier based on Logistic Regression with a `TfidfVectorizer` as a vectorizer.
    ```python
    classifier = TextClassifier(
        model=LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            penalty='l2', 
            C=1, 
        ), 
        vectorizer=TfidfVectorizer(
            max_features=5000, 
            stop_words='english'
        )
    )
    ```
- **Results**: 
    - Accuracy: 59.08%
    - Recall for Class 2: 33%
    - Global F1 Score: 51% (macro avg)

## Deep Learning
Turning to deep learning reveals an overfitting problem that is exacerbated by a lack of data.
### Experiment and result
- A sequential model with an embedding layer, 1D convolutional layer with L2 regularization, global max pooling, and a dense layer with L2 regularization was employed.

    ```python
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM,     input_length=MAX_SEQUENCE_LENGTH),
        Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01)),
        GlobalMaxPooling1D(),
        Dense(5, activation='softmax', kernel_regularizer=l2(0.01))
    ])
    ```
- **Results**: 
    - Accuracy: 55%
    - Recall for Class 2: 42%
    - Global F1 Score: 48% (macro avg)
![image](https://github.com/AntoineKANG/NLP_analysis/blob/main/assets/output2.png?raw=true)
The curve is not very satisfactory. The reason is that there is too little data.

## Final Results
### Remapping
In order to simplify the problem and get better results, I will consider negative and positive(Selection based on the amount of data), I do the following class remapping:
```python
mapping = {1: 'negative/bad', 2: 'negative/bad', 3: 'negative/bad', 4: 'positive/good', 5: 'positive/good'}
```

- **Results with Remapping**:
    - Accuracy: 86.46%
    - Recall for 'negative' category:  75%
    - Global F1 Score: 85% (macro avg)
 
  The model also gives good results after remapping

## Conclusion
Class remapping leads to better results in terms of class recall of interest, global F1 scores, and accuracy. In this case, the main reason is due to the lack of data, and there are only two classes after remapping, which simplifies the complexity of the problem and makes it easier for the model to train and predict.

## Author
- Zhuodong KANG(zhuodong.kang@epfedu.fr)