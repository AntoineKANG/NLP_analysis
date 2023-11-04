# Restaurant Reviews
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

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


