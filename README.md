# Logistic Regression for Airline Customer Satisfaction Prediction

## Project Overview
This project analyzes passenger survey data from **Invistico Airline** to predict customer satisfaction (`satisfied` or `dissatisfied`) using **Binomial Logistic Regression**.  
We preprocess the data, handle missing values, encode categorical features, train/test split, evaluate model performance (confusion matrix, precision, recall), and interpret coefficients to derive actionable business insights.

---

## Step 1: Load and Inspect the Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

# Load data (assumes file is in current directory)
df = pd.read_csv('Invistico_Airline.csv')

print("Dataset shape:", df.shape)
df.head()
