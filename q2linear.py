# -*- coding: utf-8 -*-
"""Q2linear.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gKMb89KaGwLJ7V63CCraVotZFMZFe5lC
"""

import datetime as dt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

expense_df = pd.read_csv("expense_data_1.csv")

expense_df=expense_df.drop(['Subcategory','Note.1','Account.1','Currency','INR'],axis=1)

expense_df.head(4)

expense_df=expense_df.dropna()

expense_df=expense_df.drop(expense_df.loc[expense_df['Income/Expense'].isin(['Income'])].index)

expense_df['Date'] = pd.to_datetime(expense_df['Date'])
expense_df['Date'] = expense_df['Date'].map(dt.datetime.toordinal)

model = LinearRegression()
category_code={'Food':0, 'Other':1, 'Transportation':2, 'Apparel':3, 'Household':4,
       'Social Life':5, 'Education':6, 'Self-development':7, 'Beauty':8, 'Gift':9
}

expense_df['category_code']=expense_df.Category.map(category_code)

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

inputs, targets = expense_df[['category_code','Date']], expense_df['Amount']
model = LinearRegression().fit(inputs, targets)
predictions = model.predict(inputs)

loss = rmse(targets, predictions)
print('Loss:', loss)

def predict_expense(date, category_code):
  date_ordinal = dt.datetime.toordinal(pd.to_datetime(date))
  input_data = np.array([[category_code, date_ordinal]])
  predicted_expense = model.predict(input_data)[0]
  return predicted_expense

# Get user input
date = input("Enter the date (YYYY-MM-DD): ")
category_code = int(input("Enter the category code: "))

predicted_expense = predict_expense(date, category_code)
print("Predicted expense:", predicted_expense)



def recommend_category(expense_df):
  average_expenses = expense_df.groupby('Category')['Amount'].mean()
  recommended_category = average_expenses.sort_values().index[0]
  return recommended_category
recommended_category = recommend_category(expense_df)
print("Recommended category:", recommended_category)



