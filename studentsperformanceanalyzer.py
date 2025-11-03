# -*- coding: utf-8 -*-
"""StudentsPerformanceanalyzer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("StudentsPerformance.csv")
df.head()

print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

le = LabelEncoder()
for col in df.select_dtypes (include='object').columns:
    df[col] = le.fit_transform(df[col])

x = df.drop('math score', axis=1)
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nLinear Regression RMSE:", rmse)

dt = DecisionTreeRegressor (random_state=42)
rf = RandomForestRegressor (random_state=42, n_estimators=100)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
print("Decision Tree RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

scores = cross_val_score(rf, x, y, cv=5, scoring='neg_root_mean_squared_error')
print("\n Random Forest 5-Fold CV RMSE:", -scores.mean())
