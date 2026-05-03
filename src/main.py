import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset
data = pd.DataFrame({
    'income':[10000,20000,30000,30000,np.nan,50000,60000,70000],
    'credit_score':[500,550,500,600,np.nan,700,750,850],
    'loan_status':[0,0,0,0,1,1,1,1]
})

# Handle missing data
data['income'].fillna(data['income'].median(),inplace=True)
data['credit_score'].fillna(data['credit_score'].mean(),inplace=True)

 #Feature & target
X = data[['income','credit_score']]
y = data['loan_status']

# Model
model = LogisticRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[49000,650]])
print("Loan Approval (1=Yes, 0=No):", pred[0])