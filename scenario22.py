import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
print("Swasthika M 24BAD121")
df = pd.read_csv("LIC_stock.csv")
print(df.columns)

open_col = [col for col in df.columns if "open" in col.lower()][0]
close_col = [col for col in df.columns if "close" in col.lower()][0]
high_col = [col for col in df.columns if "high" in col.lower()][0]
low_col = [col for col in df.columns if "low" in col.lower()][0]
volume_col = [col for col in df.columns if "vol" in col.lower()][0]

df['Price_Movement'] = np.where(df[close_col] > df[open_col], 1, 0)

features = [open_col, high_col, low_col, volume_col]
target = 'Price_Movement'

df = df[features + [target]]

df.fillna(df.mean(), inplace=True)

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(C=1.0, penalty='l2')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = " + str(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

importance = model.coef_[0]
plt.bar(features, importance)
plt.title("Feature Importance")
plt.show()
