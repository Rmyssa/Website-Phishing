import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings("ignore")



df = pd.read_csv("C:/Users/RÜMEYSA/Desktop/web-page-phishing.csv")
df.head()
df.info()
plt.figure(figsize=(18, 14))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
plt.figure(figsize=(6, 4))
sns.countplot(x='phishing', data=df)
plt.title('Phishing vs. Non-Phishing Counts')
plt.xlabel('Phishing')
plt.ylabel('Count')
plt.show()
X = df.drop('phishing', axis=1)
y = df['phishing']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    LogisticRegression(random_state=42),
    RandomForestClassifier(random_state=42),
    KNeighborsClassifier(),
    GradientBoostingClassifier(random_state=42),
    GaussianNB(),
    AdaBoostClassifier(random_state=42),
    MLPClassifier(random_state=42)
]


results = []

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results.append({
        "Model": model.__class__.__name__,
        "Accuracy": accuracy,
        "Precision (0)": report['0']['precision'],
        "Recall (0)": report['0']['recall'],
        "F1-score (0)": report['0']['f1-score'],
        "Precision (1)": report['1']['precision'],
        "Recall (1)": report['1']['recall'],
        "F1-score (1)": report['1']['f1-score']
    })


sns.set(font_scale=1.2)
plt.figure(figsize=(10, 6))
table = sns.heatmap(pd.DataFrame(results).set_index('Model'), annot=True, cmap="plasma", fmt=".2f", linewidths=.5, cbar=False)
plt.title("Classification Report for Different Models")
plt.show()

