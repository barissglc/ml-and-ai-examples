import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r"")

data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

M = data[data["diagnosis"] == "M"]
B = data[data["diagnosis"] == "B"]

plt.figure(figsize=(16, 8))
plt.scatter(M.radius_mean, M.texture_mean, color="red", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="green", alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()

data["diagnosis"] = [1 if diag == "M" else 0 for diag in data["diagnosis"]]

y = data["diagnosis"].values
X = data.drop(["diagnosis"], axis=1)

X = (X - X.min()) / (X.max() - X.min())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Score : {knn.score(X_test, y_test)}")

for i in range(min(len(y_test), 30)):
    print(f"Örnek No: {i}\tBeklenen: {y_test[i]}\tModel Sonucu: {y_pred[i]}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", cmap='coolwarm')
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Confusion Matrix")
plt.show()

score_list = []
for each in range(1, 15, 2):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(X_train, y_train)
    score_list.append(knn2.score(X_test, y_test))

plt.plot(range(1, 15, 2), score_list)
plt.xlabel("k Değeri")
plt.ylabel("Score")
plt.title("k Değerine Göre Model Başarısı")
plt.show()
