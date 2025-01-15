import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"")
print(data.head(3))

plt.figure(figsize=(12, 6))
plt.scatter(data["YearsExperience"], data["Salary"])
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()

X = data["YearsExperience"].values
y = data["Salary"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='k', label="Train Data")
plt.scatter(X_test, y_test, color='r', alpha=0.5, label="Test Data")
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction - Train vs Test')
plt.legend()
plt.show()

lr = LinearRegression()

X_train_ = X_train.reshape(-1, 1)
X_test_ = X_test.reshape(-1, 1)
y_train_ = y_train.reshape(-1, 1)
y_test_ = y_test.reshape(-1, 1)

lr = lr.fit(X_train_, y_train_)

y_pred = lr.predict(X_test_)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='k', label="Train Data")
plt.scatter(X_test, y_test, color='r', alpha=0.5, label="Test Data")
plt.plot(X_test, y_pred, color='b', label="Linear Model")
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction - Linear Regression')
plt.legend()
plt.show()

print("Coefficient (Slope):", lr.coef_)
print("Intercept:", lr.intercept_)

fig, axs = plt.subplots(3, 3)
fig.set_figheight(15)
fig.set_figwidth(15)

for i in range(9):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=i)
    X_train_ = X_train.reshape(-1, 1)
    X_test_ = X_test.reshape(-1, 1)
    y_train_ = y_train.reshape(-1, 1)
    y_test_ = y_test.reshape(-1, 1)
    lr = LinearRegression().fit(X_train_, y_train_)
    y_pred = lr.predict(X_test_)
    axs[i//3, i%3].scatter(X_train, y_train, color='k', label="Train Data")
    axs[i//3, i%3].scatter(X_test, y_test, color='r', alpha=0.5, label="Test Data")
    axs[i//3, i%3].plot(X_test, y_pred, color='b', label="Linear Model")
    axs[i//3, i%3].set_title(f"y = {round(lr.coef_[0][0], 2)}*X + {round(lr.intercept_[0], 2)}")
    axs[i//3, i%3].legend()

plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test_, y_pred)
mse = mean_squared_error(y_test_, y_pred)
mae = mean_absolute_error(y_test_, y_pred)

print("R² Skoru:", r2)
print("Ortalama Kare Hatası (MSE):", mse)
print("Ortalama Mutlak Hata (MAE):", mae)
