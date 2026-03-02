from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs("models", exist_ok=True)

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Regression Results")
print("MSE:", mean_squared_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

joblib.dump(model, "models/regression.pkl")

print("Regression model saved successfully!")