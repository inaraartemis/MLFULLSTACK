from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ensure models folder exists
os.makedirs("models", exist_ok=True)

# load dataset
data = load_digits()
X = data.data
y = data.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# evaluate model
preds = model.predict(X_test)

print("Classification Results")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# save model
joblib.dump(model, "models/classifier.pkl")

print("Classification model saved successfully!")