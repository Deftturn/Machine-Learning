from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


iris_data = load_iris()
X,y = iris_data.data , iris_data.target # type:ignore

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42)

model = RandomForestClassifier(n_estimators= 100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred,target_names=iris_data.target_names)# type: ignore

print(f"Model accuracy: {accuracy}")
print("Classification Report")
print(report)

# joblib.dump(model, "iris_model.pkl")