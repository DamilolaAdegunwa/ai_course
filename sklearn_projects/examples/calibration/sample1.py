from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an uncalibrated classifier
base_clf = SVC(probability=False)  # SVC does not output probabilities by default

# Wrap it with CalibratedClassifierCV
calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='sigmoid', cv=5)

# Train the calibrated classifier
calibrated_clf.fit(X_train, y_train)

# Predict probabilities
y_prob = calibrated_clf.predict_proba(X_test)

# Predict labels
y_pred = calibrated_clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
