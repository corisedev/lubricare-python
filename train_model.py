import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np

# Load dataset
data = pd.read_csv('oiltypes.csv')

# Preprocess data
X = data[['Engine CC', 'minMilage', 'maxMilage']]
y_oil = data['Recommended Engine Oil']
y_brand = data['Brand Names']

# Split data into training and testing sets
X_train, X_test, y_oil_train, y_oil_test, y_brand_train, y_brand_test = train_test_split(X, y_oil, y_brand, test_size=0.2, random_state=42)

# Train models for Engine Oil
oil_model = RandomForestClassifier()
oil_model.fit(X_train, y_oil_train)

# Train models for Brand Names
class_weights = compute_class_weight('balanced', classes=np.unique(y_brand_train), y=y_brand_train)
class_weight_dict = dict(zip(np.unique(y_brand_train), class_weights))

brand_model = GradientBoostingClassifier()  # Alternative model
brand_model.fit(X_train, y_brand_train)

# Save the models
joblib.dump(oil_model, 'oil_model.pkl')
joblib.dump(brand_model, 'brand_model.pkl')

# Make predictions
y_oil_pred = oil_model.predict(X_test)
y_brand_pred = brand_model.predict(X_test)

# Evaluate models
print("Oil Model Accuracy:", accuracy_score(y_oil_test, y_oil_pred))
print("Brand Model Classification Report:\n", classification_report(y_brand_test, y_brand_pred))
