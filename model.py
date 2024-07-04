import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from joblib import dump

# Load the dataset
data = pd.read_csv('/Users/khushpatel/Desktop/IBM/Api/diabetes.csv')

# Check for missing values and basic statistics
print(data.info())
print(data.describe())

# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and model pipeline
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

# Train the model
pipeline.fit(X_train, y_train)

# Save the model
dump(pipeline, 'diabetes_model.joblib')

# Evaluate the model (optional)
print("Training accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))
