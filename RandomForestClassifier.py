import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'dataset.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(file_path)

# Filter out rows with missing critic scores or global sales
df = df.dropna(subset=['Critic_Score', 'Global_Sales'])

# Define hits as games with sales above 1 million units
df['Hit'] = (df['Global_Sales'] > 1).astype(int)

# Features (X) and target variable (y)
X = df[['Critic_Score']]
y = df['Hit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))