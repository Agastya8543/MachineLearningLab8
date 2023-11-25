import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load the table data
table_data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(table_data)

# Assuming 'buys_computer' is the target variable
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Convert categorical variables to numerical using one-hot encoding
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Build and train the Naïve-Bayes classifier
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Make predictions on the test set
predictions = model.predict(Te_X)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Te_y, predictions)
print(f"Accuracy: {accuracy}")
