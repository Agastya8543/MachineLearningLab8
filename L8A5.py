import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_excel('embeddingsdata.xlsx')

# 'embed_0' and 'embed_1' are the features and 'Label' is the target variable
X = data[['embed_0', 'embed_1']]
y = data['Label']

# Split the data into training and testing sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Naïve-Bayes classifier
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Make predictions on the test set
predictions = model.predict(Te_X)

# Evaluate the accuracy of the model
accuracy = accuracy_score(Te_y, predictions)
print(f"Accuracy: {accuracy}")    