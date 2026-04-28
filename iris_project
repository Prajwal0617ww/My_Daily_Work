import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data file
df = pd.read_csv("iris.csv")
print("Dataset Loaded Successfully")

# just checking some rows to get idea of dataset
# print(df.head())

# taking input and output
X = df.drop("species", axis=1)
y = df["species"]

# converting text into numbers
y = y.astype('category').cat.codes

# splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# making model
model = LogisticRegression(max_iter=200)

# training the model
model.fit(X_train, y_train)

# predicting test data
y_pred = model.predict(X_test)

# checking accuracy
acc = model.score(X_test, y_test)
print("Accuracy:", acc)
      
