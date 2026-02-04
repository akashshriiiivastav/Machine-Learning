1. Importing the necessary libraries. Import pandas for data manipulation, numpy for numerical operations, and the necessary classes from scikit-learn for preprocessing.

# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
2. Load the dataset. The Titanic dataset is loaded into a pandas DataFrame from a CSV file.

# Load the dataset
df = pd.read_csv('titanic.csv')
3. Identify the categorical data. Specify which features in our dataset are categorical. In this case, 'Sex', 'Embarked', and 'Pclass' are the categorical features.

# Identify the categorical data
categorical_features = ['Sex', 'Embarked', 'Pclass']
4. Implement an instance of the ColumnTransformer clas. Initialize a ColumnTransformer that will apply a OneHotEncoder to the categorical features. The remainder='passthrough' argument ensures that the non-transformed features are not discarded.

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')


5. Apply the fit_transform method. Fit the ColumnTransformer to our DataFrame and transform the data. This applies one-hot encoding to our categorical features, converting them into numerical data suitable for a machine-learning model.

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(df)
6. Convert the output into a NumPy array. Convert the output to a NumPy array: The output of the ColumnTransformer is a sparse matrix - convert this to a dense NumPy array for easier manipulation.

# Convert the output into a NumPy array
X = np.array(X)
7. Use LabelEncoder to encode binary categorical data. The 'Survived' feature is our dependent variable. Since it is a binary categorical feature, we use LabelEncoder to transform it into numerical data.

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
y = le.fit_transform(df['Survived'])
8. Print the transformed feature matrix and dependent variable vector to verify that our preprocessing steps have been applied correctly.

# Print the updated matrix of features and the dependent variable vector
print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)
