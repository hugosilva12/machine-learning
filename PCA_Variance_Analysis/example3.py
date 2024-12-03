import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv('dataset.csv')
df.columns = df.columns.str.replace(' ', '')

# Independent variables
X = df.iloc[:, :8]

# Normalize the data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Replace eigenvector calculation and observe the same result as the previous exercise (7 components)
pca = PCA(0.949)
pca.fit(X)
print("Number of components:", pca.n_components_)
X = pca.transform(X)

#############################################
# Apply Neural Networks Algorithm
#############################################

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, df['class'], test_size=0.2, random_state=2017)

# Standardize data and fit only to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Apply transformations to the data
X_train_scaled = scaler.transform(X_train)

# Initialize MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(200, 300), activation='logistic', max_iter=2000)

# Train the classifier with the training data
mlp.fit(X_train_scaled, y_train)

# Apply transformations to the data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluate the model
print("Training set score: %f" % mlp.score(X_train_scaled, y_train))
print("Test set score: %f" % mlp.score(X_test_scaled, y_test))

# Results:
# Neural networks (exercise 6) achieved approximately 0.83
# After applying PCA, the score is approximately 0.82
