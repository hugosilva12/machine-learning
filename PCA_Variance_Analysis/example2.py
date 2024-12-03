import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

# Load the dataset
df = pd.read_csv('dataset.csv')
df.columns = df.columns.str.replace(' ', '')  # Remove spaces in column names

# Independent variables X (first 8 columns)
X = df.iloc[:, :8]

### Normalize the data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Apply PCA: Retain components that explain 94.9% of the variance
pca = PCA(0.949)
pca.fit(X)
print("Number of components:", pca.n_components_)

# Transform the data using the selected components
X = pca.transform(X)

############################################# 
# Apply the algorithm from K-Means
#############################################

# K-means clustering with 2 clusters
model = KMeans(n_clusters=2, random_state=11)
model.fit(X)

# Map the predicted labels to binary class labels (0 or 1)
df['pred_class'] = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

# Calculate the accuracy score
print('Accuracy: ', metrics.accuracy_score(df['class'], df['pred_class']))

# Value for Exercise 2: 0.6744791666666666
# Value for Exercise 3: 0.6744791666666666
# Conclusion: Even with one fewer component, the accuracy remains the same. Does this indicate that reducing dimensions does not affect the performance?
