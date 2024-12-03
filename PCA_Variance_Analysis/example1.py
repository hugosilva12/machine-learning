import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')
df.columns = df.columns.str.replace(' ', '')

# Independent variables
X = df.iloc[:, :8]

# Normalize the data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Create the covariance matrix
cov_mat = np.cov(X.T)
print('Covariance matrix \n%s' % cov_mat)

# Calculate how much each component (eigenvectors) contributes to the data variability
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' % eigenvectors)
print('Eigenvalues \n%s' % eigenvalues)

# Sort and select the eigenvectors with the highest eigenvalues so their cumulative sum captures a certain amount of information
tot = sum(eigenvalues)
var_exp = [(i/tot) * 100 for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# With 7 components, approximately 95% of the variance is explained
print('Cumulative Variance Explained [1,2,3,4,5,6,7,8] \n', cum_var_exp)

# Plot explained variance
plt.figure(figsize=(6, 4))
plt.bar(range(8), var_exp, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(range(8), cum_var_exp, where='mid', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('diabetes_ex5.png')
plt.show()

# Conclusion: It is possible to reduce the dataset dimensions from 8 to 7 while losing only about 5% of the total variance. However, the actual reduction slightly exceeds 5%.
