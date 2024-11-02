
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Diabetes.csv')
df.columns = df.columns.str.replace (' ', '')

# variaveis independentes
X = df.iloc[:, :8]

### Normalizar os dados
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# Criar a matriz de covariancia
cov_mat = np.cov(X.T)
print ('Covariance matrix \n%s' % cov_mat)

# Calcular quanto cada componente (auto-vetores) representa na variabilidade dos nossos dados
# Calcular a variância de cada vetor
autovalores, autovetores = np.linalg.eig(cov_mat)
print('Auto-vetores \n%s' % autovetores)
print('Auto-valores \n%s' % autovalores)

#Ordenar os pares e selecionar os vetores próprios com maior valor próprio, de modo a que a sua soma acumulada capture informação acima de um determinado valor
tot = sum(autovalores)
var_exp = [(i/tot) * 100 for i in sorted (autovalores, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

## Com 7 temos cerca de 95 por cento da variância
# [ 26.17974932  47.81987607  60.69024944  71.63436249  81.16366731 89.69652215  94.9442244  100.]
print('Cummulative Variance Explained [1,2,3,4,5,6,7,8] \n', cum_var_exp)

plt.figure(figsize =(6, 4))
plt.bar(range (8), var_exp , alpha =0.5, align ='center', label ='Individual Explained Variance')
plt.step(range (8), cum_var_exp , where ='mid', label ='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc ='best')
plt.tight_layout()
plt.savefig('diabetes_ex5.png')
plt.show()

#Resposta: É possivel diminuir o data set de 8 para 7 dimensões diminuindo a qualidade em apenas cerca de 5%, para ser justo ultrapassa algo os 5 %
