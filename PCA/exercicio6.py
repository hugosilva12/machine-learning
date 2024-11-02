import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.metrics as metrics

df = pd.read_csv('Diabetes.csv')
df.columns = df.columns.str.replace (' ', '')

# variaveis independentes
X = df.iloc[:, :8]

### Normalizar os dados
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#  Substituir o calculo dos auto-vetores , é possivel observar que tem o mesmo resultado do exercicio anterior (7 componentes)
pca = PCA(.949)
pca.fit(X)
print("Nr componentes:" , pca.n_components_)
X = pca.transform(X)

############################################# Aplicar algoritimo do Exercicio 2 ##################################
# k- means
model = KMeans ( n_clusters =2, random_state =11)
model.fit(X)
df['pred_class'] = np.choose(model.labels_,[1, 0, 2]). astype (np. int64 )
print ('Accuracy : ', metrics.accuracy_score(df['class'] , df['pred_class']))

## Valor Ex2: 0.6744791666666666
## Valor Ex3: 0.6744791666666666
## Conclusão, com menos um componente mantém se a mesma precisão exata? duvida