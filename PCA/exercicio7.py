import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


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

############################################# Aplicar algoritimo Redes neuronais ##################################

# Separar dados em Treino e Teste neste caso 80% treino  e 20 teste
X_train , X_test , y_train , y_test = train_test_split(X,df['class'], test_size =0.2, random_state = 2017)

# Standardise data , and fit only to the training data
scaler = StandardScaler ()
scaler.fit(X_train)
# Apply the transformations to the data
X_train_scaled = scaler.transform(X_train)

mlp = MLPClassifier (hidden_layer_sizes =(200, 300), activation = 'logistic', max_iter = 2000 )

# Train the classifier with the traning data
mlp.fit ( X_train_scaled , y_train )

# Apply the transformations to the data
X_train_scaled = scaler.transform ( X_train )
X_test_scaled = scaler.transform ( X_test )

print (" Training set score : %f" % mlp . score ( X_train_scaled , y_train ))
print (" Test set score : %f" % mlp . score ( X_test_scaled , y_test ))

###Redes neuronais ficha 6 obteve cerca de 0.83 mais ou menos
### Após aplicação do algoritimo PCA obteve cerca de 0.82