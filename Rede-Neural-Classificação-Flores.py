
#Bibliotecas
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.models import model_from_json 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# É preciso dividir essa base de dados entre as entradas e saídas
base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# Corrigindo erro que vem com a base de dados, transformando atributo categorico para atributo númerico
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size = 0.25)

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu',  kernel_initializer = 'normal', input_dim = 4))
classificador.add(Dense(units = 8, activation = 'relu',  kernel_initializer = 'normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])

# Treinamento
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
classe_teste2= [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

# Usando matrix para verificar melhor os dados
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classe_teste2)

# Salvando a rede em formato json
classificador_salvo_json = classificador.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_salvo_json)
    
# Salvando os pesos
classificador.save_weights('classificador_iris_pesos.h5')

file = open('classificador_iris.json', 'r')
arquitetura_rede = file.read()
file.close()


# Criando rede igual ao do arquivo json
classificador = model_from_json(arquitetura_rede)
# Carregando os pesos
classificador.load_weights('classificador_iris_pesos.h5')

# Criar e classificar novo registro
# Exemplo de novo registro
novo_registro = np.array([[1.4, 5.5, 2.8, 7.22]])
previsao = classificador.predict(novo_registro)

previsao = (previsao >0.5)

# Mostrando o resultado da rede neural
if (previsao[0][0]) == True and (previsao[0][1]) == False and (previsao[0][2]) == False:
    print("iris setosa")
    
elif (previsao[0][0]) == False and (previsao[0][1]) == True and (previsao[0][2]) == False:
    print("iris virginica")
    
elif (previsao[0][0]) == False and (previsao[0][1]) == False and (previsao[0][2]) == True:
    print("iris virginica")
    
else:
    print("iris não identificada")