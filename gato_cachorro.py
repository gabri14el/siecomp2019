from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow as tf
import math

#seta as sementes dos pesos iniciais
np.random.seed(0) #semente numpy
tf.set_random_seed(0) #semente tensorflow

tam = 64
batch=32
epocas = 20
#metodo que substitui 0 e 1s por nomes
def substituir_gatos(x, labels):
    valor = list(labels.keys())[list(labels.values()).index(x)]
    return valor

#metodo que plota a acuracia e perda do modelo
def plot_informations(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    
#o modelo sequencial eh um modelo que empilha camadas das redes neurais
#basicamente ele liga a entradas de umas camadas em outras camadas
classificador = Sequential()

#filters: quantidade de filtros convolucionais que a camada convolucional tera
#kernel size: representa o tamanho desses filtros 
#input shape: utilizada porque a camada convolucional eh de entrada, vale a pena
#esclarecer que o input shape representa o tamanho da imagem que será utilizada e a quantidade de canais 
#e que os canais estao assim, pois na configuracao do keras ta como 'channels last'
classificador.add(Conv2D(filters=32, kernel_size=(3,3), input_shape = (tam, tam, 3), activation = 'relu'))

#batchnormalization: Normalize the activations of the previous layer at each batch, 
#i.e. applies a transformation that maintains 
#the mean activation close to 0 and the activation standard deviation close to 1.


classificador.add(BatchNormalization())

#max pooling de 2x2
classificador.add(MaxPooling2D(pool_size = (2,2)))

#classificador.add(Conv2D(32,(3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(Conv2D(32,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

#camada de flatten
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

#back propagation busca minimizar a função de perda, a função de perde é especificado
#em perda. As métricas são como as funções de perda, seus valores são exibidos durante o log de treinamento
#mas não sao utilizados em nenhum etapa.
classificador.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])



gerador_treinamento = ImageDataGenerator (rescale = 1./255,
                                          rotation_range = 7,
                                          horizontal_flip = True,
                                          shear_range = 0.2,
                                          height_shift_range = 0.07,
                                          zoom_range = 0.2)

gerador_validacao = ImageDataGenerator(rescale = 1./255)
gerador_teste = ImageDataGenerator(rescale=1./255)


base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (tam,tam),
                                                           batch_size = batch,
                                                           class_mode = 'binary',
                                                           shuffle=True)


base_validation = gerador_validacao.flow_from_directory('dataset/validation_set',
                                               target_size = (tam, tam),
                                               batch_size = batch,
                                               class_mode = 'binary',
                                               shuffle=True)

base_teste = gerador_validacao.flow_from_directory('dataset/test_set',
                                               target_size = (tam, tam),
                                               batch_size = batch,
                                               class_mode = 'binary', 
                                               shuffle=False)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

history = classificador.fit_generator(base_treinamento, steps_per_epoch = 4000/batch,
                            epochs = epocas,
                            validation_data = base_validation,
                            validation_steps = 1000/batch,
                            callbacks=[tensorboard_callback])




predictions = classificador.predict_generator(base_teste)
predictions = predictions[:,0]  
predictions = list(map(lambda x: round(x), predictions))

steps = len(base_teste.filenames)/batch
steps = math.ceil(steps)
test_labels = []

for i in range(steps):
    test_imgs, test_labels1 = next(base_teste)
    test_labels.extend(test_labels1)

#labels = (base_teste.class_indices)

count = 0
for i in range(len(test_labels)):
    if test_labels[i] == predictions[i]:
        count += 1


acerto = count/len(test_labels)
erro = 1 - acerto

print('taxa de acerto de %f' % (acerto*100))
print('taxa de erro de %f' % (erro*100))
    






