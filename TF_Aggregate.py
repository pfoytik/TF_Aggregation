import csv
import getopt
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras import backend as K

class SimpleKeras:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        #model._name = 'Name0_1'
        model.add(Conv2D(16,kernel_size=(2,2),strides=(1,1),
                    padding="same", input_shape=(32,32,3),activation='relu',name='Conv1'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),name='Pool1'))
        model.add(Conv2D(32,kernel_size=(2,2),strides=(1,1),
                    padding="same", activation='relu',name='Conv2'))
    
        model.add(Flatten()) 
        model.add(Dense(250,activation='relu',name='Dense1'))
        model.add(Dense(100, activation ='relu',name='Dense2')) # 3 classes
        model.add(Dense(50, activation ='relu',name='Dense3'))  
        model.add(Dense(3, activation ='softmax',name='Output'))
        #model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
       # model.summary()    
        return model

## Receives a list of models
def aggregate_model(modelList):
    scaled_local_weight_list = list()
    smlp_global = SimpleKeras()
    global_model = smlp_global.build((32,32,3),3)
    for client in modelList:            
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(modelList[client].get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        #clear session to free memory after each communication round
        K.clear_session()
    
    #to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    #update global model 
    global_model.set_weights(average_weights)
    
    return global_model

def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    #print(global_count)
    #print(local_count)
    return local_count/global_count

def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def batch_data(data_shard, bs=16):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'x')
if len(args) >= 2:
    X1_file = args[0]
    y1_file = args[1]
else:
    X1_file = 'testX1.npy'
    y1_file = 'testy1.npy'

modelList = {}
models = []

# Global data test set
X1 = np.load(X1_file, allow_pickle=True)
y1 = np.load(y1_file, allow_pickle=True)

data = list(zip(X1, y1))

with open('modelList.txt', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        models = models + row

clients_batched = dict()
for client_name in models:
    clients_batched[client_name] = batch_data(data)        
        
print(models)        

for m in models:
    modelList[m] = tf.keras.models.load_model(m+'_recognition.model')
    
globalModel =  aggregate_model(modelList)
globalModel.save('global_recognition.model')
