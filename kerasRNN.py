import keras
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd  
from random import random

def _load_data(data, n_prev = 3):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())

    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY
def bitfield(n):
    digits = [int(digit) for digit in bin(n)[2:]] # [2:] to chop off the "0b" part 
    return [0 for i in range(0, 8 - len(digits))] + digits
def seq(n):
	return np.array([[[n],[n+1],[n+2]]])

in_out_neurons = 1  
hidden_neurons = 300
model = Sequential()
#model.add(Masking(mask_value=0, input_shape=(10,)))
model.add(LSTM(input_dim=in_out_neurons, output_dim = hidden_neurons, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dense(in_out_neurons))  
model.add(Activation('linear'))

#dat = [np.array(bitfield(i),ndmin=8) for i in range(1,100) ]



dataFrame = pd.DataFrame({'a': range(1,10)})
#dt = [bitfield(i) for i in range(1,10)]
#dataFrame = pd.DataFrame({'a': dt})

X, y = _load_data(dataFrame)

model.compile(loss="mean_squared_error", optimizer="rmsprop") 
#model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X, y, batch_size=450, nb_epoch=10)  

print(model.predict(np.array([[[1],[2],[3]]])))
print([model.predict(seq(i)) for i in range(1,7)])
print(model.predict(np.array([[[6],[2],[3]]])))
