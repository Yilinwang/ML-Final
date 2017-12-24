from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector
from keras.layers import Embedding, Bidirectional



def basic(voc_size):
    model = Sequential()

    model.add(Embedding(voc_size, 64, input_length = 19))
    model.add(Bidirectional(LSTM(128, return_sequences = False)))
    model.add(Dense(128, activation = 'relu'))
    model.add(RepeatVector(19))
    model.add(LSTM(128, return_sequences = True))
    model.add(TimeDistributed(Dense(voc_size, activation='softmax')))

    return model


def get_model(model, voc_size):
    model = globals()[model](voc_size)
    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop')
    return model
