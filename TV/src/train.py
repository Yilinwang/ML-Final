import numpy as np
np.random.seed(1126)

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
from . import data_processor, models
import pickle



def my_to_categorical(Y):
    Y_cat = np.zeros((len(Y), 19, 3506))
    for idx in range(len(Y)):
        for char_idx in range(19):
            Y_cat[idx][char_idx][Y[idx][char_idx]] = 1
    return Y_cat


def batch_generator(X, Y):
    while True:
        permu = np.random.permutation(len(X))
        for idx in range(0, len(X), 32):
            yield (X[permu[idx: idx+32]], to_categorical(Y[permu[idx: idx+32]], num_classes = 3506))


def train(args):
    '''
    data = data_processor.Data('data/training_data/')
    pickle.dump(data, open('model/Data.pkl', 'wb'))
    '''
    data = pickle.load(open('model/Data.pkl', 'rb'))
    X = data.get_training_data()
    model = models.get_model(args.model, data.get_vocab_size())
    X, Y = [np.array(x) for x in zip(*zip(X, X[1:]))]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 1126)

    filepath = 'model/' + args.model + '_{epoch:03d}_{val_loss:.2f}'
    checkpoint = ModelCheckpoint(filepath)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks_list = [checkpoint, earlyStopping]

    #history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), callbacks = callbacks_list, batch_size = 32, epochs = 100)
    history = model.fit_generator(batch_generator(X_train, Y_train), validation_data = batch_generator(X_test, Y_test), callbacks = callbacks_list, epochs = 100, steps_per_epoch = int(len(X_train) / 32) + 1, validation_steps = int(len(X_test) / 32) + 1)


def infer(args):
    data = pickle.load(open('model/Data.pkl', 'rb'))
    sentences = list()
    options = list()
    for idx, line in enumerate(open(args.test_path)):
        if idx != 0:
            dataid, sentence, option = line.strip().split(',')
            sentences.append(''.join([s.split(':')[1] for s in sentence.split('\t')]))
            import ipdb
            ipdb.set_trace()

    model = load_model(args.weight)

    X = data.process_sentence([line.strip() for line in open(args.test_path)])

    with open(args.output, 'w') as fp:
        fp.write('TestDataID,Rating\n')
        for idx, p in enumerate(model.predict(unpack(X))):
            rating = max(min(p[0], 5), 0)
            fp.write(str(idx+1) + ',' + str(rating) + '\n')


def main(args):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    if args.infer:
        infer(args)
    else:
        train(args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/training_data')
    parser.add_argument('--test_path', default='data/testing_data.csv')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--weight')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
