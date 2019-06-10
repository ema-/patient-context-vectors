import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
import csv
import gensim
import numpy as np
import os, sys

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.convolutional import Conv1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras import metrics
from keras.models import load_model

TRAINING_DATA_FILE = os.environ.get("TRAINING_DATA_FILE")
DICT_FILE = os.environ.get("DICT_FILE")
EMBEDDINGS_MODEL_FILE = os.environ.get("EMBEDDINGS_MODEL_FILE")
CNN_MODEL_FILE = os.environ.get("CNN_MODEL_FILE")
ICD_EMBEDDINGS_FILE = os.environ.get("ICD_EMBEDDINGS_FILE")

if None in (TRAINING_DATA_FILE, DICT_FILE, EMBEDDINGS_MODEL_FILE, CNN_MODEL_FILE, ICD_EMBEDDINGS_FILE):
    print("Please set the following env vars: TRAINING_DATA_FILE,DICT_FILE,EMBEDDINGS_MODEL_FILE,CNN_MODEL_FILE,"
          "ICD_EMBEDDINGS_FILE")
    sys.exit(1)

DICTIONARY = gensim.corpora.Dictionary.load(DICT_FILE)

EMBEDDING_DIM = 100
n_layers = 2
hidden_units = 500
batch_size = 100
vocab_size = len(DICTIONARY)
pretrained_embedding = True
patience = 3
dropout_rate = 0.3
n_filters = 100
window_size = 8
dense_activation = "relu"
l2_penalty = 0.0003
epochs = 30


def token_to_index(token, dictionary):
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1


def texts_to_indices(text, dictionary):
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))


def read_data():
    lengths = []
    csv_reader = csv.reader(open(TRAINING_DATA_FILE))
    train_texts = []
    train_labels = []
    for row in csv_reader:
        hamdid, preeva_text, preeva_label = row
        text = eval(preeva_text)
        label = eval(preeva_label)
        lengths.append(len(text))
        train_texts.append(text)
        train_labels.append(label)
    a = np.array(lengths)
    MAX_SEQUENCE_LENGTH = np.percentile(a, 90)
    return train_texts, train_labels, MAX_SEQUENCE_LENGTH


def train(train_texts, train_labels, MAX_SEQUENCE_LENGTH, dictionary):
    MAX_SEQUENCE_LENGTH = int(MAX_SEQUENCE_LENGTH)
    ss = list(map(lambda x: texts_to_indices(x, dictionary), train_texts))
    x_data = pad_sequences(ss, maxlen=int(MAX_SEQUENCE_LENGTH))
    y_data = train_labels
    # create embeddings matrix from word2vec pre-trained embeddings
    embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
    embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
    for word, i in dictionary.token2id.items():
        embedding_vector = embeddings_index[word] if word in embeddings_index else None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    if pretrained_embedding:
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
    else:
        model.add(Embedding(vocab_size,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=n_filters,
                     kernel_size=window_size,
                     activation='relu'))
    model.add(MaxPooling1D(MAX_SEQUENCE_LENGTH - window_size + 1))
    model.add(Flatten())
    for _ in range(n_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units,
                        activation=dense_activation,
                        kernel_regularizer=l2(l2_penalty),
                        bias_regularizer=l2(l2_penalty),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(train_labels[0]),
                    activation='linear',
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

    print(model.summary())

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=[metrics.mse])

    # TRAIN THE MODEL
    early_stopping = EarlyStopping(patience=patience)
    Y = np.array(y_data)
    fit = model.fit(x_data,
                    Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=[early_stopping])

    val_mean_squared_error = fit.history['val_mean_squared_error'][-1]
    print(val_mean_squared_error)
    model.save(CNN_MODEL_FILE)


def predict_one(text, model, MAX_SEQUENCE_LENGTH=785):
    MAX_SEQUENCE_LENGTH = int(MAX_SEQUENCE_LENGTH)
    test_texts = [text]
    ss_test = list(map(lambda x: texts_to_indices(x, DICTIONARY), test_texts))
    x_data_test = pad_sequences(ss_test, maxlen=int(MAX_SEQUENCE_LENGTH))
    predictions = model.predict(x_data_test, verbose=1)
    return predictions[0]


if __name__ == '__main__':
    print("Reading data")
    train_texts, train_labels, MAX_SEQUENCE_LENGTH = read_data()
    print("MAX_SEQUENCE_LENGTH = %s" % MAX_SEQUENCE_LENGTH)
    print('Training...')
    train(train_texts, train_labels, MAX_SEQUENCE_LENGTH, DICTIONARY)
