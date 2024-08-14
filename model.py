#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import LSTM, GRU, ConvLSTM2D
from keras.layers import Bidirectional
from keras.layers import multiply, concatenate


def get_lstm(units):

    model = Sequential()
    layersize = len(units)
    if layersize <= 3:
        model.add(LSTM(units[1], input_shape=(units[0], 1)))
    else:
        model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
        for layer in range(2, layersize - 2):
            model.add(LSTM(units[layer], return_sequences=True))
        model.add(LSTM(units[-2]))

    model.add(Dropout(0.2))
    model.add(Dense(units[-1], activation='sigmoid'))

    return model


def get_lstm_2(units):

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_bilstm(units):

    model = Sequential()
    model.add(Bidirectional(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True)))
    model.add(Bidirectional(LSTM(units[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_filstm(units, features):

    featurelayer = Input(shape=(features[0],), name='factors')
    dens1 = Dense(features[1], activation='tanh', name='factor_1')(featurelayer)
    dens2 = Dense(features[2], activation='softmax', name='factor_2')(dens1)

    inputlayer = Input(shape=(units[0], 1), name='series')
    lstm1 = LSTM(units[1], return_sequences=True, name='lstm_1')(inputlayer)
    lstm2 = LSTM(units[2], name='lstm_2')(lstm1)

    merge = multiply([dens2, lstm2])

    md1 = Dense(units[2], activation='tanh', name='merge_1')(merge)

    dropout = Dropout(0.2, name='dropout')(md1)

    outputlayer = Dense(units[3], activation='sigmoid', name='Output')(dropout)

    model = Model(inputs=[featurelayer, inputlayer], outputs=outputlayer)

    return model


def get_figru(units, features):
    
    featurelayer = Input(shape=(features[0],), name='factors')
    dens1 = Dense(features[1], activation='tanh', name='factor_1')(featurelayer)
    dens2 = Dense(features[2], activation='softmax', name='factor_2')(dens1)

    inputlayer = Input(shape=(units[0], 1), name='series')
    gru1 = GRU(units[1], return_sequences=True, name='gru_1')(inputlayer)
    gru2 = GRU(units[2], name='gru_2')(gru1)

    merge = multiply([dens2, gru2])

    md1 = Dense(units[2], activation='tanh', name='merge_1')(merge)

    dropout = Dropout(0.2, name='dropout')(md1)

    outputlayer = Dense(units[3], activation='sigmoid', name='Output')(dropout)

    model = Model(inputs=[featurelayer, inputlayer], outputs=outputlayer)

    return model


def get_convlstm(units):

    model = Sequential()
    model.add(ConvLSTM2D(units[1], kernel_size=(1, 3), input_shape=(units[0], 1, 1, 1), padding='same', data_format="channels_last", return_sequences=True))
    model.add(ConvLSTM2D(units[2], kernel_size=(1, 3), input_shape=(units[0], 1, 1, 1), padding='same', data_format="channels_last"))  # "channels_last" default
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(units[3], activation='sigmoid'))

    return model


def get_gru(units):

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    return model
