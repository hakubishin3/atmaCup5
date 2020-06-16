from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, Conv1D, GlobalMaxPooling1D, LeakyReLU, BatchNormalization, Input, ReLU
from tensorflow.keras.layers import concatenate


def build_model(signal_size=(512, 1), meta_size=5, output_size=1):
    input_signal = Input(shape=signal_size, name='input_signal')
    input_meta = Input(shape=(meta_size,), name='input_meta')

    x = Conv1D(256, 8, padding='same', name='Conv1')(input_signal)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256, 5, padding='same', name='Conv2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(256, 3, padding='same', name='Conv5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = GlobalMaxPooling1D()(x)

    x1 = Dense(16, activation='relu', name='dense0')(input_meta)
    x1 = Dense(32, activation='relu', name='dense1')(x1)
    xc = concatenate([x, x1], name='concat')

    x = Dense(256, activation='relu', name='features')(xc)
    output = Dense(output_size, activation='sigmoid', name='out')(x)

    model = Model([input_signal, input_meta], output)
    
    return model
