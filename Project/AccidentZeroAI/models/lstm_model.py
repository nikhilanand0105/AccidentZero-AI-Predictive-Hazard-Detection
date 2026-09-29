import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def train_lstm(X_train, y_train):
    print("[INFO] Training LSTM...")
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=10, verbose=0)
    print("[OK] LSTM trained")
    return model