import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import (Input, Dense)
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import Model
# noinspection PyUnresolvedReferences
from tensorflow.keras.optimizers import Adam

X = np.random.randint(0, 11, size=(10000, 2))
y = X.sum(axis=1)

X_train = X[:5000]
X_test = X[5000:]
y_train = y[:5000]
y_test = y[5000:]

def model_maker():
    input_layer = Input(shape=(2,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(1, activation='linear')(x)
    return Model(inputs=input_layer, outputs=output_layer)

model = model_maker()

model.compile(loss='mse',
              optimizer=Adam(),
              metrics=['mae'])


history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {test_mae:.2f}")


def predict_sum(a, b):
    prediction = model.predict(np.array([[a, b]]))[0][0]
    return prediction
print(predict_sum(6,4))