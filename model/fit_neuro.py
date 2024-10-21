import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris = load_iris()
X = iris.data[:, :3]
print(X)
y = iris.target.reshape(-1, 1)
print(y)

encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
print(X_train.shape[1], y_train.shape[1])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])
#magic
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X_train, y_train, epochs=300, batch_size=32)

model.save('classification_model.h5')
