import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras # Standalone keras
import numpy as np
class NN(keras.Model):
    def __init__(self):
        super(NN,self).__init__()
        self.fc1 = keras.layers.Dense(units=5, activation=keras.activations.gelu)
        self.fc2 = keras.layers.Dense(4, activation=keras.activations.gelu)
        self.fc3 = keras.layers.Dense(3, activation=keras.activations.gelu)
        self.fc4 =  keras.layers.Dense(2, activation=keras.activations.sigmoid)
        

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)    

# Number of samples and feature shape
num_samples = 1000
feature_shape = (5,)

# Generate random features
features = np.random.rand(num_samples, *feature_shape)

# Add noise to the first half of the dataset
noise = np.random.normal(loc=0.0, scale=0.1, size=features[:num_samples//2].shape)
features[:num_samples//2] += noise

# Create labels: 0 for noisy, 1 for non-noisy
labels = np.ones(num_samples)
labels[:num_samples//2] = 0

# Create a TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

# Shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=num_samples).batch(32)

# Split into training and validation datasets
train_size = int(0.8 * num_samples)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Instantiate and compile the model
model = NN()
model.build((32, 5))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Test the model call
model.fit(train_dataset,)

# Save the model
model.save('./modelpy',overwrite=True,)