import tensorflow as tf

# Load the TensorFlow model
model = tf.saved_model.load('model.pb')

# Perform inference (adjust input data according to your model's input)
input_data = tf.random.normal([1, 224, 224, 3])  # Example input
result = model(input_data)
print(result)
