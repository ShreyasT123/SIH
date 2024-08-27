import * as tf from '@tensorflow/tfjs';

// Define the model
const createModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [12]}));  // Adjust inputShape
    model.add(tf.layers.dense({units: 32, activation: 'relu'}));
    model.add(tf.layers.dense({units: 16, activation: 'relu'}));
    model.add(tf.layers.dense({units: 2, activation: 'softmax'}));  // Two classes: human (0) and bot (1)
    model.compile({
        optimizer: 'adam',
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
};

// Create and compile the model
const model = createModel();

// Example data
const X_train = tf.tensor2d([1,2,3,4]);  // Replace with actual training data
const y_train = tf.tensor1d([2,4,6,8]);  // Replace with actual labels

// Train the model
const trainModel = async () => {
    await model.fit(X_train, y_train, {
        epochs: 10,
        batchSize: 32,
        validationSplit: 0.3,
        callbacks: tf.callbacks.earlyStopping({monitor: 'loss'})
    });
    console.log('Model trained successfully');
};

// Save the model
const saveModel = async () => {
    await model.save('localstorage://my-model-tfjs');
    console.log('Model saved to local storage');
};

// Train and save the model
trainModel().then(saveModel).catch(console.error);