// Import TensorFlow.js for Node.js
const tf = require('@tensorflow/tfjs-node');

// Step 1: Create a synthetic dataset
function createSyntheticDataset(size) {
    const X = [];
    const y = [];
    for (let i = 0; i < size; i++) {
        const xVal = Math.random() * 10;
        const noise = Math.random() * 0.5;
        const yVal = 2 * xVal + 3 + noise; // Linear function y = 2x + 3 with noise
        X.push(xVal);
        y.push(yVal);
    }
    return { X: tf.tensor2d(X, [size, 1]), y: tf.tensor2d(y, [size, 1]) };
}

// Step 2: Create a model
function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, inputShape: [1], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError',
        metrics: ['mae'], // Mean Absolute Error
    });

    return model;
}

// Step 3: Train the model
async function trainModel(model, X_train, y_train, X_val, y_val) {
    await model.fit(X_train, y_train, {
        epochs: 50,
        validationData: [X_val, y_val],
        batchSize: 16,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Val Loss = ${logs.val_loss}`);
            }
        }
    });
}

// Step 4: Save the model
async function saveModel(model, savePath) {
    await model.save(`file://${savePath}`);
    console.log("Model saved to", savePath);
}

// Step 5: Load the model
async function loadModel(savePath) {
    const model = await tf.loadLayersModel(`file://${savePath}/model.json`);
    console.log("Model loaded from", savePath);
    return model;
}

// Step 6: Test the model
async function testModel(model, X_test, y_test) {
    const result = await model.evaluate(X_test, y_test);
    const loss = result[0].dataSync();
    const mae = result[1].dataSync();
    console.log(`Test Loss: ${loss}, Test MAE: ${mae}`);
}

// Main function
async function run() {
    // Create synthetic dataset
    const { X: X_all, y: y_all } = createSyntheticDataset(200);

    // Split into train, validation, and test sets
    const X_train = X_all.slice([0, 0], [140, -1]);
    const y_train = y_all.slice([0, 0], [140, -1]);
    const X_val = X_all.slice([140, 0], [30, -1]);
    const y_val = y_all.slice([140, 0], [30, -1]);
    const X_test = X_all.slice([170, 0], [30, -1]);
    const y_test = y_all.slice([170, 0], [30, -1]);

    // Create model
    const model = createModel();

    // Train model
    await trainModel(model, X_train, y_train, X_val, y_val);

    // Save model
    const savePath = './my_tfjs_model';
    await saveModel(model, savePath);

    // Load model
    const loadedModel = await loadModel(savePath);

    // Test loaded model
    await testModel(loadedModel, X_test, y_test);
}

run();
