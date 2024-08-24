const tf = require('@tensorflow/tfjs');
const loadModel = async () => {
    try {
        // Use the raw URL from GitHub
        const modelUrl = 'https://raw.githubusercontent.com/ShreyasT123/SIH/main/model/my-model.json';
        const model = await tf.loadLayersModel(modelUrl);
        console.log('Model loaded successfully');

        // Predict with some new data
        const input = tf.tensor2d([[1,2,3,4,5,6,7,7,8,8,9,5,6]], [1, 13]);  // Predict y for x = 10
        const prediction = model.predict(input);
        prediction.print();
    } catch (error) {
        console.error(error);
    }
};

loadModel().catch(console.error);

