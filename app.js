let model;

async function loadModel() {
    console.log('Loading the model...');
    model = await tf.loadGraphModel('model/model.json');
    console.log('Model loaded successfully.');
}


async function loadAndPredict() {
    const fileInput = document.getElementById('upload-image');
    if (fileInput.files.length > 0) {
        console.log('Image file selected, loading image...');
        const image = await loadImage(fileInput.files[0]);
        console.log('Image loaded, preprocessing image...');
        const processedInput = preprocessImage(image);
        console.log('Image preprocessed, input tensor shape:', processedInput.shape);
        console.log('Predicting...');
        const output = model.predict(processedInput);
        console.log('Prediction complete, output tensor shape:', output.shape);
        console.log('Rendering output...');
        renderOutput(output);
    } else {
        console.log('No image file selected.');
        alert('Please upload an image.');
    }
}


function preprocessImage(image) {
    const inputHeight = 240;  // Model's expected input height
    const inputWidth = 480;   // Model's expected input width

    console.log('Converting RGB image to tensor...');
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([inputHeight, inputWidth])
        .toFloat()
        .div(tf.scalar(255.0));

    console.log('Converting to YCrCb and extracting the Y channel...');
    const YCrCb = tensor.mul(tf.tensor1d([0.299, 0.587, 0.114])).sum(2);
    const Y = YCrCb.expandDims(-1);  // Add a channel dimension

    console.log('Transposing the tensor to match the NCHW format...');
    const batchedY = Y.expandDims(0); // Add a batch dimension
    const transposedY = batchedY.transpose([0, 3, 1, 2]);  // Change from [batch, height, width, channels] to [batch, channels, height, width]

    console.log('Preprocessed tensor shape:', transposedY.shape);
    return transposedY;
}


async function renderOutput(outputTensor) {
    const canvas = document.getElementById('canvas');
    const fileInput = document.getElementById('upload-image');

    console.log('Loading original image for Cr and Cb extraction...');
    const img = await loadImage(fileInput.files[0]);
    let inputTensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([240, 480])  // Original dimensions before model scaling
        .toFloat()
        .div(tf.scalar(255.0));

    // Convert the RGB image to YCrCb format
    let [R, G, B] = tf.split(inputTensor, 3, 2);
    let Y = R.mul(tf.scalar(0.299)).add(G.mul(tf.scalar(0.587))).add(B.mul(tf.scalar(0.114)));
    let Cr = R.sub(Y).mul(tf.scalar(0.713)).add(tf.scalar(0.5));
    let Cb = B.sub(Y).mul(tf.scalar(0.564)).add(tf.scalar(0.5));

    // Upscale Cr and Cb channels using bilinear interpolation
    Cr = Cr.resizeBilinear([720, 1440]).squeeze();
    Cb = Cb.resizeBilinear([720, 1440]).squeeze();

    // Merge with upscaled Y channel from the model's output
    Y = outputTensor.squeeze();

    // Convert YCrCb back to RGB
    R = Y.add(Cr.sub(tf.scalar(0.5)).mul(tf.scalar(1.403)));
    G = Y.sub(Cr.sub(tf.scalar(0.5)).mul(tf.scalar(0.714))).sub(Cb.sub(tf.scalar(0.5)).mul(tf.scalar(0.344)));
    B = Y.add(Cb.sub(tf.scalar(0.5)).mul(tf.scalar(1.773)));

    // Clip the values to ensure they are between 0 and 1
    R = R.clipByValue(0, 1);
    G = G.clipByValue(0, 1);
    B = B.clipByValue(0, 1);

    // Stack the single channel images to create a 3-channel image
    let RGB = tf.stack([R, G, B], 2);

    console.log('Rendering output on canvas...');
    await tf.browser.toPixels(RGB, canvas);
    console.log('Output successfully rendered on canvas.');
}


function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                console.log('Image loaded into HTMLImageElement.');
                resolve(img);
            };
            img.onerror = reject;
            img.src = event.target.result;
        };
        reader.onerror = error => {
            console.error('Error reading file:', error);
            reject(error);
        };
        console.log('Starting file read operation...');
        reader.readAsDataURL(file);
    });
}

window.onload = () => loadModel();
