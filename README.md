# AI-TensorFlow-FirstSteps
## TFJS-UserClassifier 🚀

My very first Neural Network project built with **TensorFlow.js** for Node.js. 
This project focuses on understanding the fundamentals of data normalization, one-hot encoding, and tensor manipulation.

## 📌 Project Overview
The goal of this project is to classify users into service tiers (**Premium**, **Medium**, or **Basic**) based on their profile features such as age, color preference, and location.

## 🧠 Concepts Applied
- **Data Normalization:** Scaling numerical values (age) to a 0-1 range.
- **One-Hot Encoding:** Converting categorical data (colors and cities) into binary vectors that the neural network can process.
- **Tensor Operations:** Creating 2D tensors for both input features (Xs) and labels (Ys).

## 🛠️ Tech Stack
- [Node.js](https://nodejs.org)
- [TensorFlow.js for Node](https://tensorflow.org)

## 📋 Data Structure
The model processes a normalized input vector representing:
`[Age, Blue, Red, Green, São Paulo, Rio, Curitiba]`

And predicts a category:
- `[1, 0, 0]` -> **Premium**
- `[0, 1, 0]` -> **Medium**
- `[0, 0, 1]` -> **Basic**

## 🚀 How to Run
1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the project:
   ```bash
   npm start
   ```

## 📖 Code Explanation

### Data Preparation

The code starts by defining the training dataset with three user profiles:

**Original User Data:**
- **Erick:** Age 30, Color Blue, Location São Paulo → Premium tier
- **Ana:** Age 25, Color Red, Location Rio → Medium tier
- **Carlos:** Age 40, Color Green, Location Curitiba → Basic tier

**Input Features (`tensorPessoasNormalizado`):**

Each row represents a user with 7 features in this order:
```
[normalized_age, blue, red, green, são_paulo, rio, curitiba]
```

- **Age normalization:** Ages are scaled to 0-1 range (25=0, 30=0.33, 40=1)
- **Color encoding:** One-hot encoded (only one color is 1, others are 0)
- **Location encoding:** One-hot encoded (only one city is 1, others are 0)

Example for Erick: `[0.33, 1, 0, 0, 1, 0, 0]`
- Age: 0.33 (normalized from 30)
- Color: Blue (1, 0, 0)
- Location: São Paulo (1, 0, 0)

**Output Labels (`tensorLabels`):**

Each row represents the service tier using one-hot encoding:
```
[premium, medium, basic]
```

- Erick → `[1, 0, 0]` (Premium)
- Ana → `[0, 1, 0]` (Medium)
- Carlos → `[0, 0, 1]` (Basic)

### Neural Network Architecture

**Model Creation:**
```javascript
const model = tf.sequential();
```
Creates a sequential model where layers are stacked linearly.

**Hidden Layer:**
```javascript
model.add(tf.layers.dense({ units: 80, inputShape: [7], activation: "relu" }));
```
- **units: 80** - Creates 80 neurons in this layer
- **inputShape: [7]** - Expects 7 input features (age + 3 colors + 3 cities)
- **activation: "relu"** - ReLU (Rectified Linear Unit) activation function that outputs only positive values, helping the network learn complex patterns

**Output Layer:**
```javascript
model.add(tf.layers.dense({ units: 3, activation: "softmax" }));
```
- **units: 3** - Three neurons for three categories (Premium, Medium, Basic)
- **activation: "softmax"** - Converts outputs to probabilities that sum to 1, making it perfect for multi-class classification

### Model Compilation

```javascript
model.compile({
  optimizer: "adam",
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});
```

- **optimizer: "adam"** - Adam (Adaptive Moment Estimation) algorithm that adjusts learning rates automatically based on training history, making training more efficient
- **loss: "categoricalCrossentropy"** - Loss function designed for multi-class classification problems with one-hot encoded labels
- **metrics: ["accuracy"]** - Tracks the percentage of correct predictions during training

### Model Training

```javascript
await model.fit(inputXs, outputYs, {
  epochs: 100,
  batchSize: 10,
  shuffle: true,
  verbose: 0,
  callbacks: {
    onEpochEnd: (epoch, logs) =>
      console.log(`Epoch: ${epoch} - loss: ${logs.loss} - accuracy: ${logs.accuracy}`),
  },
});
```

- **epochs: 100** - The model will see the entire dataset 100 times
- **batchSize: 10** - Number of samples processed before updating model weights (though our dataset only has 3 samples)
- **shuffle: true** - Randomizes the order of training samples each epoch to prevent learning order-dependent patterns
- **verbose: 0** - Suppresses default training output (we use custom callback instead)
- **callbacks** - Custom function that prints loss and accuracy after each epoch, allowing us to monitor training progress

### Training Process

During each epoch:
1. The model makes predictions on the input data
2. Calculates the loss (how wrong the predictions are)
3. Uses backpropagation to adjust neuron weights
4. Repeats for 100 epochs, gradually improving accuracy

The console output shows:
- **Epoch number** - Current training iteration
- **Loss** - Error metric (lower is better)
- **Accuracy** - Percentage of correct predictions (higher is better)
