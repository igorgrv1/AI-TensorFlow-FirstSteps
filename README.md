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
   npm install @tensorflow/tfjs-node
