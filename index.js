import tf from "@tensorflow/tfjs-node";

const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0],
  [0, 0, 1, 0, 0, 1, 0],
  [1, 0, 0, 1, 0, 0, 1],
];

const labelsNomes = ["premium", "medium", "basic"];
const tensorLabels = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

const models = trainModel(inputXs, outputYs);

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({ units: 80, inputShape: [7], activation: "relu" }),
  );

  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  await model.fit(inputXs, outputYs, {
    epochs: 100,
    batchSize: 10,
    shuffle: true,
    verbose: 0,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        console.log(
          `Epoch: ${epoch} - loss: ${logs.loss} - accuracy: ${logs.accuracy}`,
        ),
    },
  });

  return model;
}
