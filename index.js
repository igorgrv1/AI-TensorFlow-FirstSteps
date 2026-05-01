import tf from "@tensorflow/tfjs-node";

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

async function predict(model, personTensor) {
  const tfInput = tf.tensor2d(personTensor);
  const prediction = await model.predict(tfInput).array();
  return prediction[0].map((prob, index) => ({ prob, index }));
}

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

const model = await trainModel(inputXs, outputYs);

const newPerson = {
  nome: "Igor",
  idade: 30,
  cor: "vermelho",
  localizacao: "Sao Paulo",
};
const newPersonTensor = [
  [
    0.2, // idade normalizada
    0, // cor azul
    0, // cor vermelho
    1, // cor verde
    1, // localização São Paulo
    0, // localização Rio
    0, // localização Curitiba
  ],
];

const prediction = await predict(model, newPersonTensor);
const results = prediction
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`);
console.log(results);
