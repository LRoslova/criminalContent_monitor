const tf = require('@tensorflow/tfjs');

// подключение express
const express = require("express");
// создаем объект приложения
const app = express();
// определяем обработчик для маршрута "/"
app.get("/", function(request, response){
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
    let tmp;
    // Train the model using the data.
    model.fit(xs, ys, {epochs: 10}).then(() => {
    // Use the model to do inference on a data point the model hasn't seen before:
        tmp = model.predict(tf.tensor2d([5], [1, 1]));
        model.predict(tf.tensor2d([5], [1, 1])).print();
    // Open the browser devtools to see the output
    });
     
    // отправляем ответ
    response.send(`<h2>Привет Express!</h2>`);
});
// начинаем прослушивать подключения на 3000 порту
app.listen(3000);