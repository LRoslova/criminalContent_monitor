const tf = require('@tensorflow/tfjs');
// import * as tf from '@tensorflow/tfjs';
// подключение express
const express = require("express");
const bodyParser = require('body-parser');
canvas = require('canvas')


// создаем объект приложения
const app = express();
const port = 3000;
let class_names = {
    0: "alcohol",
    1: "drugs",
    2: "ordinary",
    3: "porn",
    4: "weapon"
}
let img_height = 180
let img_width = 180
const IMAGE_SIZE = 180;

// The minimum image size to consider classifying.  Below this limit the
// extension will refuse to classify the image.
const MIN_IMG_SIZE = 128;


app.use(bodyParser.json() );
app.use(bodyParser.urlencoded({ extended: true })); 
var cors = require('cors')
app.use(cors()) // Use this after the variable declaration
const model = tf.loadLayersModel('/models/5category_180MNV2/model.json');



// определяем обработчик для маршрута "/"
app.get("/", function(request, response){
    
    response.send({msg: "rabotaet!"});
});

app.post("/", function(request, response){
    console.log(request.body.data);
    // let src = request.body.data[0].result[0]
    let src = 'https://s15.stc.all.kpcdn.net/family/wp-content/uploads/2023/02/top-v-luchshie-porody-krupnykh-sobak-960x540-1-560x420.jpg'
    console.log(src);

    

    




    response.send({msg: "rabotaet!"});
});

// начинаем прослушивать подключения на 3000 порту
// app.listen(3000);
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
  });


