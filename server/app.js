const tf = require('@tensorflow/tfjs');
// import * as tf from '@tensorflow/tfjs';
// подключение express
const express = require("express");
const bodyParser = require('body-parser');


// создаем объект приложения
const app = express();
const port = 3000;

const IMAGE_SIZE = 180;


app.use(bodyParser.json({limit: "50mb"}));
app.use(bodyParser.urlencoded({limit: "50mb", extended: true, parameterLimit:50000}));
app.use(express.static(__dirname + "/models"));
app.use(bodyParser.urlencoded({ extended: true })); 
var cors = require('cors')
app.use(cors()) // Use this after the variable declaration

let example;
let model_tf;

app.get("/", function(request, response){
    
    response.send({msg: "rabotaet!"});
});
app.post("/parse_img", function(request, response){
    let tmp = request.body
    tmp = Uint8ClampedArray.from(tmp)
    console.log(tmp);
    const numChannels = 3;
    const numPixels = 180 * 180;
    const values = new Int32Array(numPixels * numChannels);

    for (let i = 0; i < numPixels; i++) {
     for (let channel = 0; channel < numChannels; ++channel) {
        values[i * numChannels + channel] = tmp[i * 4 + channel];
     }
    }
    example = tf.tensor3d(values, [180,180,3])
    
    // example = tmp
    response.send({data: "картинка обработана!"});
});

app.get("/predict", function(request, response){
    async function classificImg(){
        let tmp = await imageClassifier.analyzeImage(example)
        // answer = {percent: tmp.percent, index: tmp.index}
        console.log(tmp);
        response.send(tmp);
    }
    
    classificImg()
    
});

app.get("/initModel", function(request, response){
    let answer;
    const model = imageClassifier.loadModel();
    if(!model){
        answer = {code: false}
        
    }else{
        model_tf = model
        answer = {code: true}
    }
    response.send(answer);
});



// начинаем прослушивать подключения на 3000 порту
// app.listen(3000);
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
  });





  class ImageClassifier {
    constructor() {
      this.loadModel();
    }
  
    async loadModel() {
      try {
        this.model = await tf.loadLayersModel('http://localhost:3000//5category_ENV2M/model.json');
        // Warms up the model by causing intermediate tensor values
        // to be built and pushed to GPU.
        tf.tidy(() => {
          this.model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));
        });
        return true
      } catch (e) {
        return false
        console.error('Unable to load model', e);
      }
    }
  
    async analyzeImage(img) {
      let data2= await this.model.predict(img.reshape([1,180,180,3]))
      let data3
      data2= await tf.softmax(data2).data().then(data=>{
        data3 = data
      })
      console.log(data3);
      let arr = [];
      for (let i=0; i < data3.length; i++) arr[i] = (data3[i]*100)/100;
      console.log(arr);
      let maxIndex = await arr.indexOf(Math.max.apply(null, arr));
      let percent = Math.max.apply(null, arr);

      let predictions = {index: maxIndex, percent: percent}
      return predictions
    }
  }
  
  const imageClassifier = new ImageClassifier();