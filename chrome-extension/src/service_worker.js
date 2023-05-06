/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';


// Size of the image expected by mobilenet.
const IMAGE_SIZE = 180;

// How many predictions to take.
const TOPK_PREDICTIONS = 2;
const FIVE_SECONDS_IN_MS = 5000;
/**
 * What action to take when someone clicks the right-click menu option.
 * Here it takes the url of the right-clicked image and the current tabId, and
 * send them to the content script where the ImageData will be retrieved and
 * sent back here. After that, imageClassifier's analyzeImage method.
 */
function clickMenuCallback(info, tab) {
  const message = { action: 'IMAGE_CLICKED', url: info.srcUrl };
  chrome.tabs.sendMessage(tab.id, message, (resp) => {
    if (!resp.rawImageData) {
      console.error(
        'Failed to get image data. ' +
        'The image might be too small or failed to load. ' +
        'See console logs for errors.');
      return;
    }
    const imageData = new ImageData(
      Uint8ClampedArray.from(resp.rawImageData), resp.width, resp.height);
    imageClassifier.analyzeImage(imageData, info.srcUrl, tab.id);
  });
}

/**
 * Adds a right-click menu option to trigger classifying the image.
 * The menu option should only appear when right-clicking an image.
 */
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'contextMenu0',
    title: 'Анализировать изображение',
    contexts: ['image'],
  });
});

chrome.contextMenus.onClicked.addListener(clickMenuCallback);

/**
 * Async loads a mobilenet on construction.  Subsequently handles
 * requests to classify images through the .analyzeImage API.
 * Successful requests will post a chrome message with
 * 'IMAGE_CLICK_PROCESSED' action, which the content.js can
 * hear and use to manipulate the DOM.
 */
class ImageClassifier {
  constructor() {
    this.loadModel();
  }

  /**
   * Loads mobilenet from URL and keeps a reference to it in the object.
   */
  async loadModel() {
    console.log('Loading model...');
    const startTime = performance.now();
    try {
      this.model = await tf.loadLayersModel('http://localhost:3000//5category_180MNV2_2/model.json');
      // Warms up the model by causing intermediate tensor values
      // to be built and pushed to GPU.
      tf.tidy(() => {
        this.model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));
      });
      const totalTime = Math.floor(performance.now() - startTime);
      console.log(`Model loaded and initialized in ${totalTime} ms...`);
    } catch (e) {
      console.error('Unable to load model', e);
    }
  }

  /**
   * Triggers the model to make a prediction on the image referenced by the
   * image data. After a successful prediction a IMAGE_CLICK_PROCESSED message
   * when complete, for the content.js script to hear and update the DOM with
   * the results of the prediction.
   *
   * @param {ImageData} imageData ImageData of the image to analyze.
   * @param {string} url url of image to analyze.
   * @param {number} tabId which tab the request comes from.
   */
  async analyzeImage(imageData, url, tabId) {
    if (!tabId) {
      console.error('No tab.  No prediction.');
      return;
    }
    if (!this.model) {
      console.log('Waiting for model to load...');
      setTimeout(
        () => { this.analyzeImage(imageData, url, tabId) }, FIVE_SECONDS_IN_MS);
      return;
    }
    console.log('Predicting...');
    const startTime = performance.now();
    const input = imageToInput(imageData, 3)
    // const input = tf.tensor3d(imageData, [IMAGE_SIZE, IMAGE_SIZE, 3], 'int32');
    const example = tf.browser.fromPixels(imageData);
    // console.log(input);
    // console.log(example);  // for example
    // model.predict(tensor.reshape([1,224,224,3]))
    // const prediction = model.predict(example);
    // const predictions = await this.model.predict(input);
    let data2= this.model.predict(example.reshape([1,180,180,3]))
    // .data().then(data => {
    //   data2 = data
    //   console.log(data)});
    let data3
    data2= await tf.softmax(data2).data().then(data=>{
      data3 = data
      console.log(data);
    })
    let arr = [];
    for (let i=0; i<data3.length; i++) arr[i] = (data3[i]*100)/100;
    console.log(arr);
    console.log(Math.max.apply(null, arr));
    let maxIndex = await arr.indexOf(Math.max.apply(null, arr));
    let percent = Math.max.apply(null, arr);
    console.log(maxIndex);
    console.log(percent);
    let predictions = {index: maxIndex, percent: percent}
    // console.log(predictions);
    // console.log(predictions2);

    const totalTime = performance.now() - startTime;
    console.log(`Done in ${totalTime.toFixed(1)} ms `);
    const message = { action: 'IMAGE_CLICK_PROCESSED', url, predictions};
    chrome.tabs.sendMessage(tabId, message);
  }
}
const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels)
  const outShape = [1, image.height, image.width, numChannels];
  const input = tf.tensor4d(values, outShape, 'int32');


  return input
}
const imageByteArray = (image, numChannels) => {
  const pixels = image.data
  const numPixels = image.width * image.height;
  const values = new Int32Array(numPixels * numChannels);


  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel];
    }
  }


  return values
}

const imageClassifier = new ImageClassifier();