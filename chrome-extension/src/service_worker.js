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

const FIVE_SECONDS_IN_MS = 5000;

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
    
    analyzeImage(resp.rawImageData, info.srcUrl, tab.id);
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



async function analyzeImage(imageData, url, tabId) {
   
  console.log('Преобразуем изображение к нужному входному формату...');
  const startTime3 = performance.now();
  const response3 = await fetch("http://localhost:3000/parse_img", {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(imageData)
  });
  const jsonData = await response3.json();
  const imgInput = jsonData.data
  console.log(imgInput);
  const totalTime3 = Math.floor(performance.now() - startTime3);
  console.log(`Картинка преобразована за ${totalTime3} ms...`);


  console.log('Подключение к модели...');
  const startTime2 = performance.now();
  const response = await fetch("http://localhost:3000/initModel");
  const res = await response.json()
  if (!tabId) {
    console.error('No tab.  No prediction.');
    return;
  }
  if (!res.code) {
    console.log('Waiting for model to load...');
    setTimeout(
      () => { this.analyzeImage( url, tabId) }, FIVE_SECONDS_IN_MS);
    return;
  }
  const totalTime2 = Math.floor(performance.now() - startTime2);
  console.log(`Подключились за ${totalTime2} ms...`);


  console.log('Классифицируем...');
  const startTime = performance.now();
  const responseAnalyse = await fetch("http://localhost:3000/predict");
  const predictions = await responseAnalyse.json()
  console.log(predictions);
 
  
  const totalTime = performance.now() - startTime;
  console.log(`Анализ завершен за ${totalTime.toFixed(1)} ms `);

  const message = { action: 'IMAGE_CLICK_PROCESSED', url, predictions};
  chrome.tabs.sendMessage(tabId, message);
}
