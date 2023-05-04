const analysBtn = document.getElementById("analysBtn");
analysBtn.addEventListener("click",() => {    
    // Получить активную вкладку браузера
    chrome.tabs.query({active: true}, function(tabs) {
        var tab = tabs[0];
        // и если она есть, то выполнить на ней скрипт
        if (tab) {
            execScript(tab);
        } else {
            alert("Нет активного окна браузера")
        }
    })
})
/**
 * Выполняет функцию grabImages() на веб-странице указанной
 * вкладки и во всех ее фреймах,
 * @param tab {Tab} Объект вкладки браузера
 */
function execScript(tab) {
    // Выполнить функцию на странице указанной вкладки
    // и передать результат ее выполнения в функцию onResult
    chrome.scripting.executeScript(
        {
            target:{tabId: tab.id, allFrames: true},
            func:grabImages
        },
        onResult
    )
}

/**
 * Получает список абсолютных путей всех картинок
 * на удаленной странице
 * 
 *  @return Array Массив URL
 */
function grabImages() {
    const images = document.querySelectorAll("img");
    let tmp = Array.from(images).map(image=>image.src);
    // let height = Array.from(images).map(image=>image.height);
    // let width = Array.from(images).map(image=>{height: image.height, width: image.width});
    console.log(tmp);
    // console.log(height[0]);
    // console.log(width[0]);
    return tmp
      
}
let img_height = 180
let img_width = 180
const IMAGE_SIZE = 180;

// The minimum image size to consider classifying.  Below this limit the
// extension will refuse to classify the image.
const MIN_IMG_SIZE = 128;
/**
 * Выполняется после того как вызовы grabImages 
 * выполнены во всех фреймах удаленной web-страницы.
 * Функция объединяет результаты в строку и копирует  
 * список путей к изображениям в буфер обмена
 * 
 * @param {[]InjectionResult} frames Массив результатов
 * функции grabImages
 */
function onResult(frames) {
    // Если результатов нет
    // if (!frames || !frames.length) { 
    //     alert("Could not retrieve images from specified page");
    //     return;
    // }
    // // Объединить списки URL из каждого фрейма в один массив
    // const imageUrls = frames.map(frame=>frame.result)
    //                         .reduce((r1,r2)=>r1.concat(r2));
    // // Скопировать в буфер обмена полученный массив  
    // // объединив его в строку, используя возврат каретки 
    // // как разделитель  
    // console.log('hi');
    // window.navigator.clipboard
    //       .writeText(imageUrls.join("\n"))
    //       .then(()=>{
    //          // закрыть окно расширения после 
    //          // завершения
    //          window.close();
    //       });
    let src = 'https://habrastorage.org/r/w1560/webt/vt/sm/-2/vtsm-2o5y-t3vhqx5-qaomhxnfw.jpeg'
    let response;
    const img = new Image();
    img.src = src;
    // return response
    img.crossOrigin = 'anonymous';
    img.onload = function(e) {
      if ((img.height && img.height > MIN_IMG_SIZE) ||
          (img.width && img.width > MIN_IMG_SIZE)) {
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        // When image is loaded, render it to a canvas and send its ImageData back
        // to the service worker.
        const canvas = new OffscreenCanvas(img.width, img.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        response = {
          rawImageData: Array.from(imageData.data),
          width: img.width,
          height: img.height,
        };
        // return response;
      }
      // Fail out if either dimension is less than MIN_IMG_SIZE.
      alert(`Image size too small. [${img.height} x ${
          img.width}] vs. minimum [${MIN_IMG_SIZE} x ${MIN_IMG_SIZE}]`);
        response = {rawImageData: undefined};
    };
    img.onerror = function(e) {
      alert(`Could not load image from external source ${src}.`);
      response = {rawImageData: undefined};
    };
    

    let xhr = new XMLHttpRequest();
    let json = JSON.stringify({
    data: response
    });
    xhr.open("POST", 'http://localhost:3000/')
    xhr.setRequestHeader('Content-type', 'application/json');
    xhr.send(json)
    // тело ответа {"сообщение": "Привет, мир!"}
    xhr.onload = function() {
        let responseObj = xhr.response;
        alert(responseObj);
        // console.log(responseObj);
        // console.log(responseObj.msg); // Привет, мир!
    };
}

function loadImageAndSendDataBack(src) {
    // Load image (with crossOrigin set to anonymouse so that it can be used in a
    // canvas later).
    
  }