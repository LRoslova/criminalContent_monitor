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
 * Выполняет функцию analysText() на веб-странице указанной
 * вкладки и во всех ее фреймах,
 * @param tab {Tab} Объект вкладки браузера
 */
function execScript(tab) {
    // Выполнить функцию на странице указанной вкладки
    // и передать результат ее выполнения в функцию onResult
    chrome.scripting.executeScript(
        {
            target:{tabId: tab.id, allFrames: true},
            func:analysText
        },
        // onResult
    )
}

/**
 * Функция исполняется на удаленной странице браузера,
 * получает список изображений и возвращает массив
 * путей к ним
 * 
 *  @return Array массив строк
 */
function analysText() {
    // стягиваем весь текст с открытой страницы браузера
    let text= document.documentElement.innerText;
    console.log(text);
    return text;    
}
