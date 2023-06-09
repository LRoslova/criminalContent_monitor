import time
from selenium import webdriver
from PIL import Image 
from selenium.webdriver.common.by import By
import os
import hashlib
import requests
import io


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)

    # составляем поисковой запрос
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"
    # загружаем страницу
    wd.get(search_url.format(q=query))
    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)
        # собираем все полученные результаты
        thumbnail_results = wd.find_elements(By.CSS_SELECTOR, "img.rg_i")
        number_results = len(thumbnail_results)
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        for img in thumbnail_results[results_start:number_results]:
            # кликаем по предпросмотрам для получения полных изображений
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue
            # собираем ссылки на изображения
            actual_images = wd.find_elements(By.CSS_SELECTOR, 'img.r48jcc')
            for actual_image in actual_images:
                if actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))
        image_count = len(image_urls)
        if len(image_urls) >= max_links_to_fetch:
            print(f"Found: {len(image_urls)} image links, done!")
            break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(1)
            load_more_button = wd.find_element(By.CSS_SELECTOR, ".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")
                # двигаемся дальше
                results_start = len(thumbnail_results)
    return image_urls

def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content
    except Exception as e:
        print(f"ERROR — Could not download {url} — {e}")
    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
            print(f"SUCCESS — saved {url} — as {file_path}")
    except Exception as e:
        print(f"ERROR — Could not save {url} — {e}")

def search_and_download(search_term:str, target_path='./images', number_images=500):
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    with webdriver.Chrome() as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
        for elem in res:
            persist_image(target_folder,elem)

# search_and_download(search_term="девушка")
# search_and_download(search_term="мужчина")

# search_and_download(search_term="машина")
# search_and_download(search_term="домашнее животное")

# search_and_download(search_term="корабль")
# search_and_download(search_term="одежда")

# search_and_download(search_term="фрукт")
# search_and_download(search_term="овощ")

# search_and_download(search_term="лес")
# search_and_download(search_term="озеро")

# search_and_download(search_term="сок")
# search_and_download(search_term="птицы")

# search_and_download(search_term="посуда")
# search_and_download(search_term="дом")

# search_and_download(search_term="самолет")
search_and_download(search_term="документы")

search_and_download(search_term="бытовая техника")
search_and_download(search_term="рыба")





