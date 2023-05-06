import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CIFAR-10 classes
class_names = {
    0: "alcohol",
    1: "drugs",
    2: "ordinary",
    3: "porn",
    4: "weapon"
}

img_height = 180
img_width = 180

model = keras.models.load_model('mymodel-v4.h5')


img_arr = [
    "https://static.insales-cdn.com/files/1/3228/11226268/original/mceclip0-1580992808851.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/b3/Hipop%C3%B3tamo_%28Hippopotamus_amphibius%29%2C_parque_nacional_de_Chobe%2C_Botsuana%2C_2018-07-28%2C_DD_82.jpg",
    "https://images.thevoicemag.ru/upload/img_cache/1c8/1c83f6f695855deceec54ebd48a3d40b_cropped_666x833.jpg",
    "https://img.championat.com/c/900x900/news/big/p/x/neveroyatno-krasivaya-devushka-podpischiki-ocenili-novoe-foto-gimnastki-sevastyanovoj_1640409929144929227.jpg",
    "https://birdinflight.com/wp-content/uploads/2018/12/lsdmdma.jpg",
    "https://static.ogorodniki.com/ogorod/af/l_1f284dde-9c23-4116-8769-7cfb35328aaf.jpg"
]

for i in range(len(img_arr)):
    test_url = img_arr[i]
    file_name = 'test_{}.txt'.format(i)
    test_path = tf.keras.utils.get_file(file_name, origin=test_url)
    img = tf.keras.utils.load_img(
        test_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "{} with a {:.2f} percent \n"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )