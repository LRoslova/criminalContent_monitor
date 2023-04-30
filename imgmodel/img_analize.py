import numpy as np
import os
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import pathlib
dataset_url2 = "https://vk.com/doc193464385_661088124?hash=K1cSG8Amb64pKKvlsZUSlNxOCpzBQq5KyMMjSh9NhAP&dl=gW3E15a3cPOiXwGSqHjXSto8JYIu4VrBpJVauEzvtuL"
data_dir = tf.keras.utils.get_file('images', origin=dataset_url2, untar=True)
data_dir = pathlib.Path(data_dir)
# cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
# print(os.listdir(cache_dir))
# remove_dir = os.path.join(train_dir, 'unsup')
# shutil.rmtree(remove_dir)
# sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
# with open(sample_file) as f:
#   print(f.read())

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# alcohol = list(data_dir.glob('alcohol/*'))
# PIL.Image.open(str(alcohol[0]))
# PIL.Image.open(str(alcohol[1]))

batch_size = 96
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)



# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs=25
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)
model.save("mymodel-v2.h5")
model.save("results/mymodel-v2.h5")

test_url = "https://www.agora-group.ru/assets/images/resources/195/2020.png"
test_path = tf.keras.utils.get_file('wine', origin=test_url)

img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

test2_url = "https://s14.stc.all.kpcdn.net/family/wp-content/uploads/2023/02/top-v-luchshie-porody-krupnykh-sobak-960x540-1-560x420.jpg"
test2_path = tf.keras.utils.get_file('goodboy', origin=test2_url)

img = tf.keras.utils.load_img(
    test2_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


test3_url = "https://apikabu.ru/img_n/2012-04_1/jyf.jpg"
test3_path = tf.keras.utils.get_file('kalash', origin=test3_url)

img = tf.keras.utils.load_img(
    test3_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

test4_url = "https://probirkalab.ru/wp-content/uploads/2020/05/%D0%A8%D0%BF%D1%80%D0%B8%D1%86-%D0%BE%D0%B4%D0%BD%D0%BE%D1%80%D0%B0%D0%B7%D0%BE%D0%B2%D1%8B%D0%B9-50-%D0%BC%D0%BB-%D1%81-%D0%B8%D0%B3%D0%BB%D0%BE%D0%B9-12x40-%D0%BC%D0%BC-18Gx1-12-LUER-Binano.jpg"
test4_path = tf.keras.utils.get_file('shpric', origin=test3_url)

img = tf.keras.utils.load_img(
    test4_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


test5_url = "https://img.the-steppe.com/images/news/7374-LQUJNJ5E.jpg"
test5_path = tf.keras.utils.get_file('eda', origin=test3_url)

img = tf.keras.utils.load_img(
    test5_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

