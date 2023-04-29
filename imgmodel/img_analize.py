from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.layers import Reshape
from keras import backend as K
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Гиперпараметры
batch_size = 64
# 10 категорий для изображений  (CIFAR-10)
num_classes = 3
# количество эпох для обучения
epochs = 10

img_height = 180
img_width = 180
print("")
print("")
if K.image_data_format() != 'channels_first':
     input_shape = (3, 300, 300)
else:
     input_shape = (3, 300, 300) 

def load_data():
    """
    Эта функция загружает набор данных CIFAR-10 dataset и делает предварительную обработку
    """
    def preprocess_image(image, label):
        # преобразуем целочисленный диапазон [0, 255] в диапазон действительных чисел [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    
    url = 'https://vk.com/doc193464385_660941651?hash=P3vMn1oXuo6ZghGYclzzVaalNbt1tI9OgjWK9Wa0y2P&dl=H6zeXLu9gzdAwwBcpNCJiiMmGvohQdUqZRF0sLWxxB8'
    dataset = tf.keras.utils.get_file('images.tar', url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'images')
    train_dir = os.path.join(dataset_dir, 'train')
    print("")
    print("")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'images/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=123)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'images/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=123)
    
    print("")
    print("")
    
    # test_ds = tf.keras.utils.text_dataset_from_directory(
    #     'aclImdb/test',
    #     batch_size=batch_size)

    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    train_size = train_ds.cardinality().numpy()
    val_size = val_ds.cardinality().numpy()

    # # загружаем набор данных CIFAR-10, разделяем его на обучающий и тестовый
    # ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    # ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    # # повторять набор данных, перемешивая, предварительно обрабатывая, разделяем по пакетам
    train_ds = train_ds.repeat().shuffle(1024).batch(batch_size)
    val_ds = val_ds.repeat().shuffle(1024).batch(batch_size)
    return train_ds, val_ds, train_size, val_size

def create_model():
    # построение модели
    model = Sequential()
    # model.add(Reshape((224,224))),
    model.add(Resizing(300,300)),
    model.add(Rescaling(1./255)),
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # сглаживание неровностей
    model.add(Flatten())
    # полносвязный слой
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # печатаем итоговую архитектуру модели
    # model.summary()
    # обучение модели с помощью оптимизатора Адама
    model.compile(
                    optimizer='adam',
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # загружаем данные
    ds_train, ds_test, train_size, val_size = load_data()
    # конструируем модель
    model = create_model()
    # несколько хороших обратных вызовов
    # logdir = os.path.join("logs", "cifar10-model-v1")
    # tensorboard = TensorBoard(log_dir=logdir)
    # убедимся, что папка с результатами существует
    if not os.path.isdir("results"):
        os.mkdir("results")
    # обучаем
    model.fit(ds_train, 
              epochs=epochs, 
              validation_data=ds_test, 
              steps_per_epoch=train_size // batch_size,
              validation_steps=val_size // batch_size)
    # сохраняем модель на диске
    model.save("results/cifar10-model-v1.h5")