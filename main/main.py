from tensorflow.keras import utils
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential


def main():
    # данные
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # преобразование размерности изображений
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    # нормализация данных
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # преобразуем метки в категории
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    model = Sequential()

    # слои
    # model.add(Dense(800, input_dim=784, activation="relu"))
    # model.add(Dense(10, activation="softmax"))
    # model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
    # model.add(Conv2D(32, kernel_size=3, activation="relu"))
    # model.add(Flatten())
    # model.add(Dense(10, activation="softmax"))
    model.add(
        Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # Компилируем модель
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    print(model.summary())

    # Обучаем сеть
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=20,
                        verbose=1)

    # Оцениваем качество обучения сети на тестовых данных
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))


if __name__ == '__main__':
    main()
