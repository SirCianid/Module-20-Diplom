"""
1. Загрузка необходимых библиотек.
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


def load_and_preprocess_data():
    """
    2. Загрузка датасета MNIST для задачи классификации, определение обучающей и тестовой выборок.

    3. Предобработка данных. Метод astype - нормализует данные к диапазону [0, 1].
       Метод reshape - разворачивает изображения в вектора.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28 * 28))
    x_test = x_test.reshape((-1, 28 * 28))
    return x_train, y_train, x_test, y_test


def create_and_compile_model():
    """
    4. Создание модели с тремя слоями (входным, скрытым и выходным).  Выходной слой имеет 10 нейронов,
    что соответсвует 10 классам в используемом датасете. Функция активации ReLU - добавляет нелинейность,
    позволяя модели учиться более сложным зависимостям. Функция активации Softmax - преобразует выходные значения
    в вероятности, сумма которых равна 1, что удобно для задачи классификации.

    5. Компиляция модели (настройка перед обучением). Оптимизатор Adam - определяет,
    как обновлять веса модели во время обучения. Функция потерь - sparse categorical crossentropy,
    подходит для задач многоклассовой классификации, когда метки классов представленны целыми числами (что мы и имеем).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28 * 28,)),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train):
    """
    6. Обучение модели
    """
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    return model


def evaluate_model(model, x_test, y_test):
    """
    7. Оценка качества модели.
        7.1 Дополнительная оценка. Получение вероятностей для каждого класса.
        7.2. Получение отчета о классификации.
        7.3. Создание матрицы ошибок, и ее визуализация.
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", conf_matrix)

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title('Матрица ошибок')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)])
    plt.yticks(tick_marks, [str(i) for i in range(10)])
    plt.xlabel('Прогнозируемые значения')
    plt.ylabel('Реальные значения')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(model, x_test, y_test):
    """
    7.4. Рассчитываем ROC-кривую и AUC для каждого класса.
    А также строим график ROC-кривых для всех классов.
    """
    y_pred_probs = model.predict(x_test, verbose=0)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-Кривая для каждого класса')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    8. Основная функция, которая выполняет весь пайплайн.
    """
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = create_and_compile_model()
    model = train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    plot_roc_curves(model, x_test, y_test)


if __name__ == "__main__":
    main()
