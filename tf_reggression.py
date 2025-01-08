"""
1. Загрузка необходимых библиотек.
"""
import tensorflow as tf
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np


def generate_dataset():
    """
    2. Создание обучающего датасета. Т.к tensorflow.keras.dataset не имеет, явно выраженных,
       датасетов для регрессии, прибегнем к самостоятельному созданию датасета. В качестве примера создадим датасет для
       задачи многомерной регрессии, где будет создаваться зависимость итоговой оценки, по условному предмету,
       от времени  затраченного на изучение предмета, уровня начальной подготовки и взаимодействия этих двух признаков.
     2.1. Создаем два признака x1 - часы подготовки (от 0 до 20 часов); x2 - уровень нач. подготовки (от 0 до 10).
     2.2. Создание целевой переменной y - итоговая оценка (макс. 100 баллов).
     P. S. Вносим случайный шум как иммитацию внешних факторов,
           влияющих на изучение предмета (настроение, болезнь и т.п.). Ограничиваем итоговую оценку от 0 до 100
     2.3. Объединяем признаки.
    """
    x1 = np.random.rand(1000) * 20
    x2 = np.random.rand(1000) * 10
    y = (5 * x1 + 8 * x2 + 0.5 * x1 * x2) + np.random.rand(1000) * 10
    y = np.clip(y, 0, 100)
    x = np.column_stack((x1, x2))
    return x, y


def split_dataset(x, y):
    """
    3. Разделение данных на обучающую и тестовую выборки. В данном примере, воспользуемся ручным разделением выборок
       с помощью срезов массивов NumPy.
     3.1. Выбираем индексы для тестовой выборки.
     3.2. Создаем маски для тестовой и обучающей выборок.
     3.3. Разделяем данные с помощью созданных масок.
     """
    dataset_size = x.shape[0]
    test_size = int(dataset_size * 0.2)
    test_indices = np.random.choice(dataset_size, size=test_size, replace=False)
    test_mask = np.zeros(dataset_size, dtype=bool)
    test_mask[test_indices] = True
    train_mask = np.logical_not(test_mask)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]
    return x_train, y_train, x_test, y_test


def create_and_train_model(x_train, y_train):
    """
    4. Создание, компиляция и обучение модели. Модель имеет три слоя - входной, скрытый и выходной слой.
    Отмечу, что выходной слой имеет один нейрон, что нужно для регрессии.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[x_train.shape[1]]),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.metrics.MeanAbsoluteError])
    model.fit(x_train, y_train, epochs=100, verbose=0)
    return model


def evaluate_model(model, x_test, y_test):
    """
    5. Оценка качества модели c использованием метрик MSE, MAE и R2.
    """
    results = model.evaluate(x_test, y_test, verbose=0)
    loss = results[0]
    mae = results[1]
    print(f"Test MSE: {loss}")
    print(f"Test MAE: {mae}")
    y_pred = model.predict(x_test).flatten()
    r2 = r2_score(y_test, y_pred)
    print(f"Test R2: {r2}")
    return y_pred, y_test


def visualize_results(y_test, y_pred, x_test):
    """
    5.2. Выводим визуализацию результатов.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title('График предсказанных значений от фактических')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_test[:, 0], y_test, color='blue', label='Фактические', alpha=0.6)
    plt.scatter(x_test[:, 0], y_pred, color='red', label='Предсказания', alpha=0.6)
    plt.xlabel('Часы подготовки (X1)')
    plt.ylabel('Итоговая оценка (y)')
    plt.title('Предсказания по часам подготовки')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(x_test[:, 1], y_test, color='blue', label='Фактические', alpha=0.6)
    plt.scatter(x_test[:, 1], y_pred, color='red', label='Предсказания', alpha=0.6)
    plt.xlabel('Начальный уровень подготовки (X2)')
    plt.ylabel('Итоговая оценка (y)')
    plt.title('Предсказания по уровню подготовки')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    6. Основная функция, которая выполняет весь пайплайн.
    """
    x, y = generate_dataset()
    x_train, y_train, x_test, y_test = split_dataset(x, y)
    model = create_and_train_model(x_train, y_train)
    y_pred, y_test = evaluate_model(model, x_test, y_test)
    visualize_results(y_test, y_pred, x_test)


if __name__ == '__main__':
    main()
