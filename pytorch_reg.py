"""
1. Загрузка необходимых библиотек.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def generate_and_split_data():
    """
    2. Генерируем данные для обучения. Для более удобного понимания, модель будет предсказывать стоимость
    квартиры основываясь на ее площади.
    3. Разделяем данные на обучающую и тестовую выборки.
    """
    np.random.seed(42)
    num_samples = 200
    house_sizes = np.random.uniform(75, 250, num_samples).reshape(-1, 1)
    usd_to_rub = 104
    house_prices = (640 * house_sizes + 7640 + np.random.normal(0, 10000, num_samples).reshape(-1, 1)) * usd_to_rub
    x_train, x_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def scale_and_convert_data(x_train, x_test, y_train, y_test):
    """
    4. Производим масштабирование входных и выходных данных.
     4.1. Преобразуем массивы NumPy в тензоры PyTorch.
    """
    scaler_x = StandardScaler()
    x_train_scaled = scaler_x.fit_transform(x_train)
    x_test_scaled = scaler_x.transform(x_test)
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    x_train = torch.from_numpy(x_train_scaled).float()
    x_test = torch.from_numpy(x_test_scaled).float()
    y_train = torch.from_numpy(y_train_scaled).float()
    y_test = torch.from_numpy(y_test_scaled).float()
    return x_train, x_test, y_train, y_test, scaler_x, scaler_y


def create_and_train_model(x_train, y_train):
    """
    5. Определение модели. В PyTorch модели определяются как, так называемые, классовые модули наследуемые от базового
    класса всех нейросетевых слоев и моделей - nn.Module.
    В функции __init__ - оределяем линейный слой который и будет обучаться.
    В функции forward - метод, который определяет как данные будут обрабатываться. Задаем просто линейный слой.
    6. Задаем функцию потерь и оптимизатора.
    7. Запускаем обучение модели.
    """

    class HousePriceRegression(nn.Module):
        def __init__(self):
            super(HousePriceRegression, self).__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = HousePriceRegression()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 2000
    for epoch in range(num_epochs):
        y_predicted = model(x_train)
        loss = criterion(y_predicted, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
    return model


def evaluate_model(model, x_test, y_test, scaler_y, scaler_x, x_train, y_train):
    """
    8. Оценка модели и визуализация результатов.
     8.1. Переводим предсказанные данные в оригинальный масштаб.
     8.4. Рассчитываем метрики и выводим результаты.
     8.3. Визуализация результатов работы модели.
    """
    with torch.no_grad():
        y_predicted_test = model(x_test)

    y_predicted_test_orig = scaler_y.inverse_transform(y_predicted_test.numpy())
    y_test_orig = scaler_y.inverse_transform(y_test.numpy())
    y_train_orig = scaler_y.inverse_transform(y_train.numpy())
    x_test_orig = scaler_x.inverse_transform(x_test.numpy())
    x_train_orig = scaler_x.inverse_transform(x_train.numpy())

    mse = mean_squared_error(y_test_orig, y_predicted_test_orig)
    mae = mean_absolute_error(y_test_orig, y_predicted_test_orig)
    r2 = r2_score(y_test_orig, y_predicted_test_orig)

    print(f"MSE (Mean Squared Error) на тестовых данных: {mse:.2f}")
    print(f"MAE (Mean Absolute Error) на тестовых данных: {mae:.2f}")
    print(f"R^2 (Coefficient of Determination) на тестовых данных: {r2:.2f}")

    plt.scatter(x_train_orig, y_train_orig, label="Train data")
    plt.scatter(x_test_orig, y_test_orig, label="Test data")
    plt.plot(x_test_orig, y_predicted_test_orig, color="red", label="Regression Line")
    plt.xlabel("Площадь жилья (кв. м)")
    plt.ylabel("Стоимость жилья (тыс. руб.)")
    plt.legend()
    plt.show()


def main():
    """
    9.Основная функция, которая выполняет весь пайплайн.
    """
    x_train, x_test, y_train, y_test = generate_and_split_data()
    x_train, x_test, y_train, y_test, scaler_x, scaler_y = scale_and_convert_data(x_train, x_test, y_train, y_test)
    model = create_and_train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test, scaler_y, scaler_x, x_train, y_train)


if __name__ == '__main__':
    main()
