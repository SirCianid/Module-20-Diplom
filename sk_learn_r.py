from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_and_prepare_data():
    """
    1. Установка необходимых библиотек, и загрузка датасета.
    2. Задаем переменную для датасета.
    3. Получение данных и целевой переменной. x - данные; y - целевая переменная, соответсвенно.
    """
    diabetes = load_diabetes()
    x = diabetes.data
    y = diabetes.target
    return x, y


def split_data(x, y):
    """
    4. Разделяем данные на обучающую и тестовую выборки.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):
    """
    5. Масштабирование данных. Т.к. выбранный метод обучения чувствителен к масштабу признаков,
    нужно стандартизировать и нормализовать признаки.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def train_svr_model(x_train_scaled, y_train):
    """
    6. Настраиваем гиперпараметры для повышения точности предсказания.
    Обучение модели с помощью метода опорных векторов (Support Vector Regression).
    P.S. gamma актуально для rbf и poly.
    """
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1],
        'epsilon': [0.01, 0.1, 0.2]
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(SVR(), param_grid, cv=kf, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_scaled, y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    return grid_search.best_estimator_


def predict_and_evaluate(best_model, x_test_scaled, y_test):
    """
    8. Предсказание на тестовых данных.
    9. Оценка качества модели. Поскольку SVR - это регрессионная модель,
    оценка точности будет отличаться от классификации.
    """
    y_pred = best_model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("R^2:", r2)


def visualize_results(y_test, y_pred):
    """
    10. Визуализация результатов.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Истинные значения")
    plt.ylabel("Предсказанные значения")
    plt.title("Предсказанные vs. Истинные значения")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.grid(True)
    plt.show()


def main():
    """
    11. Основная функция, которая выполняет весь пайплайн.
    """
    x, y = load_and_prepare_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    best_model = train_svr_model(x_train_scaled, y_train)
    predict_and_evaluate(best_model, x_test_scaled, y_test)
    visualize_results(y_test, best_model.predict(x_test_scaled))


if __name__ == '__main__':
    main()
