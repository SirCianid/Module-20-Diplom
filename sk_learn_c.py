from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import figure, ylabel, plot, legend, grid, show, xlabel, title


def load_and_prepare_data():
    """
    1. Установка необходимых библиотек, и загрузка датасета.
    2. Задаем переменную для датасета.
    3. Получение данных и целевой переменной. x - данные; y - целевая переменная, соответсвенно.
    """
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def scale_data(x_train, x_test):
    """
    4. Масштабирование данных. Хотя логистическая регрессия и не так чувствительна к не масштабированым данным,
       но все равно способна выдавать лучший результат если признаки масштабированы.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def train_logistic_regression(x_train_scaled, y_train):
    """
    5. Настройка гиперпараметров, и обучение модели логистической регрессии.
    Этот метод идеально подходит в данной ситуации, т.к. выбранный датасет является задачей бинарной классификации.
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'class_weight': [None, 'balanced']
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=8000), param_grid, cv=skf,
                               scoring='accuracy')
    grid_search.fit(x_train_scaled, y_train)

    print("Лучшие параметры:", grid_search.best_params_)
    return grid_search.best_estimator_


def predict_and_evaluate(best_model, x_test_scaled, y_test):
    """
    6. Предсказываем результаты на тестовых данных.
    7. Оцениваем качество модели.
    """
    y_pred = best_model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def visualize_roc_curve(best_model, x_test_scaled, y_test):
    """
    8. Визуализация данных с помощью ROC-кривой.
    """
    y_prob = best_model.predict_proba(x_test_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    figure(figsize=(8, 6))
    plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
    plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    xlabel('False Positive Rate (FPR)')
    ylabel('True Positive Rate (TPR)')
    title('ROC-Кривая')
    legend(loc="lower right")
    grid(True)
    show()


def main():
    """
    9. Основная функция, которая выполняет весь пайплайн.
    """
    x_train, x_test, y_train, y_test = load_and_prepare_data()
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)
    best_model = train_logistic_regression(x_train_scaled, y_train)
    predict_and_evaluate(best_model, x_test_scaled, y_test)
    visualize_roc_curve(best_model, x_test_scaled, y_test)


if __name__ == '__main__':
    main()
