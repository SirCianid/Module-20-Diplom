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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


def generate_and_split_data():
    """
    2. Генерируем данные для обучения и разделяем выборки. В данной модели будем иммитировать классификацию -
       спам/не спам. Основываться будем на длине условных сообщений, и колличестве восклицательных знаков.
     2.1. Разделяем данные на обучающую и тестовую выборки. Также воспользуемся train_test_split от scikit-learn.
    """
    np.random.seed(42)
    num_samples = 200
    message_lengths = np.random.uniform(10, 100, num_samples)
    exclamation_marks = np.random.randint(0, 10, num_samples)
    spam_probability = 0.2 + 0.005 * message_lengths + 0.05 * exclamation_marks
    spam_labels = (np.random.rand(num_samples) < spam_probability).astype(int)
    x = np.column_stack((message_lengths, exclamation_marks))
    y = spam_labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def scale_and_convert_data(x_train, x_test, y_train, y_test):
    """
    3. Масштабируем входные признаки.
     3.1. Преобразуем данные в тензоры PyTorch.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train = torch.from_numpy(x_train_scaled).float()
    x_test = torch.from_numpy(x_test_scaled).float()
    y_train = torch.from_numpy(y_train).float().reshape(-1, 1)
    y_test = torch.from_numpy(y_test).float().reshape(-1, 1)
    return x_train, x_test, y_train, y_test


def create_and_train_model(x_train, y_train):
    """
    4. Определение модели. В функции __init__ - оределяем линейный слой который и будет обучаться.
       Функции forward - метод, который определяет как данные будут обрабатываться. Сначала происходит линейное
       преобразование, а затем к результату применяется сигмоидальная функция.
    5. Задаем функцию потерь и оптимизатора.
    6. Запускаем обучение модели, с выводом информации о прогрессе через каждые 200 эпох.
    """

    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

    model = LogisticRegression(2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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


def evaluate_model(model, x_test, y_test):
    """
     7. Оценка модели и визуализация результатов.
      7.1. Вычисляем матрицу ошибок и визуализируем ее.
      7.2. Получаем отчет о классификации.
      7.3. Визуализация результатов.
    """
    with torch.no_grad():
        y_predicted_test = model(x_test)
        y_predicted_test_cls = (y_predicted_test >= 0.5).float()
        accuracy = accuracy_score(y_test.numpy(), y_predicted_test_cls.numpy())
        print(f"Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_test.numpy(), y_predicted_test_cls.numpy())
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    plt.xlabel('Прогнозируемые значения')
    plt.ylabel('Реальные значения')
    plt.title('Матрица ошибок')
    plt.show()

    report = classification_report(y_test.numpy(), y_predicted_test_cls.numpy())
    print("Classification Report:")
    print(report)

    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.squeeze().numpy(), label="Data", cmap='RdBu')
    with torch.no_grad():
        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                                torch.linspace(y_min, y_max, 100))
        grid = torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1)
        z = model(grid).reshape(xx.shape)
        plt.contourf(xx.numpy(), yy.numpy(), z.detach().numpy(), cmap='RdBu', alpha=0.4)

    plt.xlabel("Длинна сообщения")
    plt.ylabel("Восклицательные знаки")
    plt.title("Классификация")
    plt.show()


def main():
    """
    8. Основная функция, которая выполняет весь пайплайн.
    """
    x_train, x_test, y_train, y_test = generate_and_split_data()
    x_train, x_test, y_train, y_test = scale_and_convert_data(x_train, x_test, y_train, y_test)
    model = create_and_train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
