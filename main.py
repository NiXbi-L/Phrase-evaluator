import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import label
import math
import test
from itertools import chain


def plot_combined_graph(data):
    """
    Функция для построения одного общего графика на основе входного списка.

    Параметры:
    - data: список списков точек, где каждый подсписок представляет график.
            Пример: [[(x1, y1), (x2, y2), ...], [(x1, y1), (x2, y2), ...], ...]

    Возвращает:
    - None (строит один общий график с помощью matplotlib)
    """

    # Функция для вычисления корня из суммы квадратов координат
    def calculate_y(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    # Инициализация переменных
    x_values = []  # Список для хранения координат x
    y_values = []  # Список для хранения новых значений y'
    x_counter = 0  # Счетчик для координаты x

    # Обработка всех графиков
    for points in data:
        for x, y in points:
            x_values.append(x_counter)  # Добавляем текущее значение счетчика
            y_values.append(calculate_y(x, y))  # Вычисляем новое значение y'
            x_counter += 1  # Увеличиваем счетчик

    # Построение общего графика
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Общий график")

    # Настройка графика
    plt.title("Общий график с новыми координатами y'")
    plt.xlabel("Индекс точки")
    plt.ylabel("sqrt(x^2 + y^2)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_values


def find_line_skeleton(binary_image):
    """
    Функция принимает бинарное изображение и возвращает список линий,
    где каждая линия представлена списком точек (координат).

    :param binary_image: Бинарное изображение (numpy array размером 100x100)
    :return: Список линий, где каждая линия — это список точек [(x1, y1), (x2, y2), ...]
    """
    # Выполняем скелетизацию
    skeleton = skeletonize(binary_image)

    # Разделяем скелет на отдельные связные компоненты
    labeled_array, num_features = label(skeleton)

    # Извлекаем точки для каждой связной компоненты
    lines = []
    for i in range(1, num_features + 1):
        # Находим координаты точек для текущей связной компоненты
        coords = np.column_stack(np.where(labeled_array == i))
        lines.append(coords)  # Добавляем точки в список линий

    return lines


def approximate_line(line, epsilon=0.01):
    """
    Аппроксимация линии с помощью cv2.approxPolyDP.

    :param line: Массив точек линии (shape: [N, 2])
    :param epsilon: Параметр точности аппроксимации
    :return: Аппроксимированная линия (массив точек)
    """
    # Преобразуем точки в формат OpenCV
    contour = line.reshape(-1, 1, 2).astype(np.int32)

    # Вычисляем периметр контура
    perimeter = cv2.arcLength(contour, closed=False)

    # Аппроксимируем контур
    approx = cv2.approxPolyDP(contour, epsilon * perimeter, closed=False)

    # Преобразуем обратно в массив точек
    approx = approx.reshape(-1, 2)
    return approx


def plot_lines(original_lines, approximated_lines=None):
    """
    Функция строит графики исходных и аппроксимированных линий на одном холсте.

    :param original_lines: Список исходных линий
    :param approximated_lines: Список аппроксимированных линий
    """
    plt.figure(figsize=(8, 8))

    colors = plt.cm.tab10.colors  # Цветовая палитра

    # Рисуем исходные линии
    for i, line in enumerate(original_lines):
        plt.plot(line[:, 1], line[:, 0], '-', color=colors[i % len(colors)], label=f"Original Line {i + 1}")

    # # Рисуем аппроксимированные линии
    # for i, line in enumerate(approximated_lines):
    #     plt.plot(line[:, 1], line[:, 0], '--', color=colors[i % len(colors)], label=f"Approximated Line {i + 1}")

    # Настройка графика
    plt.gca().invert_yaxis()  # Инвертируем ось Y
    plt.title("Line Skeletons and Approximations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Сохраняем пропорции осей
    plt.show()


def flatten_points(points):
    """
    Рекурсивно разворачивает вложенные списки точек в один плоский список.

    :param points: Список точек, возможно с вложенными списками
    :return: Плоский список точек [(x1, y1), (x2, y2), ...]
    """
    flat_list = []
    for item in points:
        if isinstance(item, (list, np.ndarray)):
            # Если элемент — список или массив NumPy, проверяем его содержимое
            if len(item) > 0 and len(item[0]) == 2:
                # Если это точка (x, y), добавляем её
                flat_list.append(tuple(item[0]))  # Преобразуем в кортеж
            else:
                # Если это вложенный список, рекурсивно обрабатываем его
                flat_list.extend(flatten_points(item))
    return flat_list


def find_nearest_point(current_point, available_points):
    """
    Находит ближайшую точку к current_point из available_points.
    :param current_point: Текущая точка [x, y].
    :param available_points: Список доступных точек [[x1, y1], [x2, y2], ...].
    :return: Ближайшая точка и ее индекс.
    """
    min_distance = float('inf')
    nearest_point = None
    nearest_index = -1

    for i, point in enumerate(available_points):
        distance = math.sqrt((point[0] - current_point[0]) ** 2 + (point[1] - current_point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_point = point
            nearest_index = i

    return nearest_point, nearest_index


def transform_points(points):
    """
    Применяет преобразования к списку точек:
    1. Поворот на 90 градусов по часовой стрелке.
    2. Отражение относительно оси X.
    3. Отражение относительно оси Y.
    :param points: Список точек [[x1, y1], [x2, y2], ...].
    :return: Преобразованный список точек.
    """
    # 1. Поворот на 90 градусов по часовой стрелке
    rotated_points = [[-point[1], point[0]] for point in points]

    # 2. Отражение относительно оси X
    reflected_x_points = [[point[0], -point[1]] for point in rotated_points]

    # 3. Отражение относительно оси Y
    reflected_y_points = [[-point[0], point[1]] for point in reflected_x_points]

    return reflected_y_points


def plot_graphs(graphs):
    """
    Визуализирует графики.
    :param graphs: Список графиков, где каждый график — список точек [[x1, y1], [x2, y2], ...].
    """
    plt.figure(figsize=(8, 8))

    for graph in graphs:
        x = [point[0] for point in graph]
        y = [point[1] for point in graph]
        plt.plot(x, y, marker='o')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title("Графики после преобразований")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def connect_and_transform(points, max_distance=20, min_points=5):
    """
    Алгоритм соединения точек, преобразования и визуализации.
    :param points: Исходный список точек [[x1, y1], [x2, y2], ...].
    :param max_distance: Максимальное расстояние между точками для соединения.
    :param min_points: Минимальное количество точек в графиках для отображения.
    :return: Список графиков, удовлетворяющих условиям.
    """
    used_points = set()  # Набор использованных точек
    graphs = []  # Список графиков

    while len(used_points) < len(points):
        # Ищем первую непримененную точку
        start_point = None
        for point in points:
            if tuple(point) not in used_points:
                start_point = point
                break

        if start_point is None:
            break

        # Создаем новый график
        current_graph = [start_point]
        used_points.add(tuple(start_point))
        current_point = start_point

        while True:
            available_points = [point for point in points if tuple(point) not in used_points]
            if not available_points:
                break

            nearest_point, nearest_index = find_nearest_point(current_point, available_points)

            if nearest_point is None:
                break

            distance = math.sqrt((nearest_point[0] - current_point[0]) ** 2 + (nearest_point[1] - current_point[1]) ** 2)
            if distance > max_distance:
                break

            current_graph.append(nearest_point)
            used_points.add(tuple(nearest_point))
            current_point = nearest_point

        # Добавляем график, если он содержит достаточно точек
        if len(current_graph) >= min_points:
            graphs.append(current_graph)

    # Применяем преобразования к каждому графику
    transformed_graphs = [transform_points(graph) for graph in graphs]

    return transformed_graphs


def binaied(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (700, 700))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.bilateralFilter(img, 11, 15, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    alpha = 1.5  # Коэффициент контраста (>1 увеличивает контраст)
    beta = 0  # Смещение (изменяет яркость)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    binary = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    karnel = np.ones((4, 4), np.uint8)
    binary = cv2.dilate(binary, karnel, iterations=1)
    return binary


binary1 = binaied('img/p.jpg')
binary1 = cv2.resize(binary1, (225, 225))

binary2 = binaied('img/p_normal.jpg')
binary2 = cv2.resize(binary2, (225, 225))

# Получаем скелеты линий
original_lines = flatten_points(find_line_skeleton(binary1))
original_lines2 = flatten_points(find_line_skeleton(binary2))

result = test.process_points(original_lines, max_distance=20, min_points=2, segment_size=50, epsilon=0.2, show_plot=True)
graph = []
for i in range(225, 0, -1):
    graph.append((i, 1))
for i in range(2, 151):
    graph.append((1, i))
for i in range(1, 226):
    graph.append((i, 151))

graph1 = [graph]
print(plot_combined_graph(result['normal']))
binary1 = cv2.resize(binary1, (700, 700))
cv2.imshow('asd', binary1)
cv2.waitKey(0)

