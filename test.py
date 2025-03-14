import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_nearest_point(current_point, points, used_points, max_distance):
    """
    Находит ближайшую точку к текущей, которая не используется и находится в пределах max_distance.
    :param current_point: Текущая точка [x, y].
    :param points: Список всех точек [[x1, y1], [x2, y2], ..., [xn, yn]].
    :param used_points: Список уже использованных точек.
    :param max_distance: Максимальное расстояние для соединения точек.
    :return: Ближайшая точка или None, если подходящей точки нет.
    """
    nearest_point = None
    min_distance = float('inf')
    for point in points:
        if point not in used_points:
            distance = math.sqrt((point[0] - current_point[0]) ** 2 + (point[1] - current_point[1]) ** 2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_point = point
    return nearest_point

def build_graphs(points, max_distance=20, min_points=5):
    """
    Строит графики на основе заданного алгоритма.
    :param points: Список всех точек [[x1, y1], [x2, y2], ..., [xn, yn]].
    :param max_distance: Максимальное расстояние для соединения точек.
    :param min_points: Минимальное количество точек для отображения графика.
    :return: Список графиков, каждый из которых содержит список точек.
    """
    graphs = []
    used_points = set()

    while len(used_points) < len(points):
        # Находим первую доступную точку
        start_point = None
        for point in points:
            if tuple(point) not in used_points:
                start_point = point
                break

        if start_point is None:
            break

        # Начинаем новый график
        graph = [start_point]
        used_points.add(tuple(start_point))
        current_point = start_point

        while True:
            nearest_point = find_nearest_point(current_point, points, used_points, max_distance)
            if nearest_point is None:
                break
            graph.append(nearest_point)
            used_points.add(tuple(nearest_point))
            current_point = nearest_point

        # Добавляем график, если он содержит достаточно точек
        if len(graph) >= min_points:
            graphs.append(graph)

    return graphs

def approximate_line(line, epsilon=0.01):
    """
    Аппроксимация линии с помощью cv2.approxPolyDP.
    :param line: Массив точек линии (shape: [N, 2]).
    :param epsilon: Параметр точности аппроксимации.
    :return: Аппроксимированная линия (массив точек).
    """
    contour = np.array(line).reshape(-1, 1, 2).astype(np.int32)
    perimeter = cv2.arcLength(contour, closed=False)
    approx = cv2.approxPolyDP(contour, epsilon * perimeter, closed=False)
    return approx.reshape(-1, 2)

def approximate_graphs(graphs, segment_size=10, epsilon=0.2):
    """
    Аппроксимирует графики по сегментам.
    :param graphs: Список графиков.
    :param segment_size: Размер сегмента для аппроксимации.
    :param epsilon: Параметр точности аппроксимации.
    :return: Список аппроксимированных графиков.
    """
    approximated_graphs = []
    for graph in graphs:
        approximated_graph = []
        for i in range(0, len(graph), segment_size):
            segment = graph[i:i + segment_size]
            if len(segment) > 1:  # Аппроксимируем только если сегмент содержит более одной точки
                approx_segment = approximate_line(segment, epsilon)
                approximated_graph.extend(approx_segment)
        approximated_graphs.append(approximated_graph)
    return approximated_graphs

def plot_graphs(graphs, approximated_graphs, show_plot=True):
    """
    Отображает графики и их аппроксимированные версии.
    :param graphs: Список обычных графиков.
    :param approximated_graphs: Список аппроксимированных графиков.
    :param show_plot: Флаг для отображения графиков.
    """
    if not show_plot:
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(graphs)))

    for i, (graph, approx_graph) in enumerate(zip(graphs, approximated_graphs)):
        color = colors[i]
        graph = np.array(graph)
        approx_graph = np.array(approx_graph)

        # Рисуем обычный график
        plt.scatter(graph[:, 0], graph[:, 1], color=color, s=10, label=f'Graph {i+1}')
        plt.plot(graph[:, 0], graph[:, 1], color=color, alpha=0.5)

        # Рисуем аппроксимированный график
        plt.plot(approx_graph[:, 0], approx_graph[:, 1], color=color, linestyle='--', linewidth=2)

    plt.legend()
    plt.title("Graphs and Their Approximations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def process_points(original_lines, max_distance=20, min_points=5, segment_size=10, epsilon=0.2, show_plot=True):
    """
    Основная функция для обработки точек.
    :param original_lines: Список всех точек [[x1, y1], [x2, y2], ..., [xn, yn]].
    :param max_distance: Максимальное расстояние для соединения точек.
    :param min_points: Минимальное количество точек для отображения графика.
    :param segment_size: Размер сегмента для аппроксимации.
    :param epsilon: Параметр точности аппроксимации.
    :param show_plot: Флаг для отображения графиков.
    :return: Словарь с обычными и аппроксимированными графиками.
    """
    graphs = build_graphs(original_lines, max_distance, min_points)
    approximated_graphs = approximate_graphs(graphs, segment_size, epsilon)
    plot_graphs(graphs, approximated_graphs, show_plot)

    # Преобразуем графики в требуемый формат
    result = {
        'normal': [graph for graph in graphs],
        'aprox': [approx_graph for approx_graph in approximated_graphs]
    }
    return result