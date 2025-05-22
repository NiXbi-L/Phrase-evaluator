import cv2
from preprocessing import preprocess
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d


def find_line_skeleton(binary_image):
    skeleton = skeletonize(binary_image)

    labeled_array, num_features = label(skeleton)

    lines = []
    for i in range(1, num_features + 1):
        coords = np.column_stack(np.where(labeled_array == i))
        lines.append(coords)

    return lines


def flatten_points(points):
    flat_list = []
    for item in points:
        if isinstance(item, (list, np.ndarray)):
            if len(item) > 0 and len(item[0]) == 2:
                flat_list.append(tuple(item[0]))
            else:
                flat_list.extend(flatten_points(item))
    return flat_list


def combined_graph(data, show=False):
    # Собираем все точки для вычисления центра
    all_points = data

    if not all_points:
        return []  # Если данных нет, возвращаем пустой список

    # Находим крайние точки для определения центра
    min_x = min(p[0] for p in all_points)
    max_x = max(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_y = max(p[1] for p in all_points)

    # Центр облака точек (пересечение диагоналей)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Вектор направления к (0,0) от центра
    dir_to_origin_x = -center_x
    dir_to_origin_y = -center_y

    # Функция для вычисления угла для сортировки по часовой стрелке
    def sort_key(point):
        x, y = point
        dx = x - center_x
        dy = y - center_y

        # Угол направления к (0,0)
        angle_dir = np.arctan2(dir_to_origin_y, dir_to_origin_x)

        # Угол текущей точки
        angle_point = np.arctan2(dy, dx)

        # Разность углов и нормализация
        relative_angle = (angle_point - angle_dir) % (2 * np.pi)

        # Преобразуем в угол по часовой стрелке
        clock_angle = (2 * np.pi - relative_angle) % (2 * np.pi)

        return clock_angle

    # Собираем все точки и сортируем
    sorted_points = sorted(all_points, key=sort_key)

    # Формируем y_values в порядке отсортированных точек
    y_values = []
    x_counter = 0
    x_values = []

    for point in sorted_points:
        x, y = point
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        x_values.append(x_counter)
        y_values.append(distance)
        x_counter += 1

    return y_values


def evaluate_smoothness(y):
    y_array = np.array(y)

    x = np.arange(len(y_array))
    coeffs = np.polyfit(x, y_array, 1)
    y_detrended = y_array - (coeffs[0] * x + coeffs[1])

    y_normalized = (y_detrended - np.mean(y_detrended)) / np.std(y_detrended)

    dy = np.diff(y_normalized)

    metrics = {
        'RMS_derivative': np.sqrt(np.mean(dy ** 2)),
        'MAE_derivative': np.mean(np.abs(dy)),
        'Max_derivative': np.max(np.abs(dy)),
        'Zero_crossings': np.sum(np.diff(np.sign(dy)) != 0)
    }

    return metrics, y_normalized, dy


def interpolate_to_length(arr, target_length=1000):
    """
    Масштабирует массив до указанной длины с помощью кубической интерполяции
    Если исходная длина больше целевой - возвращает исходный массив
    """
    arr = np.asarray(arr)
    original_length = len(arr)

    if original_length >= target_length:
        return arr

    # Создаем базовые оси для интерполяции
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)

    # Создаем интерполяционную функцию
    interp_func = interp1d(
        x_original,
        arr,
        kind='cubic',
        fill_value='extrapolate'
    )

    # Интерполируем значения
    return interp_func(x_target)


def evaluate(image_path):
    img = cv2.imread(image_path)

    contour_images = preprocess(img)

    results = []

    for contour in contour_images:
        print('Скелетонизация')
        res = flatten_points(find_line_skeleton(contour))

        print('Строим оценочные графики')
        res = combined_graph(res)

        res = interpolate_to_length(res)

        results.append(res)

    metrics = []
    graphs = {
        'y_normalized': [],
        'yd': []
    }
    for result in results:
        sm, y_normalized, yd = evaluate_smoothness(result)
        graphs['y_normalized'].append(y_normalized)
        graphs['yd'].append(yd)
        metrics.append(sm)

    return metrics, graphs
