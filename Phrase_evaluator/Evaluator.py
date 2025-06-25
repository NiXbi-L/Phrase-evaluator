import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d
import math


def get_skeleton_points(binary_image):
    """Возвращает плоский массив точек скелета в формате (y, x)"""
    # Скелетизация
    skeleton = skeletonize(binary_image, method='lee')

    # Маркировка компонентов
    labeled, num_features = label(skeleton, structure=np.ones((3, 3)))

    # Быстрое извлечение всех точек скелета
    points = np.column_stack(np.nonzero(labeled))

    # Если нужно сохранить принадлежность к линиям (опционально)
    # labels = labeled[points[:,0], points[:,1]
    # return points, labels

    return [(p[1], -p[0]) for p in points]


def normalize_points(points) -> list:
    """Нормализация облака точек для анализа лучей"""
    if not points:
        return []

    # Конвертация в numpy array для векторных операций
    arr = np.array(points, dtype=np.float32)

    # 1. Центрирование относительно центра масс
    centroid = np.mean(arr, axis=0)
    centered = arr - centroid

    # 2. Масштабирование в единичный квадрат
    max_range = np.max(np.abs(centered)) * 1.2  # +20% запаса
    if max_range == 0:
        return arr.tolist()

    normalized = centered / max_range

    return normalized.tolist()


def combined_graph(data):
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


def interpolate_to_length(arr, target_length=5000):
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


def evaluate_tremor(contour_images):
    results = []

    for contour in contour_images:
        res = normalize_points(get_skeleton_points(contour))

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


def auto_project_points(points, margin=0.2):
    """
    Автоматически определяет границы бокса и проектирует точки в зонах:
    - Левая зона: [min_x, min_x + margin]
    - Правая зона: [max_x - margin, max_x]

    Параметры:
    points (list): Список точек в формате [(x1, y1), (x2, y2), ...]
    margin (float): Ширина зоны проекции от границ (по умолчанию 0.2)

    Возвращает:
    tuple: (projected_points, left_bound, right_bound)
    """
    if not points:
        return [], 0, 0

    # Определяем границы бокса
    all_x = [x for x, y in points]
    min_x = min(all_x)
    max_x = max(all_x)

    # Рассчитываем зоны проекции
    left_zone = (min_x, min_x + margin)
    right_zone = (max_x - margin, max_x)

    projected = []

    for x, y in points:
        # Проекция на левую границу
        if left_zone[0] <= x <= left_zone[1]:
            projected.append((min_x, y))

        # Проекция на правую границу
        elif right_zone[0] <= x <= right_zone[1]:
            projected.append((max_x, y))

        # Точки вне зон игнорируются (не включаются в результат)

    return projected


def group_points_by_y(points, y_epsilon=0.005):
    """
    Группирует точки сначала по X, затем по Y с заданной точностью

    Параметры:
    points (list): Список точек в формате [(x1, y1), (x2, y2), ...]
    y_epsilon (float): Максимальная разница Y для попадания в одну группу

    Возвращает:
    list: Список групп точек [[(x1, y1), (x2, y2)], ...]
    """
    if not points:
        return []

    # 1. Группировка по X
    x_groups = {}
    for x, y in points:
        if x not in x_groups:
            x_groups[x] = []
        x_groups[x].append((x, y))

    # 2. Группировка по Y внутри каждой X-группы
    result = []
    for x, group in x_groups.items():
        # Сортируем по Y
        sorted_group = sorted(group, key=lambda p: p[1])

        # Создаем подгруппы по Y
        current_subgroup = [sorted_group[0]]
        for p in sorted_group[1:]:
            if abs(p[1] - current_subgroup[-1][1]) <= y_epsilon:
                current_subgroup.append(p)
            else:
                result.append(current_subgroup)
                current_subgroup = [p]
        result.append(current_subgroup)

    return result


def recursive_auto_project(points, margin=0.1, depth=0, max_depth=10):
    """
    Рекурсивно проецирует точки на границы зоны, пока ширина не станет меньше margin

    Параметры:
    points (list): Список точек в формате [(x1, y1), (x2, y2), ...]
    margin (float): Ширина зоны проекции от границ
    depth (int): Текущая глубина рекурсии
    max_depth (int): Максимальная глубина рекурсии

    Возвращает:
    list: Спроецированные точки
    """
    if not points or depth > max_depth:
        return []

    # Определяем границы текущего облака
    xs = [x for x, y in points]
    min_x = min(xs)
    max_x = max(xs)
    width = max_x - min_x

    # Базовый случай: ширина зоны меньше margin
    if width < margin:
        return []

    # Определяем зоны проекции
    left_zone = (min_x, min_x + margin)
    right_zone = (max_x - margin, max_x)

    projected = []
    remaining = []

    # Проецируем точки в зонах
    for x, y in points:
        if left_zone[0] <= x <= left_zone[1]:
            projected.append((min_x, y))
        elif right_zone[0] <= x <= right_zone[1]:
            projected.append((max_x, y))
        else:
            remaining.append((x, y))

    # Рекурсивно обрабатываем оставшиеся точки
    recursive_projected = recursive_auto_project(remaining, margin, depth + 1, max_depth)

    return projected + recursive_projected


def sort_and_group_by_x(point_groups):
    sort = {}
    count_el = {}
    for point_group in point_groups:
        if point_group[0][0] in list(sort):
            sort[point_group[0][0]].append(point_group)
            count_el[point_group[0][0]] += 1
        else:
            sort[point_group[0][0]] = [point_group]
            count_el[point_group[0][0]] = 1

    return sort, count_el


def find_max_group_count(data):
    # Создаем словарь для подсчета частот
    freq = {}
    for count in data.values():
        freq[count] = freq.get(count, 0) + 1

    # Сортируем уникальные значения по убыванию
    sorted_counts = sorted(freq.keys(), reverse=True)

    # Ищем первое значение, которое встречается >= 2 раз
    for count in sorted_counts:
        if freq[count] >= 2:
            return count

    # Если не нашли, возвращаем 0
    return 0


def find_groups_dict(data, n):
    # Создаем копию данных
    remaining_data = {}
    for x, groups in data.items():
        # Создаем глубокую копию и сразу сортируем группы по y
        sorted_groups = sorted(
            groups,
            key=lambda group: min(point[1] for point in group),
            reverse=True  # Сортируем от ВЕРХНЕЙ к НИЖНЕЙ
        )
        remaining_data[x] = sorted_groups

    # Словари для хранения найденных групп
    left_groups = {}
    right_groups = {}

    # Левый проход (слева направо)
    found_left_levels = set()
    sorted_x_left = sorted(remaining_data.keys())

    for x in sorted_x_left:
        if len(found_left_levels) >= n:
            break

        groups = remaining_data[x]
        if not groups:
            continue

        # Группы уже отсортированы от верхней (уровень 0) к нижней
        # Проходим по уровням в порядке сортировки
        for level in range(len(groups)):
            if level in found_left_levels:
                continue

            if len(found_left_levels) < n:
                group = groups[level]
                left_groups[level] = group
                found_left_levels.add(level)
                # Удаляем найденную группу
                remaining_data[x] = [g for g in remaining_data[x] if g != group]
            else:
                break

    # Правый проход (справа налево)
    found_right_levels = set()
    sorted_x_right = sorted(remaining_data.keys(), reverse=True)

    for x in sorted_x_right:
        if len(found_right_levels) >= n:
            break

        groups = remaining_data[x]
        if not groups:
            continue

        # Группы уже отсортированы от верхней (уровень 0) к нижней
        # Проходим по уровням в порядке сортировки
        for level in range(len(groups)):
            if level in found_right_levels:
                continue

            if len(found_right_levels) < n:
                group = groups[level]
                right_groups[level] = group
                found_right_levels.add(level)
                # Удаляем найденную группу
                remaining_data[x] = [g for g in remaining_data[x] if g != group]
            else:
                break

    # Формируем результат
    result = {}
    for level in range(n):
        # Уровни начинаются с 0, но в результате нумеруем с 1
        result[level + 1] = [
            right_groups.get(level, []),  # правая группа
            left_groups.get(level, [])  # левая группа
        ]

    return result


def calculate_angles(result_dict):
    """
    Вычисляет углы между нижними точками левой и правой групп для каждого уровня.

    Параметры:
    result_dict - словарь в формате {level: [right_group, left_group]}

    Возвращает:
    Словарь углов в градусах для каждого уровня и среднее значение модуля углов
    """
    angles = {}
    angle_values = []

    for level, groups in result_dict.items():
        right_group = groups[0]
        left_group = groups[1]

        # Пропускаем уровень, если отсутствует одна из групп
        if not right_group or not left_group:
            angles[level] = None
            continue

        # Находим нижнюю точку в правой группе (с минимальной y-координатой)
        right_bottom = min(right_group, key=lambda point: point[1])

        # Находим нижнюю точку в левой группе (с минимальной y-координатой)
        left_bottom = min(left_group, key=lambda point: point[1])

        # Вычисляем разницу координат
        dx = right_bottom[0] - left_bottom[0]
        dy = right_bottom[1] - left_bottom[1]

        # Вычисляем угол в радианах и преобразуем в градусы
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Сохраняем угол и его абсолютное значение
        angles[level] = angle_deg
        angle_values.append(abs(angle_deg))

    # Вычисляем среднее значение, если есть вычисленные углы
    average_angle = np.mean(angle_values) if angle_values else 0.0

    return angles, average_angle


def process_points(bin_image):
    points = normalize_points(get_skeleton_points(bin_image))

    xs = [x for x, y in points]
    min_x = min(xs)
    max_x = max(xs)

    points1 = group_points_by_y(recursive_auto_project(points))
    points1 = [point_group for point_group in points1 if len(point_group) > 25]

    sort, count_el = sort_and_group_by_x(points1)
    res = find_groups_dict(sort, find_max_group_count(count_el))

    return res, (min_x, max_x), sort


def evaluate_angle(bin_image):
    res, _, __ = process_points(bin_image)
    angles, mean_angle = calculate_angles(res)
    angle_std, _ = calculate_std_dev([angle for _, angle in angles.items()])
    return mean_angle, angle_std


def calculate_std_dev(data):
    n = len(data)
    if n <= 1:
        return 0.0  # Недостаточно данных для расчёта
    mean = sum(data) / n
    squared_diffs = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diffs) / (n - 1)  # Исправленная дисперсия для выборки
    return math.sqrt(variance), mean


def evaluate_indent(bin_image):
    res, box, __ = process_points(bin_image)

    left_indent = [res[i][1][0][0] for i in list(res)]
    right_indent = [res[i][0][0][0] for i in list(res)]

    left_indent_delta = [indent - box[0] for indent in left_indent]
    right_indent_delta = [box[1] - indent for indent in right_indent]

    left_indent_std, left_indent_mean = calculate_std_dev(left_indent_delta)
    right_indent_std, right_indent_mean = calculate_std_dev(right_indent_delta)

    if left_indent_mean:
        left_indent_cv = left_indent_std / left_indent_mean
    else:
        left_indent_cv = 0

    if right_indent_mean:
        right_indent_cv = right_indent_std / right_indent_mean
    else:
        right_indent_cv = 0

    return left_indent_cv, right_indent_cv


def calculate_average_range(data):
    all_ranges = []  # Список для хранения диапазонов всех линий

    # Перебираем все группы по x-координате
    for x_coord, lines in data.items():
        # Перебираем каждую линию в группе
        for line in lines:
            if not line:  # Пропустить пустые линии
                continue

            # Извлекаем y-координаты
            y_coords = [point[1] for point in line]

            # Вычисляем диапазон
            min_y = min(y_coords)
            max_y = max(y_coords)
            y_range = max_y - min_y

            all_ranges.append(y_range)

    # Вычисляем среднее значение
    if not all_ranges:  # Если нет данных
        return 0.0

    char_height_std, char_height_mean = calculate_std_dev(all_ranges)
    if char_height_mean:
        char_height_cv = char_height_std / char_height_mean
    else:
        char_height_cv = 0

    return char_height_cv


def evaluate_width(bin_image):
    _, __, points = process_points(bin_image)
    return calculate_average_range(points)


def calculate_line_spacing(data):
    all_distances = []  # Список для хранения всех расстояний

    # Перебираем все вертикальные линии
    for x_coord, segments in data.items():
        if len(segments) < 2:
            continue  # Нужно минимум 2 отрезка для расчета расстояний

        # Список для характеристик отрезков: (min_Y, max_Y)
        segment_props = []

        # Собираем характеристики каждого отрезка
        for segment in segments:
            y_coords = [point[1] for point in segment]
            min_y = min(y_coords)
            max_y = max(y_coords)
            segment_props.append((min_y, max_y))

        # Сортируем отрезки по Y (сверху вниз)
        segment_props.sort(key=lambda x: x[0])

        # Рассчитываем расстояния между соседними отрезками
        for i in range(len(segment_props) - 1):
            # Верхний отрезок: segment_props[i]
            # Нижний отрезок: segment_props[i+1]
            distance = segment_props[i + 1][0] - segment_props[i][1]
            all_distances.append(distance)

    # Рассчитываем метрики
    if not all_distances:
        return 0.0, 0.0  # Если нет данных

    mean_spacing = np.mean(all_distances)

    # Коэффициент вариации (в процентах)
    if mean_spacing == 0:
        cv = 0.0
    else:
        cv = (np.std(all_distances) / mean_spacing)

    return mean_spacing, cv


def evaluate_line_spacing(bin_image):
    _, __, points = process_points(bin_image)
    line_spacing_mean, line_spacing_cv = calculate_line_spacing(points)

    return line_spacing_mean, line_spacing_cv


def evaluate_stroke_discontinuity(contour_images):
    count_contours = len(contour_images)
    words = 14
    return count_contours - words
