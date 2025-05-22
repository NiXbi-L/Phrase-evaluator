import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import interp1d
from scipy.ndimage import label


def show_image(images, titles=None, cmaps=None, figsize=(10, 6)):
    """
    Функция для отображения одного или нескольких изображений.

    Параметры:
    - images: Одно изображение (ndarray) или список изображений.
    - titles: Список заголовков для каждого изображения (опционально).
    - cmaps: Список цветовых карт для каждого изображения (опционально).
    - figsize: Размер фигуры (ширина, высота) в дюймах.
    """
    if not isinstance(images, list):
        images = [images]

    if titles is None:
        titles = ["Image"] * len(images)
    elif isinstance(titles, str):
        titles = [titles]

    if cmaps is None:
        cmaps = [None] * len(images)
    elif isinstance(cmaps, str):
        cmaps = [cmaps]

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    if num_images == 1:
        axes = [axes]

    for i, (image, title, cmap) in enumerate(zip(images, titles, cmaps)):
        if cmap is None and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image, cmap=cmap)
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


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


def dilate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    binary_skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_skeleton, kernel, iterations=1)

    return dilated


def find_contours(dilated, image):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]

    # Фильтрация и вычисление центроидов
    min_contour_area = 150
    filtered_contours = []

    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            # Вычисление моментов для центроида
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
            else:
                cX = 0  # Если момент нулевой, используем левую границу
            filtered_contours.append((cX, contour))

    # Сортировка по X-координате центроида
    filtered_contours.sort(key=lambda x: x[0])

    # Создание изображений контуров в отсортированном порядке
    images = []
    for cX, contour in filtered_contours:
        contour_image = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(contour_image, [contour], -1, 255, thickness=2)
        images.append(contour_image)

    return images


def plot_combined_graph(data, show=False):
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

    if show:
        # Построение общего графика
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label="Общий график")

        # Настройка графика
        plt.title("Общий график с расстояниями от центра (по часовой стрелке)")
        plt.xlabel("Индекс точки")
        plt.ylabel("Расстояние от центра")
        plt.legend()
        plt.grid(True)
        plt.show()

    return y_values


def plot_overlaid_graphs(y1, y2, title='Наложение графиков',
                         labels=('График 1', 'График 2'),
                         xlabel='Индекс', ylabel='Значение'):
    # Преобразование в numpy массивы
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    # Проверка одинаковой длины
    if len(y1) != len(y2):
        raise ValueError("Массивы должны быть одинаковой длины")

    # Создание оси X на основе индексов
    x = np.arange(len(y1))

    # Настройка стиля
    plt.figure(figsize=(12, 6))
    plt.grid(True, alpha=0.3)
    plt.title(title)

    # Отрисовка графиков
    plt.plot(x, y1, label=labels[0], alpha=0.8, linewidth=2, color='blue')
    plt.plot(x, y2, label=labels[1], alpha=0.8, linewidth=2, color='orange')

    # Подписи осей и легенда
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_smoothness(y, plot=False):
    y_array = np.array(y)

    # 1. Удаление тренда
    x = np.arange(len(y_array))
    coeffs = np.polyfit(x, y_array, 1)
    y_detrended = y_array - (coeffs[0] * x + coeffs[1])

    # 2. Z-score нормализация
    y_normalized = (y_detrended - np.mean(y_detrended)) / np.std(y_detrended)

    # 3. Вычисление производных (БЕЗ abs!)
    dy = np.diff(y_normalized)

    # 4. Метрики плавности
    metrics = {
        'RMS_derivative': np.sqrt(np.mean(dy ** 2)),
        'MAE_derivative': np.mean(np.abs(dy)),
        'Max_derivative': np.max(np.abs(dy)),
        'Zero_crossings': np.sum(np.diff(np.sign(dy)) != 0)
    }

    # 5. Визуализация
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_normalized, label='Нормализованный сигнал')
        plt.plot(np.arange(len(dy)), dy, label='Производная')
        plt.legend()
        plt.show()

    return metrics, y_normalized, dy


def preprocess(cv_image):
    print('Ищем контуры первого изображения')
    dilated = dilate_image(cv_image)
    contour_images = find_contours(dilated, cv_image)

    # show_image([dilated], titles=['Контур'])
    
    return contour_images


def evaluate(image_path):
    img = cv2.imread(image_path)

    contour_images = preprocess(img)

    results = []

    for contour in contour_images:
        show_image([contour], titles=['Контур'])

        print('Скелетонизация')
        res = flatten_points(find_line_skeleton(contour))

        print('Строим оценочные графики')
        res = plot_combined_graph(res)

        res = interpolate_to_length(res)

        results.append(res)

    metrics = []
    graphs = {
        'y_normalized': [],
        'yd': []
    }
    for result in results:
        sm, y_normalized, yd = evaluate_smoothness(result, True)
        graphs['y_normalized'].append(y_normalized)
        graphs['yd'].append(yd)
        metrics.append(sm)

    return metrics, graphs
