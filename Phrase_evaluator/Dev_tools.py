import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from matplotlib.colors import ListedColormap


def plot_multiple_graphs(y_list, labels=None):
    """
    Отрисовывает несколько графиков на одной плоскости.

    Параметры:
    y_list (list of arrays): Список массивов значений y для каждого графика.
    labels (list of str, optional): Список меток для легенды. По умолчанию None.
    """
    plt.figure(figsize=(10, 6))  # Размер графика

    # Построение каждого графика
    for i, y in enumerate(y_list):
        x = range(len(y))  # Индексы как значения x
        if labels and i < len(labels):
            plt.plot(x, y, label=labels[i])
        else:
            plt.plot(x, y)

    # Настройка оформления
    plt.title('Наложенные графики')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    if labels:
        plt.legend()  # Добавление легенды, если есть метки
    plt.grid(True)
    plt.show()


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


def plot_multiple_points(points_list, labels=None, colors=None,
                         box=False, box_color='red', box_alpha=0.3,
                         centre=False, centre_color='lime', centre_size=100):
    """
    Отрисовывает точки с опциональным bounding box и центром

    Параметры:
    points_list (list): Список наборов точек
    labels (list): Метки для наборов точек
    colors (list): Цвета для наборов точек
    box (bool): Рисовать bounding box вокруг всех точек
    box_color (str): Цвет рамки bounding box
    box_alpha (float): Прозрачность заливки бокса (0-1)
    centre (bool): Рисовать центр bounding box
    centre_color (str): Цвет точки центра
    centre_size (int): Размер точки центра
    """
    plt.figure(figsize=(10, 6))

    # Отрисовка основных точек
    default_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, points in enumerate(points_list):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        color = colors[i] if colors and i < len(colors) else default_colors[i % 7]
        label = labels[i] if labels and i < len(labels) else None
        plt.scatter(x, y, c=color, alpha=0.7, label=label, edgecolors='none')

    # Расчет границ для box и центра
    if box or centre:
        all_points = [p for points in points_list for p in points]
        all_x = [p[0] for p in all_points]
        all_y = [p[1] for p in all_points]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

    # Отрисовка bounding box
    if box:
        plt.gca().add_patch(Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=2,
            edgecolor=box_color,
            facecolor='none',
            alpha=box_alpha,
            zorder=1
        ))

    # Отрисовка центра
    if centre:
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        plt.scatter(
            [center_x], [center_y],
            c=centre_color,
            s=centre_size,
            marker='X',
            edgecolors='black',
            zorder=2,
            label='Center' if labels else None
        )

    plt.title('Точки на плоскости с Bounding Box')
    plt.xlabel('X')
    plt.ylabel('Y')
    if labels:
        plt.legend()
    plt.grid(True)
    plt.show()


def plot_axis(sorted_points, center, all_points):
    plt.figure(figsize=(12, 6))

    # 1. Исходные данные
    plt.subplot(1, 2, 1)
    plt.scatter(*zip(*all_points), c='blue', s=10, label='Исходные точки')
    plt.scatter(*center, c='red', s=100, marker='X', label='Центр')

    # Рисуем лучи
    for p in sorted_points:
        plt.plot([center[0], p[0]], [center[1], p[1]],
                 color='gray', alpha=0.3, linewidth=0.5)

    plt.title('Визуализация лучей')
    plt.axis('equal')
    plt.legend()

    # 2. График расстояний
    distances = [np.hypot(p[0] - center[0], p[1] - center[1]) for p in sorted_points]

    plt.subplot(1, 2, 2)
    plt.plot(distances, marker='o', markersize=3, linestyle='-', linewidth=1)
    plt.title('Распределение расстояний от центра')
    plt.xlabel('Номер точки')
    plt.ylabel('Расстояние')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_paired_groups(result_dict):
    """
    Визуализирует парные группы точек (левые и правые) для каждого уровня.

    Параметры:
    result_dict - словарь в формате {level: [right_group, left_group]}
    """
    plt.figure(figsize=(12, 8))

    # Создаем цветовую карту по количеству уровней
    levels = list(result_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(levels)))
    cmap = ListedColormap(colors)

    # Собираем все точки для правильного масштабирования осей
    all_points = []
    for level, groups in result_dict.items():
        all_points.extend(groups[0])  # правая группа
        all_points.extend(groups[1])  # левая группа
    all_x = [p[0] for p in all_points]
    all_y = [p[1] for p in all_points]

    # Отрисовываем каждую пару групп
    for i, level in enumerate(levels):
        color = cmap(i)
        right_group = result_dict[level][0]
        left_group = result_dict[level][1]

        # Рисуем правую группу
        if right_group:
            right_x = [p[0] for p in right_group]
            right_y = [p[1] for p in right_group]
            plt.scatter(right_x, right_y, color=color, s=100,
                        edgecolors='k', zorder=3, label=f'Уровень {level} (правая)')

            # Подписываем правую группу
            center_x = min(right_x) if len(set(right_x)) > 1 else right_x[0]
            center_y = np.mean(right_y)
            plt.text(center_x + 0.02, center_y, str(level),
                     fontsize=12, color=color, weight='bold',
                     verticalalignment='center')

        # Рисуем левую группу
        if left_group:
            left_x = [p[0] for p in left_group]
            left_y = [p[1] for p in left_group]
            plt.scatter(left_x, left_y, color=color, s=100,
                        edgecolors='k', zorder=3, marker='s',
                        label=f'Уровень {level} (левая)')

            # Подписываем левую группу
            center_x = max(left_x) if len(set(left_x)) > 1 else left_x[0]
            center_y = np.mean(left_y)
            plt.text(center_x - 0.02, center_y, str(level),
                     fontsize=12, color=color, weight='bold',
                     horizontalalignment='right', verticalalignment='center')

    # Настройка внешнего вида графика
    plt.title('Парные группы точек по уровням', fontsize=14)
    plt.xlabel('X координата', fontsize=12)
    plt.ylabel('Y координата', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Автоматическая настройка осей с небольшим запасом
    x_margin = (max(all_x) - min(all_x)) * 0.2
    y_margin = (max(all_y) - min(all_y)) * 0.2
    plt.xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    plt.ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Убираем дубликаты в легенде
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.values(), loc='best', title='Группы')

    plt.tight_layout()
    plt.show()


def plot_segment(A, B):
    import math
    plt.figure(figsize=(10, 5))
    plt.plot([A[0], B[0]], [A[1], B[1]], 'b-o', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.axis('equal')  # Критически важно!
    plt.grid(True)
    plt.title(f"Угол наклона: {math.degrees(math.atan(abs(A[1] - B[1]) / abs(A[0] - B[0]))):.2f}°")
    plt.show()
