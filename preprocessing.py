import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


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


def preprocess(cv_image):
    print('Ищем контуры изображения')
    dilated = dilate_image(cv_image)
    contour_images = find_contours(dilated, cv_image)

    return contour_images
