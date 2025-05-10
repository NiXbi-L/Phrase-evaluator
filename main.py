import cv2
from preprocessing import preprocess

img = cv2.imread('img/G_normal.jpg')

metrics, graphs = preprocess(img)

for i in range(len(metrics)):
    print(f"Оценка графика: {i}")
    # print(graphs)
    print(metrics[i])
