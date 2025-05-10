import cv2
from preprocessing import preprocess

img = cv2.imread('img/test_p.png')

metrics, graphs = preprocess(img)

for i in range(len(metrics)):
    print(f"Оценка графика: {i}")
    # print(graphs)
    print(metrics[i])
