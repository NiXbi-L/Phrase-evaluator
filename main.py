from preprocessing import evaluate

metrics, graphs = evaluate('img/P_test.jpg')

for i in range(len(metrics)):
    print(f"Оценка графика: {i}")
    # print(graphs)
    print(metrics[i])
