# Документация для модуля анализа тремора при письме

## Архитектура проекта
```
.
├── Evaluator.py          # Основной модуль оценки тремора
├── preprocessing.py      # Модуль предобработки изображений
└── main.py               # Пример использования
```

## Установка
1. Требуется Python 3.8+
2. Установите зависимости:
```bash
pip install opencv-python scikit-image scipy matplotlib numpy
```

## Основные компоненты

### 1. Модуль предобработки (`preprocessing.py`)
#### Функции:
- **`dilate_image(image)`** - бинаризация и морфологическая обработка
- **`find_contours(dilated, image)`** - поиск и сортировка контуров
- **`preprocess(cv_image)`** - главная функция обработки:
  ```python
  def preprocess(cv_image):
      # Возвращает список изображений контуров
      return contour_images
  ```

### 2. Модуль оценки (`Evaluator.py`)
#### Основной пайплайн:
```python
def evaluate(image_path):
   # Возвращает:
   # metrics - список метрик для каждого контура
   # graphs - сырые данные для анализа
   return metrics, graphs
```

#### Ключевые функции:
- **`find_line_skeleton()`** - скелетизация контура
- **`combined_graph()`** - нормализация и сортировка точек
- **`evaluate_smoothness()`** - расчет метрик плавности
- **`interpolate_to_length()`** - унификация длины сигналов

### 3. Пример использования (`main.py`)
```python
from Evaluator import evaluate

metrics, graphs = evaluate('img/P_test.jpg')

for i, metric in enumerate(metrics):
    print(f"Контур {i}:")
    print(f"  RMS: {metric['RMS_derivative']:.3f}")
    print(f"  MAE: {metric['MAE_derivative']:.3f}")
    print(f"  Max: {metric['Max_derivative']:.3f}")
    print(f"  Zero: {metric['Zero_crossings']}")
```

## Этапы обработки данных
1. **Загрузка изображения** (через OpenCV)
2. **Предобработка**:
   - Бинаризация
   - Удаление шумов
   - Скелетизация
   - Поиск контуров
3. **Анализ контуров**:
   - Нормализация координат
   - Интерполяция до 1000 точек
4. **Расчет метрик**:
   - RMS/MAE/Max производной
   - Подсчет пересечений нуля

## Интерпретация метрик
| Метрика             | Норма          | Тремор         |
|----------------------|----------------|----------------|
| RMS_derivative      | < 0.1         | > 0.5          |
| MAE_derivative      | < 0.05        | > 0.3          |
| Max_derivative      | < 0.5         | > 2.0          |
| Zero_crossings      | < 50          | > 200          |

## Работа с результатами
### Структура выходных данных
```python
metrics = [
    {   # Для каждого контура
        'RMS_derivative': float,
        'MAE_derivative': float,
        'Max_derivative': float,
        'Zero_crossings': int
    },
    ...
]

graphs = {
    'y_normalized': [np.array, ...],  # Нормализованные расстояния
    'yd': [np.array, ...]            # Производные сигналов
}
```

### Пример визуализации
```python
import matplotlib.pyplot as plt

idx = 0  # Номер контура
plt.figure(figsize=(12, 4))
plt.plot(graphs['y_normalized'][idx], label='Нормализованный сигнал')
plt.plot(graphs['yd'][idx], label='Производная', alpha=0.7)
plt.title(f'Анализ контура {idx}')
plt.legend()
plt.grid(True)
plt.show()
```

## Требования к изображениям
- Формат: JPG/PNG
- Разрешение: мин. 300x300 пикселей
- Контраст: мин. 70% между текстом и фоном
- Ориентация: горизонтальная/вертикальная без искажений

## Советы по улучшению точности
1. Используйте образцы с четкими линиями
2. Избегайте теней на фоне
3. Делайте снимки при равномерном освещении
4. Для калибровки используйте эталонные изображения

## Дополнительные возможности
- Кастомный анализ через `graphs`
- Экспорт данных в CSV
- Интеграция с ML-моделями
- Сравнительный анализ нескольких образцов