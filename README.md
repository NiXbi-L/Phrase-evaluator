```markdown
# Документация для модуля анализа тремора при письме

## Установка
1. Убедитесь, что установлен Python 3.8+
2. Установите зависимости:
```bash
pip install opencv-python scikit-image scipy matplotlib numpy
```
```

## Основные компоненты
### 1. Основной скрипт (`main.py`)
- Загружает изображение
- Вызывает обработчик
- Выводит результаты

Пример использования:
```python
import cv2
from preprocessing import preprocess

# Загрузка изображения
img = cv2.imread('path/to/image.jpg')

# Обработка и получение результатов
metrics, _ = preprocess(img)

# Вывод метрик
for i, metric in enumerate(metrics):
    print(f"Буква {i}:")
    print(f"  RMS: {metric['RMS_derivative']:.3f}")
    print(f"  MAE: {metric['MAE_derivative']:.3f}")
    print(f"  Max: {metric['Max_derivative']:.3f}")
    print(f"  Zero: {metric['Zero_crossings']}")
```

### 2. Модуль предобработки (`preprocessing.py`)
#### Основные функции:
- **`preprocess(cv_image)`** - главная функция обработки
  - Возвращает: 
    - `metrics`: список словарей с метриками для каждого контура
    - `graphs`: словарь с нормализованными данными и производными

- **`evaluate_smoothness(y, plot=False)`** - анализ плавности
  - Возвращает:
    - Метрики: RMS, MAE, Max производной, пересечения нуля
    - Нормализованные данные
    - Производную

#### Ключевые этапы обработки:
1. **Дилатация изображения** (`dilate_image`)
2. **Поиск контуров** (`find_contours`)
3. **Скелетизация** (`find_line_skeleton`)
4. **Нормализация данных** (`plot_combined_graph`)
5. **Интерполяция** (`interpolate_to_length`)

## Интерпретация метрик
| Метрика             | Норма          | Тремор         |
|----------------------|----------------|----------------|
| RMS_derivative      | < 0.1         | > 0.5          |
| MAE_derivative      | < 0.05        | > 0.3          |
| Max_derivative      | < 0.5         | > 2.0          |
| Zero_crossings      | < 50          | > 200          |

## Визуализация
Используйте встроенные функции для отладки:
```python
# Показать изображение с контурами
show_image(contour_image, title='Пример контура')

# Построить графики сигнала и производной
evaluate_smoothness(data, plot=True)
```

## Примеры данных
Тестовые изображения должны:
- Быть в формате JPG/PNG
- Иметь разрешение не менее 300x300 пикселей
- Содержать текст на светлом фоне
- Иметь контраст не менее 70%

```markdown
## Работа с графиками данных

Модуль предоставляет доступ к нормализованным данным и производным для кастомного анализа:

### Структура объекта `graphs`
```python
graphs = {
    'y_normalized': [  # Нормализованные данные расстояний
        np.array([0.12, 0.35, ...]),  # Для буквы 1
        np.array([0.08, 0.41, ...])   # Для буквы 2
    ],
    'yd': [            # Производные сигналов 
        np.array([-0.02, 0.15, ...]), # Для буквы 1
        np.array([0.03, -0.11, ...])  # Для буквы 2
    ]
}
```

### Примеры использования
1. **Визуализация конкретной буквы**
```python
import matplotlib.pyplot as plt

# Выбор буквы (например, первая в списке)
letter_idx = 0 

plt.figure(figsize=(12, 4))
plt.plot(graphs['y_normalized'][letter_idx], label='Нормализованный сигнал')
plt.plot(graphs['yd'][letter_idx], label='Производная', alpha=0.7)
plt.title(f'Анализ буквы {letter_idx+1}')
plt.legend()
plt.grid(True)
plt.show()
```

2. **Собственные метрики на основе графиков**
```python
def custom_metric(y, yd):
    """Пользовательский показатель резкости"""
    return np.percentile(np.abs(yd), 95)

for i, (y, yd) in enumerate(zip(graphs['y_normalized'], graphs['yd'])):
    print(f"Буква {i}: Пиковая резкость = {custom_metric(y, yd):.3f}")
```

3. **Экспорт данных для ML**
```python
import pandas as pd

# Создание датафрейма с признаками
features = []
for y, yd in zip(graphs['y_normalized'], graphs['yd']):
    features.append({
        'mean_y': np.mean(y),
        'std_yd': np.std(yd),
        'max_yd': np.max(yd),
        'zero_cross': np.sum(np.diff(np.sign(yd)) != 0)
    })

df = pd.DataFrame(features)
df.to_csv('tremor_features.csv', index=False)
```

### Ключевые особенности данных
1. **Нормализация**:
   - `y_normalized`: Z-score нормализация + удаление тренда
   - Диапазон значений: ~[-3σ, +3σ]

2. **Производные**:
   - Рассчитаны через `np.diff()`
   - Сохраняют исходную временную динамику
   - Размерность: `len(yd) = len(y_normalized) - 1`

3. **Интерполяция**:
   - Все сигналы приведены к 1000 точкам
   - Гарантированная одинаковная длина массивов

### Типовые сценарии использования
- Построение кастомных визуализаций
- Обучение ML-моделей на сырых данных
- Сравнение паттернов между разными образцами
- Анализ временных характеристик тремора
