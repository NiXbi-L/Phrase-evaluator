# Визуализация: функции из Dev_tools.py

Функции визуализации нужны только для разработки и анализа промежуточных результатов. Импортировать их рекомендуется непосредственно перед использованием.

---

## Соответствие типов данных, переменных и функций визуализации

| Тип данных                | Примеры переменных из кода                | Рекомендуемая функция визуализации         |
|--------------------------|-------------------------------------------|--------------------------------------------|
| numpy-массив (1 изображение) | `dilated`, любые бинарные/обработанные изображения | `show_image(dilated)`                      |
| список numpy-массивов     | `contour_images`                         | `show_image(contour_images)`               |
| список точек [(x, y), ...]| `points`, `pts`, `coords`, любые списки точек | `plot_multiple_points([points])`           |
| словарь уровней с группами| `res` из `process_points`                | `plot_paired_groups(res)`                  |
| массив/список чисел       | `y_values`, метрики, сигналы, любые массивы чисел | `plot_multiple_graphs([y1, y2, ...])`      |
| две точки (x, y)          | `A`, `B`                                 | `plot_segment(A, B)`                       |
| списки точек + центр      | `sorted_points`, `center`, `all_points`  | `plot_axis(sorted_points, center, all_points)` |

---

## Описание функций визуализации

### 1. show_image
**Назначение:** Показать одно или несколько изображений (например, результат предобработки, бинаризации, контуры).

**Когда использовать:** После этапа предобработки, чтобы посмотреть, как выглядит обработанное изображение или контуры.

**Пример:**
```python
from Phrase_evaluator.Dev_tools import show_image
show_image(dilated, titles='Dilated')
show_image(contour_images, titles=['Contour 1', 'Contour 2', ...])
```

---

### 2. plot_multiple_graphs
**Назначение:** Построить несколько графиков на одной оси (например, сравнение разных метрик или сигналов).

**Когда использовать:** Для сравнения нескольких массивов данных (например, нормализованные значения разных строк).

**Пример:**
```python
from Phrase_evaluator.Dev_tools import plot_multiple_graphs
plot_multiple_graphs([y1, y2, y3], labels=['Line 1', 'Line 2', 'Line 3'])
```

---

### 3. plot_multiple_points
**Назначение:** Визуализировать наборы точек, опционально с bounding box и центром.

**Когда использовать:** Для анализа распределения точек (например, после скелетизации или кластеризации), для отладки группировки точек.

**Пример:**
```python
from Phrase_evaluator.Dev_tools import plot_multiple_points
plot_multiple_points([points], box=True, centre=True)
```

---

### 4. plot_axis
**Назначение:** Показать исходные точки, центр и лучи, а также график расстояний от центра.

**Когда использовать:** Для анализа распределения точек относительно центра (например, после нормализации).

**Пример:**
```python
from Phrase_evaluator.Dev_tools import plot_axis
plot_axis(sorted_points, center, all_points)
```

---

### 5. plot_paired_groups
**Назначение:** Визуализировать парные группы точек (левые и правые) для каждого уровня.

**Когда использовать:** После получения результата из process_points для визуального анализа группировки строк.

**Пример:**
```python
from Phrase_evaluator.Dev_tools import plot_paired_groups
from Phrase_evaluator.Evaluator import process_points
res, _, _ = process_points(dilated)
plot_paired_groups(res)
```

---

### 6. plot_segment
**Назначение:** Показать отрезок между двумя точками и его угол наклона.

**Когда использовать:** Для анализа углов между точками (например, при анализе наклона строки).

**Пример:**
```python
from Phrase_evaluator.Dev_tools import plot_segment
plot_segment(A, B)
```