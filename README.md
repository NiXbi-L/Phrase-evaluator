# Phrase Evaluator Module

Модуль для анализа рукописного текста на изображениях с вычислением различных метрик качества письма.

## Основные функции

Модуль позволяет вычислять следующие метрики:
- Наклон строки (угол наклона и стабильность)
- Стабильность отступов (поля)
- Межстрочные интервалы
- Размер символов
- Связность линий (отсутствие разрывов)

## Структура модуля

```
Phrase_evaluator/
├── Evaluator.py        # Основной модуль вычисления метрик
├── preprocessing.py    # Предобработка изображений
├── Dev_tools.py        # Инструменты визуализации
```

## Требования

- Python 3.7+
- Зависимости:
  ```
  opencv-python
  numpy
  scipy
  scikit-image
  matplotlib
  easyocr
  ```

Установите зависимости командой:
```bash
pip install opencv-python numpy scipy scikit-image matplotlib easyocr
```

## Функции модуля и взаимодействие

### Предобработка изображений

- **read_img(img_path: str) -> np.array**
  - Загружает изображение из файла.
  - **Параметры:** путь к изображению.
  - **Возвращает:** изображение в формате numpy array.

- **preprocess(cv_image: np.array) -> (list[np.array], np.array)**
  - Выполняет морфологическую обработку и выделяет контуры.
  - **Параметры:** исходное изображение.
  - **Возвращает:** список изображений контуров, обработанное бинарное изображение.

### Метрики качества письма

- **evaluate_angle(bin_image: np.array) -> (float, float)**
  - Оценивает средний угол наклона строк и его стандартное отклонение.
  - **Параметры:** бинаризованное изображение.
  - **Возвращает:** (mean_angle, angle_std).

- **evaluate_indent(bin_image: np.array) -> (float, float)**
  - Оценивает стабильность левого и правого отступов (коэффициенты вариации).
  - **Параметры:** бинаризованное изображение.
  - **Возвращает:** (left_indent_cv, right_indent_cv).

- **evaluate_width(bin_image: np.array) -> float**
  - Оценивает средний размер символов (высоту).
  - **Параметры:** бинаризованное изображение.
  - **Возвращает:** средняя высота символов.

- **evaluate_line_spacing(bin_image: np.array) -> (float, float)**
  - Оценивает средний межстрочный интервал и его вариацию.
  - **Параметры:** бинаризованное изображение.
  - **Возвращает:** (line_spacing_mean, line_spacing_cv).

- **evaluate_stroke_discontinuity(contour_images: list[np.ndarray]) -> int**
  - Оценивает количество разрывов в штрихах.
  - **Параметры:** список изображений контуров.
  - **Возвращает:** количество разрывов.

### Пример использования

```python
from Phrase_evaluator.Evaluator import (
    evaluate_angle,
    evaluate_indent,
    evaluate_width,
    evaluate_line_spacing,
    evaluate_stroke_discontinuity
)
from Phrase_evaluator.preprocessing import preprocess, read_img

img = read_img('img/Screenshot_82.png')

contour_images, dilated = preprocess(img)

mean_angle, angle_std = evaluate_angle(dilated)
left_indent_std, right_indent_std = evaluate_indent(dilated)
char_height_mean = evaluate_width(dilated)
line_spacing_mean, line_spacing_cv = evaluate_line_spacing(dilated)
stroke_discontinuity = evaluate_stroke_discontinuity(contour_images)

print(f'1. Наклон строки:\n'
      f'mean_angle: {mean_angle}\n'
      f'angle_std: {angle_std}\n\n'
      f'3. Межстрочные интервалы:\n'
      f'line_spacing_mean: {line_spacing_mean}\n'
      f'line_spacing_cv: {line_spacing_cv}\n\n'
      f'4. Поля:\n'
      f'left_indent_std: {left_indent_std}\n'
      f'right_indent_std: {right_indent_std}\n\n'
      f'6. Размер символов:\n'
      f'char_height_mean: {char_height_mean}\n\n'
      f'8. Связность линий:\n'
      f'stroke_discontinuity: {stroke_discontinuity}')


```

## Описание метрик

1. **Наклон строки**
   - `mean_angle`: Средний угол наклона строк (градусы) [90; -90]
   - `angle_std`: Стандартное отклонение угла наклона [90; -90]

2. **Стабильность отступов**
   - `left_indent_cv`: Коэффициент вариации левого отступа [0; 1]
   - `right_indent_cv`: Коэффициент вариации правого отступа [0; 1]

3. **Межстрочные интервалы**
   - `line_spacing_mean`: Среднее расстояние между строками (0; 2)
   - `line_spacing_cv`: Коэффициент вариации интервалов [0; 1]

4. **Размер символов**
   - `char_height_cv`: Коэффициент вариации высоты символов [0; 1]

5. **Связность линий**
   - `stroke_discontinuity`: Количество разрывов в штрихах [0; +inf)

## Инструменты визуализации

Модуль `Dev_tools.py` содержит функции для визуализации:
- `plot_multiple_graphs()`: Наложение нескольких графиков
- `show_image()`: Отображение изображений
- `plot_multiple_points()`: Визуализация точек с bounding box
- `plot_axis()`: Анализ распределения точек
- `plot_paired_groups()`: Визуализация парных групп точек
- `plot_segment()`: Анализ угла наклона отрезка

Пример использования:
```python
from Phrase_evaluator.Dev_tools import show_image, plot_paired_groups

# Показать обработанное изображение
show_image(dilated, title='Processed Image')

# Визуализировать группы точек
res, _, _ = process_points(dilated)
plot_paired_groups(res)
```

## Что нового?
1. **Убрана метрика** `left_right_margin_diff`
2. **Замена метрик** (`left_indent_std` и `right_indent_std`) на (`left_indent_cv` и `right_indent_cv`)
3. **Замена метрики** `char_height_mean` на `char_height_cv`
4. **Измененные выводы смотреть в блоке функции**
5. **Добавлен подробный гайд по визуализации** — файл `Visualisation.md` содержит описание всех функций из Dev_tools.py, примеры использования и таблицу соответствия между переменными и функциями визуализации для удобного отображения любых данных из проекта.

Модуль предназначен для анализа качества рукописного текста и может использоваться в образовательных целях или системах автоматической оценки письменных работ.