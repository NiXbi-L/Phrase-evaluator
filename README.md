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

## Использование

Основной сценарий работы:

```python
from Phrase_evaluator.Evaluator import (
    evaluate_angle,
    evaluate_indent,
    evaluate_width,
    evaluate_line_spacing,
    evaluate_stroke_discontinuity
)
from Phrase_evaluator.preprocessing import preprocess, read_img

img = read_img('img/Screenshot_110.png')

contour_images, dilated = preprocess(img)

mean_angle, angle_std = evaluate_angle(dilated)
indent_std, left_right_margin_diff = evaluate_indent(dilated)
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
      f'left_right_margin_diff: {left_right_margin_diff}\n'
      f'left_indent_std: {indent_std[0]}\n'
      f'right_indent_std: {indent_std[1]}\n\n'
      f'6. Размер символов:\n'
      f'char_height_mean: {char_height_mean}\n\n'
      f'8. Связность линий:\n'
      f'stroke_discontinuity: {stroke_discontinuity}')


```

## Описание метрик

1. **Наклон строки**
   - `mean_angle`: Средний угол наклона строк (градусы)
   - `angle_std`: Стандартное отклонение угла наклона

2. **Стабильность отступов**
   - `left_indent_std`: Стандартное отклонение левого отступа (σ)
   - `right_indent_std`: Стандартное отклонение правого отступа (σ)
   - `left_right_margin_diff`: Разница между левым и правым полем (px)

3. **Межстрочные интервалы**
   - `line_spacing_mean`: Среднее расстояние между строками (px)
   - `line_spacing_cv`: Коэффициент вариации интервалов (%)

4. **Размер символов**
   - `char_height_mean`: Средняя высота символов (px)

5. **Связность линий**
   - `stroke_discontinuity`: Количество разрывов в штрихах

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

## Особенности работы

1. **Предобработка изображения**:
   - Преобразование в градации серого
   - Бинаризация с адаптивным порогом
   - Морфологические операции
   - Скелетизация текста

2. **Анализ структуры текста**:
   - Группировка точек по вертикальным линиям
   - Проекция точек на границы
   - Выделение строк и символов

3. **Вычисление метрик**:
   - Статистический анализ распределений
   - Расчет углов и расстояний
   - Оценка вариативности параметров

Модуль предназначен для анализа качества рукописного текста и может использоваться в образовательных целях или системах автоматической оценки письменных работ.