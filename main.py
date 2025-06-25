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
left_indent_cv, right_indent_cv = evaluate_indent(dilated)
char_height_cv = evaluate_width(dilated)
line_spacing_mean, line_spacing_cv = evaluate_line_spacing(dilated)
stroke_discontinuity = evaluate_stroke_discontinuity(contour_images)

print(f'1. Наклон строки:\n'
      f'mean_angle: {mean_angle}\n'
      f'angle_std: {angle_std}\n\n'
      f'3. Межстрочные интервалы:\n'
      f'line_spacing_mean: {line_spacing_mean}\n'
      f'line_spacing_cv: {line_spacing_cv}\n\n'
      f'4. Поля:\n'
      f'left_indent_cv: {left_indent_cv}\n'
      f'right_indent_cv: {right_indent_cv}\n\n'
      f'6. Размер символов:\n'
      f'char_height_cv: {char_height_cv}\n\n'
      f'8. Связность линий:\n'
      f'stroke_discontinuity: {stroke_discontinuity}')

