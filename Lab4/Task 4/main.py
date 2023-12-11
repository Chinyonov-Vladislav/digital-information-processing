import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk


def plot_filter_frequency(b, a, str_filter):
    w, h = freqz(b, a)
    plt.figure(figsize=(10, 5))
    plt.title(f"Частотная характеристика фильтра {str_filter}")
    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Амплитуда [дб]', color='b')
    plt.xlabel('Частота [рад/единица]')
    plt.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Угол (радианы)', color='g')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    zeros, poles, _ = tf2zpk(b, a)
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='red', label='Полюса')
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='blue', label='Нули')

    # Установка радиуса для круглой диаграммы
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    circle = plt.Circle((0, 0), 1, color='gray', fill=False)
    plt.gca().add_patch(circle)

    plt.title(f'Диаграмма "Полюс-ноль" для фильтра {str_filter}')
    plt.xlabel('Реальная')
    plt.ylabel('Мнимая')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


def main():
    equations = [
        ([0.41], [1, -0.8, 0.24, -0.032, 0.002],"y[n]=0,41x[n] + 0,8y[n-1] - 0,24y[n-2] + 0,032y[n-3] - 0,002y[n-4]"),
        ([0.93], [1, -0.93, 0.86], "y[n]=0,93x[n] – 0,93x[n-1] + 0,86y[n-1]"),
        ([0.32], [1, -0.68], "у[n]=0,32x[n] + 0,68y[n-1]"),
        ([0.36, 0.22, -0.85], [1],"y[n] = 0,36x[n] + 0,22x[n-1] - 0,85x[n-2]"),
        ([0.76, 0.32], [1, -0.15], "у[n] = 0,76x[n] + 0,32x[n-1] + 0,15y[n-1]"),
        ([1, 0, 0, 0, 0, -1], [1], "у[п] = х[п] - х[п-5]"),
        ([0.8], [1, 0.2, 0.3, -0.8], "y[n] = 0,8x[n] – 0,2y[n-1] – 0,3y[n-2] + 0,8y[n-3]"),
        ([1, 0, -1], [1, -0.9, 0.6], "y[n] = x[n] – x[n-2] + 0,9y[n-1] – 0,6y[n-2]")
    ]
    for index, item in enumerate(equations):
        plot_filter_frequency(item[0], item[1], item[2])


if __name__ == '__main__':
    main()
