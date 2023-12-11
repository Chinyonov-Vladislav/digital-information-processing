import matplotlib.pyplot as plt
from scipy.signal import firwin, step, TransferFunction


def design_lowpass_filter(order, cutoff_freq, fs):
    # Вычисление коэффициентов КИХ-фильтра с использованием оконной функции
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    taps = firwin(order + 1, normal_cutoff, window='hamming')
    return taps


def plot_impulse_response(taps):
    # Построение импульсной характеристики
    plt.subplot(2, 1, 1)
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Импульсная характеристика')
    plt.grid(True)


def plot_transient_response(time, response):
    plt.subplot(2, 1, 2)
    # Построение графика переходной функции
    plt.plot(time, response)
    plt.title('Переходная характеристика фильтра')
    plt.xlabel('Время (в секундах)')
    plt.ylabel('Амплитуда')
    plt.grid(True)


def main():
    # Параметры фильтра
    order = 31  # Порядок фильтра
    cutoff_freq = 1000.0  # Частота среза в Гц
    fs = 8000.0  # Частота дискретизации в Гц

    # Разработка КИХ-фильтра
    numerator = design_lowpass_filter(order, cutoff_freq, fs)
    denumerator = [1.0] + [0.0] * order # знаменатель всегда равен 0: Это связано с тем, что входной сигнал умножается на коэффициенты фильтра в конечном числе точек (taps), и фильтр не имеет обратной связи
    H = TransferFunction(numerator, denumerator)
    time, response = step(H)
    # Построение импульсной и переходной характеристик
    plt.figure(figsize=(10, 6))
    plot_impulse_response(numerator)
    plot_transient_response(time, response)
    plt.show()


if __name__ == "__main__":
    main()
