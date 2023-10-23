import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sounddevice as sd
import os
import shutil
import librosa.feature

path_folder = "audiofiles_task2_2"


def task2_2():
    f1, f2, f3 = 1500, 4500, 6100
    fs = 16000
    duration = 0.6
    print("Генерация сигнала s1")
    # Создаем временную ось от 0 до duration с шагом 1/fs
    t1 = np.arange(0, duration, 1 / fs)
    s1 = np.sin(2 * np.pi * f1 * t1) + np.sin(2 * np.pi * f2 * t1) + np.sin(2 * np.pi * f3 * t1)
    print("Генерация шумового сигнала sn")
    sn = np.random.normal(0, 1, int(duration * fs))
    print(f"Len s1 = {len(s1)}, Len sn = {len(sn)}")
    print("Генерация зашумленного сигнала - s2 путём накладывания сигнала s1 и sn")
    s2 = s1 + sn
    print("Генерация сигнала s3 - путём сдвига s1 на 300 выборок")
    s3 = np.roll(s1, 300)
    print("Генерация сигнала s4")
    fs4 = 11025
    t2 = np.arange(0, duration, 1 / fs4)
    s4 = np.sin(2 * np.pi * f1 * t2) + np.sin(2 * np.pi * f2 * t2) + np.sin(2 * np.pi * f3 * t2)
    signals = [s1, s2, s3, s4]
    print("Построение 50 выборок каждого сигнала")
    plt.figure(figsize=(12, 6))
    for index, signal in enumerate(signals):
        if len(signal) >= 50:
            plt.subplot(2, 2, index + 1)
            plt.plot(signal[:50])
            plt.title(f'Сигнал s{index + 1}')
    plt.tight_layout()
    plt.show()
    print("Сохранение каждого сигнала в файл WAVE")
    full_path_folder = os.path.join(os.getcwd(), path_folder)
    if os.path.exists(full_path_folder):
        shutil.rmtree(full_path_folder)
    os.makedirs(full_path_folder)
    for index, signal in enumerate(signals):
        if index == len(signals) - 1:
            wavfile.write(os.path.join(full_path_folder, f's{index + 1}.wav'), fs4, signal.astype(np.float32))
        else:
            wavfile.write(os.path.join(full_path_folder, f's{index + 1}.wav'), fs, signal.astype(np.float32))
    path_to_s1_wav = os.path.join(full_path_folder, 's1.wav')
    if os.path.exists(path_to_s1_wav) and os.path.isfile(path_to_s1_wav):
        print("Проигрывание файла s1.wav")
        sample_rate, audio_data = wavfile.read(path_to_s1_wav)
        sd.play(audio_data, sample_rate)
        sd.wait()
    else:
        print("Файл s1.wav не найден!")
    path_to_s4_wav = os.path.join(full_path_folder, 's4.wav')
    if os.path.exists(path_to_s4_wav) and os.path.isfile(path_to_s4_wav):
        print("Проигрывание файла s4.wav")
        sample_rate, audio_data = wavfile.read(path_to_s4_wav)
        sd.play(audio_data, sample_rate)
        sd.wait()
    else:
        print("Файл s4.wav не найден!")


def task2_3():
    full_path_folder = os.path.join(os.getcwd(), path_folder)
    path_to_files = [
        os.path.join(full_path_folder, "s1.wav"), os.path.join(full_path_folder, "s2.wav"),
        os.path.join(full_path_folder, "s3.wav"), os.path.join(full_path_folder, "s4.wav")]
    files_exists = True
    for file in path_to_files:
        if not os.path.exists(file):
            files_exists = False
            break
    if not files_exists:
        print("Файлы .wav не найдены!")
        return
    sample_rate1, signal_s1 = wavfile.read(path_to_files[0])
    sample_rate2, signal_s2 = wavfile.read(path_to_files[1])
    sample_rate3, signal_s3 = wavfile.read(path_to_files[2])
    signal_x = np.array([1, 5, 3, 2, 6])
    signal_h = np.array([2, 3, 1])
    print("Вычисление свертки")
    convolution_result_x_h = np.convolve(signal_x, signal_h, mode='full')
    print(f"Convolution using Library Method: {convolution_result_x_h}")
    print(f"Convolution using My Method: {my_convolution(signal_x, signal_h)}")
    convolution_result_s1_s2 = np.convolve(signal_s1, signal_s2, mode='full')
    convolution_result_s1_s3 = np.convolve(signal_s1, signal_s3, mode='full')
    print("Вычисление взаимной корреляции")
    correlation_result_x_h = np.correlate(signal_x, signal_h, mode='full')
    print(f"Correlation using Library Method: {correlation_result_x_h}")
    print(f"Correlation using My Method: {my_correlation(signal_x, signal_h)}")
    correlation_result_s1_s2 = np.correlate(signal_s1, signal_s2, mode='full')
    correlation_result_s1_s3 = np.correlate(signal_s1, signal_s3, mode='full')
    print("Отображение результатов")
    plt.figure(figsize=(12, 9))

    plt.subplot(3, 2, 1)
    plt.title('Convolution x and h')
    plt.stem(convolution_result_x_h, use_line_collection=True)

    plt.subplot(3, 2, 2)
    plt.title('Convolution s1 and s2')
    plt.stem(convolution_result_s1_s2, use_line_collection=True)

    plt.subplot(3, 2, 3)
    plt.title('Convolution s1 and s3')
    plt.stem(convolution_result_s1_s3, use_line_collection=True)

    plt.subplot(3, 2, 4)
    plt.title('Correlation x and h')
    plt.stem(correlation_result_x_h, use_line_collection=True)

    plt.subplot(3, 2, 5)
    plt.title('Correlation s1 and s2')
    plt.stem(correlation_result_s1_s2, use_line_collection=True)

    plt.subplot(3, 2, 6)
    plt.title('Correlation s1 and s3')
    plt.stem(correlation_result_s1_s3, use_line_collection=True)

    plt.tight_layout()
    plt.show()


def task2_4():
    path = ""
    while path is "":
        try:
            path = str(input("Введите полный путь к файлу, который содержит сигнал (.wav файл): "))
            if not os.path.exists(path):
                raise ValueError("введенный путь не существует!")
            if not os.path.isfile(path):
                raise ValueError("введенный путь не является файлом")
            if not path.endswith(".wav"):
                raise ValueError("введенный файл не является файлом .wav")
        except ValueError as e:
            print(f"Ошибка: {e}")
            path = ""
    sample_rate, signal = wavfile.read(path)
    print(f"Частота дискретизации = {sample_rate}")
    print(f"Длина сигнала = {len(signal)}")
    start_index_segment_signal = -1
    finish_index_segment_signal = -1
    if len(signal) > 0:
        while start_index_segment_signal == -1 and finish_index_segment_signal == -1:
            try:
                print(f"Выберите сегмент сигнала в пределах его длины: от 1 до {len(signal)}")
                start_index_segment_signal = int(input("Введите стартовый индекс сегмента: ")) - 1
                if start_index_segment_signal < 0 or start_index_segment_signal >= len(signal):
                    raise ValueError("Неверное значение для стартового индекса сегмента")
                finish_index_segment_signal = int(input("Введите конечный индекс сегмента: ")) - 1
                if finish_index_segment_signal <= start_index_segment_signal or finish_index_segment_signal >= len(
                        signal):
                    raise ValueError("Неверное значение для конечного индекса сегмента")
                break
            except ValueError as e:
                print(f"Ошибка: {e}")
                start_index_segment_signal = -1
                finish_index_segment_signal = -1
        segment = signal[start_index_segment_signal:finish_index_segment_signal + 1]
        print(f"Segment = {segment}")
        print("Вычисление энергии сегмента сигнала")
        print(f"Энергия сегмента сигнала = {np.sum(np.square(segment))}")
        print("Вычисление скорости пересечения нуля")
        # разобраться со скоростью пересечения нуля
        print(f"Скорость пересечения нуля (с помощью самописной функции) = {my_zero_cross_rate(segment)}")
        print(f"Скорость пересечения нуля (с помощью метода из библиотеки Librosa) = {librosa.feature.zero_crossing_rate(segment)}")
        print("Вычисление среднего значения сегмента сигнала")
        print(f"Среднее значение сегмента сигнала= {np.mean(segment)}")
        print("Вычисление дисперсии сигнала")
        print(f"Дисперсия сегмента сигнала = {np.var(segment)}")
    else:
        print("Сигнал не имеет длины!")


def my_convolution(signal1, signal2):
    len1, len2 = len(signal1), len(signal2)
    result_len = len1 + len2 - 1
    # Инициализация результата нулями
    result = [0] * result_len
    # Выполнение свертки
    for n in range(result_len):
        for k in range(max(0, n - len2 + 1), min(n + 1, len1)):
            result[n] += signal1[k] * signal2[n - k]
    return result


def my_correlation(signal1, signal2):
    len1 = len(signal1)
    len2 = len(signal2)
    result_length = len1 + len2 - 1
    # Заполним нулями массив, в который будем записывать результат
    result = [0] * result_length
    # Реверсируем signal2
    signal2 = signal2[::-1]
    # Вычисляем взаимную корреляцию
    for i in range(result_length):
        for j in range(len1):
            if 0 <= i - j < len2:
                result[i] += signal1[j] * signal2[i - j]
    return result


def my_zero_cross_rate(segment_signal):
    count = 0
    for index in range(1, len(segment_signal)):
        if segment_signal[index - 1] < 0 and segment_signal[index] == 0.0:
            count += 1
            continue
        if segment_signal[index - 1] * segment_signal[index] < 0:
            count += 1
    return count / len(segment_signal)


def main():
    while True:
        try:
            choice = int(input(
                "Выберите задание:\n1 - Упражнение 2.2\n2 - Упражнение 2.3\n3 - Упражнение 2.4\n4 - Выход\nВаш выбор: "))
            if choice not in [1, 2, 3, 4]:
                raise ValueError("введен неверный номер. Повторите попытку!")
            if choice == 1:
                task2_2()
            if choice == 2:
                task2_3()
            if choice == 3:
                task2_4()
            if choice == 4:
                break
        except ValueError as e:
            print(f"Ошибка: {e}")


if __name__ == '__main__':
    main()
