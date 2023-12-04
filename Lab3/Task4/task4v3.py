import soundfile as sf
import sys
import numpy as np
from scipy.signal import firwin
from HelperMethods import *


def design_low_pass_filter(fs, fc, num_taps):  # Проектирование КИХ-фильтра
    fir_filter = firwin(num_taps, fc / fs, pass_zero='lowpass')
    return fir_filter
    '''nyquist = 0.5 * fs
    cutoff = fc / nyquist
    taps = firwin(num_taps, cutoff, window='hamming')
    return taps'''


def pad_zeros_to(x, new_length):
    """Append new_length - x.shape[0] zeros to x's end via copy."""
    output = np.zeros((new_length,))
    output[:x.shape[0]] = x
    return output


def next_power_of_2(n):
    return 1 << (int(np.log2(n - 1)) + 1)


def fft_convolution(x, h, K=None):
    Nx = x.shape[0]  # Nx - длина входного сигнала x
    Nh = h.shape[0]  # Nh - длина фильтра h
    Ny = Nx + Nh - 1  # Ny - ожидаемая длина выходного сигнала после свертки

    # Если K не задан, выбираем его как ближайшую степень двойки к Ny
    if K is None:
        K = next_power_of_2(Ny)

    # Применяем нулевое дополнение и выполнение FFT к входному сигналу и фильтру
    X = np.fft.fft(pad_zeros_to(x, K))
    H = np.fft.fft(pad_zeros_to(h, K))

    # Поэлементное перемножение преобразованных сигналов в частотной области
    Y = np.multiply(X, H)

    y = np.real(np.fft.ifft(Y))  # Выполняем обратное FFT для получения временного сигнала

    return y[:Ny]  # Возвращаем обрезанный выходной сигнал


def overlap_add_convolution(x, h, B, K=None):
    M = len(x)  # M - длина входного сигнала x
    N = len(h)  # N - длина фильтра h
    # B - размер блока, xp - входной сигнал, дополненный нулями до кратности B
    num_input_blocks = np.ceil(M / B).astype(int)
    xp = pad_zeros_to(x, num_input_blocks * B)
    # Размер выходного сигнала после свертки
    output_size = num_input_blocks * B + N - 1
    y = np.zeros((output_size,))  # создание массива из нулей длиной output_size

    # Итерация по блокам входного сигнала
    for n in range(num_input_blocks):
        xb = xp[n * B:(n + 1) * B]  # Выделение текущего блока из входного сигнала

        u = fft_convolution(xb, h, K)  # Применение быстрой свертки (FFT) к текущему блоку и фильтру

        y[n * B:n * B + len(u)] += u  # Накопление результатов свертки в выходном сигнале

    return y[:M + N - 1]  # Возвращение обрезанного выходного сигнала


def overlap_save_convolution(x, h, B, K=None):
    M = len(x)  # M - длина входного сигнала x
    N = len(h)  # N - длина фильтра h

    # Если K не задан, выбираем его как максимум из B и ближайшей степени двойки к N
    if K is None:
        K = max(B, next_power_of_2(N))

    # Рассчитываем количество блоков входного сигнала, учитывая K
    num_input_blocks = np.ceil(M / B).astype(int) \
                       + np.ceil(K / B).astype(int) - 1

    # Дополняем входной сигнал нулями до кратности B
    xp = pad_zeros_to(x, num_input_blocks * B)
    # Рассчитываем размер выходного сигнала после свертки
    output_size = num_input_blocks * B + N - 1
    y = np.zeros((output_size,))  # создаём массив выходного сигнала y длиной output_size

    xw = np.zeros((K,))  # Инициализация массива с окном для быстрой свертки

    for n in range(num_input_blocks):  # Итерация по блокам входного сигнала
        xb = xp[n * B:n * B + B]  # Выделение текущего блока из входного сигнала

        # Сдвиг окна и обновление его последних B элементов
        xw = np.roll(xw, -B)
        xw[-B:] = xb

        # Применение быстрой свертки (FFT) к текущему блоку и фильтру
        u = fft_convolution(xw, h, K)

        y[n * B:n * B + B] = u[-B:]  # Запись результатов свертки в выходной сигнал

    return y[:M + N - 1]  # Возвращение обрезанного выходного сигнала


def save_filtered_signals(filtered_signal_overlap_add, filtered_signal_overlap_save, fs, name_audio, folder):
    name_audio_without_extension = os.path.splitext(os.path.basename(name_audio))[0]
    path_file_overlap_add = os.path.join(folder, f'filtered_{name_audio_without_extension}_overlap_add.wav')
    sf.write(path_file_overlap_add, filtered_signal_overlap_add, fs)
    path_file_overlap_save = os.path.join(folder, f'filtered_{name_audio_without_extension}_overlap_save.wav')
    sf.write(path_file_overlap_save, filtered_signal_overlap_save, fs)
    sf.write(path_file_overlap_add, filtered_signal_overlap_add, fs)


def main():
    folder_with_filtered_signals = "Filtered_audiofiles"
    folder_with_audiofiles = "Audiofiles"
    names_audiofiles = ["s1.wav", "s2.wav", "s3.wav", "s4.wav", "s5.wav"]
    success, message = checkExistAudiofiles(folder_with_audiofiles, names_audiofiles)
    if success is False:
        print(message)
        sys.exit(-1)
    fc = 1500  # частота среза
    num_taps = 101
    audiofiles = readAudioFiles(folder_with_audiofiles, names_audiofiles)
    fullPathFolder = checkFolderForFilteredSignals(folder_with_filtered_signals)
    new_path_folder_for_filtered_signals = getFullPathForNewFolder(fullPathFolder)
    block_size = 128
    for item in audiofiles:
        name_audio, audio, fs = item
        h = design_low_pass_filter(fs, fc, num_taps)
        overlap_add_filtered_signal = overlap_add_convolution(audio, h, block_size)
        overlap_save_filtered_signal = overlap_save_convolution(audio, h, block_size)
        save_filtered_signals(overlap_add_filtered_signal, overlap_save_filtered_signal, fs, name_audio,
                              new_path_folder_for_filtered_signals)
        data = list()
        data.append((f"Спектрограмма оригинального сигнала {name_audio}", audio, fs))
        data.append((f"Спектрограмма отфильтрованного сигнала с помощью КИХ-фильтра нижних частот с применением "
                     f"метода Overlap-add {name_audio}", overlap_add_filtered_signal, fs))
        data.append((
            f"Спектрограмма отфильтрованного сигнала с помощью КИХ-фильтра нижних частот с применением метода "
            f"Overlap-save {name_audio}",
            overlap_save_filtered_signal, fs))
        plot_spectrogram(data)


if __name__ == '__main__':
    main()
