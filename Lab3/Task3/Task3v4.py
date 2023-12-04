import sys
import numpy as np
from scipy.fft import fft
from scipy.signal import spectrogram
from HelperMethods import *


def main():
    folder_with_audiofiles = "Audiofiles"
    names_audiofiles = ["s1.wav", "s2.wav", "s3.wav", "s4.wav", "s5.wav"]
    success, message = checkExistAudiofiles(folder_with_audiofiles, names_audiofiles)
    if success is False:
        print(message)
        sys.exit(-1)
    audiofiles = readAudioFiles(folder_with_audiofiles, names_audiofiles)
    for item in audiofiles: # item состоит из трех элементов:
        # название файла, из которого получен аудиосигнал
        # сигнал во временной области
        # частота дискретизации
        length_signal = len(item[1])
        freqs = np.fft.fftfreq(length_signal, 1 / item[2]) # получение массива частот
        fft_values = fft(item[1]) #быстрое преобразование Фурье
        magnitude_spectrum = np.abs(fft_values)
        phase_spectrum = np.angle(fft_values)

        frequencies, times, Sxx = spectrogram(item[1], fs=item[2])

        plt.subplot(3, 1, 1)
        plt.plot(freqs[:length_signal // 2], magnitude_spectrum[:length_signal // 2])
        plt.title(f'Амплитудный спектр аудиосигнала из файла {item[0]}')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Амплитуда')

        plt.subplot(3, 1, 2)
        plt.plot(freqs[:length_signal // 2], phase_spectrum[:length_signal // 2])
        plt.title(f'Фазовый спектр аудиосигнала из файла {item[0]}')
        plt.xlabel('Частота (Гц)')
        plt.ylabel('Фаза (радианы)')

        plt.subplot(3,1,3)
        plt.pcolormesh(times, frequencies,
                       10 * np.log10(Sxx))  # используем логарифмический масштаб для улучшения визуализации
        plt.ylabel('Частота [Гц]')
        plt.xlabel('Время [сек]')
        plt.title(f'Спектрограмма сигнала из аудио файла {item[0]}')
        plt.colorbar(label='Интенсивность (dB)')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()