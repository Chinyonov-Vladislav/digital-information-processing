import os
import sys
import librosa
from matplotlib import pyplot as plt
from scipy import signal

def checkExistAudiofiles(folder_with_audiofiles, names_audiofiles):
    full_path_to_folder_with_audiofiles = os.path.join(os.getcwd(), folder_with_audiofiles)
    if not os.path.isdir(full_path_to_folder_with_audiofiles):
        return False, "Папка с аудиофайлами отсутствует"
    for item in names_audiofiles:
        full_path_to_audiofile = os.path.join(full_path_to_folder_with_audiofiles, item)
        if not os.path.exists(full_path_to_audiofile):
            return False, f"Файл по пути {full_path_to_audiofile} отсутствует"
    return True, "Все файлы найдены"


def readAudioFiles(folder_with_audiofiles, names_audiofiles):
    audiofiles = []
    full_path_to_folder_with_audiofiles = os.path.join(os.getcwd(), folder_with_audiofiles)
    for item in names_audiofiles:
        full_path_to_audiofile = os.path.join(full_path_to_folder_with_audiofiles, item)
        audio, sampleRate = librosa.load(full_path_to_audiofile)
        audiofiles.append((item, audio, sampleRate))
    return audiofiles


def create_low_pass_filter(numtaps, cutoff_hz, nyq_rate):
    return signal.firwin(numtaps, cutoff_hz / nyq_rate, window='hamming')

def create_high_pass_filter_from_low_pass_filter(taps):
    taps = -taps
    taps[len(taps)//2]+=1
    return taps

def create_high_pass_filter(numtaps, cutoff_hz, nyq_rate):
    return signal.firwin(numtaps, cutoff_hz / nyq_rate,
                         window='hamming', pass_zero=False)


def apply_filter_to_signal(fir_filter, signal_input):
    return signal.lfilter(fir_filter, 1.0, signal_input)

def show_spectrogram(data, sample_rate, title):
    plt.specgram(data, Fs=sample_rate, cmap='viridis', aspect='auto')
    plt.xlabel('Время, сек')
    plt.ylabel('Частота, Гц')
    plt.title(title)
    plt.colorbar(label='Интенсивность')
    plt.show()

def show_graphics(name_signal, signal_original, filter_signal_low_pass, filter_signal_highpass_from_low, filter_signal_high_pass):
    plt.figure(figsize=(20, 10))
    plt.subplot(411)
    plt.plot(range(len(signal_original)), signal_original)
    plt.title(f"Оригинальный сигнал из файла {name_signal}")
    plt.subplot(412)
    plt.plot(range(len(filter_signal_low_pass)), filter_signal_low_pass)
    plt.title(f"Отфильтрованный сигнал из файла {name_signal} с помощью КИХ-фильтра нижних частот")
    plt.subplot(413)
    plt.plot(range(len(filter_signal_highpass_from_low)), filter_signal_highpass_from_low)
    plt.title(f"Отфильтрованный сигнал из файла {name_signal} с помощью КИХ-фильтра верхних частот, образованного из фильтра нижних частот")
    plt.subplot(414)
    plt.plot(range(len(filter_signal_high_pass)), filter_signal_high_pass)
    plt.title(f"Отфильтрованный сигнал из файла {name_signal} с помощью КИХ-фильтра верхних частот")
    plt.show()


def main():
    folder_with_audiofiles = "Audiofiles"
    names_audiofiles = ["s1.wav", "s2.wav", "s3.wav", "s4.wav"]
    success, message = checkExistAudiofiles(folder_with_audiofiles, names_audiofiles)
    if success is False:
        print(message)
        sys.exit(-1)
    audiofiles = readAudioFiles(folder_with_audiofiles, names_audiofiles)
    num_taps = 31
    for item in audiofiles:
        name_audio, audio, fs = item
        nyq_rate = fs / 2.0
        cutoff = 1500
        taps_low = create_low_pass_filter(num_taps, cutoff, nyq_rate)
        taps_high_from_low = create_high_pass_filter_from_low_pass_filter(taps_low)
        taps_high = create_high_pass_filter(num_taps, cutoff, nyq_rate)
        filtered_signal_by_low_pass_filter = apply_filter_to_signal(taps_low, audio)
        filtered_signal_by_high_pass_filter_than_been_created_from_low_filter = apply_filter_to_signal(taps_high_from_low, audio)
        filtered_signal_by_high_pass_filter = apply_filter_to_signal(taps_high, audio)
        show_graphics(name_audio, audio, filtered_signal_by_low_pass_filter, filtered_signal_by_high_pass_filter_than_been_created_from_low_filter, filtered_signal_by_high_pass_filter)
        show_spectrogram(audio, fs, f'Спектрограмма исходного сигнала из файла {name_audio}')
        show_spectrogram(filtered_signal_by_low_pass_filter, fs, f'Спектрограмма сигнала из файла {name_audio} после фильтрации с помощью КИХ-фильтра нижних частот')
        show_spectrogram(filtered_signal_by_high_pass_filter_than_been_created_from_low_filter, fs,
                         f'Спектрограмма сигнала из файла {name_audio} после фильтрации с помощью КИХ-фильтра верхних частот, '
                         f'созданного путём применения спектральной инверсии ядра ФНЧ ')
        show_spectrogram(filtered_signal_by_high_pass_filter, fs, f"Спектрограмма сигнала из файла {name_audio} после фильтрации с помощью КИХ-фильтра верхних частот")

if __name__ == '__main__':
    main()
