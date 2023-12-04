import sys
from scipy.signal import butter, lfilter
import soundfile as sf
from HelperMethods import *


def create_lowpass_butterworth_filter(order, fs, fc):
    nyquist = 0.5 * fs
    normal_cutoff = fc / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, b, a):
    filtered_data = lfilter(b, a, data)
    return filtered_data

def save_wav(filtered_signal, fs, name_audio, folder):
    name_audio_without_extension = os.path.splitext(os.path.basename(name_audio))[0]
    path_file_filtered_signal = os.path.join(folder, f'filtered_{name_audio_without_extension}.wav')
    # wav.write(path_file_overlap_add, fs, np.int16(filtered_signal_overlap_add))
    sf.write(path_file_filtered_signal, filtered_signal, fs)

def main():
    folder_with_filtered_signals = "Filtered_audiofiles"
    folder_with_audiofiles = "Audiofiles"
    names_audiofiles = ["s1.wav", "s2.wav", "s3.wav", "s4.wav", "s5.wav"]
    success, message = checkExistAudiofiles(folder_with_audiofiles, names_audiofiles)
    if success is False:
        print(message)
        sys.exit(-1)
    while True:
        try:
            fc = int(input("Введите частоту среза для фильтра: "))
            if fc <= 0:
                raise Exception("значение частоты среза должно быть больше 0")
            break
        except Exception as ex:
            print(f"Ошибка: {ex}. Повторите попытку!")
    while True:
        try:
            order = int(input("Введите порядок фильтра (целое положительное число): "))
            if order < 0:
                raise Exception("порядок фильтра должен быть целым положительным числом")
            break
        except Exception as ex:
            print(f"Ошибка: {ex}. Повторите попытку!")
    audiofiles = readAudioFiles(folder_with_audiofiles, names_audiofiles)
    fullPathFolder = checkFolderForFilteredSignals(folder_with_filtered_signals)
    new_path_folder_for_filtered_signals = getFullPathForNewFolder(fullPathFolder)
    for item in audiofiles:
        name_audio, audio, fs = item
        b, a = create_lowpass_butterworth_filter(order, fs, fc)
        filtered_data = apply_filter(audio, b, a)
        save_wav(filtered_data, fs, name_audio,new_path_folder_for_filtered_signals )
        data = list()
        data.append((f"Спектрограмма оригинального аудиосигнала {name_audio}", audio, fs))
        data.append((f"Спектрограмма отфильтрованного аудиосигнала {name_audio}",filtered_data, fs ))
        plot_spectrogram(data)

if __name__ == '__main__':
    main()