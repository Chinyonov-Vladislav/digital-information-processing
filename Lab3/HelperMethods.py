import os
import matplotlib.pyplot as plt
import librosa


def readAudioFiles(folder_with_audiofiles, names_audiofiles):
    audiofiles = []
    full_path_to_folder_with_audiofiles = os.path.join(os.getcwd(), folder_with_audiofiles)
    for item in names_audiofiles:
        full_path_to_audiofile = os.path.join(full_path_to_folder_with_audiofiles, item)
        audio, sampleRate = librosa.load(full_path_to_audiofile)
        audiofiles.append((item, audio, sampleRate))
    return audiofiles


def checkExistAudiofiles(folder_with_audiofiles, names_audiofiles):
    full_path_to_folder_with_audiofiles = os.path.join(os.getcwd(), folder_with_audiofiles)
    if not os.path.isdir(full_path_to_folder_with_audiofiles):
        return False, "Папка с аудиофайлами отсутствует"
    for item in names_audiofiles:
        full_path_to_audiofile = os.path.join(full_path_to_folder_with_audiofiles, item)
        if not os.path.exists(full_path_to_audiofile):
            return False, f"Файл по пути {full_path_to_audiofile} отсутствует"
    return True, "Все файлы найдены"


def checkFolderForFilteredSignals(folder):
    full_path_folder = os.path.join(os.getcwd(), folder)
    if not os.path.isdir(full_path_folder):
        os.makedirs(full_path_folder, exist_ok=True)
    return full_path_folder


def getFullPathForNewFolder(directory):
    # Получить список всех файлов и папок в указанной директории
    items = os.listdir(directory)

    # Отфильтровать только папки
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]

    template = "try"
    max_try_folder = 0
    for folder in folders:
        if folder.startswith(template):
            number_try_from_folder_name = int(folder[3:])
            if number_try_from_folder_name > max_try_folder:
                max_try_folder = number_try_from_folder_name
    new_name_folder = template + str(max_try_folder + 1)
    new_full_path = os.path.join(directory, new_name_folder)
    os.makedirs(new_full_path, exist_ok=True)
    return new_full_path


def plot_spectrogram(data):
    fig, ax = plt.subplots(nrows=len(data), ncols=1)
    fig.subplots_adjust(hspace=0.5)
    for index, item in enumerate(data):
        ax[index].specgram(item[1], Fs=item[2], cmap='viridis')
        ax[index].set_title(item[0])
        ax[index].set_xlabel('Время (сек)')
        ax[index].set_ylabel('Частота (Гц)')
    plt.show()
