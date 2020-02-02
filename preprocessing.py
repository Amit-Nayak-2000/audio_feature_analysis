import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import opencv

def main():
    #df = load_csv()
    #preprocessing(df)
    #plot_classdistribution(df)
    #preprocessing_chromagram(df)
    #preprocessing_spectral(df)
    
def load_csv():
    filename = 'dataset/metadata/UrbanSound8K.csv'
    df = pd.read_csv(filename)
    return df

def plot_classdistribution(df):
    fig = df.hist(column="classID")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Density")
    plt.savefig("classdist.png")


# def load_data(testingfold):

def preprocessing_MFCC(df):
    import os
    import librosa
    import librosa.display

    audio_location = "dataset/audio/"
    mfcc_location = "dataset/mfcc_figs/"

    if not os.path.exists(mfcc_location):
        os.mkdir(mfcc_location)
    
    for index, row in df.iterrows():
        wavfilename = str(row['slice_file_name'])
        sig, fs = librosa.load(audio_location + "fold" + str(row['fold']) + "/" + wavfilename)
        save_path = mfcc_location + wavfilename.split('.')[0] + '.png'
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        S = librosa.feature.melspectrogram(y = sig, sr = fs)
        librosa.display.specshow(librosa.power_to_db(S, ref = np.max))
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
        plt.close()
        print(index)


def preprocessing_chromagram(df):
    import os
    import librosa
    import librosa.display

    audio_location = "dataset/audio/"
    chroma_location = "dataset/chroma_figs/"

    if not os.path.exists(chroma_location):
        os.mkdir(chroma_location)
    
    for index, row in df.iterrows():
        wavfilename = str(row['slice_file_name'])
        sig, fs = librosa.load(audio_location + "fold" + str(row['fold']) + "/" + wavfilename)
        save_path = chroma_location + wavfilename.split('.')[0] + '.png'
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        S = np.abs(librosa.stft(sig))
        librosa.display.specshow(librosa.power_to_db(S, ref = np.max))
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
        plt.close()
        print(index)

def preprocessing_spectral(df):
    import os
    import librosa
    import librosa.display

    audio_location = "dataset/audio/"
    spectral_location = "dataset/spectral_figs/"

    if not os.path.exists(spectral_location):
        os.mkdir(spectral_location)
    
    for index, row in df.iterrows():
        wavfilename = str(row['slice_file_name'])
        sig, fs = librosa.load(audio_location + "fold" + str(row['fold']) + "/" + wavfilename)
        save_path = spectral_location + wavfilename.split('.')[0] + '.png'
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        S = np.abs(librosa.stft(sig))
        contrast = librosa.feature.spectral_contrast(S=S, sr=fs)
        librosa.display.specshow(contrast)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
        plt.close()
        print(index)


if __name__ == "__main__":
    main()