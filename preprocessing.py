import matplotlib.pyplot as plt
import librosa
import numpy as np

def display_waveform(audio, sr):
    fig, ax = plt.subplots(
        nrows = 1,
        ncols = 1,
        figsize = (20, 5),
        sharex = True,
        sharey = True,
        squeeze = True
    )

    ax.plot(audio)
    ax.set_title('Waveform')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    plt.tight_layout()

    return fig

def display_rms(audio, frame=2048, hop=512):
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]

    fig, ax = plt.subplots(
        nrows = 1,
        ncols = 1,
        figsize = (20, 5),
        sharex = True,
        sharey = True,
        squeeze = True
    )

    ax.plot(rms)
    ax.set_title('RMS')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    plt.tight_layout()

    return fig

def display_zcr(audio, frame=2048, hop=512):
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame, hop_length=hop)[0]

    fig, ax = plt.subplots(
        nrows = 1,
        ncols = 1,
        figsize = (20, 5),
        sharex = True,
        sharey = True,
        squeeze = True
    )

    ax.plot(zcr)
    ax.set_title('ZCR')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    plt.tight_layout()

    return fig

def display_mfcc(audio, sr, frame=2048, hop=512, mfcc_num=25):
    mfcc_spectrum = librosa.feature.mfcc(y=audio, sr=sr, n_fft=frame, hop_length=hop, n_mfcc=mfcc_num)

    delt1 = librosa.feature.delta(mfcc_spectrum, order=1)
    delt2 = librosa.feature.delta(mfcc_spectrum, order=2)

    mfcc_feature = np.concatenate((np.mean(mfcc_spectrum, axis=1), np.mean(delt1, axis=1), np.mean(delt2, axis=1)))

    fig, ax = plt.subplots(
        nrows = 3,
        ncols = 1,
        figsize = (10, 15),
        sharex = True,
        sharey = True,
        squeeze = True
    )

    ax[0].plot(delt1)
    ax[0].set_title('MFCC')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(delt2)
    ax[1].set_title('MFCC')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Amplitude')

    ax[2].plot(mfcc_spectrum)
    ax[2].set_title('MFCC')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Amplitude')

    plt.tight_layout()

    return fig