import librosa
import pickle
import numpy as np

def zcr(audio, frame=2048, hop=512):
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame, hop_length=hop)[0]

    return zcr

def rms(audio, frame=2048, hop=512):
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]

    return rms

def mfcc(audio, sr, frame=2048, hop=512, mfcc_num=25):
    mfcc_spectrum = librosa.feature.mfcc(y=audio, sr=sr, n_fft=frame, hop_length=hop, n_mfcc=mfcc_num)

    delt1 = librosa.feature.delta(mfcc_spectrum, order=1)
    delt2 = librosa.feature.delta(mfcc_spectrum, order=2)

    mfcc_feature = np.concatenate((np.mean(mfcc_spectrum, axis=1), np.mean(delt1, axis=1), np.mean(delt2, axis=1)))

    return mfcc_feature

def extract_feature(audio, sr, frame=2048, hop=512, mfcc_num=25):
    audio_mfcc, audio_zcr, audio_rmse = [],[],[]

    mfcc_score = mfcc(audio, sr, frame=frame, hop=hop, mfcc_num=mfcc_num)
    zcr_score = np.mean(zcr(audio, frame=frame, hop=hop))
    rmse_score = np.mean(rms(audio, frame=frame, hop=hop))

    audio_mfcc.append(mfcc_score)
    audio_zcr.append(zcr_score)
    audio_rmse.append(rmse_score)
    
    feature = np.column_stack((audio_mfcc, audio_zcr, audio_rmse))

    return feature

def prediction_data(model, df):
    y_pred = model.predict(df)

    labels = {
        0: "Bahagia",
        1: "Sedih"
    }
    pred = labels[y_pred[0]]

    return pred

def normalize_feature(feature):
    scaler = pickle.load(open("scaler_svm_2000data_new_1.pkl", "rb"))
    feature = scaler.transform(feature)

    return feature

def prediction(audio, sr):
    model = pickle.load(open("svm_2000data_new_1_fix.pkl", "rb"))
  
    df = extract_feature(audio, sr)
    x = normalize_feature(df)
    pred = prediction_data(model, x)

    return pred