import streamlit as st
#from audio_recorder_streamlit import audio_recorder
from svm import extract_feature, prediction, normalize_feature
from preprocessing import display_waveform, display_rms, display_zcr, display_mfcc
import librosa
import pandas as pd

def tab_hasil(audio, sr):
  tab_hasil, tab_preprocessing, tab_feature = st.tabs(['Hasil Identifikasi', 'Preprocessing Audio', 'Feature Extraction'])

  with tab_hasil:
    st.subheader("Hasil Identifikasi Emosi")
    pred = prediction(audio, sr)

    st.write("Emosi yang dihasilkan merupakan emosi")
    st.write(f"<h4>{pred}</h4>", unsafe_allow_html=True)

  with tab_preprocessing:
    st.subheader("Preprocessing Audio")
    st.markdown('<br>', unsafe_allow_html=True)

    st.write('Waveform')
    waveform = display_waveform(audio, sr)
    st.pyplot(waveform)

    st.write('---')

    st.write('Root Mean Square Energy (RMSE)')
    disp_rms = display_rms(audio)
    st.pyplot(disp_rms)

    st.write('---')

    st.write('Zero Crossing Rate (ZCR)')
    disp_zcr = display_zcr(audio)
    st.pyplot(disp_zcr)

    st.write('---')

    st.write('Display Mel Spectrogram(MFCC)')
    disp_mfcc = display_mfcc(audio, sr)
    st.pyplot(disp_mfcc)
  
  with tab_feature:
    st.subheader("Feature Extraction")
    st.markdown('<br>', unsafe_allow_html=True)

    st.write('Feature Extraction')
    feature = extract_feature(audio, sr)
    feature = pd.DataFrame(feature)
    st.dataframe(feature)

    st.write('Hasil Normalize Feature Extraction')
    normalize = normalize_feature(feature)
    st.dataframe(normalize)

    st.write("Jumlah fitur yang dihasilkan")
    st.subheader(feature.shape)

# def rekam_suara():
#   st.write('Rekam suara yang akan diidentifikasi.')
#   st.markdown('<br>', unsafe_allow_html=True)

#   audio_bytes = audio_recorder()
#   if audio_bytes:
#     st.audio(audio_bytes, format='audio/wav')

#     audio, sr = librosa.load(audio_bytes)

#     tab_hasil(audio, sr)
    
#   st.write('Hasil identifikasi emosi:')
#   st.markdown('<br>', unsafe_allow_html=True)

def upload_file(uploaded_file):
  st.audio(uploaded_file, format='audio/ogg')
  st.markdown('<br>', unsafe_allow_html=True)

  audio, sr = librosa.load(uploaded_file)

  tab_hasil(audio, sr)


def content_tab_one(title):
  st.subheader(title)
  
  st.write('Upload atau rekam suara untuk dilakukan identifikasi emosi.')
  st.markdown('<br>', unsafe_allow_html=True)

  select = st.selectbox('Pilih Metode', ['Upload Suara'])
  st.markdown('<br>', unsafe_allow_html=True)

  if select == 'Upload Suara':
    st.write('Upload suara yang akan diidentifikasi.')

    audio_types = ["wav", "mp3", "ogg", "flac"]
    uploaded_file = st.file_uploader("Pilih file suara", type=audio_types)

    if uploaded_file is not None:
      upload_file(uploaded_file)
      

