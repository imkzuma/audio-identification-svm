import streamlit as st

def preprocessing():
  st.success("Menampilkan Waveform Audio")
  waveform = '''
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
  '''

  st.code(waveform, language="python")
  st.write("Kodingan ini untuk menampilkan plot dari audio yang akan di analisis, untuk mengetahui Amplitude vs Time nya")
  st.markdown("---")

  st.success("Process RMSe")
  disp_rms = '''
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
  '''
  st.code(disp_rms, language='python')
  st.write("Kodingan ini untuk mencari RMSe audio dan menampilkan plot audio sesudah dilakukan proses RMSe")
  st.markdown("---")

  st.success("Process Zero Crossing Rate")
  disp_zcr = '''
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
  '''
  st.code(disp_zcr, language='python')
  st.write("Kodingan ini untuk mencari ZCR audio dan menampilkan plot audio sesudah dilakukan proses ZCR")
  st.markdown("---")

  st.success("Menampilkan Hasil MFCC yang sudah dilakukan")
  disp_mfcc = '''
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
  '''
  st.code(disp_mfcc, language='python')
  st.write("Kodingan ini untuk menampilkan plot sesudah dilakukan prosess MFCC")

def svm():
  st.success("Ekstraksi Fitur Audio")
  extract = '''
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
  '''
  st.code(extract, language="python")
  st.write("Fungsi ini untuk melakukan ekstraksi fitur dari audio yang akan di prediksi, fitur yang sudah di extract ini nantinya akan diambil mfcc nya, mfcc yang digunakan yaitu sebanyak 25")
  st.write("Hasil yang akan didapatkan yaitu feature mfcc, feature zcr, dan juga feature rmse. Yang diana masing-masing sebanyak 25 matriks, dan akan di gabungkan menjadi 75 matriks feature")
  st.markdown("---")

  st.success("Normalisasi Feature")
  normalize = '''
    def normalize_feature(feature):
      scaler = pickle.load(open("scaler_svm_2000data_new_1.pkl", "rb"))
      feature = scaler.transform(feature)

      return feature
  '''
  st.code(normalize, language="python")
  st.write("Fungsi ini untuk melakukan normalisasi fitur yang sudah di extract. Bertujuan agar prediksi data akurat, dan fitur tidak bernilai besar, yang dapat mempengaruhi hasil prediksi")
  st.write("Normalisasi menggunakan scaler yang sudah dibuat pada saat proses training data model")
  st.markdown('---')

  st.success("Prediction Label Decode")
  label = '''
    def prediction_data(model, df):
      y_pred = model.predict(df)

      labels = {
          0: "Bahagia",
          1: "Sedih"
      }
      pred = labels[y_pred[0]]

      return pred
  '''
  st.code(label, language="python")
  st.write("Fungsi ini untuk melakukan decode label, yang dimana hasil prediksi yang bernilai 0 atau 1 akan dilakukan decode untuk menampilkan hasil emosi, apakah sedih atau bahagia. sesuai dengan prediksi dari model yang sudah dibuat")
  st.markdown('---')

  st.success("Fungsi Prediksi")
  prediction = '''
    def prediction(audio, sr):
      model = pickle.load(open("svm_2000data_new_1_fix.pkl", "rb"))
    
      df = extract_feature(audio, sr)
      x = normalize_feature(df)
      pred = prediction_data(model, x)

      return pred
  '''
  st.code(prediction, language='python')
  st.write("Fungsi ini untuk melakukan prediksi dari audio yang akan dilakukan prediksi. Dengan memanggil fungsi-fungsi yang sudah dibuat tadinya. yaitu ekstraksi fitur audio, dan dilakukan normalisasi lalu dilakukan label decode dan hasil prediksi akan ditampillkan.")
  st.write("Menggunakan model yang sudah di training sebelumya. Yang sudah diexport dan dipanggil menggunakan library pickle")

def content_tab_two(title):
  st.subheader(title)
  st.write("Berikut merupakan dokumentasi dari kodingan dari tahapan preprocessing sampai dengan SVM")
  st.markdown("<br>", unsafe_allow_html=True)

  tab_preprocessing, tab_svm = st.tabs(["Tahapan Preprocessing", "Penggunaan SVM"])

  with tab_preprocessing:
    preprocessing()

  with tab_svm:
    svm()