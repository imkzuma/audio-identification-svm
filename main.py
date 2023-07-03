import streamlit as st
from tab_one import content_tab_one
from tab_two import content_tab_two

st.set_page_config(
    page_title="Audio Processing | SVM Algorithm",
    page_icon=":musical_note:",
    layout="wide",
)

st.title('Identifikasi Emosi berdasarkan Suara Orang')
st.write('Aplikasi ini dibuat dengan menggunakan algoritma SVM (Support Vector Machine) untuk mengidentifikasi emosi berdasarkan suara orang.')
st.markdown('<br> <br>', unsafe_allow_html=True)

with st.sidebar:
  st.header("Kelompok 2")
  author = st.multiselect(
    'Author',
    ['Gung Krisna', 'Alex'],
    ['Gung Krisna', 'Alex'])
  
  library = st.multiselect(
    'Library Used',
    ['Librosa', 'Streamlit', 'Sklearn', 'Numpy', 'Pandas'],
    ['Librosa', 'Streamlit', 'Sklearn', 'Numpy', 'Pandas'])

  dataset = st.multiselect(
    'Dataset Used :blue[2000 audio dataset]',
    ['1000 Happy', '1000 Sad'],
    ['1000 Happy', '1000 Sad']
  )

tab1, tab2 = st.tabs(['Identifikasi Emosi','Dokumentasi'])

with tab1:
  content_tab_one('Identifikasi Emosi')

with tab2:
  content_tab_two('Dokumentasi')