import streamlit as st
import pickle 
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plost
import requests

#MODEL_PATH = f'./model/lr_model.pkl'
#SCALER_PATH = f'./model/scaler.pkl'
IMG_SIDEBAR_PATH = "./assets/img.jpg"
BGR_PATH = "./assets/background.png"

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

#model = load_pkl(MODEL_PATH)
#scaler = load_pkl(SCALER_PATH)

def get_clean_data():
  data = pd.read_csv("./dataset/data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def add_sidebar():
  st.sidebar.header("Breast Cancer Predictor `App 👧🏻`")
  image = np.array(Image.open(IMG_SIDEBAR_PATH))
  st.sidebar.image(image)
  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
  st.sidebar.write("Please connect this App to your Citology Lab to help diagnose Breast Cancer from your tissue sample.")

  st.sidebar.subheader('Select Lab Parameters ✅:')
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )

  st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

  return input_dict

def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict

def get_radar_chart(input_data):
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig

def add_predictions(input_data) :
    input_array = np.array(list(input_data.values())).reshape(1, -1).tolist()

    data = {'array': input_array}

    #resp = requests.post("http://127.0.0.1:5000", json=data)

    #http://18.223.133.31:5000/

    resp = requests.post("http://18.219.34.29:5000/", json=data)

    pred_result = resp.json()["Results"]["result"]
    prob_beg = resp.json()["Results"]["prob_beg"]
    prob_mag = resp.json()["Results"]["prob_mag"]

    st.markdown("### Cell Cluster Prediction ✅")
    st.write("<span class='diagnosis-label'>Machine Learning Model Result:</span>",  unsafe_allow_html=True)
    
    if pred_result == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1 :
        st.metric("Probability of Being Benign:", f"{prob_beg}%", "Benign")
    
    with col2:
        st.metric("Probability of Being Malicious:", f"{prob_mag}%", "-Malignant")

    st.write("`This Artificial Intelligence can Assist Medical Professionals in making a Diagnosis, but Should Not be used as a Substitute for a Professional Diagnosis.`")

def main() :  
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    input_data = add_sidebar()

    st.markdown(
        """
        <style>
        [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }

        </style>
        """,
        unsafe_allow_html=True
    )    

    with st.container() :
        st.title("Breast Cancer Predictor 🏥")
        st.write("This App predicts using a Support Vector Machine Learning Model whether a Breast Mass is Benign or Malignant bases on the measurements it receives from your Cytosis Lab. You can also Update the measurements by hand using sliders in the sidebar.")
        st.markdown("<hr/>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    df = pd.read_csv("./dataset/chart_tumor_diagnosed.csv")

    with col1:
        st.markdown('### Radar Chart 📊')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        st.markdown('### Bar Chart 📉')
        st.markdown("---", unsafe_allow_html=True)
        plost.bar_chart(
            data=df,
            bar='Tumor',
            value='Diagnosed', 
            legend='bottom',
            use_container_width=True,
            color='Tumor')        
        

    with col2:
        st.markdown('### Donut Chart 📈')
        plost.donut_chart(
            data=df,
            theta="Diagnosed",
            color='Tumor',
            legend='bottom', 
            use_container_width=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        add_predictions(input_data)

if __name__ == "__main__" :
    main()

    print("App Running!")
