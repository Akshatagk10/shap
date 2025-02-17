import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, Model
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="SHAP ECG Anomaly Detection", page_icon="💓", layout="wide")

# Load Data
@st.cache_data
def load_data(file=None):
    if file:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
    return df

# Define Autoencoder Model
class ECG_Autoencoder(Model):
    def __init__(self):
        super(ECG_Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(140, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load Model
@st.cache_resource
def load_model():
    model = ECG_Autoencoder()
    model.compile(optimizer='adam', loss='mse')
    return model

# Load dataset
uploaded_file = st.sidebar.file_uploader("Upload your ECG data (CSV)", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values.astype(bool)

    # Split Data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)
    
    # Normalize Data
    min_val, max_val = np.min(train_data), np.max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)
    
    normal_train = train_data[train_labels]
    normal_test = test_data[test_labels]
    anomaly_test = test_data[~test_labels]

    # Train Autoencoder
    autoencoder = load_model()
    autoencoder.fit(normal_train, normal_train, epochs=50, batch_size=256, validation_data=(normal_test, normal_test), verbose=1)
    
    # Compute Anomaly Scores
    def compute_anomaly_score(data):
        reconstructions = autoencoder(data)
        loss = tf.keras.losses.mse(data, reconstructions)
        return np.mean(loss, axis=1)
    
    # SHAP Explanation
    def shap_explanation(data, index):
        explainer = shap.DeepExplainer(autoencoder.encoder, normal_train[:100])
        shap_values = explainer.shap_values(data[index:index+1])
        shap.summary_plot(shap_values, data[index:index+1], feature_names=[f"Feature {i}" for i in range(data.shape[1])])
        st.pyplot()
    
    # Sidebar Inputs
    st.sidebar.title("ECG Anomaly Detection")
    ecg_type = st.sidebar.selectbox("Select ECG Type", ["Normal ECG", "Abnormal ECG"])
    ecg_index = st.sidebar.slider("Select ECG Index", 0, len(normal_test)-1, 0)
    show_shap = st.sidebar.checkbox("Show SHAP Explanation", False)

    # Display Results
    st.write("### ECG Data Sample")
    st.line_chart(test_data[ecg_index])
    
    if show_shap:
        st.write("### SHAP Explanation")
        shap_explanation(normal_test if ecg_type == "Normal ECG" else anomaly_test, ecg_index)
