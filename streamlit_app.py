import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tensorflow.keras import layers, Model, losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings
warnings.filterwarnings("ignore")

# Set Streamlit page config
st.set_page_config(page_title="SHAP ECG Anomaly Detection", page_icon="💓", layout="wide")

# Enable GPU memory growth if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Cache function to load data
@st.cache_data
def load_data(file=None):
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv", header=None)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Cache function to load the model
@st.cache_resource
def load_model():
    try:
        class Detector(Model):
            def __init__(self):
                super(Detector, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Dense(32, activation='relu'),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(8, activation='relu')
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Dense(16, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(140, activation='sigmoid')
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        model = Detector()
        model.compile(optimizer='adam', loss='mae')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar: Upload Data
uploaded_file = st.sidebar.file_uploader("Upload ECG Data (CSV)", type=["csv"])

# Load Data
df = load_data(uploaded_file)

if df is not None:
    # Prepare data
    data = df.iloc[:, :-1].values  # Features
    labels = df.iloc[:, -1].values  # Labels

    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

    # Normalize the data
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Convert to tensors
    train_data, test_data = map(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), [train_data, test_data])

    # Convert labels to boolean
    train_labels, test_labels = train_labels.astype(bool), test_labels.astype(bool)

    # Separate normal and abnormal ECG data
    n_train_data, n_test_data = train_data[train_labels], test_data[test_labels]
    an_train_data, an_test_data = train_data[~train_labels], test_data[~test_labels]

    # Load model
    autoencoder = load_model()
    if autoencoder:
        try:
            autoencoder.fit(n_train_data, n_train_data, epochs=20, batch_size=32, validation_data=(n_test_data, n_test_data))
        except Exception as e:
            st.error(f"Error during model training: {e}")
            autoencoder = None
else:
    st.warning("No ECG data available. Please upload a dataset.")

# Function to plot original vs reconstructed ECG
def plot_ecg(data, index):
    fig, ax = plt.subplots()
    encoded = autoencoder.encoder(data)
    reconstructed = autoencoder.decoder(encoded)

    ax.plot(data[index], 'b', label='Input')
    ax.plot(reconstructed[index], 'r', label='Reconstruction')
    ax.fill_between(np.arange(140), data[index], reconstructed[index], color='lightcoral', alpha=0.5, label='Error')
    ax.legend()
    st.pyplot(fig)

# Function to explain predictions with SHAP
def shap_explanation(data, index):
    explainer = shap.Explainer(autoencoder, n_train_data)
    shap_values = explainer(data[index:index+1])

    # Plot SHAP Explanation
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, data[index:index+1], plot_type="bar", show=False)
    st.pyplot(fig)

# Sidebar Controls
st.sidebar.title("ECG Anomaly Detection")
ecg_type = st.sidebar.selectbox("Select ECG Type", ["Normal ECG", "Abnormal ECG"])
ecg_index = st.sidebar.slider("Select ECG Index", 0, len(n_test_data) - 1, 0)
use_shap = st.sidebar.checkbox("Show SHAP Explanation", False)

# Anomaly Detection Threshold
if autoencoder:
    reconstructed = autoencoder(n_train_data)
    train_loss = losses.mae(reconstructed, n_train_data)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)

    # Prediction function
    def is_anomaly(model, data, threshold):
        rec = model(data)
        loss = losses.mae(rec, data)
        return tf.math.greater(loss, threshold)  # True if anomaly

    # Show ECG Plot
    plot_ecg(n_test_data, ecg_index)

    # Show SHAP Explanation
    if use_shap:
        shap_explanation(n_test_data, ecg_index)

    # Show Anomaly Detection Results
    if st.sidebar.button("Make Predictions"):
        pred = is_anomaly(autoencoder, n_test_data, threshold)
        acc = np.sum(pred.numpy()) / len(pred.numpy()) * 100
        st.sidebar.write(f"Anomaly Detection Accuracy: {acc:.2f}%")
