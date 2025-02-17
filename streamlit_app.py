import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="ECG Anomaly Detection", page_icon="💓", layout="wide")

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
@st.cache_data
def load_data(file=None):
    url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(url, header=None)
    return df

# Allow user to upload data; otherwise, load default dataset.
uploaded_file = st.sidebar.file_uploader("Upload your ECG data (CSV)", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    # Assume the last column is the label: True (normal) and False (anomaly)
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values.astype(bool)

    # Split into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )

    # Normalize: scale features to [0, 1]
    min_val, max_val = np.min(train_data), np.max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    # Use only normal samples for training (i.e. where label==True)
    normal_train = train_data[train_labels]
    # For testing, separate normal and anomalous examples
    normal_test = test_data[train_labels]
    anomaly_test = test_data[~train_labels]

    # ---------------------------
    # Model Definition & Training
    # ---------------------------
    class ECG_Autoencoder(Model):
        def __init__(self):
            super(ECG_Autoencoder, self).__init__()
            self.encoder = tf.keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(64, activation='relu'),
                layers.Dense(128, activation='relu'),
                layers.Dense(140, activation='sigmoid')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    @st.cache_resource
    def load_model():
        model = ECG_Autoencoder()
        model.compile(optimizer='adam', loss='mse')
        # Train only on normal ECG data
        model.fit(normal_train, normal_train, epochs=100, batch_size=256, validation_split=0.1, verbose=1)
        return model

    autoencoder = load_model()

    # ---------------------------
    # Anomaly Score & Threshold
    # ---------------------------
    # Compute anomaly score as the mean squared error between input and reconstruction.
    def compute_anomaly_score(x):
        reconstructions = autoencoder(x)
        # Using tf.keras.losses.mean_squared_error returns a 1D tensor (one score per sample)
        loss = tf.keras.losses.mean_squared_error(x, reconstructions)
        return loss.numpy()  # Shape: (batch,)

    # Set a dynamic threshold: 99th percentile of anomaly scores on normal training data.
    train_loss = compute_anomaly_score(normal_train)
    threshold = np.percentile(train_loss, 99)

    # Function to decide if a sample is anomalous.
    def is_anomaly(x):
        scores = compute_anomaly_score(x)
        return scores > threshold

    # Compute overall test accuracy:
    # For normal samples, we expect is_anomaly → False;
    # For anomalies, we expect is_anomaly → True.
    test_preds = is_anomaly(test_data)
    overall_accuracy = np.mean(test_preds == (~test_labels)) * 100
    st.sidebar.write(f"**Overall Anomaly Detection Accuracy: {overall_accuracy:.2f}%**")

    # ---------------------------
    # SHAP Explanation Setup
    # ---------------------------
    # Define a wrapper model that outputs a scalar anomaly score for each sample.
    def anomaly_score_model(x):
        reconstruction = autoencoder(x)
        return tf.reduce_mean(tf.square(x - reconstruction), axis=1, keepdims=True)

    # Use a background of 100 normal samples for DeepExplainer.
    background = normal_train[:100].astype(np.float32)

    def shap_explanation(data, index):
        sample = data[index:index+1].astype(np.float32)
        explainer = shap.DeepExplainer(anomaly_score_model, background)
        shap_values = explainer.shap_values(sample)
        # shap_values is a list with one array (since the model outputs a scalar)
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values[0], sample,
                          feature_names=[f"Feature {i}" for i in range(data.shape[1])],
                          show=False)
        st.pyplot(fig)

    # ---------------------------
    # Visualization
    # ---------------------------
    st.sidebar.title("ECG Anomaly Detection")
    ecg_type = st.sidebar.selectbox("Select ECG Type", ["Normal ECG", "Anomalous ECG"])
    # Select display data based on type
    display_data = normal_test if ecg_type == "Normal ECG" else anomaly_test
    ecg_index = st.sidebar.slider("Select ECG Index", 0, len(display_data)-1, 0)
    show_shap = st.sidebar.checkbox("Show SHAP Explanation", False)

    # Plot the selected ECG sample and its reconstruction.
    def plot_ecg(data, index):
        sample = data[index:index+1]
        reconstruction = autoencoder(sample).numpy().squeeze()
        original = sample.squeeze()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(original, label="Original", color="blue")
        ax.plot(reconstruction, label="Reconstruction", color="red", linestyle="--")
        ax.fill_between(range(len(original)), original, reconstruction, color="lightcoral", alpha=0.5, label="Error")
        ax.legend()
        st.pyplot(fig)

    st.write("### ECG Sample")
    plot_ecg(display_data, ecg_index)

    if show_shap:
        st.write("### SHAP Explanation for Anomaly Score")
        shap_explanation(display_data, ecg_index)
