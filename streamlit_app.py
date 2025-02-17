import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
from tensorflow.keras import layers, losses, Model
from sklearn.model_selection import train_test_split

# Streamlit UI config
st.set_page_config(page_title="ECG Anomaly Detection", page_icon="💓", layout="wide")

# Load ECG Data
@st.cache_data
def load_data(file=None):
    url = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'
    df = pd.read_csv(file) if file else pd.read_csv(url, header=None)
    return df

# Load Autoencoder Model
@st.cache_resource
def load_model():
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

# Upload ECG data
uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
df = load_data(uploaded_file)

if df is not None:
    data = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # Train-test split
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21)

    # Normalize data
    min_val, max_val = np.min(train_data), np.max(train_data)
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_labels, test_labels = train_labels.astype(bool), test_labels.astype(bool)

    n_train_data, n_test_data = train_data[train_labels], test_data[test_labels]

    # Load and train autoencoder
    autoencoder = load_model()
    autoencoder.fit(n_train_data, n_train_data, epochs=20, batch_size=512, validation_data=(n_test_data, n_test_data))

    # Calculate anomaly detection threshold
    reconstructed = autoencoder(n_train_data)
    train_loss = losses.mae(reconstructed, n_train_data)
    threshold = np.mean(train_loss) + 2 * np.std(train_loss)

    # Define prediction function
    def prediction(model, data, threshold):
        rec = model(data)
        loss = losses.mae(rec, data)
        return tf.math.less(loss, threshold)

    # SHAP Explainer
    explainer = shap.Explainer(autoencoder, n_train_data[:500])  # Sample subset to speed up SHAP

    # Plot ECG with SHAP highlights
    def plot_with_shap(data, index):
        fig, ax = plt.subplots(figsize=(10, 4))
        enc_img = autoencoder.encoder(data)
        dec_img = autoencoder.decoder(enc_img)

        ax.plot(data[index], 'b', label='Input Signal')
        ax.plot(dec_img[index], 'r', linestyle='dashed', label='Reconstructed Signal')

        # Generate SHAP values
        shap_values = explainer(data[index:index+1])
        shap_importance = np.abs(shap_values.values[0])

        # Highlight top anomaly features
        top_anomalies = np.argsort(shap_importance)[-5:]  # 5 most important time steps
        ax.scatter(top_anomalies, data[index][top_anomalies], color='red', marker='o', label='Anomalous Points')
        
        ax.legend()
        st.pyplot(fig)

        # SHAP Force Plot
        st.subheader("SHAP Force Plot")
        shap_fig, shap_ax = plt.subplots()
        shap.force_plot(
            explainer.expected_value[0],
            shap_values.values[0],
            feature_names=[f"Time {i}" for i in range(140)],
            matplotlib=True,
            show=False
        )
        st.pyplot(shap_fig)

        # SHAP Summary Plot
        st.subheader("SHAP Summary Plot")
        shap.summary_plot(shap_values, data, feature_names=[f"Time {i}" for i in range(140)], show=False)
        st.pyplot()

    # Interactive Streamlit UI
    st.sidebar.title("ECG Anomaly Detection")
    ecg_index = st.sidebar.slider("Select ECG Index", 0, len(n_test_data) - 1, 0)
    use_shap = st.sidebar.checkbox("Show SHAP Explanation", False)

    # Display ECG and SHAP explanation
    if use_shap:
        plot_with_shap(n_test_data, ecg_index)

    # Prediction button
    if st.sidebar.button("Make Predictions"):
        pred = prediction(autoencoder, n_test_data, threshold)
        accuracy = np.sum(pred.numpy()) / len(pred.numpy()) * 100
        st.sidebar.write(f"Anomaly Detection Accuracy: {accuracy:.2f}%")
