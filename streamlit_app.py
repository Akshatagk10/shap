import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Set Streamlit page configuration
st.set_page_config(page_title="ECG Anomaly Detection", page_icon="💓", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
    df = pd.read_csv(url, header=None)
    return df

# Build improved Autoencoder model
class ECGAutoencoder(Model):
    def __init__(self):
        super(ECGAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
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

@st.cache_resource
def train_model(train_data):
    model = ECGAutoencoder()
    model.compile(optimizer='adam', loss='mae')
    model.fit(train_data, train_data, epochs=50, batch_size=512, validation_split=0.1, verbose=1)
    return model

# Load dataset
df = load_data()
data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split data
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

# Separate normal and abnormal data
normal_train = train_data[train_labels == 0]
normal_test = test_data[test_labels == 0]
anomaly_test = test_data[test_labels == 1]

# Train model
autoencoder = train_model(normal_train)

# Compute anomaly scores
def compute_anomaly_score(data):
    reconstructed = autoencoder(data)
    loss = losses.mae(reconstructed, data)
    return loss.numpy()

# Determine threshold
train_loss = compute_anomaly_score(normal_train)
threshold = np.mean(train_loss) + 3 * np.std(train_loss)

# Define anomaly detection function
def is_anomaly(data):
    scores = compute_anomaly_score(data)
    return scores > threshold

# Sidebar controls
st.sidebar.title("ECG Anomaly Detection")
ecg_type = st.sidebar.selectbox("Select ECG Type", ["Normal ECG", "Anomalous ECG"])
ecg_index = st.sidebar.slider("Select ECG Index", 0, len(normal_test) - 1, 0)
show_shap = st.sidebar.checkbox("Show SHAP Explanation", False)

# Select the ECG data
if ecg_type == "Normal ECG":
    selected_data = normal_test
else:
    selected_data = anomaly_test

# Plot ECG data
def plot_ecg(data, index):
    fig, ax = plt.subplots()
    reconstructed = autoencoder(data)
    ax.plot(data[index], label="Original", color="blue")
    ax.plot(reconstructed[index], label="Reconstructed", color="red")
    ax.fill_between(range(140), data[index], reconstructed[index], color="lightcoral", alpha=0.5, label="Reconstruction Error")
    ax.legend()
    st.pyplot(fig)

plot_ecg(selected_data, ecg_index)

# SHAP Explanation
def shap_explanation(data, index):
    explainer = shap.KernelExplainer(compute_anomaly_score, normal_train[:50])
    shap_values = explainer.shap_values(data[index:index+1])

    # SHAP Summary Plot
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, data[index:index+1], feature_names=[f"Feature {i}" for i in range(data.shape[1])], show=False)
    st.pyplot(fig)

if show_shap:
    shap_explanation(selected_data, ecg_index)

# Accuracy Calculation
predictions = is_anomaly(test_data)
accuracy = np.mean(predictions == (test_labels == 1)) * 100
st.sidebar.write(f"Anomaly Detection Accuracy: {accuracy:.2f}%")
