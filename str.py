"""
CareWatch Pro - Streamlit Interactive Demo
Live Demo für Ahead Care GmbH Bewerbung
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------
# 🏗️ Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="CareWatch Pro - Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# 🎨 Custom Styling
# -----------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------
# 📦 Utility: Check Model Path
# -----------------------------------------------------
def model_exists():
    model_path = "models/lstm_autoencoder.h5"
    threshold_path = "models/lstm_autoencoder_threshold.npy"
    exists = os.path.exists(model_path) and os.path.exists(threshold_path)
    if not exists:
        st.warning(f"⚠️ Modelldateien nicht gefunden im Ordner `models/`.\n\nErwartete Dateien:\n- {model_path}\n- {threshold_path}")
    return exists


# -----------------------------------------------------
# 🧠 Load Model and Data
# -----------------------------------------------------
@st.cache_resource
def load_model():
    """Lädt trainiertes Modell"""
    if not model_exists():
        return None, None
    try:
        model = keras.models.load_model('models/lstm_autoencoder.h5')
        threshold = np.load('models/lstm_autoencoder_threshold.npy')
        return model, threshold
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None


@st.cache_data
@st.cache_data
def load_data():
    """Lädt Test-Daten"""
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        test_errors = np.load('test_errors.npy')
        test_predictions = np.load('test_predictions.npy')
        return X_test, y_test, test_errors, test_predictions
    except Exception as e:
        st.error(f"Fehler beim Laden der Testdaten: {e}")
        return None, None, None, None


# -----------------------------------------------------
# 📊 Plot Functions
# -----------------------------------------------------
def plot_signal_interactive(data, title, feature_names):
    fig = make_subplots(rows=3, cols=1, subplot_titles=feature_names, vertical_spacing=0.1)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for i, (feature, color) in enumerate(zip(feature_names, colors)):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(data[:, i]))),
                y=data[:, i],
                mode='lines',
                name=feature,
                line=dict(color=color, width=2),
                hovertemplate=f'{feature}: %{{y:.3f}}<extra></extra>'
            ),
            row=i+1, col=1
        )
        fig.update_xaxes(title_text="Zeitschritt", row=i+1, col=1)
        fig.update_yaxes(title_text="Wert", row=i+1, col=1)
    fig.update_layout(height=600, title_text=title, showlegend=False, hovermode='x unified')
    return fig


def plot_reconstruction_comparison(original, reconstructed, feature_names):
    fig = make_subplots(rows=3, cols=1, subplot_titles=feature_names, vertical_spacing=0.1)
    colors_orig = ['#e74c3c', '#3498db', '#2ecc71']
    for i, (feature, color) in enumerate(zip(feature_names, colors_orig)):
        fig.add_trace(go.Scatter(
            x=list(range(len(original[:, i]))),
            y=original[:, i],
            mode='lines',
            name='Original',
            line=dict(color=color, width=2)
        ), row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=list(range(len(reconstructed[:, i]))),
            y=reconstructed[:, i],
            mode='lines',
            name='Rekonstruiert',
            line=dict(color=color, width=2, dash='dash')
        ), row=i+1, col=1)
        fig.update_xaxes(title_text="Zeitschritt", row=i+1, col=1)
        fig.update_yaxes(title_text="Wert", row=i+1, col=1)
    fig.update_layout(height=600, title_text="Original vs. Rekonstruktion", showlegend=True, hovermode='x unified')
    return fig


# -----------------------------------------------------
# 🚀 Main App
# -----------------------------------------------------
def main():
    st.title("🏥 CareWatch Pro")
    st.markdown("### Deep Learning Anomalie-Erkennung für Vitaldaten")
    st.markdown("*Entwickelt für Ahead Care GmbH (moio.care)*")
    st.markdown("---")

    # Load model and data
    model, threshold = load_model()
    X_test, y_test, test_errors, test_predictions = load_data()

    if model is None or X_test is None:
        st.error("⚠️ Modell oder Daten nicht gefunden! Bitte stelle sicher, dass die Dateien im Ordner `models/` liegen.")
        st.stop()

    # Sidebar
    st.sidebar.title("⚙️ Einstellungen")
    st.sidebar.markdown("### 📊 Sample Auswahl")

    sample_type = st.sidebar.radio("Wähle Sample-Typ:", ["Normal", "Anomalie", "Zufällig"])

    if sample_type == "Normal":
        available_indices = np.where(y_test == 0)[0]
    elif sample_type == "Anomalie":
        available_indices = np.where(y_test == 1)[0]
    else:
        available_indices = np.arange(len(y_test))

    selected_idx = st.sidebar.selectbox(
        "Sample Index:",
        available_indices,
        format_func=lambda x: f"Sample {x} ({'Normal' if y_test[x]==0 else 'Anomalie'})"
    )

    adjusted_threshold = st.sidebar.slider(
        "Anomalie-Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=float(threshold),
        step=0.01,
        help="Threshold für Anomalie-Erkennung"
    )

    # Main content
    selected_sample = X_test[selected_idx]
    true_label = y_test[selected_idx]
    reconstruction_error = test_errors[selected_idx]
    prediction = 1 if reconstruction_error > adjusted_threshold else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ground Truth", "Anomalie" if true_label == 1 else "Normal")
    with col2:
        st.metric("Prediction", "Anomalie" if prediction == 1 else "Normal")
    with col3:
        st.metric("Reconstruction Error", f"{reconstruction_error:.4f}")
    with col4:
        st.metric("Confidence", f"{abs(reconstruction_error - adjusted_threshold):.4f}")

    st.markdown("---")

    feature_names = ['Herzfrequenz', 'Bewegung', 'Atmung']

    tab1, tab2 = st.tabs(["📊 Signal Visualisierung", "🔄 Rekonstruktion"])

    with tab1:
        st.markdown("### Vitaldaten - Sample")
        st.plotly_chart(plot_signal_interactive(selected_sample, f"Sample {selected_idx}", feature_names), use_container_width=True)

    with tab2:
        st.markdown("### Original vs. Rekonstruiert")
        reconstructed = model.predict(selected_sample.reshape(1, 100, 3), verbose=0)[0]
        st.plotly_chart(plot_reconstruction_comparison(selected_sample, reconstructed, feature_names), use_container_width=True)

    st.markdown("---")
    st.markdown("*Entwickelt für Ahead Care GmbH (moio.care) - Data Science Engineer Position*")


# -----------------------------------------------------
# 🔧 Run
# -----------------------------------------------------
if __name__ == "__main__":
    main()
