"""
CareWatch Pro - Streamlit Interactive Demo
Live Demo für Ahead Care GmbH Bewerbung
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page Config
st.set_page_config(
    page_title="CareWatch Pro - Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# Load Model and Data
@st.cache_resource
def load_model():
    """Lädt trainiertes Model"""
    try:
        model = keras.models.load_model('models/lstm_autoencoder.h5')
        threshold = np.load('models/lstm_autoencoder_threshold.npy')
        return model, threshold
    except:
        return None, None

@st.cache_data
def load_data():
    """Lädt Test-Daten"""
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
        test_errors = np.load('test_errors.npy')
        test_predictions = np.load('test_predictions.npy')
        return X_test, y_test, test_errors, test_predictions
    except:
        return None, None, None, None

def plot_signal_interactive(data, title, feature_names):
    """Erstellt interaktiven Plot mit Plotly"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=feature_names,
        vertical_spacing=0.1
    )
    
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
    
    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def plot_reconstruction_comparison(original, reconstructed, feature_names):
    """Vergleicht Original vs. Rekonstruktion"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=feature_names,
        vertical_spacing=0.1
    )
    
    colors_orig = ['#e74c3c', '#3498db', '#2ecc71']
    
    for i, (feature, color) in enumerate(zip(feature_names, colors_orig)):
        # Original
        fig.add_trace(
            go.Scatter(
                x=list(range(len(original[:, i]))),
                y=original[:, i],
                mode='lines',
                name='Original',
                line=dict(color=color, width=2),
                hovertemplate='Original: %{y:.3f}<extra></extra>'
            ),
            row=i+1, col=1
        )
        
        # Rekonstruiert
        fig.add_trace(
            go.Scatter(
                x=list(range(len(reconstructed[:, i]))),
                y=reconstructed[:, i],
                mode='lines',
                name='Rekonstruiert',
                line=dict(color=color, width=2, dash='dash'),
                hovertemplate='Rekonstruiert: %{y:.3f}<extra></extra>'
            ),
            row=i+1, col=1
        )
        
        fig.update_xaxes(title_text="Zeitschritt", row=i+1, col=1)
        fig.update_yaxes(title_text="Wert", row=i+1, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Original vs. Rekonstruktion",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

# Main App
def main():
    # Header
    st.title("🏥 CareWatch Pro")
    st.markdown("### Deep Learning Anomalie-Erkennung für Vitaldaten")
    st.markdown("*Entwickelt für Ahead Care GmbH (moio.care)*")
    
    st.markdown("---")
    
    # Load Data
    model, threshold = load_model()
    X_test, y_test, test_errors, test_predictions = load_data()
    
    if model is None or X_test is None:
        st.error("⚠️ Model oder Daten nicht gefunden! Bitte erst `train.py` ausführen.")
        st.info("📝 Führe folgende Commands aus:\n\n```bash\npython train.py\n```")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Einstellungen")
    
    # Sample Selection
    st.sidebar.markdown("### 📊 Sample Auswahl")
    
    sample_type = st.sidebar.radio(
        "Wähle Sample-Typ:",
        ["Normal", "Anomalie", "Zufällig"]
    )
    
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
    
    # Threshold Adjustment
    st.sidebar.markdown("### 🎯 Threshold Anpassung")
    adjusted_threshold = st.sidebar.slider(
        "Anomalie-Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=float(threshold),
        step=0.01,
        help="Threshold für Anomalie-Erkennung"
    )
    
    st.sidebar.markdown("---")
    
    # Info Box
    st.sidebar.markdown("### ℹ️ Über das Projekt")
    st.sidebar.info(
        "**LSTM Autoencoder** lernt normale Vitaldaten-Muster. "
        "Anomalien haben einen hohen **Reconstruction Error**."
    )
    
    # Model Info
    st.sidebar.markdown("### 🧠 Model Info")
    st.sidebar.metric("Parameter", "64,227")
    st.sidebar.metric("Model Size", "~250 KB")
    st.sidebar.metric("Training Samples", "3,776")
    
    # Main Content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get selected sample
    selected_sample = X_test[selected_idx]
    true_label = y_test[selected_idx]
    reconstruction_error = test_errors[selected_idx]
    prediction = 1 if reconstruction_error > adjusted_threshold else 0
    
    # Metrics
    with col1:
        st.metric(
            "Ground Truth",
            "Anomalie" if true_label == 1 else "Normal",
            delta=None
        )
    
    with col2:
        st.metric(
            "Prediction",
            "Anomalie" if prediction == 1 else "Normal",
            delta="✅ Korrekt" if prediction == true_label else "❌ Falsch"
        )
    
    with col3:
        st.metric(
            "Reconstruction Error",
            f"{reconstruction_error:.4f}",
            delta=f"{((reconstruction_error/adjusted_threshold - 1)*100):.0f}% vs. Threshold"
        )
    
    with col4:
        st.metric(
            "Confidence",
            f"{abs(reconstruction_error - adjusted_threshold):.4f}",
            delta="Hoch" if abs(reconstruction_error - adjusted_threshold) > 0.1 else "Niedrig"
        )
    
    st.markdown("---")
    
    # Tabs für verschiedene Visualisierungen
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Signal Visualisierung",
        "🔄 Rekonstruktion",
        "📈 Alle Samples",
        "📋 Performance"
    ])
    
    feature_names = ['Herzfrequenz', 'Bewegung', 'Atmung']
    
    # Tab 1: Signal Visualization
    with tab1:
        st.markdown("### Ausgewähltes Sample - Vitaldaten")
        
        # Plotly Interactive Plot
        fig = plot_signal_interactive(
            selected_sample,
            f"Sample {selected_idx} - {'ANOMALIE' if true_label == 1 else 'NORMAL'}",
            feature_names
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample Info
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 📊 Statistiken")
            stats_df = pd.DataFrame({
                'Feature': feature_names,
                'Mean': selected_sample.mean(axis=0),
                'Std': selected_sample.std(axis=0),
                'Min': selected_sample.min(axis=0),
                'Max': selected_sample.max(axis=0)
            })
            st.dataframe(stats_df.style.format({
                'Mean': '{:.3f}',
                'Std': '{:.3f}',
                'Min': '{:.3f}',
                'Max': '{:.3f}'
            }), use_container_width=True)
        
        with col_b:
            st.markdown("#### 🎯 Entscheidung")
            
            if reconstruction_error > adjusted_threshold:
                st.error(f"🚨 **ANOMALIE ERKANNT**")
                st.write(f"Error ({reconstruction_error:.4f}) > Threshold ({adjusted_threshold:.4f})")
            else:
                st.success(f"✅ **NORMAL**")
                st.write(f"Error ({reconstruction_error:.4f}) ≤ Threshold ({adjusted_threshold:.4f})")
            
            # Anomalie-Typ raten
            if true_label == 1:
                hr_spike = selected_sample[:, 0].max() > 2
                motion_spike = selected_sample[:, 1].max() > 3
                resp_irregular = selected_sample[:, 2].std() > 1.5
                
                st.markdown("**Mögliche Anomalie:**")
                if hr_spike:
                    st.write("❤️ Herzfrequenz-Anomalie (Tachykardie)")
                if motion_spike:
                    st.write("🚨 Bewegungs-Anomalie (Sturz)")
                if resp_irregular:
                    st.write("🫁 Atem-Anomalie (Unregelmäßig)")
    
    # Tab 2: Reconstruction
    with tab2:
        st.markdown("### Original vs. Rekonstruierte Signale")
        
        # Reconstruct
        reconstructed = model.predict(selected_sample.reshape(1, 100, 3), verbose=0)[0]
        
        # Plotly Comparison
        fig = plot_reconstruction_comparison(selected_sample, reconstructed, feature_names)
        st.plotly_chart(fig, use_container_width=True)
        
        # Error Analysis
        st.markdown("#### 🔍 Error-Analyse pro Feature")
        col_x, col_y, col_z = st.columns(3)
        
        for i, (col, feature) in enumerate(zip([col_x, col_y, col_z], feature_names)):
            with col:
                feature_error = np.mean(np.square(selected_sample[:, i] - reconstructed[:, i]))
                st.metric(feature, f"{feature_error:.4f}", delta=None)
    
    # Tab 3: All Samples Overview
    with tab3:
        st.markdown("### Alle Test-Samples - Anomaly Scores")
        
        # Scatter Plot
        fig = go.Figure()
        
        # Normal Samples
        normal_mask = y_test == 0
        fig.add_trace(go.Scatter(
            x=np.where(normal_mask)[0],
            y=test_errors[normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='green', size=6, opacity=0.6),
            hovertemplate='Sample %{x}<br>Error: %{y:.4f}<extra></extra>'
        ))
        
        # Anomaly Samples
        anomaly_mask = y_test == 1
        fig.add_trace(go.Scatter(
            x=np.where(anomaly_mask)[0],
            y=test_errors[anomaly_mask],
            mode='markers',
            name='Anomalie',
            marker=dict(color='red', size=6, opacity=0.6),
            hovertemplate='Sample %{x}<br>Error: %{y:.4f}<extra></extra>'
        ))
        
        # Threshold Line
        fig.add_hline(
            y=adjusted_threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Threshold ({adjusted_threshold:.3f})"
        )
        
        # Selected Sample Highlight
        fig.add_trace(go.Scatter(
            x=[selected_idx],
            y=[reconstruction_error],
            mode='markers',
            name='Ausgewählt',
            marker=dict(color='yellow', size=15, symbol='star', 
                       line=dict(color='black', width=2)),
            hovertemplate='Selected<br>Error: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Reconstruction Errors - Alle Test Samples",
            xaxis_title="Sample Index",
            yaxis_title="Reconstruction Error (MSE)",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("#### Normal Samples")
            st.metric("Count", f"{np.sum(y_test==0)}")
            st.metric("Mean Error", f"{np.mean(test_errors[y_test==0]):.4f}")
            st.metric("Std Error", f"{np.std(test_errors[y_test==0]):.4f}")
        
        with col_s2:
            st.markdown("#### Anomaly Samples")
            st.metric("Count", f"{np.sum(y_test==1)}")
            st.metric("Mean Error", f"{np.mean(test_errors[y_test==1]):.4f}")
            st.metric("Std Error", f"{np.std(test_errors[y_test==1]):.4f}")
        
        with col_s3:
            st.markdown("#### Threshold Stats")
            st.metric("False Positives", f"{np.sum((y_test==0) & (test_errors>adjusted_threshold))}")
            st.metric("False Negatives", f"{np.sum((y_test==1) & (test_errors<=adjusted_threshold))}")
            
            accuracy = np.mean((test_errors > adjusted_threshold) == y_test)
            st.metric("Accuracy", f"{accuracy:.1%}")
    
    # Tab 4: Performance Metrics
    with tab4:
        st.markdown("### Model Performance - Test Set")
        
        # Recalculate with adjusted threshold
        adjusted_predictions = (test_errors > adjusted_threshold).astype(int)
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, adjusted_predictions)
        
        col_p1, col_p2 = st.columns([1, 1])
        
        with col_p1:
            st.markdown("#### Confusion Matrix")
            
            fig_cm, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['Normal', 'Anomalie'],
                       yticklabels=['Normal', 'Anomalie'],
                       ax=ax, annot_kws={'size': 16})
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)
            st.pyplot(fig_cm)
        
        with col_p2:
            st.markdown("#### Metriken")
            
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            met_col1, met_col2 = st.columns(2)
            
            with met_col1:
                st.metric("Accuracy", f"{accuracy:.1%}")
                st.metric("Precision", f"{precision:.1%}")
            
            with met_col2:
                st.metric("Recall", f"{recall:.1%}")
                st.metric("F1-Score", f"{f1:.1%}")
            
            st.markdown("---")
            st.markdown("**Confusion Matrix Breakdown:**")
            st.write(f"✅ True Positives: {tp}")
            st.write(f"✅ True Negatives: {tn}")
            st.write(f"❌ False Positives: {fp}")
            st.write(f"❌ False Negatives: {fn}")
        
        # Classification Report
        st.markdown("#### Detaillierter Report")
        report = classification_report(y_test, adjusted_predictions, 
                                       target_names=['Normal', 'Anomalie'],
                                       output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            'precision': '{:.3f}',
            'recall': '{:.3f}',
            'f1-score': '{:.3f}',
            'support': '{:.0f}'
        }), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🎯 Über dieses Projekt")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("**Architektur**")
        st.write("LSTM Autoencoder")
        st.write("64,227 Parameter")
        st.write("~250 KB Model Size")
    
    with col_f2:
        st.markdown("**Training**")
        st.write("3,776 normale Samples")
        st.write("50 Epochs (Early Stop)")
        st.write("Adam Optimizer")
    
    with col_f3:
        st.markdown("**Features**")
        st.write("Herzfrequenz (HR)")
        st.write("Bewegung (Motion)")
        st.write("Atmung (Respiration)")
    
    st.markdown("---")
    st.markdown("*Entwickelt für Ahead Care GmbH (moio.care) - Data Science Engineer Position*")

if __name__ == "__main__":
    main()
