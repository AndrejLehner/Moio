"""
CareWatch Pro - Data Preprocessing Pipeline
Bereitet Sensordaten für LSTM Autoencoder vor
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple

class DataProcessor:
    """Preprocessing Pipeline für Zeitreihen-Daten"""
    
    def __init__(self, window_size: int = 100, overlap: float = 0.5):
        """
        Args:
            window_size: Anzahl Zeitschritte pro Window (2 Sekunden bei 50Hz)
            overlap: Überlappung zwischen Windows (0.0 - 1.0)
        """
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.scaler = StandardScaler()
        
    def create_windows(self, df: pd.DataFrame, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Erstellt Sliding Windows aus Zeitreihen
        
        Args:
            df: DataFrame mit Sensordaten und sample_id
            labels: Array mit Labels pro Sample
            
        Returns:
            X: Array shape (n_windows, window_size, n_features)
            y: Array shape (n_windows,) mit Labels
        """
        n_samples = df['sample_id'].nunique()
        features = ['heartrate', 'motion', 'respiration']
        
        all_windows = []
        all_labels = []
        
        print(f"Erstelle Windows (size={self.window_size}, step={self.step_size})...")
        
        for sample_id in range(n_samples):
            # Hole Daten für dieses Sample
            sample_data = df[df['sample_id'] == sample_id][features].values
            sample_label = labels[sample_id]
            
            # Erstelle Sliding Windows
            n_steps = len(sample_data)
            for start_idx in range(0, n_steps - self.window_size + 1, self.step_size):
                window = sample_data[start_idx:start_idx + self.window_size]
                all_windows.append(window)
                all_labels.append(sample_label)
        
        X = np.array(all_windows)
        y = np.array(all_labels)
        
        print(f"✅ {len(X)} Windows erstellt")
        print(f"   Shape: {X.shape}")
        print(f"   Normal: {np.sum(y==0)}, Anomalien: {np.sum(y==1)}")
        
        return X, y
    
    def normalize_data(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z-Score Normalisierung pro Feature
        
        Args:
            X_train: Training data (n_samples, window_size, n_features)
            X_test: Test data
            
        Returns:
            X_train_norm, X_test_norm: Normalisierte Arrays
        """
        print("\nNormalisiere Daten (Z-Score)...")
        
        # Reshape für Scaler: (n_samples * window_size, n_features)
        n_train, window_size, n_features = X_train.shape
        n_test = X_test.shape[0]
        
        X_train_flat = X_train.reshape(-1, n_features)
        X_test_flat = X_test.reshape(-1, n_features)
        
        # Fit auf Training, Transform beide
        X_train_norm = self.scaler.fit_transform(X_train_flat)
        X_test_norm = self.scaler.transform(X_test_flat)
        
        # Zurück zu Original-Shape
        X_train_norm = X_train_norm.reshape(n_train, window_size, n_features)
        X_test_norm = X_test_norm.reshape(n_test, window_size, n_features)
        
        print("✅ Normalisierung abgeschlossen")
        print(f"   Train Mean: {X_train_norm.mean(axis=(0,1))}")
        print(f"   Train Std:  {X_train_norm.std(axis=(0,1))}")
        
        return X_train_norm, X_test_norm
    
    def prepare_dataset(self, csv_path: str = 'sensor_data.csv',
                       labels_path: str = 'labels.npy',
                       test_size: float = 0.2) -> dict:
        """
        Komplette Pipeline: Load -> Windows -> Normalize -> Split
        
        Returns:
            dict mit X_train, X_test, y_train, y_test, X_train_norm, X_test_norm
        """
        print("=== DATA PREPROCESSING PIPELINE ===\n")
        
        # 1. Load Data
        print("📂 Lade Daten...")
        df = pd.read_csv(csv_path)
        labels = np.load(labels_path)
        print(f"✅ {len(df)} Datenpunkte, {labels.shape[0]} Samples")
        
        # 2. Create Windows
        X, y = self.create_windows(df, labels)
        
        # 3. Train/Test Split
        print(f"\n📊 Train/Test Split ({int((1-test_size)*100)}/{int(test_size*100)})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"✅ Train: {len(X_train)} Windows, Test: {len(X_test)} Windows")
        
        # 4. Normalize
        X_train_norm, X_test_norm = self.normalize_data(X_train, X_test)
        
        # 5. Save processed data
        print("\n💾 Speichere preprocessed data...")
        np.save('X_train.npy', X_train_norm)
        np.save('X_test.npy', X_test_norm)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        print("✅ Gespeichert: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
        
        return {
            'X_train': X_train_norm,
            'X_test': X_test_norm,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_raw': X_train,
            'X_test_raw': X_test
        }
    
    def plot_window_examples(self, X: np.ndarray, y: np.ndarray, n_examples: int = 4):
        """Visualisiert Beispiel-Windows"""
        feature_names = ['Herzfrequenz', 'Bewegung', 'Atmung']
        
        fig, axes = plt.subplots(n_examples, 3, figsize=(15, 3*n_examples))
        fig.suptitle('Beispiel Windows (Normalisiert)', fontsize=14, fontweight='bold')
        
        # Je 2 normale und 2 Anomalien
        normal_idx = np.where(y == 0)[0][:n_examples//2]
        anomaly_idx = np.where(y == 1)[0][:n_examples//2]
        selected_idx = np.concatenate([normal_idx, anomaly_idx])
        
        for i, idx in enumerate(selected_idx):
            window = X[idx]
            label = "NORMAL" if y[idx] == 0 else "ANOMALIE"
            
            for j, feature_name in enumerate(feature_names):
                axes[i, j].plot(window[:, j], linewidth=1.0)
                axes[i, j].set_ylabel(feature_name)
                axes[i, j].grid(True, alpha=0.3)
                
                if i == 0:
                    axes[i, j].set_title(feature_name, fontweight='bold')
                if i == len(selected_idx) - 1:
                    axes[i, j].set_xlabel('Zeitschritt')
                    
                # Label auf rechter Seite
                if j == 2:
                    axes[i, j].text(1.05, 0.5, label, 
                                   transform=axes[i, j].transAxes,
                                   fontsize=10, fontweight='bold',
                                   va='center', rotation=270)
        
        plt.tight_layout()
        plt.savefig('window_examples.png', dpi=150, bbox_inches='tight')
        print("📊 Visualisierung gespeichert: window_examples.png")
        return fig
    
    def plot_feature_distributions(self, X_train: np.ndarray, X_test: np.ndarray):
        """Visualisiert Feature-Verteilungen vor/nach Normalisierung"""
        feature_names = ['Herzfrequenz', 'Bewegung', 'Atmung']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Feature Distributions', fontsize=14, fontweight='bold')
        
        for i, feature_name in enumerate(feature_names):
            # Vor Normalisierung (aus raw data)
            train_feature = X_train[:, :, i].flatten()
            test_feature = X_test[:, :, i].flatten()
            
            axes[0, i].hist(train_feature, bins=50, alpha=0.6, label='Train', color='blue')
            axes[0, i].hist(test_feature, bins=50, alpha=0.6, label='Test', color='orange')
            axes[0, i].set_title(f'{feature_name} (normalisiert)')
            axes[0, i].set_xlabel('Wert')
            axes[0, i].set_ylabel('Häufigkeit')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Statistiken
            axes[1, i].axis('off')
            stats_text = f"""
            Train:
              Mean: {train_feature.mean():.3f}
              Std:  {train_feature.std():.3f}
              Min:  {train_feature.min():.3f}
              Max:  {train_feature.max():.3f}
            
            Test:
              Mean: {test_feature.mean():.3f}
              Std:  {test_feature.std():.3f}
              Min:  {test_feature.min():.3f}
              Max:  {test_feature.max():.3f}
            """
            axes[1, i].text(0.1, 0.5, stats_text, fontsize=10, 
                          verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
        print("📊 Visualisierung gespeichert: feature_distributions.png")
        return fig


# Main Execution
if __name__ == "__main__":
    processor = DataProcessor(window_size=100, overlap=0.5)
    
    # Run complete pipeline
    data = processor.prepare_dataset()
    
    # Visualizations
    print("\n📊 Erstelle Visualisierungen...")
    processor.plot_window_examples(data['X_train'], data['y_train'])
    processor.plot_feature_distributions(data['X_train'], data['X_test'])
    
    # Summary
    print("\n" + "="*50)
    print("✅ SPRINT 2 ABGESCHLOSSEN - DATA PREPROCESSING")
    print("="*50)
    print(f"\nDatensatz bereit für Training:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_test:  {data['X_test'].shape}")
    print(f"  y_train: {data['y_train'].shape} (Normal: {np.sum(data['y_train']==0)}, Anomalien: {np.sum(data['y_train']==1)})")
    print(f"  y_test:  {data['y_test'].shape} (Normal: {np.sum(data['y_test']==0)}, Anomalien: {np.sum(data['y_test']==1)})")
    print("\n🎯 Nächster Schritt: anomaly_model.py + train.py")
    
    plt.show()