"""
CareWatch Pro - LSTM Autoencoder für Anomalieerkennung
Encoder-Decoder Architektur für Zeitreihen-Rekonstruktion
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

class LSTMAnomalyDetector:
    """LSTM Autoencoder für unsupervised Anomalie-Detection"""
    
    def __init__(self, window_size: int = 100, n_features: int = 3, 
                 latent_dim: int = 32, lstm_units: tuple = (64, 32)):
        """
        Args:
            window_size: Länge der Input-Sequenz (Zeitschritte)
            n_features: Anzahl Features (3: HR, Motion, Respiration)
            latent_dim: Dimension des Latent Space
            lstm_units: Tuple mit LSTM Layer Sizes (Encoder Layers)
        """
        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None
        self.threshold = None
        
    def build_model(self) -> Model:
        """
        Baut LSTM Autoencoder Architektur
        
        Architecture:
            INPUT (batch, 100, 3)
                ↓
            ENCODER: LSTM(64) → LSTM(32) → Dense(latent_dim)
                ↓
            Latent Space (32D)
                ↓
            DECODER: RepeatVector → LSTM(32) → LSTM(64) → Dense(3)
                ↓
            OUTPUT (batch, 100, 3)
        """
        print("\n🏗️  Baue LSTM Autoencoder...")
        
        # Input Layer
        inputs = layers.Input(shape=(self.window_size, self.n_features))
        
        # ==================== ENCODER ====================
        # Erste LSTM Schicht
        x = layers.LSTM(self.lstm_units[0], return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        
        # Zweite LSTM Schicht (kein return_sequences → gibt letzten Hidden State)
        encoded = layers.LSTM(self.lstm_units[1])(x)
        encoded = layers.Dropout(0.2)(encoded)
        
        # Latent Space Representation
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent_space')(encoded)
        
        # ==================== DECODER ====================
        # RepeatVector: Kopiert latent vector für alle Zeitschritte
        x = layers.RepeatVector(self.window_size)(latent)
        
        # Erste LSTM Schicht (spiegelt Encoder)
        x = layers.LSTM(self.lstm_units[1], return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Zweite LSTM Schicht
        x = layers.LSTM(self.lstm_units[0], return_sequences=True)(x)
        x = layers.Dropout(0.2)(x)
        
        # Output Layer: Rekonstruiert Original-Features
        outputs = layers.TimeDistributed(layers.Dense(self.n_features))(x)
        
        # Model kompilieren
        self.model = Model(inputs=inputs, outputs=outputs, name='LSTM_Autoencoder')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error für Rekonstruktion
            metrics=['mae']  # Mean Absolute Error als zusätzliche Metrik
        )
        
        print("✅ Model erstellt!")
        return self.model
    
    def summary(self):
        """Zeigt Model-Architektur"""
        if self.model is None:
            raise ValueError("Model muss erst mit build_model() erstellt werden!")
        
        print("\n" + "="*60)
        print("LSTM AUTOENCODER ARCHITEKTUR")
        print("="*60)
        self.model.summary()
        
        # Parameter zählen
        trainable_params = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        print(f"\n📊 Trainierbare Parameter: {trainable_params:,}")
        print(f"💾 Model Size (ca.): {trainable_params * 4 / 1024:.1f} KB")
        
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None,
              epochs: int = 50, batch_size: int = 32, 
              patience: int = 10, verbose: int = 1) -> dict:
        """
        Trainiert den Autoencoder
        
        Args:
            X_train: Training data (n_samples, window_size, n_features)
            X_val: Optional validation data (falls None, wird 20% von X_train verwendet)
            epochs: Max Anzahl Epochen
            batch_size: Batch size für Training
            patience: Early Stopping Patience
            verbose: Keras verbosity level
            
        Returns:
            history: Training history dict
        """
        if self.model is None:
            raise ValueError("Model muss erst mit build_model() erstellt werden!")
        
        print("\n🎓 Starte Training...")
        print(f"   Epochs: {epochs}, Batch Size: {batch_size}")
        print(f"   Training Samples: {len(X_train)}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/lstm_autoencoder_best.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Validation Split
        validation_data = None
        validation_split = 0.0
        if X_val is not None:
            validation_data = (X_val, X_val)
            print(f"   Validation Samples: {len(X_val)}")
        else:
            validation_split = 0.2
            print(f"   Validation Split: 20%")
        
        # Training (Autoencoder lernt Input zu rekonstruieren)
        history = self.model.fit(
            X_train, X_train,  # Input = Target (Autoencoder!)
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("\n✅ Training abgeschlossen!")
        
        return history.history
    
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Berechnet Reconstruction Error (MSE) für jedes Sample
        
        Args:
            X: Input data (n_samples, window_size, n_features)
            
        Returns:
            errors: Array mit MSE pro Sample (n_samples,)
        """
        if self.model is None:
            raise ValueError("Model muss erst trainiert werden!")
        
        # Rekonstruktion
        X_reconstructed = self.model.predict(X, verbose=0)
        
        # MSE pro Sample (über alle Zeitschritte und Features)
        errors = np.mean(np.square(X - X_reconstructed), axis=(1, 2))
        
        return errors
    
    def set_threshold(self, X_train: np.ndarray, method: str = 'percentile', 
                     percentile: float = 95, std_multiplier: float = 2.0):
        """
        Setzt Anomalie-Threshold basierend auf Training Data
        
        Args:
            X_train: Training data (nur normale Samples!)
            method: 'percentile' oder 'std'
            percentile: Wenn method='percentile', welches Percentile (95-99 typisch)
            std_multiplier: Wenn method='std', wieviele STDs über Mean
        """
        print(f"\n🎯 Berechne Anomalie-Threshold ({method})...")
        
        # Berechne Reconstruction Errors auf Training Data
        train_errors = self.compute_reconstruction_error(X_train)
        
        if method == 'percentile':
            self.threshold = np.percentile(train_errors, percentile)
            print(f"   Threshold (P{percentile}): {self.threshold:.6f}")
        elif method == 'std':
            mean_error = np.mean(train_errors)
            std_error = np.std(train_errors)
            self.threshold = mean_error + std_multiplier * std_error
            print(f"   Mean Error: {mean_error:.6f}")
            print(f"   Std Error:  {std_error:.6f}")
            print(f"   Threshold (μ + {std_multiplier}σ): {self.threshold:.6f}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Statistiken
        n_train_anomalies = np.sum(train_errors > self.threshold)
        print(f"   False Positives auf Training: {n_train_anomalies}/{len(train_errors)} "
              f"({100*n_train_anomalies/len(train_errors):.1f}%)")
        
        return self.threshold
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        Vorhersage: Normal (0) oder Anomalie (1)
        
        Args:
            X: Input data (n_samples, window_size, n_features)
            
        Returns:
            predictions: Binary predictions (0/1)
            errors: Reconstruction errors für jedes Sample
        """
        if self.threshold is None:
            raise ValueError("Threshold muss erst mit set_threshold() gesetzt werden!")
        
        errors = self.compute_reconstruction_error(X)
        predictions = (errors > self.threshold).astype(int)
        
        return predictions, errors
    
    def save_model(self, filepath: str = 'models/lstm_autoencoder.h5'):
        """Speichert trainiertes Model"""
        if self.model is None:
            raise ValueError("Model muss erst trainiert werden!")
        
        import os
        os.makedirs('models', exist_ok=True)
        
        self.model.save(filepath)
        
        # Speichere auch Threshold
        np.save(filepath.replace('.h5', '_threshold.npy'), self.threshold)
        
        print(f"✅ Model gespeichert: {filepath}")
        print(f"✅ Threshold gespeichert: {filepath.replace('.h5', '_threshold.npy')}")
    
    def load_model(self, filepath: str = 'models/lstm_autoencoder.h5'):
        """Lädt gespeichertes Model"""
        self.model = keras.models.load_model(filepath)
        
        # Lade auch Threshold
        threshold_path = filepath.replace('.h5', '_threshold.npy')
        if os.path.exists(threshold_path):
            self.threshold = np.load(threshold_path)
            print(f"✅ Threshold geladen: {self.threshold:.6f}")
        
        print(f"✅ Model geladen: {filepath}")


# Test Code
if __name__ == "__main__":
    print("=== CareWatch Pro - LSTM Autoencoder Test ===\n")
    
    # Erstelle Model
    detector = LSTMAnomalyDetector(
        window_size=100,
        n_features=3,
        latent_dim=32,
        lstm_units=(64, 32)
    )
    
    # Build
    detector.build_model()
    detector.summary()
    
    print("\n✅ Model bereit für Training!")
    print("🎯 Nächster Schritt: train.py ausführen")
