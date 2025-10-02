"""
CareWatch Pro - Training Script
Trainiert LSTM Autoencoder auf Vitaldaten
"""

import numpy as np
import matplotlib.pyplot as plt
from anomaly_model import LSTMAnomalyDetector
import os

def plot_training_history(history: dict, save_path: str = 'training_history.png'):
    """Visualisiert Training Loss Curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('MSE Loss', fontsize=12)
    axes[0].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training History gespeichert: {save_path}")
    
    return fig

def plot_reconstruction_errors(train_errors: np.ndarray, 
                               test_errors: np.ndarray,
                               y_train: np.ndarray,
                               y_test: np.ndarray,
                               threshold: float,
                               save_path: str = 'reconstruction_errors.png'):
    """Visualisiert Reconstruction Error Distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reconstruction Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training Errors - Normal vs Anomaly
    train_normal = train_errors[y_train == 0]
    train_anomaly = train_errors[y_train == 1]
    
    axes[0, 0].hist(train_normal, bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
    axes[0, 0].hist(train_anomaly, bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
    axes[0, 0].axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 0].set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Test Errors - Normal vs Anomaly
    test_normal = test_errors[y_test == 0]
    test_anomaly = test_errors[y_test == 1]
    
    axes[0, 1].hist(test_normal, bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
    axes[0, 1].hist(test_anomaly, bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
    axes[0, 1].axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 1].set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 3. Box Plot - Train
    data_train = [train_normal, train_anomaly]
    axes[1, 0].boxplot(data_train, labels=['Normal', 'Anomaly'], patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
    axes[1, 0].axhline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 0].set_ylabel('Reconstruction Error (MSE)', fontsize=11)
    axes[1, 0].set_title('Training Set Box Plot', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Box Plot - Test
    data_test = [test_normal, test_anomaly]
    axes[1, 1].boxplot(data_test, labels=['Normal', 'Anomaly'], patch_artist=True,
                       boxprops=dict(facecolor='lightcoral'))
    axes[1, 1].axhline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1, 1].set_ylabel('Reconstruction Error (MSE)', fontsize=11)
    axes[1, 1].set_title('Test Set Box Plot', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Error Analysis gespeichert: {save_path}")
    
    return fig

def main():
    """Haupt-Training Pipeline"""
    print("="*60)
    print("  CAREWATCH PRO - LSTM AUTOENCODER TRAINING")
    print("="*60)
    
    # 1. Daten laden
    print("\n📂 Lade preprocessed data...")
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    print(f"✅ Data loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   y_train: Normal={np.sum(y_train==0)}, Anomaly={np.sum(y_train==1)}")
    print(f"   y_test:  Normal={np.sum(y_test==0)}, Anomaly={np.sum(y_test==1)}")
    
    # 2. Model erstellen
    print("\n🏗️  Erstelle LSTM Autoencoder...")
    detector = LSTMAnomalyDetector(
        window_size=100,
        n_features=3,
        latent_dim=32,
        lstm_units=(64, 32)
    )
    
    detector.build_model()
    detector.summary()
    
    # 3. Training
    # WICHTIG: Wir trainieren nur auf NORMALEN Samples!
    # Autoencoder lernt "normale" Muster, Anomalien haben hohen Reconstruction Error
    X_train_normal = X_train[y_train == 0]
    print(f"\n🎯 Training nur auf normalen Samples: {len(X_train_normal)}")
    
    history = detector.train(
        X_train=X_train_normal,
        X_val=None,  # 20% Validation Split automatisch
        epochs=50,
        batch_size=32,
        patience=10,
        verbose=1
    )
    
    # 4. Training History Plot
    print("\n📊 Erstelle Training Visualisierungen...")
    plot_training_history(history)
    
    # 5. Threshold bestimmen
    # Wir verwenden ALLE Training-Samples (inkl. bekannte Anomalien) für realistische Threshold
    detector.set_threshold(
        X_train_normal, 
        method='percentile',  # Alternativ: 'std'
        percentile=95
    )
    
    # 6. Reconstruction Errors analysieren
    print("\n🔍 Berechne Reconstruction Errors...")
    train_errors = detector.compute_reconstruction_error(X_train)
    test_errors = detector.compute_reconstruction_error(X_test)
    
    print(f"\nError Statistics:")
    print(f"  Training Set:")
    print(f"    Normal:  Mean={np.mean(train_errors[y_train==0]):.6f}, "
          f"Std={np.std(train_errors[y_train==0]):.6f}")
    print(f"    Anomaly: Mean={np.mean(train_errors[y_train==1]):.6f}, "
          f"Std={np.std(train_errors[y_train==1]):.6f}")
    print(f"  Test Set:")
    print(f"    Normal:  Mean={np.mean(test_errors[y_test==0]):.6f}, "
          f"Std={np.std(test_errors[y_test==0]):.6f}")
    print(f"    Anomaly: Mean={np.mean(test_errors[y_test==1]):.6f}, "
          f"Std={np.std(test_errors[y_test==1]):.6f}")
    
    # Plot Error Distributions
    plot_reconstruction_errors(train_errors, test_errors, y_train, y_test, 
                              detector.threshold)
    
    # 7. Quick Evaluation auf Test Set
    print("\n🎯 Quick Evaluation auf Test Set...")
    predictions, _ = detector.predict(X_test)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print("\n" + "="*60)
    print("  TEST SET PERFORMANCE")
    print("="*60)
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print("="*60)
    
    # 8. Model speichern
    print("\n💾 Speichere trainiertes Model...")
    detector.save_model('models/lstm_autoencoder.h5')
    
    # Zusätzlich: Predictions speichern für Demo
    np.save('test_predictions.npy', predictions)
    np.save('test_errors.npy', test_errors)
    print("✅ Predictions gespeichert: test_predictions.npy, test_errors.npy")
    
    print("\n" + "="*60)
    print("✅ SPRINT 3+4 ABGESCHLOSSEN - TRAINING ERFOLGREICH!")
    print("="*60)
    print("\n🎯 Nächster Schritt: demo.py für finale Visualisierungen")
    
    plt.show()

if __name__ == "__main__":
    # Models Ordner erstellen
    os.makedirs('models', exist_ok=True)
    
    # Training starten
    main()
