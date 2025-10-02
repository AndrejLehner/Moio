"""
CareWatch Pro - Demo & Visualisierung
Zeigt Model Performance und Beispiel-Predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from anomaly_model import LSTMAnomalyDetector

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Erstellt Confusion Matrix Visualisierung"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                ax=ax, annot_kws={'size': 16})
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set', fontsize=15, fontweight='bold')
    
    # Zusätzliche Metriken anzeigen
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"""
    Accuracy:  {accuracy:.3f}
    Precision: {precision:.3f}
    Recall:    {recall:.3f}
    F1-Score:  {f1:.3f}
    """
    
    plt.text(1.3, 0.5, metrics_text, transform=ax.transAxes,
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Confusion Matrix gespeichert: {save_path}")
    
    return fig

def plot_roc_curve(y_true, y_scores, save_path='roc_curve.png'):
    """Erstellt ROC Curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - Anomaly Detection', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 ROC Curve gespeichert: {save_path}")
    
    return fig

def plot_reconstruction_examples(detector, X_test, y_test, n_examples=6, 
                                 save_path='reconstruction_examples.png'):
    """Zeigt Original vs. Rekonstruierte Signale"""
    
    # Wähle Beispiele: 3 Normal, 3 Anomaly
    normal_idx = np.where(y_test == 0)[0][:n_examples//2]
    anomaly_idx = np.where(y_test == 1)[0][:n_examples//2]
    selected_idx = np.concatenate([normal_idx, anomaly_idx])
    
    # Rekonstruktion
    X_reconstructed = detector.model.predict(X_test[selected_idx], verbose=0)
    
    feature_names = ['Herzfrequenz', 'Bewegung', 'Atmung']
    
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 2.5*n_examples))
    fig.suptitle('Original vs. Rekonstruierte Signale', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(selected_idx):
        original = X_test[idx]
        reconstructed = X_reconstructed[i]
        label = "NORMAL" if y_test[idx] == 0 else "ANOMALIE"
        
        # Berechne Reconstruction Error für dieses Sample
        mse = np.mean(np.square(original - reconstructed))
        
        for j, feature_name in enumerate(feature_names):
            axes[i, j].plot(original[:, j], label='Original', linewidth=1.5, alpha=0.8)
            axes[i, j].plot(reconstructed[:, j], label='Rekonstruiert', 
                          linewidth=1.5, linestyle='--', alpha=0.8)
            
            if i == 0:
                axes[i, j].set_title(feature_name, fontsize=12, fontweight='bold')
            
            axes[i, j].set_ylabel('Normalisierter Wert', fontsize=9)
            axes[i, j].grid(True, alpha=0.3)
            
            if i == n_examples - 1:
                axes[i, j].set_xlabel('Zeitschritt', fontsize=10)
            
            if j == 0:
                axes[i, j].legend(loc='upper left', fontsize=8)
            
            # Label und Error rechts
            if j == 2:
                error_text = f"{label}\nMSE: {mse:.4f}"
                color = 'green' if label == "NORMAL" else 'red'
                axes[i, j].text(1.05, 0.5, error_text, 
                              transform=axes[i, j].transAxes,
                              fontsize=9, fontweight='bold', color=color,
                              va='center', rotation=270,
                              bbox=dict(boxstyle='round', facecolor='white', 
                                      edgecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Reconstruction Examples gespeichert: {save_path}")
    
    return fig

def plot_anomaly_score_timeline(errors, y_true, threshold, 
                                save_path='anomaly_score_timeline.png'):
    """Zeigt Anomaly Scores über Zeit"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Farben basierend auf Ground Truth
    colors = ['green' if label == 0 else 'red' for label in y_true]
    
    ax.scatter(range(len(errors)), errors, c=colors, alpha=0.6, s=20, 
              label='Samples')
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, 
              label=f'Threshold ({threshold:.3f})')
    
    # Bereiche markieren
    ax.fill_between(range(len(errors)), 0, threshold, alpha=0.2, color='green',
                    label='Normal Range')
    ax.fill_between(range(len(errors)), threshold, max(errors)*1.1, alpha=0.2, 
                    color='red', label='Anomaly Range')
    
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Anomaly Scores - Test Set Timeline', fontsize=14, fontweight='bold')
    ax.set_ylim([0, min(max(errors)*1.1, 5)])  # Cap bei 5 für Lesbarkeit
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Custom Legend für Farben
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='True Normal'),
        Patch(facecolor='red', alpha=0.6, label='True Anomaly'),
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, 
                  label=f'Threshold')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Anomaly Score Timeline gespeichert: {save_path}")
    
    return fig

def generate_classification_report(y_true, y_pred, save_path='classification_report.txt'):
    """Erstellt detaillierten Classification Report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Anomaly'],
                                   digits=3)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("  CAREWATCH PRO - CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
    
    print(f"📄 Classification Report gespeichert: {save_path}")
    print("\n" + report)
    
    return report

def main():
    """Haupt-Demo Pipeline"""
    print("="*60)
    print("  CAREWATCH PRO - DEMO & VISUALIZATION")
    print("="*60)
    
    # 1. Lade Model und Daten
    print("\n📂 Lade Model und Test Data...")
    detector = LSTMAnomalyDetector()
    detector.load_model('models/lstm_autoencoder.h5')
    
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    test_predictions = np.load('test_predictions.npy')
    test_errors = np.load('test_errors.npy')
    
    print(f"✅ Geladen:")
    print(f"   Model: models/lstm_autoencoder.h5")
    print(f"   Test Samples: {len(X_test)}")
    print(f"   Threshold: {detector.threshold:.6f}")
    
    # 2. Confusion Matrix
    print("\n📊 Erstelle Confusion Matrix...")
    plot_confusion_matrix(y_test, test_predictions)
    
    # 3. ROC Curve
    print("\n📊 Erstelle ROC Curve...")
    plot_roc_curve(y_test, test_errors)
    
    # 4. Reconstruction Examples
    print("\n📊 Erstelle Reconstruction Examples...")
    plot_reconstruction_examples(detector, X_test, y_test, n_examples=6)
    
    # 5. Anomaly Score Timeline
    print("\n📊 Erstelle Anomaly Score Timeline...")
    plot_anomaly_score_timeline(test_errors, y_test, detector.threshold)
    
    # 6. Classification Report
    print("\n📄 Generiere Classification Report...")
    generate_classification_report(y_test, test_predictions)
    
    # 7. Summary Statistics
    print("\n" + "="*60)
    print("  DEMO SUMMARY")
    print("="*60)
    
    tp = np.sum((y_test == 1) & (test_predictions == 1))
    tn = np.sum((y_test == 0) & (test_predictions == 0))
    fp = np.sum((y_test == 0) & (test_predictions == 1))
    fn = np.sum((y_test == 1) & (test_predictions == 0))
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Positives:  {tp:4d} (Korrekt erkannte Anomalien)")
    print(f"  True Negatives:  {tn:4d} (Korrekt erkannte Normale)")
    print(f"  False Positives: {fp:4d} (Fehlalarme)")
    print(f"  False Negatives: {fn:4d} (Verpasste Anomalien)")
    
    print(f"\nError Statistics:")
    print(f"  Normal Samples:")
    print(f"    Mean Error: {np.mean(test_errors[y_test==0]):.6f}")
    print(f"    Max Error:  {np.max(test_errors[y_test==0]):.6f}")
    print(f"  Anomaly Samples:")
    print(f"    Mean Error: {np.mean(test_errors[y_test==1]):.6f}")
    print(f"    Max Error:  {np.max(test_errors[y_test==1]):.6f}")
    
    print("\n" + "="*60)
    print("✅ SPRINT 5 ABGESCHLOSSEN - DEMO KOMPLETT!")
    print("="*60)
    print("\nGenerierte Visualisierungen:")
    print("  ✓ confusion_matrix.png")
    print("  ✓ roc_curve.png")
    print("  ✓ reconstruction_examples.png")
    print("  ✓ anomaly_score_timeline.png")
    print("  ✓ classification_report.txt")
    
    print("\n🎯 Nächster Schritt: README.md finalisieren & GitHub Push")
    
    plt.show()

if __name__ == "__main__":
    main()
