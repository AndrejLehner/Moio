"""
CareWatch Pro - Sensor Data Simulator
Simuliert realistische Vitaldaten für Pflegemonitoring
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt

class VitalSignSimulator:
    """Simuliert Herzfrequenz, Bewegung und Atmung mit Anomalien"""
    
    def __init__(self, sampling_rate: int = 50, seed: int = 42):
        """
        Args:
            sampling_rate: Samples pro Sekunde (Hz)
            seed: Random seed für Reproduzierbarkeit
        """
        self.sampling_rate = sampling_rate
        np.random.seed(seed)
        
    def generate_heartrate(self, duration_sec: int, base_hr: float = 70.0, 
                          add_anomaly: bool = False) -> np.ndarray:
        """Generiert Herzfrequenz-Signal mit realistischer Variabilität"""
        n_samples = duration_sec * self.sampling_rate
        t = np.linspace(0, duration_sec, n_samples)
        
        # Base heartrate mit realistischer HRV
        hr = base_hr + 5 * np.sin(2 * np.pi * 0.1 * t)  # Langsame Schwankung
        hr += 2 * np.sin(2 * np.pi * 0.25 * t)  # Atemabhängige Variation
        hr += np.random.normal(0, 1, n_samples)  # Rauschen
        
        # Anomalie: Plötzlicher Anstieg (z.B. Stress, Sturz)
        if add_anomaly:
            anomaly_start = n_samples // 2
            anomaly_duration = self.sampling_rate * 10  # 10 Sekunden
            hr[anomaly_start:anomaly_start + anomaly_duration] += 30
            
        return hr
    
    def generate_motion(self, duration_sec: int, activity_level: float = 0.3,
                       add_anomaly: bool = False) -> np.ndarray:
        """Generiert Bewegungssignal (Accelerometer-ähnlich)"""
        n_samples = duration_sec * self.sampling_rate
        t = np.linspace(0, duration_sec, n_samples)
        
        # Normale Bewegung: Ruhig mit gelegentlichen Spikes
        motion = activity_level * np.random.exponential(0.5, n_samples)
        motion += 0.1 * np.sin(2 * np.pi * 0.05 * t)  # Langsame Bewegungen
        
        # Anomalie: Sturz (plötzlicher großer Spike)
        if add_anomaly:
            fall_idx = n_samples // 2
            motion[fall_idx:fall_idx + 50] = 5.0 + np.random.normal(0, 0.5, 50)
            # Nach Sturz: Keine Bewegung
            motion[fall_idx + 50:fall_idx + 200] = 0.05 * np.random.normal(1, 0.1, 150)
            
        return motion
    
    def generate_respiration(self, duration_sec: int, breaths_per_min: float = 15.0,
                            add_anomaly: bool = False) -> np.ndarray:
        """Generiert Atmungssignal"""
        n_samples = duration_sec * self.sampling_rate
        t = np.linspace(0, duration_sec, n_samples)
        
        # Normale Atmung
        freq = breaths_per_min / 60.0
        resp = np.sin(2 * np.pi * freq * t)
        resp += 0.1 * np.random.normal(0, 1, n_samples)
        
        # Anomalie: Unregelmäßige Atmung
        if add_anomaly:
            anomaly_start = n_samples // 2
            anomaly_duration = self.sampling_rate * 20
            t_anom = t[anomaly_start:anomaly_start + anomaly_duration]
            # Unregelmäßiger Rhythmus
            resp[anomaly_start:anomaly_start + anomaly_duration] = \
                np.sin(2 * np.pi * freq * t_anom * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t_anom)))
            
        return resp
    
    def generate_dataset(self, n_normal: int = 100, n_anomaly: int = 20,
                        duration_sec: int = 60) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generiert kompletten Datensatz
        
        Returns:
            df: DataFrame mit allen Sensordaten
            labels: 0 = normal, 1 = anomaly
        """
        all_data = []
        labels = []
        
        print(f"Generiere {n_normal} normale Samples...")
        for i in range(n_normal):
            hr = self.generate_heartrate(duration_sec, add_anomaly=False)
            motion = self.generate_motion(duration_sec, add_anomaly=False)
            resp = self.generate_respiration(duration_sec, add_anomaly=False)
            
            df_sample = pd.DataFrame({
                'heartrate': hr,
                'motion': motion,
                'respiration': resp,
                'sample_id': i,
                'timestamp': np.arange(len(hr)) / self.sampling_rate
            })
            all_data.append(df_sample)
            labels.append(0)
        
        print(f"Generiere {n_anomaly} Anomalie-Samples...")
        for i in range(n_anomaly):
            # Zufällig eine Art von Anomalie wählen
            anomaly_type = np.random.choice(['heartrate', 'motion', 'respiration'])
            
            hr = self.generate_heartrate(duration_sec, add_anomaly=(anomaly_type=='heartrate'))
            motion = self.generate_motion(duration_sec, add_anomaly=(anomaly_type=='motion'))
            resp = self.generate_respiration(duration_sec, add_anomaly=(anomaly_type=='respiration'))
            
            df_sample = pd.DataFrame({
                'heartrate': hr,
                'motion': motion,
                'respiration': resp,
                'sample_id': n_normal + i,
                'timestamp': np.arange(len(hr)) / self.sampling_rate
            })
            all_data.append(df_sample)
            labels.append(1)
        
        df = pd.concat(all_data, ignore_index=True)
        labels = np.array(labels)
        
        print(f"✅ Datensatz erstellt: {len(df)} Datenpunkte, {n_normal + n_anomaly} Samples")
        return df, labels
    
    def plot_sample(self, sample_id: int, df: pd.DataFrame, label: int = 0):
        """Visualisiert ein einzelnes Sample"""
        sample_data = df[df['sample_id'] == sample_id]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        title_suffix = "ANOMALIE" if label == 1 else "NORMAL"
        fig.suptitle(f'Sample {sample_id} - {title_suffix}', fontsize=14, fontweight='bold')
        
        axes[0].plot(sample_data['timestamp'], sample_data['heartrate'], 'r-', linewidth=0.8)
        axes[0].set_ylabel('Herzfrequenz (bpm)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(sample_data['timestamp'].min(), sample_data['timestamp'].max())
        
        axes[1].plot(sample_data['timestamp'], sample_data['motion'], 'b-', linewidth=0.8)
        axes[1].set_ylabel('Bewegung (g)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(sample_data['timestamp'].min(), sample_data['timestamp'].max())
        
        axes[2].plot(sample_data['timestamp'], sample_data['respiration'], 'g-', linewidth=0.8)
        axes[2].set_ylabel('Atmung')
        axes[2].set_xlabel('Zeit (s)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(sample_data['timestamp'].min(), sample_data['timestamp'].max())
        
        plt.tight_layout()
        return fig


# Test-Code
if __name__ == "__main__":
    print("=== CareWatch Pro - Sensor Simulator ===\n")
    
    simulator = VitalSignSimulator(sampling_rate=50)
    
    # Generiere Datensatz
    df, labels = simulator.generate_dataset(n_normal=80, n_anomaly=20, duration_sec=60)
    
    # Speichere Daten
    df.to_csv('sensor_data.csv', index=False)
    np.save('labels.npy', labels)
    print("\n💾 Daten gespeichert: sensor_data.csv, labels.npy")
    
    # Visualisiere Beispiele
    print("\n📊 Erstelle Visualisierungen...")
    
    # Normales Sample
    fig1 = simulator.plot_sample(0, df, label=0)
    plt.savefig('sample_normal.png', dpi=150, bbox_inches='tight')
    print("   ✅ sample_normal.png")
    
    # Anomalie Sample
    fig2 = simulator.plot_sample(80, df, label=1)
    plt.savefig('sample_anomaly.png', dpi=150, bbox_inches='tight')
    print("   ✅ sample_anomaly.png")
    
    plt.show()
    
    print("\n✅ SPRINT 1 ABGESCHLOSSEN!")
    print("   Nächster Schritt: data_processor.py implementieren")