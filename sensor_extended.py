"""
CareWatch Pro - Extended Sensor Simulator
Liegeposition & Sturzgefahr Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class ExtendedVitalSignSimulator:
    """Erweiterte Sensor-Simulation mit Liegeposition & Sturzgefahr"""
    
    def __init__(self, sampling_rate: int = 50, seed: int = 42):
        self.sampling_rate = sampling_rate
        np.random.seed(seed)
        
        # Liegeposition Definitionen (Accelerometer-ähnlich)
        self.positions = {
            'ruecken': {'x': 0.0, 'y': 0.0, 'z': 9.8},     # Flach auf Rücken
            'seite_links': {'x': -9.8, 'y': 0.0, 'z': 0.0}, # Auf linker Seite
            'seite_rechts': {'x': 9.8, 'y': 0.0, 'z': 0.0}, # Auf rechter Seite
            'bauch': {'x': 0.0, 'y': 0.0, 'z': -9.8},      # Auf Bauch
            'sitzen': {'x': 0.0, 'y': 9.8, 'z': 0.0},      # Sitzend
        }
    
    def generate_position_sensor(self, duration_sec: int, 
                                 position: str = 'ruecken',
                                 add_transitions: bool = False) -> Dict[str, np.ndarray]:
        """
        Generiert 3-Achsen Accelerometer-Daten für Liegeposition
        
        Args:
            duration_sec: Dauer in Sekunden
            position: Startposition ('ruecken', 'seite_links', etc.)
            add_transitions: Fügt Positionswechsel hinzu
            
        Returns:
            Dict mit x, y, z Accelerometer-Daten
        """
        n_samples = duration_sec * self.sampling_rate
        t = np.linspace(0, duration_sec, n_samples)
        
        # Basis Position
        pos = self.positions[position]
        x = np.ones(n_samples) * pos['x'] + np.random.normal(0, 0.2, n_samples)
        y = np.ones(n_samples) * pos['y'] + np.random.normal(0, 0.2, n_samples)
        z = np.ones(n_samples) * pos['z'] + np.random.normal(0, 0.2, n_samples)
        
        # Atemsimulation (kleine periodische Bewegung)
        breath_freq = 0.25  # 15 Atemzüge/Min
        breath_amplitude = 0.3
        z += breath_amplitude * np.sin(2 * np.pi * breath_freq * t)
        
        # Positionswechsel simulieren
        if add_transitions:
            # Wechsel bei 1/3 und 2/3 der Zeit
            transition_points = [n_samples // 3, 2 * n_samples // 3]
            positions_list = ['ruecken', 'seite_links', 'seite_rechts']
            
            for i, trans_point in enumerate(transition_points):
                new_pos = self.positions[positions_list[(i + 1) % 3]]
                transition_duration = self.sampling_rate * 2  # 2 Sekunden Übergang
                
                # Smooth transition
                for j in range(transition_duration):
                    alpha = j / transition_duration
                    idx = trans_point + j
                    if idx < n_samples:
                        x[idx] = (1 - alpha) * x[idx] + alpha * new_pos['x']
                        y[idx] = (1 - alpha) * y[idx] + alpha * new_pos['y']
                        z[idx] = (1 - alpha) * z[idx] + alpha * new_pos['z']
                
                # Neue Position nach Übergang
                if trans_point + transition_duration < n_samples:
                    x[trans_point + transition_duration:] = new_pos['x'] + \
                        np.random.normal(0, 0.2, n_samples - (trans_point + transition_duration))
                    y[trans_point + transition_duration:] = new_pos['y'] + \
                        np.random.normal(0, 0.2, n_samples - (trans_point + transition_duration))
                    z[trans_point + transition_duration:] = new_pos['z'] + \
                        np.random.normal(0, 0.2, n_samples - (trans_point + transition_duration))
        
        return {'x': x, 'y': y, 'z': z, 'timestamp': t}
    
    def generate_fall_risk_scenario(self, duration_sec: int, 
                                    risk_level: str = 'low') -> Dict:
        """
        Generiert Sensor-Daten für verschiedene Sturzrisiko-Szenarien
        
        Args:
            duration_sec: Dauer in Sekunden
            risk_level: 'low', 'medium', 'high'
            
        Returns:
            Dict mit Sensor-Daten und Risk-Score
        """
        n_samples = duration_sec * self.sampling_rate
        t = np.linspace(0, duration_sec, n_samples)
        
        if risk_level == 'low':
            # Normales Liegen, stabil
            position_data = self.generate_position_sensor(
                duration_sec, position='ruecken', add_transitions=False
            )
            movement_intensity = 0.1 + np.random.uniform(0, 0.2, n_samples)
            risk_score = np.ones(n_samples) * 10  # 10% Risiko
            
        elif risk_level == 'medium':
            # Unruhiges Liegen, häufige Positionswechsel
            position_data = self.generate_position_sensor(
                duration_sec, position='ruecken', add_transitions=True
            )
            movement_intensity = 0.5 + np.random.uniform(0, 0.5, n_samples)
            risk_score = np.ones(n_samples) * 40  # 40% Risiko
            
        elif risk_level == 'high':
            # Aufstehversuch, Bett verlassen
            position_data = self.generate_position_sensor(
                duration_sec, position='sitzen', add_transitions=False
            )
            
            # Kritischer Event in der Mitte
            event_start = n_samples // 2
            event_duration = self.sampling_rate * 5  # 5 Sekunden Event
            
            movement_intensity = np.ones(n_samples) * 0.3
            risk_score = np.ones(n_samples) * 30
            
            # Aufstehversuch simulieren
            for i in range(event_duration):
                idx = event_start + i
                if idx < n_samples:
                    # Starke Bewegung
                    movement_intensity[idx] = 3.0 + np.random.uniform(0, 2.0)
                    risk_score[idx] = 85 + np.random.uniform(0, 15)  # 85-100% Risiko
                    
                    # Position ändert sich zu aufrecht
                    position_data['z'][idx] += 2.0 * (i / event_duration)
        
        else:
            raise ValueError(f"Unknown risk level: {risk_level}")
        
        # Herzfrequenz korreliert mit Risiko
        base_hr = 70
        hr = base_hr + (risk_score / 100) * 30  # HR steigt mit Risiko
        hr += 5 * np.sin(2 * np.pi * 0.1 * t)  # Natürliche Variation
        hr += np.random.normal(0, 2, n_samples)
        
        # Atemfrequenz
        resp_rate = 15 + (risk_score / 100) * 10  # Atmung beschleunigt sich
        resp = np.sin(2 * np.pi * (resp_rate / 60) * t)
        resp += np.random.normal(0, 0.1, n_samples)
        
        return {
            'position_x': position_data['x'],
            'position_y': position_data['y'],
            'position_z': position_data['z'],
            'movement_intensity': movement_intensity,
            'heartrate': hr,
            'respiration': resp,
            'risk_score': risk_score,
            'timestamp': t
        }
    
    def classify_position(self, x: float, y: float, z: float) -> str:
        """Klassifiziert Liegeposition basierend auf Accelerometer-Daten"""
        # Finde nächste Position
        min_dist = float('inf')
        closest_pos = 'unknown'
        
        for pos_name, pos_vals in self.positions.items():
            dist = np.sqrt(
                (x - pos_vals['x'])**2 + 
                (y - pos_vals['y'])**2 + 
                (z - pos_vals['z'])**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_pos = pos_name
        
        return closest_pos
    
    def calculate_fall_risk(self, sensor_data: Dict) -> float:
        """
        Berechnet Sturzrisiko basierend auf Sensor-Daten
        
        Features für Risiko-Berechnung:
        - Bewegungsintensität (hohe Bewegung = Aufstehversuch)
        - Positionsänderungen (häufig = unruhig)
        - Sitzende Position (Vorbereitung zum Aufstehen)
        - Herzfrequenz-Anstieg
        
        Returns:
            Risk score 0-100
        """
        # Feature Extraction
        movement_mean = np.mean(sensor_data['movement_intensity'])
        movement_std = np.std(sensor_data['movement_intensity'])
        hr_mean = np.mean(sensor_data['heartrate'])
        
        # Position analysieren
        current_pos = self.classify_position(
            sensor_data['position_x'][-1],
            sensor_data['position_y'][-1],
            sensor_data['position_z'][-1]
        )
        
        # Risk Score Berechnung
        risk = 0
        
        # Bewegung Faktor (40% Gewicht)
        if movement_mean > 2.0:
            risk += 40
        elif movement_mean > 1.0:
            risk += 20
        
        # Positionsfaktor (30% Gewicht)
        if current_pos == 'sitzen':
            risk += 30
        elif current_pos in ['seite_links', 'seite_rechts']:
            risk += 10
        
        # Herzfrequenz Faktor (20% Gewicht)
        if hr_mean > 90:
            risk += 20
        elif hr_mean > 80:
            risk += 10
        
        # Variabilität Faktor (10% Gewicht)
        if movement_std > 1.0:
            risk += 10
        
        return min(risk, 100)
    
    def plot_extended_sensors(self, data: Dict, title: str = "Extended Sensor Data"):
        """Visualisiert alle Sensor-Daten inkl. Liegeposition & Risk Score"""
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        t = data['timestamp']
        
        # 1. Position Sensors (3-Achsen)
        axes[0].plot(t, data['position_x'], label='X-Achse', alpha=0.7)
        axes[0].plot(t, data['position_y'], label='Y-Achse', alpha=0.7)
        axes[0].plot(t, data['position_z'], label='Z-Achse', alpha=0.7)
        axes[0].set_ylabel('Beschleunigung (m/s²)')
        axes[0].set_title('Liegeposition Sensor (Accelerometer)', fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Movement Intensity
        axes[1].plot(t, data['movement_intensity'], color='blue', linewidth=1.5)
        axes[1].set_ylabel('Intensität')
        axes[1].set_title('Bewegungsintensität', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(2.0, color='red', linestyle='--', label='Kritischer Schwellwert')
        axes[1].legend()
        
        # 3. Herzfrequenz
        axes[2].plot(t, data['heartrate'], color='red', linewidth=1.5)
        axes[2].set_ylabel('bpm')
        axes[2].set_title('Herzfrequenz', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Atmung
        axes[3].plot(t, data['respiration'], color='green', linewidth=1.5)
        axes[3].set_ylabel('Amplitude')
        axes[3].set_title('Atmung', fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # 5. Risk Score
        axes[4].plot(t, data['risk_score'], color='orange', linewidth=2)
        axes[4].fill_between(t, 0, data['risk_score'], alpha=0.3, color='orange')
        axes[4].set_ylabel('Risk Score (%)')
        axes[4].set_xlabel('Zeit (s)')
        axes[4].set_title('Sturzrisiko', fontweight='bold')
        axes[4].set_ylim([0, 100])
        axes[4].grid(True, alpha=0.3)
        
        # Risiko-Zonen markieren
        axes[4].axhline(70, color='red', linestyle='--', label='Hohes Risiko')
        axes[4].axhline(40, color='orange', linestyle='--', label='Mittleres Risiko')
        axes[4].legend()
        
        plt.tight_layout()
        return fig


# Test & Demo
if __name__ == "__main__":
    print("=== CareWatch Pro - Extended Sensor Simulator ===\n")
    
    simulator = ExtendedVitalSignSimulator(sampling_rate=50)
    
    # Beispiel 1: Niedriges Risiko (stabiles Liegen)
    print("📊 Generiere LOW RISK Szenario...")
    data_low = simulator.generate_fall_risk_scenario(60, risk_level='low')
    fig1 = simulator.plot_extended_sensors(data_low, "Low Risk - Stabiles Liegen")
    plt.savefig('sensor_low_risk.png', dpi=150, bbox_inches='tight')
    print("   ✅ sensor_low_risk.png")
    
    # Beispiel 2: Mittleres Risiko (unruhig)
    print("📊 Generiere MEDIUM RISK Szenario...")
    data_med = simulator.generate_fall_risk_scenario(60, risk_level='medium')
    fig2 = simulator.plot_extended_sensors(data_med, "Medium Risk - Unruhiges Liegen")
    plt.savefig('sensor_medium_risk.png', dpi=150, bbox_inches='tight')
    print("   ✅ sensor_medium_risk.png")
    
    # Beispiel 3: Hohes Risiko (Aufstehversuch)
    print("📊 Generiere HIGH RISK Szenario...")
    data_high = simulator.generate_fall_risk_scenario(60, risk_level='high')
    fig3 = simulator.plot_extended_sensors(data_high, "High Risk - Aufstehversuch")
    plt.savefig('sensor_high_risk.png', dpi=150, bbox_inches='tight')
    print("   ✅ sensor_high_risk.png")
    
    # Risk Score Test
    print("\n🎯 Risk Score Berechnung:")
    risk_low = simulator.calculate_fall_risk(data_low)
    risk_med = simulator.calculate_fall_risk(data_med)
    risk_high = simulator.calculate_fall_risk(data_high)
    
    print(f"   Low Risk Scenario:    {risk_low:.1f}%")
    print(f"   Medium Risk Scenario: {risk_med:.1f}%")
    print(f"   High Risk Scenario:   {risk_high:.1f}%")
    
    # Positions-Test
    print("\n📍 Position Classification Test:")
    for pos_name, pos_vals in simulator.positions.items():
        detected = simulator.classify_position(pos_vals['x'], pos_vals['y'], pos_vals['z'])
        print(f"   {pos_name:15s} → {detected}")
    
    plt.show()
    
    print("\n✅ Extended Sensor Simulation komplett!")
    print("🎯 Nächster Schritt: Integration in Streamlit App & Kubernetes Deployment")
