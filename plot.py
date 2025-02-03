import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Physikalische Konstanten
rho = 1.225  # kg/m³ (Luftdichte auf Meereshöhe)
g = 9.81  # Erdbeschleunigung in m/s²

# Parameterwerte aus Desmos
V0 = 1  # Anfangsvolumen in m³
mr = 10  # Masse der Rakete ohne Treibstoff in kg
b = 1  # Dichte des Treibstoffs in kg/m³ 
r = 0.005  # Rohrradius in m 
z = 1  # Zähigkeitsfaktor
L = 0.06  # Länge der Nozzle in m
Pa = 100000  # Atmosphärendruck in Pascal
P0 = 5000000  # Anfangsdruck im Tank in Pascal
V10 = 2  # Anfangsvolumen des Treibstoffs in m³
A = np.pi * r**2  # Querschnittsfläche der Düse

# Zeitbereich
t_values = np.linspace(0, 20, 2000)  # Erhöhte Zeitspanne für Sinkflug

def P(Vl_values):
    """Druck als Funktion des Volumens."""
    return P0 * (V0 / Vl_values)

def c(P_values):
    """Strömungsgeschwindigkeit (Bernoulli)."""
    return np.sqrt(2 * np.maximum(P_values - Pa, 0) / rho)  # Schutz vor negativen Werten

def Vl(t_values, V10, A, c_values):
    """Volumenveränderung über Zeit."""
    return np.maximum(V10 - A * cumulative_trapezoid(c_values, t_values, initial=0), V0)  # Verhindert negatives Volumen

# Erste Näherung für Vl
c_initial = np.ones_like(t_values)  # Dummy-Werte für Startwert
Vl_values = Vl(t_values, V10, A, c_initial)

# Druck und Strömungsgeschwindigkeit mit echten Werten berechnen
P_values = P(Vl_values)
c_values = c(P_values)

# Erneute Berechnung von Vl mit korrektem c(t)
Vl_values = Vl(t_values, V10, A, c_values)

# Weitere Berechnungen
Vdot_values = A * c_values  # Volumenflussrate
mdot_values = Vdot_values * b  # Massenflussrate
m_values = np.maximum(mr + cumulative_trapezoid(mdot_values, t_values, initial=0), mr)  # Verhindert unrealistische Massen
F_values = mdot_values * c_values  # Schubkraft

a_values = np.where(m_values > 0, (F_values / m_values) - g, -g)  # Schwerkraft berücksichtigen, Fall vermeiden
v_values = cumulative_trapezoid(a_values, t_values, initial=0)
h_values = cumulative_trapezoid(v_values, t_values, initial=0)

# Zeitpunkt bestimmen, wenn die Rakete 50 m erreicht
index_50m = np.where(h_values >= 50)[0]
t_50m = t_values[index_50m[0]] if len(index_50m) > 0 else None
print(f"Die Rakete erreicht 50 m nach {t_50m:.2f} Sekunden." if t_50m else "Die Rakete erreicht keine 50 m.")
h_max = np.max(h_values)
print(f"Die maximale Höhe beträgt {h_max:.2f} m.")


# Plot verbessern, um Sinkflug darzustellen
plt.figure(figsize=(12, 8))
plt.plot(t_values, h_values, label='Höhe h(t) [m]', color='r')

plt.axhline(50, color="purple", linewidth=1, linestyle="--", label="50m Höhe")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.xlim(0, np.max(t_values))
plt.ylim(0, np.max(h_values) * 1.1)  # Etwas Platz nach oben lassen

plt.legend()
plt.xlabel("Zeit t [s]")
plt.ylabel("Höhe [m]")
plt.title("Simulation der Raketenhöhe mit Sinkflug")
plt.grid()
plt.show()
