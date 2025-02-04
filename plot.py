import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Physikalische Konstanten
rho = 1.225  # kg/m³ (Luftdichte auf Meereshöhe)
g = 9.81  # Erdbeschleunigung in m/s²

# Parameterwerte
V0 = 0.0001  # Anfangsvolumen in m³
mr = 10  # Masse der Rakete ohne Treibstoff in kg
b = 1  # Dichte des Treibstoffs in kg/m³
r = 0.015  # Rohrradius in m
z = 1  # Zähigkeitsfaktor
L = 0.06  # Länge der Nozzle in m
Pa = 100000  # Atmosphärendruck in Pascal
P0 = 5000000  # Anfangsdruck im Tank in Pascal
V10 = 0.0002  # Anfangsvolumen des Treibstoffs in m³
A = np.pi * r ** 2  # Querschnittsfläche der Düse


t_values = np.linspace(0, 20, 200000)


def P(Vl_values):
    """Druck als Funktion des Volumens mit Schutz vor Division durch Null."""
    return P0 * (V0 / Vl_values)


def c(P_values):
    """Strömungsgeschwindigkeit (Bernoulli) mit Schutz vor negativen Argumenten."""
    return np.sqrt(2*g*L + 2 * np.maximum(P_values - Pa, 0) / rho)


def Vl(t_vals, V10, A, c_vals):
    """Volumenveränderung über Zeit."""
    return 0.0003 - A * cumulative_trapezoid(c_vals, t_vals, initial=0)


# Erste Näherung für Vl
c_initial = np.zeros_like(t_values)
Vl_values = Vl(t_values, V10, A, c_initial)

# Druck und Strömungsgeschwindigkeit mit echten Werten berechnen
P_values = P(Vl_values)
c_values = c(P_values)

# Erneute Berechnung von Vl mit korrektem c(t)
Vl_values = Vl(t_values, V10, A, c_values)

# Weitere Berechnungen
Vdot_values = A * c_values  # Volumenflussrate
mdot_values = Vdot_values * b * A # Massenflussrate
m_values = (V10 - (Vdot_values * t_values)) + mr
F_values = (mdot_values * c_values) - (9.81 * m_values)
a_values = F_values / m_values
v_values = cumulative_trapezoid(a_values, t_values, initial=0)
h_values = cumulative_trapezoid(v_values, t_values, initial=0)

# Zeitpunkt bestimmen, wenn die Rakete 50 m erreicht
index_50m = np.where(h_values >= 50)[0]
t_50m = t_values[index_50m[0]] if len(index_50m) > 0 else None
if t_50m is not None:
    print(f"Die Rakete erreicht 50 m nach {t_50m:.2f} Sekunden.")
else:
    print("Die Rakete erreicht keine 50 m.")

h_max = np.nanmax(h_values)
print(f"Die maximale Höhe beträgt {h_max:.2f} m.")
print(F_values)

# Plot
plt.figure(figsize=(12, 8))
plt.plot(t_values, h_values, label="Höhe h(t) [m]", color="r")
plt.axhline(50, color="purple", linewidth=1, linestyle="--", label="50m Höhe")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.xlim(0, np.max(t_values))

# Nur plotten, wenn h_max endlich und > 0
if np.isfinite(h_max) and h_max > 0:
    plt.ylim(0, h_max * 1.1)

plt.legend()
plt.xlabel("Zeit t [s]")
plt.ylabel("Höhe [m]")
plt.title("Simulation der Raketenhöhe")
plt.grid()
plt.show()
