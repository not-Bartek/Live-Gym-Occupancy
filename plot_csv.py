import pandas as pd
import matplotlib.pyplot as plt

# 1. Wczytanie danych z parsowaniem daty
df = pd.read_csv("occupancy_log_medium_nobufor.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# 2. Konfiguracja stylu wykresu
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# --- Wykres 1: Liczba osób ---
ax1.plot(df["timestamp"], df["people_in_frame"], label="Osoby w kadrze", color="#9B59B6", linewidth=2, alpha=0.7)
ax1.plot(df["timestamp"], df["people_in_zone"], label="Osoby w strefie", color="#2ECC71", linewidth=2.5)

# Półprzezroczyste wypełnienie pod strefą dla czytelności
ax1.fill_between(df["timestamp"], df["people_in_zone"], color="#2ECC71", alpha=0.15)

ax1.set_title("Analiza obłożenia w czasie (CCTV Monitoring)", fontsize=14, fontweight="bold", pad=12)
ax1.set_ylabel("Liczba osób", fontsize=11)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
ax1.yaxis.get_major_locator().set_params(integer=True)  # tylko liczby całkowite na osi Y

# --- Wykres 2: Stabilność FPS ---
ax2.plot(df["timestamp"], df["fps"], label="FPS", color="#E67E22", linewidth=1.5, linestyle="--")
ax2.set_ylabel("FPS", fontsize=11)
ax2.set_xlabel("Czas", fontsize=11)
ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

# Formatowanie osi X (czas)
fig.autofmt_xdate()
plt.tight_layout()

# Zapis i wyświetlenie
plt.savefig("wykres_oblozenia.png", dpi=300)
plt.show()