import numpy as np
import matplotlib.pyplot as plt

# --- Parametry zadané v Úloze 1 ---
L = 5.0
V0 = 5.0

# Vypočtené energie vázaných stavů (v Hartree) z předchozího řešení
E_n = [0.1668, 0.6698, 1.5172, 2.7242, 4.3263]

# Nastavení osy x rozdělené na dvě oblasti
# Oblast I: 0 až L
x1 = np.linspace(0, L, 500)
# Oblast II: L až kousek za bariéru pro ukázku exponenciálního útlumu
x2 = np.linspace(L, 10, 500)

plt.figure(figsize=(10, 8))

# 1. Vykreslení tvaru samotného potenciálu
# V x=0 je nekonečná stěna, v x=L skok z 0 na V0
x_pot = [0, 0, L, L, 10]
y_pot = [6, 0, 0, V0, V0] # 6 je jen vizuální hranice pro nekonečno
plt.plot(x_pot, y_pot, 'k-', linewidth=2.5, label='Potenciál V(x)')

# 2. Vykreslení vlnových funkcí pro každý stav
# Faktor pro zmenšení amplitudy funkcí, aby se lépe vešly do jámy (čistě vizuální úprava)
scale = 0.8 

for i, E in enumerate(E_n):
    # Výpočet vlnového vektoru (k) a koeficientu útlumu (kappa) pro danou energii
    k = np.sqrt(2 * E)
    kappa = np.sqrt(2 * (V0 - E))
    
    # Pro vizualizaci tvaru nepotřebujeme funkce přísně normovat, 
    # položíme amplitudu A = 1 a koeficient C dopočítáme ze spojitosti v x = L
    A = 1.0
    C = A * np.sin(k * L)
    
    # Definice vlnových funkcí v oblastech
    psi_1 = A * np.sin(k * x1)
    psi_2 = C * np.exp(-kappa * (x2 - L))
    
    # Získání barvy z aktuálního cyklu pro konzistenci
    color = plt.cm.tab10(i)
    
    # Kreslíme vlnovou funkci: y = E + (škálovaná_psi). 
    # Funkce se tak "vznáší" na své energetické hladině.
    plt.plot(x1, E + scale * psi_1, color=color, label=f'Stav n={i+1} (E = {E:.4f} Ha)')
    plt.plot(x2, E + scale * psi_2, color=color) # Druhá část stejnou barvou
    
    # Vyznačení horizontální linie energetické hladiny E_n
    plt.hlines(E, 0, 10, colors='gray', linestyles='dashed', alpha=0.4)

# --- Formátování grafu ---
plt.title('Vlnové funkce vázaných stavů (Semi-infinite potential)', fontsize=14)
plt.xlabel('Poloha x (Bohrovy poloměry)', fontsize=12)
plt.ylabel('Energie (Hartree) / Vlnová funkce $\psi(x)$', fontsize=12)

# Zobrazení os s rozumnými limity
plt.xlim(-1, 10)
plt.ylim(-0.5, 6)

# Zapnutí mřížky a legendy
plt.grid(True, alpha=0.2)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

# Finální zobrazení
plt.show()