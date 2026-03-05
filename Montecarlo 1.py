import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from arch import arch_model
cartera = {
    "MSCI": "IWDA.AS",
    "S&P500": "^GSPC",
    "IBEX35": "^IBEX",
    "DAX": "^GDAXI",
    "BONDSAM20Y": "IS04.DE",
    "GOLD": "EGLN.L",
    "COBRE": "HG=F",
    "PETROLEO": "BZ=F",
    "SANTANDER": "SAN.MC",
    "REDEIA": "RED.MC",
    "BBVA": "BBVA.MC",
    "ACERINOX": "ACE1.F",
    "IBERDROLA": "IBE.MC",
    "COCA COLA": "KO",
    "INTEL": "INTC",
    "AMD": "AMD",
    "ARM": "ARM", 
    "NVIDIA": "NVDA",
    "TSMC": "TSMC34.SA",
    "ASML": "ASML",
    "BTC": "BTC-EUR", 
    "ETH": "ETH-EUR",
    "MONERO": "XMR-EUR",
    "ALGORAND": "ALGO-USD",
    "NEXO": "NEXO-USD",
    "FASTTOKEN": "FTN-USD",
    "SOLANA": "SOL-EUR",
    "DOGECOIN": "DOGE-EUR"
}
tiempo = {
    "diario": 1,
    "semanal": 5,
    "mensual": 21,
    "trimestral": 63,
    "anual": 252,
    "lustro": 1260,
    "decada": 2520
}

def precio(Valor):
    precio = (yf.Ticker(cartera[Valor])).history(period="max", interval="1d")
    precio = precio["Close"]
    return precio
def retorno(precio, Tiempo):
    ret = precio.pct_change(periods=tiempo[Tiempo]).dropna()*100
    return ret
def GARCH(ret):
    modelo = arch_model(ret, vol="GARCH", p=1,  q=1)
    entre = modelo.fit(disp="off")
    return entre
def simulacion(entre, preci):
    simul = entre.forecast(horizon=252, method="simulation", simulations=1000)
    retor = (simul.simulations.values[-1, :, :]).T
    ult_precio = preci.iloc[-1]
    precio_sim = ult_precio * (1+(retor/100)).cumprod(axis=0)
    return precio_sim
def driftcero(entre, preci):
    simul = entre.forecast(horizon=252, method="simulation", simulations=1000)
    retor = (simul.simulations.values[-1, :, :]-entre.params["mu"]).T
    ult_precio = preci.iloc[-1]
    precio_sim = ult_precio * (1+(retor/100)).cumprod(axis=0)
    return precio_sim

preci = precio("REDEIA")
ret = retorno(preci, "diario")
garch = GARCH(ret)
precio_sim = simulacion(garch, preci)
preci_sim = driftcero(garch, preci)

res = (precio_sim[-1, :]).mean()
print(res/(preci.iloc[-1]))

# --- VISUALIZACIÓN ---
plt.figure(figsize=(12, 6))

# Pintamos las 1000 líneas (con transparencia para ver la densidad)
plt.plot(precio_sim, color='gray', alpha=0.05)

# Pintamos la media (El escenario base)
plt.plot(precio_sim.mean(axis=1), color='red', linewidth=2, label='Precio Esperado')

# Pintamos los cuantiles (El "Cono de Probabilidad" al 95%)
q05 = np.percentile(precio_sim, 5, axis=1)
q95 = np.percentile(precio_sim, 95, axis=1)
print(q95[-1]/preci.iloc[-1])
print(q05[-1]/preci.iloc[-1])
plt.plot(q05, color='green', linestyle='--', linewidth=2, label='Peor caso (5%)')
plt.plot(q95, color='green', linestyle='--', linewidth=2, label='Mejor caso (95%)')

plt.title(f'Simulación Montecarlo GARCH: 1 año vista')
plt.xlabel('Días Futuros')
plt.ylabel('Precio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

resi = (preci_sim[-1, :]).mean()
print(resi/(preci.iloc[-1]))

# --- VISUALIZACIÓN ---
plt.figure(figsize=(12, 6))

# Pintamos las 1000 líneas (con transparencia para ver la densidad)
plt.plot(preci_sim, color='gray', alpha=0.05)

# Pintamos la media (El escenario base)
plt.plot(preci_sim.mean(axis=1), color='red', linewidth=2, label='Precio Esperado')

# Pintamos los cuantiles (El "Cono de Probabilidad" al 95%)
q05 = np.percentile(preci_sim, 5, axis=1)
q95 = np.percentile(preci_sim, 95, axis=1)
print(q95[-1]/preci.iloc[-1])
print(q05[-1]/preci.iloc[-1])
plt.plot(q05, color='green', linestyle='--', linewidth=2, label='Peor caso (5%)')
plt.plot(q95, color='green', linestyle='--', linewidth=2, label='Mejor caso (95%)')

plt.title(f'Simulación Montecarlo GARCH: 1 año vista')
plt.xlabel('Días Futuros')
plt.ylabel('Precio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


