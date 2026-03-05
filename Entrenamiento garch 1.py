from arch import arch_model
import pandas as pd
import yfinance as yf

def precio(Valor):
    precio = (yf.Ticker(cartera[Valor])).history(period="max", interval="1d")
    precio = precio["Close"]
    return precio

def retorno(precio, Tiempo):
    ret = precio.pct_change(periods=tiempo[Tiempo]).dropna()*100
    return ret

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

preci = precio("REDEIA")
ret = retorno(preci, "diario")

modelo = arch_model(ret, vol="GARCH", p=1, q=1)

resultado = modelo.fit(disp="on")

prediccion = resultado.forecast(horizon=1)

print(resultado.summary())