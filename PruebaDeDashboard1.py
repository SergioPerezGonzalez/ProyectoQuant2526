"""
Mejorar el rendimiento
Optimizador de sharpe ratio diversificando
Analizador de parámetro

    
"""
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from arch import arch_model
#Activos que me interesan
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
#Funciones con las que trabajar
@st.cache_data(ttl=3600)
#Saca las evoluciones de los precios
def precio(Valor):
    precio = (yf.Ticker(cartera[Valor])).history(period="max", interval="1d")
    precio = precio["Close"]
    return precio
#Saca los retornos dados en cada período de tiempo
def retorno(precio, Tiempo):
    ret = precio.pct_change(periods=tiempo[Tiempo]).dropna()*100
    return ret
#Saca la volatilidad media respecto al tiempo
def volatilidad(ret):
    vol = np.sqrt(((np.where(ret > 0, 0, ret)**2).mean()))
    return vol
#Saca el sortino ratio
def sortino_ratio(ret, vol):
    #Retorno/volatilidad negativa, no penaliza la volatilidad positiva
    sort = ret.mean()/vol
    return sort
#Es un cálculo muy muy simplista que determina la cantidad de dinero que hay que invertir en el activo
def halfkelly(ret, vol):
    halfk = ((ret.mean()/100)/(vol/100)**2)/2
    return halfk
#Es un algoritmo simple que busca la caida más brusca que haya habido en el periodo de tiempo estipulado
#Devuelve el drawdownmaximo
def drawdownmax(ret):
    dmax = 0
    for val in ret:
        if dmax > val:
            dmax=val
    #OJO: Lo ponemos en negativo para invertir el signo, al ser perdidas, se devolvía en negativo
    return -dmax
#Devuelve el VaR, es decir, lo máximo que arriesgas matemáticamente el 99% de los dias 
def VaR(vol):
    #Z Score del 99%, el estandar de la industria es del 95%
    z = 2.33
    var = vol*z
    return var
#Analiza el riesgo medio el peor 1% de los días
def CVaR(ret, var):
    cvar = -(ret[ret < -abs(var)].mean())
    return cvar
#La curtosis mide lo estable que se comporta un activo, cuanto mayor sea el valor, más loco es el activo analizado
def curtosis(ret):
    #Se utiliza la desviaciones y la desviación estandar elevado a 4 para castigar lo máximo posible las colas largas
    #Se le resta tres ya que se compara con la curtosis de la campana de gauss que es 3
    curt = (((ret - ret.mean())**4).mean())/((ret.std())**4)-3
    return curt
#Crea la media del precio respecto al tiempo estipulado
def mediamovilsimple(precio, n):
    medmovs = precio.rolling(window=n).mean()
    return medmovs
#Es el multiplicador basado en el n días seleccionados para el proceso
def k(n):
    k = 2/(n+1)
    return k
#Es una herramienta básica de trading que mide la aceleración del precio
def MACD(precio):
    macd = pd.DataFrame()
    n1 = 12
    n2 = 26
    macd["Rapida"]= (precio.ewm(span=n1).mean() - precio.ewm(span=n2).mean())/(precio.ewm(span=n2).mean())*100
    n3 = 9
    macd["Señal"]= (macd["Rapida"]).ewm(span=n3).mean()
    macd["Conclusión"]= macd["Rapida"] - macd["Señal"]
    return macd

#El modelo garch trata la volatilidad como una mezcla entre la volatilidad del pasado, la influencia de las noticias presentes y un mínimo de volatilidad
def GARCH(ret):
    modelo = arch_model(ret, vol="GARCH", p=1,  q=1)
    entre = modelo.fit(disp="off")
    return entre
#Esto hace una simulación de montecarlo, hace recorridos aleatorios basado en la rentabilidad pasada y en la volatilidad del activo
def simulacion(entre, preci):
    #Esta pensado para utilizarlo en dias, hace 1000 simulaciones
    simul = entre.forecast(horizon=252, method="simulation", simulations=1000)
    retor = (simul.simulations.values[-1, :, :]).T
    ult_precio = preci.iloc[-1]
    precio_sim = ult_precio * (1+(retor/100)).cumprod(axis=0)
    return precio_sim
#Es tambien una simulación de montecarlo, pero es una versión pesmista, elimina cualquier tendencia alcista
def driftcero(entre, preci):
    simul = entre.forecast(horizon=252, method="simulation", simulations=1000)
    retor = (simul.simulations.values[-1, :, :]-entre.params["mu"]).T
    ult_precio = preci.iloc[-1]
    precio_sim = ult_precio * (1+(retor/100)).cumprod(axis=0)
    return precio_sim
#Dashboard
st.title("Dashboard de inversión: Monte su cartera💲💲💲")
col1, col2= st.columns(2)
with col1:
    Valor = st.selectbox("Valores", list(cartera.keys()))
with col2: 
    Tiempo = st.selectbox("Tiempo", list(tiempo.keys()))
if st.button("Calcular"):
    st.subheader("General")
    preci = precio(Valor)
    ret = retorno(preci, Tiempo)
    vol = volatilidad(ret)
    sort = sortino_ratio(ret, vol)
    halfk = halfkelly(ret, vol)
    drawmax = drawdownmax(ret)
    var = VaR(vol)
    cvar = CVaR(ret, var)
    curt = curtosis(ret)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Precio")
        st.line_chart(preci)
    with col2:
        st.subheader("Rentabilidad")
        st.line_chart(ret)
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(label="Volatilidad", value=str(vol))
        with st.container(border=True):
            st.metric(label="Rentabilidad media", value=str(ret.mean()))
    with col2:
        with st.container(border=True):
            st.metric(label="Sortino Ratio", value=str(sort))
        with st.container(border=True):
            st.metric(label="Half Kelly", value=str(halfk))
        with st.container(border=True):
            st.metric(label="Drawdown máximo", value=str(drawmax))
    with col3:
        with st.container(border=True):
            st.metric(label="VaR", value=str(var))
        with st.container(border=True):
            st.metric(label="CVaR", value=str(cvar))
        with st.container(border=True):
            st.metric(label="Curtosis", value=str(curt))
    st.divider()
    st.subheader("Tendencia")
    medmovs20 = mediamovilsimple(preci, 20)
    medmovs50 = mediamovilsimple(preci, 50)
    medmovs200 = mediamovilsimple(preci, 200)

    medmovs = pd.DataFrame()
    medmovs["Precio"]= preci
    medmovs["Media móvil 20"] = medmovs20
    medmovs["Media móvil 50"] = medmovs50
    medmovs["Media móvil 200"] = medmovs200

    bandasbollin  = pd.DataFrame()
    bandasbollin["Precio"]= preci
    bandasbollin["Alto"] = medmovs20 + 2*(preci.rolling(window=20).std())
    bandasbollin["Centro"] = medmovs20
    bandasbollin["Bajo"] = medmovs20 - 2*(preci.rolling(window=20).std())

    macd = MACD(preci)

    st.subheader("Medias móviles")
    st.line_chart(medmovs)

    st.subheader("Bandas de Bollinger")
    st.line_chart(bandasbollin)

    st.subheader("MACD")
    st.line_chart(macd)

    st.divider()

    st.subheader("Riesgo")
    garch = GARCH(ret)
    omega = garch.params['omega']
    alfa = garch.params['alpha[1]']
    beta = garch.params['beta[1]']
    pers = alfa + beta
    vidmedgarch = np.log(0.5)/np.log(pers)

    col1, col2, col3= st.columns(3)
    with col1:
        with st.container(border=True):
            st.metric(label="Alfa", value=str(alfa))
        with st.container(border=True):
            st.metric(label="Persistencia", value=str(pers))  
    with col2:
        with st.container(border=True):
            st.metric(label="Beta", value=str(beta)) 
        with st.container(border=True):
            st.metric(label="Vida media", value=str(vidmedgarch)) 
    with col3:
        with st.container(border=True):
            st.metric(label="Omega", value=str(omega))
    st.subheader("Volatilidad evolución")
    st.line_chart(garch.conditional_volatility)
    st.divider()



    st.subheader("Simulación")
    precio_sim = simulacion(garch, preci)
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

    drift = driftcero(garch, preci)
    # --- VISUALIZACIÓN ---
    plt.figure(figsize=(12, 6))

    # Pintamos las 1000 líneas (con transparencia para ver la densidad)
    plt.plot(drift, color='gray', alpha=0.05)

    # Pintamos la media (El escenario base)
    plt.plot(drift.mean(axis=1), color='red', linewidth=2, label='Precio Esperado')

    # Pintamos los cuantiles (El "Cono de Probabilidad" al 95%)
    dq05 = np.percentile(drift, 5, axis=1)
    dq95 = np.percentile(drift, 95, axis=1)
    print(dq95[-1]/preci.iloc[-1])
    print(dq05[-1]/preci.iloc[-1])
    plt.plot(q05, color='green', linestyle='--', linewidth=2, label='Peor caso (5%)')
    plt.plot(q95, color='green', linestyle='--', linewidth=2, label='Mejor caso (95%)')

    plt.title(f'Simulación Montecarlo GARCH: 1 año vista')
    plt.xlabel('Días Futuros')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    st.divider()