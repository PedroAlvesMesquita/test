"""
ElecCommSphere - Generador del dataset del caso de negocio.
Laboratorio de Modelos Analiticos para la Toma de Decisiones - ISTEA
Uso docente. Genera datos sinteticos reproducibles (semilla fija).

Salida:
    eleccommsphere_transacciones.csv  (nivel transaccion, con "suciedad" intencional)
    eleccommsphere_clientes.csv       (nivel cliente, alta y atributos)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20262)

INICIO = pd.Timestamp("2025-01-01")
FIN = pd.Timestamp("2026-06-30")
N_CLIENTES = 4000

PROVINCIAS = ["Buenos Aires", "CABA", "Cordoba", "Santa Fe", "Mendoza",
              "Tucuman", "Salta", "Entre Rios", "Chaco", "Neuquen"]
P_PROV = np.array([.30, .22, .12, .10, .06, .05, .04, .04, .04, .03])

CATEGORIAS = ["Electronica", "Indumentaria", "Hogar", "Electrodomesticos", "Accesorios"]
P_CAT = np.array([.30, .25, .18, .12, .15])
TICKET_CAT = {"Electronica": 145_000, "Indumentaria": 38_000, "Hogar": 62_000,
              "Electrodomesticos": 310_000, "Accesorios": 22_000}

CANALES = ["Web", "App", "WhatsApp", "Instagram"]
P_CANAL = np.array([.46, .34, .12, .08])
PAGOS = ["Tarjeta credito", "Tarjeta debito", "Transferencia", "Billetera virtual"]
P_PAGO = np.array([.52, .21, .12, .15])
ADQUISICION = ["Organico", "Google Ads", "Meta Ads", "Referido", "Email"]
P_ADQ = np.array([.34, .26, .22, .10, .08])


# ---------------------------------------------------------------- clientes
def construir_clientes():
    ids = np.arange(1, N_CLIENTES + 1)

    # Antiguedad: cohortes de alta repartidas en el periodo (mas altas al inicio)
    dias_total = (FIN - INICIO).days
    peso = np.exp(-np.linspace(0, 2.2, dias_total + 1))
    peso /= peso.sum()
    offset_alta = RNG.choice(dias_total + 1, size=N_CLIENTES, p=peso)
    fecha_alta = INICIO + pd.to_timedelta(offset_alta, unit="D")

    edad = np.clip(RNG.normal(38, 12, N_CLIENTES), 18, 79).round().astype(int)

    # Tasa base de compra (compras por semana). Lognormal -> pocos clientes muy activos.
    tasa_base = np.exp(RNG.normal(-2.35, 0.80, N_CLIENTES))
    tasa_base = np.clip(tasa_base, 0.008, 0.85)

    # Palanca del proyecto ECS-SOI: proporcion de busquedas sin resultado util.
    p_busqueda_fallida = RNG.beta(2.0, 6.0, N_CLIENTES)

    # Propension a devolver producto
    p_devolucion = np.clip(RNG.beta(1.6, 22.0, N_CLIENTES), 0, 0.5)

    df = pd.DataFrame({
        "id_cliente": ids,
        "fecha_alta": fecha_alta,
        "edad": edad,
        "genero": RNG.choice(["F", "M", "X"], N_CLIENTES, p=[.49, .49, .02]),
        "provincia": RNG.choice(PROVINCIAS, N_CLIENTES, p=P_PROV),
        "canal_adquisicion": RNG.choice(ADQUISICION, N_CLIENTES, p=P_ADQ),
        "suscripto_newsletter": RNG.choice([1, 0], N_CLIENTES, p=[.42, .58]),
    })
    oculto = pd.DataFrame({
        "tasa_base": tasa_base,
        "p_busqueda_fallida": p_busqueda_fallida,
        "p_devolucion": p_devolucion,
    })
    return df, oculto


# ------------------------------------------------------------ transacciones
def simular_transacciones(clientes, oculto):
    """Simula compras semana a semana. Los clientes que se desenganchan bajan su
    tasa de forma exponencial: eso genera senal real en recencia y frecuencia."""
    semanas = pd.date_range(INICIO, FIN, freq="W-MON")
    n_sem = len(semanas)

    tasa = oculto["tasa_base"].to_numpy()
    p_fail = oculto["p_busqueda_fallida"].to_numpy()
    p_dev = oculto["p_devolucion"].to_numpy()
    alta = clientes["fecha_alta"].to_numpy()
    news = clientes["suscripto_newsletter"].to_numpy()

    # Probabilidad de entrar en desenganche: peor experiencia de busqueda y mas
    # devoluciones -> mas chance de abandonar. El newsletter protege un poco.
    logit = -0.45 + 3.1 * p_fail + 3.4 * p_dev - 1.25 * np.log(tasa / 0.09) - 0.55 * news
    p_desenganche = 1 / (1 + np.exp(-logit))
    desengancha = RNG.random(N_CLIENTES) < p_desenganche
    # Semana en la que arranca la caida
    sem_inicio_caida = RNG.integers(int(n_sem * 0.25), n_sem, size=N_CLIENTES)
    velocidad_caida = RNG.uniform(0.045, 0.22, N_CLIENTES)

    filas = []
    tid = 1
    for s, semana in enumerate(semanas):
        activo = semana >= alta
        factor = np.ones(N_CLIENTES)
        cayendo = desengancha & (s >= sem_inicio_caida)
        factor[cayendo] = np.exp(-velocidad_caida[cayendo] * (s - sem_inicio_caida[cayendo]))
        # Estacionalidad: picos en mayo (dia de la madre/hot sale) y noviembre (black friday)
        mes = semana.month
        estac = {5: 1.35, 6: 1.10, 11: 1.55, 12: 1.30}.get(mes, 1.0)

        lam = tasa * factor * estac * activo
        n_compras = RNG.poisson(lam)
        idx = np.nonzero(n_compras)[0]

        for i in idx:
            for _ in range(n_compras[i]):
                cat = RNG.choice(CATEGORIAS, p=P_CAT)
                base = TICKET_CAT[cat]
                monto = float(np.round(base * np.exp(RNG.normal(0, 0.42)), 2))
                unidades = int(RNG.choice([1, 1, 1, 2, 2, 3], 1)[0])
                desc = float(np.round(RNG.choice([0, 0, 0, 5, 10, 15, 20, 30],
                                                 p=[.42, .12, .08, .10, .12, .08, .05, .03]), 2))
                busquedas = int(RNG.poisson(2.6) + 1)
                fallidas = int(RNG.binomial(busquedas, p_fail[i]))
                filas.append((
                    tid,
                    int(clientes.at[i, "id_cliente"]),
                    semana + pd.Timedelta(days=int(RNG.integers(0, 7))),
                    RNG.choice(CANALES, p=P_CANAL),
                    cat,
                    monto * unidades * (1 - desc / 100),
                    unidades,
                    desc,
                    RNG.choice(PAGOS, p=P_PAGO),
                    busquedas,
                    fallidas,
                    int(RNG.random() < p_dev[i]),
                ))
                tid += 1

    cols = ["id_transaccion", "id_cliente", "fecha", "canal", "categoria", "monto",
            "unidades", "descuento_pct", "metodo_pago", "busquedas_sesion",
            "busquedas_sin_resultado", "devuelto"]
    tx = pd.DataFrame(filas, columns=cols)
    tx["monto"] = tx["monto"].round(2)
    tx = tx[tx["fecha"] <= FIN]
    return tx.sort_values("fecha").reset_index(drop=True)


# --------------------------------------------------------------- ensuciar
def ensuciar(tx, clientes):
    """Introduce problemas de calidad tipicos. El TP1 (relevamiento) y la clase de
    limpieza trabajan sobre esto."""
    tx = tx.copy()
    n = len(tx)

    # 1. Nulos en monto (~2.5%) y en metodo_pago (~1.5%)
    tx.loc[RNG.choice(n, int(n * 0.025), replace=False), "monto"] = np.nan
    tx.loc[RNG.choice(n, int(n * 0.015), replace=False), "metodo_pago"] = np.nan

    # 2. Categorias inconsistentes (mayusculas, espacios, acentos)
    variantes = {"Electronica": ["electronica", "ELECTRONICA", "Electronica ", " Electrónica"],
                 "Indumentaria": ["indumentaria", "INDUMENTARIA", "Indumentaria "],
                 "Hogar": ["hogar", "HOGAR", " Hogar"],
                 "Accesorios": ["accesorios", "Accesorios "]}
    for base, alts in variantes.items():
        mask = tx["categoria"] == base
        idx = tx.index[mask]
        elegidos = RNG.choice(idx, int(len(idx) * 0.18), replace=False)
        tx.loc[elegidos, "categoria"] = RNG.choice(alts, len(elegidos))

    # 3. Outliers de monto (cargas erroneas: se tipeo de mas)
    idx = RNG.choice(n, 22, replace=False)
    tx.loc[idx, "monto"] = tx.loc[idx, "monto"] * RNG.uniform(80, 400, 22)

    # 4. Montos negativos (devoluciones mal registradas)
    idx = RNG.choice(n, 60, replace=False)
    tx.loc[idx, "monto"] = -tx.loc[idx, "monto"].abs()

    # 5. Duplicados exactos
    dups = tx.sample(180, random_state=7)
    tx = pd.concat([tx, dups], ignore_index=True)

    # 6. Un id_cliente que no existe en la tabla de clientes (integridad referencial)
    idx = RNG.choice(len(tx), 25, replace=False)
    tx.loc[idx, "id_cliente"] = 999999

    tx = tx.sample(frac=1, random_state=11).reset_index(drop=True)

    # Clientes: edades imposibles y nulos
    cl = clientes.copy()
    idx = RNG.choice(len(cl), 30, replace=False)
    cl.loc[idx, "edad"] = RNG.choice([0, 1, 137, 150, -5], 30)
    idx = RNG.choice(len(cl), int(len(cl) * 0.02), replace=False)
    cl.loc[idx, "edad"] = np.nan
    return tx, cl


if __name__ == "__main__":
    clientes, oculto = construir_clientes()
    tx = simular_transacciones(clientes, oculto)
    tx, clientes = ensuciar(tx, clientes)

    tx["fecha"] = pd.to_datetime(tx["fecha"]).dt.strftime("%Y-%m-%d")
    clientes["fecha_alta"] = pd.to_datetime(clientes["fecha_alta"]).dt.strftime("%Y-%m-%d")

    tx.to_csv("eleccommsphere_transacciones.csv", index=False)
    clientes.to_csv("eleccommsphere_clientes.csv", index=False)

    print(f"transacciones : {len(tx):,}")
    print(f"clientes      : {len(clientes):,}")
    print(f"periodo       : {tx['fecha'].min()} -> {tx['fecha'].max()}")
