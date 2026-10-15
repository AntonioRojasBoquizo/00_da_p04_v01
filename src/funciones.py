"""Funciones y constantes reutilizables del proyecto EDA de marketing bancario."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constantes y diccionarios
# ---------------------------------------------------------------------------

RENOMBRAR_BANK = {
    "age": "edad",
    "job": "trabajo",
    "marital": "estado_civil",
    "education": "educacion",
    "default": "impago",
    "housing": "hipoteca",
    "loan": "prestamo",
    "contact": "medio_contacto",
    "duration": "duracion",
    "campaign": "contactos_campana",
    "pdays": "dias_ultimo_contacto",
    "previous": "contactos_previos",
    "poutcome": "resultado_previo",
    "emp.var.rate": "tasa_var_empleo",
    "cons.price.idx": "indice_precios",
    "cons.conf.idx": "indice_confianza",
    "euribor3m": "euribor_3m",
    "nr.employed": "num_empleados",
    "y": "suscrito",
    "date": "fecha_contacto",
    "latitude": "latitud",
    "longitude": "longitud",
    "id_": "id_cliente",
}

RENOMBRAR_CLIENTES = {
    "Income": "ingresos",
    "Kidhome": "num_ninos",
    "Teenhome": "num_adolescentes",
    "Dt_Customer": "fecha_alta",
    "NumWebVisitsMonth": "visitas_web_mes",
    "ID": "id_cliente",
}

MESES = {
    "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
    "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
    "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12",
}

ORDEN_EDAD = ["<25", "25-34", "35-44", "45-54", "55-64", "65+", "desconocido"]
ORDEN_INGRESOS = ["bajo", "medio-bajo", "medio-alto", "alto"]

RUTA_IMAGENES = "../images"


# ---------------------------------------------------------------------------
# Limpieza e inspección
# ---------------------------------------------------------------------------

def resumen_de_nulos(df):
    """Devuelve una tabla con el número y el porcentaje de nulos por columna (solo las que tienen)."""
    resumen = pd.DataFrame({
        "nulos": df.isnull().sum(),
        "porcentaje": (df.isnull().mean() * 100).round(1),
    })
    return resumen[resumen["nulos"] > 0].sort_values("nulos", ascending=False)


def contar_outliers_iqr(serie):
    """Cuenta los outliers de una serie con la regla del IQR. Devuelve (n, límite_inf, límite_sup)."""
    serie = serie.dropna()
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    n = ((serie < limite_inf) | (serie > limite_sup)).sum()
    return n, limite_inf, limite_sup


# ---------------------------------------------------------------------------
# Análisis descriptivo
# ---------------------------------------------------------------------------

def tabla_frecuencias(df, columna):
    """Frecuencia absoluta y porcentaje de cada categoría de una columna (incluye nulos)."""
    frec = df[columna].value_counts(dropna=False)
    porc = (df[columna].value_counts(dropna=False, normalize=True) * 100).round(1)
    return pd.DataFrame({"frecuencia": frec, "porcentaje": porc})


def tasa_por_segmento(df, columna):
    """Clientes, suscritos, tasa de suscripción (%) y diferencia en pp frente a la tasa global."""
    tasa_global = df["suscrito_num"].mean() * 100
    tabla = (
        df.groupby(columna, observed=True)["suscrito_num"]
        .agg(clientes="count", suscritos="sum", tasa="mean")
    )
    tabla["tasa_%"] = (tabla["tasa"] * 100).round(1)
    tabla["dif_vs_global_pp"] = (tabla["tasa_%"] - tasa_global).round(1)
    return tabla[["clientes", "suscritos", "tasa_%", "dif_vs_global_pp"]].sort_values("tasa_%", ascending=False)


def pares_correlacionados(matriz, umbral=0.7):
    """Pares de variables con |correlación| >= umbral, sin repetir cada par."""
    triangulo = matriz.where(np.triu(np.ones(matriz.shape, dtype=bool), k=1))
    pares = triangulo.stack()
    pares = pares[pares.abs() >= umbral]
    return pares.sort_values(key=abs, ascending=False).round(3)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def _guardar(fig, archivo):
    """Guarda la figura en la carpeta de imágenes si se ha indicado un nombre de archivo."""
    if archivo:
        os.makedirs(RUTA_IMAGENES, exist_ok=True)
        fig.savefig(f"{RUTA_IMAGENES}/{archivo}", dpi=150, bbox_inches="tight")


def histograma(df, columna, titulo, xlabel, bins=30, linea=None):
    """Histograma de una columna numérica (sin nulos), con línea vertical de referencia opcional."""
    fig, ax = plt.subplots()
    sns.histplot(df[columna].dropna().astype(float), bins=bins, ax=ax)
    if linea is not None:
        ax.axvline(linea, color="red", linestyle="--", label=f"Referencia: {linea}")
        ax.legend()
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Número de clientes")
    plt.tight_layout()
    plt.show()


def barras_categoria(df, columna, titulo, orden=None, archivo=None):
    """Gráfico de barras horizontales con el número de clientes por categoría."""
    fig, ax = plt.subplots()
    sns.countplot(data=df, y=columna, order=orden, ax=ax, color="steelblue")
    ax.set_title(titulo)
    ax.set_xlabel("Número de clientes")
    ax.set_ylabel("")
    plt.tight_layout()
    _guardar(fig, archivo)
    plt.show()


def grafico_tasa(df, columna, titulo, orden=None, min_clientes=500, archivo=None):
    """Barras con la tasa de suscripción por segmento, la tasa global y el nº de clientes (n)."""
    tasa_global = df["suscrito_num"].mean() * 100
    tabla = tasa_por_segmento(df, columna)
    tabla = tabla[tabla["clientes"] >= min_clientes]  # se descartan segmentos con muestra pequeña

    if orden is not None:
        tabla = tabla.reindex([o for o in orden if o in tabla.index])
    else:
        tabla = tabla.sort_values("tasa_%", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x=tabla["tasa_%"], y=tabla.index.astype(str), ax=ax, color="steelblue")
    ax.axvline(tasa_global, color="red", linestyle="--", label=f"Tasa global ({tasa_global:.1f} %)")

    limite = tabla["tasa_%"].max() * 1.45
    ax.set_xlim(0, limite)
    for i, (tasa, n) in enumerate(zip(tabla["tasa_%"], tabla["clientes"])):
        ax.text(limite * 0.99, i, f"{tasa:.1f} % (n={n:,})".replace(",", "."),
                va="center", ha="right", fontsize=9)

    ax.set_title(titulo)
    ax.set_xlabel("Tasa de suscripción (%)")
    ax.set_ylabel("")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)
    plt.tight_layout()
    _guardar(fig, archivo)
    plt.show()