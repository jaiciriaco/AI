import csv
import os
import numpy as np
import matplotlib.pyplot as plt

from ee_de_base import funcion_potencias_desplazada, funcion_ackley


def leer_csv_dict(nombre_csv):
    filas = []
    with open(nombre_csv, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filas.append(row)
    return filas


def convertir_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    return float(s)


def convertir_int(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    return int(float(s))


def mejor_fila_por_metricas(filas):
    mejor = None

    i = 0
    while i < len(filas):
        te = convertir_float(filas[i]["TE"])
        vamm = convertir_float(filas[i]["VAMM"])
        pex = convertir_float(filas[i]["PEX"])

        if mejor is None:
            mejor = filas[i]
        else:
            te_m = convertir_float(mejor["TE"])
            vamm_m = convertir_float(mejor["VAMM"])
            pex_m = convertir_float(mejor["PEX"])

            if te > te_m:
                mejor = filas[i]
            elif te == te_m:
                if pex_m is None and pex is not None:
                    mejor = filas[i]
                elif pex is None and pex_m is not None:
                    pass
                else:
                    if pex is not None and pex_m is not None:
                        if pex < pex_m:
                            mejor = filas[i]
                        elif pex == pex_m:
                            if vamm < vamm_m:
                                mejor = filas[i]
                    else:
                        if vamm < vamm_m:
                            mejor = filas[i]

        i = i + 1

    return mejor


def nombre_historial_mejor_ee(funcion, fila):
    modo_sigma = fila["modo_sigma"]
    recomb = fila["recomb"]
    supervivencia = fila["supervivencia"]

    suf = supervivencia.replace("(", "").replace(
        ")", "").replace(",", "").replace("+", "plus")
    return "historial_ee_" + funcion + "_ms" + modo_sigma + "_" + recomb + "_" + suf + ".csv"


def nombre_historial_mejor_de(funcion, fila):
    variante = fila["variante"]
    F = str(fila["F"]).replace(".", "_")
    CR = str(fila["CR"]).replace(".", "_")
    return "historial_de_" + funcion + "_" + variante + "_F" + F + "_CR" + CR + ".csv"


def leer_historial_por_ejecucion(nombre_csv):
    datos = {}
    filas = leer_csv_dict(nombre_csv)

    i = 0
    while i < len(filas):
        ejec = convertir_int(filas[i]["ejecucion"])
        evals = convertir_int(filas[i]["evaluaciones"])
        fit = convertir_float(filas[i]["mejor_fitness"])

        if ejec not in datos:
            datos[ejec] = []
        datos[ejec].append((evals, fit))

        i = i + 1

    for k in datos:
        datos[k].sort(key=lambda t: t[0])

    return datos


def promedio_curva(datos_por_ejecucion):
    keys = list(datos_por_ejecucion.keys())
    if len(keys) == 0:
        return [], []

    L = len(datos_por_ejecucion[keys[0]])
    i = 1
    while i < len(keys):
        L2 = len(datos_por_ejecucion[keys[i]])
        if L2 < L:
            L = L2
        i = i + 1

    xs = []
    ys = []

    j = 0
    while j < L:
        x = datos_por_ejecucion[keys[0]][j][0]

        suma = 0.0
        k = 0
        while k < len(keys):
            suma = suma + datos_por_ejecucion[keys[k]][j][1]
            k = k + 1

        xs.append(x)
        ys.append(suma / len(keys))

        j = j + 1

    return xs, ys


def plot_curvas_progreso(nombre_fig, titulo, datos_por_ejecucion):
    plt.figure()

    keys = list(datos_por_ejecucion.keys())
    i = 0
    while i < len(keys):
        ejec = keys[i]
        xs = []
        ys = []
        j = 0
        while j < len(datos_por_ejecucion[ejec]):
            xs.append(datos_por_ejecucion[ejec][j][0])
            ys.append(datos_por_ejecucion[ejec][j][1])
            j = j + 1

        plt.plot(xs, ys, linewidth=1)
        i = i + 1

    xs_m, ys_m = promedio_curva(datos_por_ejecucion)
    if len(xs_m) > 0:
        plt.plot(xs_m, ys_m, linewidth=3)

    plt.yscale("log")
    plt.xlabel("Evaluaciones")
    plt.ylabel("Mejor fitness (log)")
    plt.title(titulo)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(nombre_fig, dpi=200)
    plt.close()


def representar_funcion_2d_guardar(funcion, titulo, limite_inferior, limite_superior, paso, nombre_png):
    x = np.arange(limite_inferior, limite_superior + paso, paso)
    y = np.arange(limite_inferior, limite_superior + paso, paso)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    i = 0
    while i < X.shape[0]:
        j = 0
        while j < X.shape[1]:
            x1 = X[i, j]
            x2 = Y[i, j]
            Z[i, j] = funcion([x1, x2])
            j = j + 1
        i = i + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)

    ax.set_title(titulo)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")

    plt.tight_layout()
    plt.savefig(nombre_png, dpi=200)
    plt.close()


def graficas_n2():
    theta = 5
    representar_funcion_2d_guardar(
        funcion=lambda x: funcion_potencias_desplazada(x, theta=theta),
        titulo="Potencias desplazada (n=2, theta=5)",
        limite_inferior=-10.0,
        limite_superior=10.0,
        paso=0.2,
        nombre_png="funcion_potencias_n2.png"
    )

    representar_funcion_2d_guardar(
        funcion=lambda x: funcion_ackley(x),
        titulo="Ackley (n=2)",
        limite_inferior=-32.77,
        limite_superior=32.77,
        paso=0.5,
        nombre_png="funcion_ackley_n2.png"
    )


def main():
    graficas_n2()

    filas_ee_pot = leer_csv_dict("resumen_ee_potencias.csv")
    filas_ee_ack = leer_csv_dict("resumen_ee_ackley.csv")
    mejor_ee_pot = mejor_fila_por_metricas(filas_ee_pot)
    mejor_ee_ack = mejor_fila_por_metricas(filas_ee_ack)

    filas_de_pot = leer_csv_dict("resumen_de_potencias.csv")
    filas_de_ack = leer_csv_dict("resumen_de_ackley.csv")
    mejor_de_pot = mejor_fila_por_metricas(filas_de_pot)
    mejor_de_ack = mejor_fila_por_metricas(filas_de_ack)

    hist_ee_pot = nombre_historial_mejor_ee("potencias", mejor_ee_pot)
    hist_ee_ack = nombre_historial_mejor_ee("ackley", mejor_ee_ack)

    hist_de_pot = nombre_historial_mejor_de("potencias", mejor_de_pot)
    hist_de_ack = nombre_historial_mejor_de("ackley", mejor_de_ack)

    if os.path.exists(hist_ee_pot):
        datos = leer_historial_por_ejecucion(hist_ee_pot)
        plot_curvas_progreso("curva_ee_potencias.png",
                             "Curvas progreso EE (Potencias) - mejor config", datos)

    if os.path.exists(hist_ee_ack):
        datos = leer_historial_por_ejecucion(hist_ee_ack)
        plot_curvas_progreso(
            "curva_ee_ackley.png", "Curvas progreso EE (Ackley) - mejor config", datos)

    if os.path.exists(hist_de_pot):
        datos = leer_historial_por_ejecucion(hist_de_pot)
        plot_curvas_progreso("curva_de_potencias.png",
                             "Curvas progreso DE (Potencias) - mejor config", datos)

    if os.path.exists(hist_de_ack):
        datos = leer_historial_por_ejecucion(hist_de_ack)
        plot_curvas_progreso(
            "curva_de_ackley.png", "Curvas progreso DE (Ackley) - mejor config", datos)


if __name__ == "__main__":
    main()
