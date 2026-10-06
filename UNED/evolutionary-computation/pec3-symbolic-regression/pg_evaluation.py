import csv
import math
import statistics
import time
import pg_base

PROBLEMAS = {
    "problema_1": {
        "nombre": "Problema 1",
        "funcion": pg_base.problema_1,
        "a": -1.0,
        "b": 3.0,
        "M": 41
    },
    "problema_2": {
        "nombre": "Problema 2",
        "funcion": pg_base.problema_2,
        "a": 0.0,
        "b": 4.0,
        "M": 41
    },
    "problema_3": {
        "nombre": "Problema 3",
        "funcion": pg_base.problema_3,
        "a": 0.0,
        "b": 4.0,
        "M": 41
    },
    "problema_4": {
        "nombre": "Problema 4",
        "funcion": pg_base.problema_4,
        "a": 2.0,
        "b": 6.0,
        "M": 41
    },
    "problema_5": {
        "nombre": "Problema 5",
        "funcion": pg_base.problema_5,
        "a": 0.0,
        "b": 10.0,
        "M": 101
    }
}


def configuracion_base():
    return {
        "tamano_poblacion": 50,
        "max_generaciones": 80,
        "probabilidad_cruce": 0.9,
        "probabilidad_mutacion": 0.1,
        "profundidad_inicial_maxima": 4,
        "profundidad_maxima": 7,
        "profundidad_subarbol_mutacion": 3,
        "tamano_torneo": 3,
        "elitismo": True,
        "U": 1e-1,
        "K0": 1.0,
        "K1": 10.0,
        "penalizacion_no_finito": 1000.0,
        "penalizacion_tamano": 0.001,
        "tamano_objetivo": 25,
        "usar_busqueda_local": True,
        "intensidad_busqueda_local": 5,
        "paso_busqueda_local": 0.25,
        "paciencia_mutacion": 5,
        "factor_subida_mutacion": 1.2,
        "factor_bajada_mutacion": 0.95,
        "pm_min": 0.02,
        "pm_max": 0.5
    }


def desviacion_tipica(lista):
    if len(lista) <= 1:
        return 0.0
    return statistics.stdev(lista)


def construir_id_configuracion(config):
    texto = "P" + str(config["tamano_poblacion"])
    texto = texto + "_G" + str(config["max_generaciones"])
    texto = texto + "_PC" + str(config["probabilidad_cruce"])
    texto = texto + "_PM" + str(config["probabilidad_mutacion"])
    texto = texto + "_DI" + str(config["profundidad_inicial_maxima"])
    return texto


def escribir_cabeceras_si_hace_falta():
    archivo = open("pg_resumen_tuning.csv", "w", newline="")
    escritor = csv.writer(archivo)
    escritor.writerow([
        "problema", "configuracion", "n_ejecuciones", "TE",
        "VAMM", "DE_VAMM", "PEX", "DE_PEX", "tiempo_medio",
        "mejor_expresion"
    ])
    archivo.close()

    archivo = open("pg_historial_tuning.csv", "w", newline="")
    escritor = csv.writer(archivo)
    escritor.writerow([
        "problema", "configuracion", "ejecucion", "generacion",
        "fitness", "pm"
    ])
    archivo.close()

    archivo = open("pg_resumen_final.csv", "w", newline="")
    escritor = csv.writer(archivo)
    escritor.writerow([
        "problema", "configuracion", "n_ejecuciones", "TE",
        "VAMM", "DE_VAMM", "PEX", "DE_PEX", "tiempo_medio",
        "mejor_expresion"
    ])
    archivo.close()

    archivo = open("pg_historial_final.csv", "w", newline="")
    escritor = csv.writer(archivo)
    escritor.writerow([
        "problema", "configuracion", "ejecucion", "generacion",
        "fitness", "pm"
    ])
    archivo.close()

    archivo = open("pg_predicciones_final.csv", "w", newline="")
    escritor = csv.writer(archivo)
    escritor.writerow([
        "problema", "configuracion", "x", "y_real", "y_pred"
    ])
    archivo.close()


def guardar_historial(nombre_archivo, problema_id, config_id, ejecucion, historial, historial_pm):
    archivo = open(nombre_archivo, "a", newline="")
    escritor = csv.writer(archivo)

    i = 0
    while i < len(historial):
        if i == 0:
            escritor.writerow(
                [problema_id, config_id, ejecucion, i, historial[i], historial_pm[i]])
        else:
            if historial[i] != historial[i - 1] or historial_pm[i] != historial_pm[i - 1]:
                escritor.writerow(
                    [problema_id, config_id, ejecucion, i, historial[i], historial_pm[i]])
        i = i + 1

    archivo.close()


def guardar_resumen(nombre_archivo, problema_id, config_id, n_ejecuciones,
                    TE, VAMM, DE_VAMM, PEX, DE_PEX, tiempo_medio,
                    mejor_expresion):
    archivo = open(nombre_archivo, "a", newline="")
    escritor = csv.writer(archivo)
    escritor.writerow([
        problema_id, config_id, n_ejecuciones, TE,
        VAMM, DE_VAMM, PEX, DE_PEX, tiempo_medio,
        mejor_expresion
    ])
    archivo.close()


def guardar_predicciones(nombre_archivo, problema_id, config_id, muestras_x, muestras_y, arbol):
    archivo = open(nombre_archivo, "a", newline="")
    escritor = csv.writer(archivo)

    i = 0
    while i < len(muestras_x):
        y_pred = pg_base.evaluar_arbol(arbol, muestras_x[i])
        escritor.writerow(
            [problema_id, config_id, muestras_x[i], muestras_y[i], y_pred])
        i = i + 1

    archivo.close()


def evaluar_configuracion(problema_id, config, numero_ejecuciones, nombre_historial, nombre_resumen):
    datos_problema = PROBLEMAS[problema_id]
    muestras_x, muestras_y = pg_base.generar_muestras(
        datos_problema["funcion"],
        datos_problema["a"],
        datos_problema["b"],
        datos_problema["M"]
    )

    config_id = construir_id_configuracion(config)

    fitness_finales = []
    pex_exitos = []
    tiempos = []
    exitos = 0
    mejor_global = None
    mejor_expresion = ""

    ejecucion = 0
    while ejecucion < numero_ejecuciones:
        semilla = 1000 + ejecucion
        inicio = time.perf_counter()
        resultado = pg_base.ejecutar_pg(
            muestras_x, muestras_y, config, semilla, False)
        fin = time.perf_counter()

        tiempo = fin - inicio
        tiempos.append(tiempo)

        fitness_final = resultado["mejor_individuo"]["fitness"]
        fitness_finales.append(fitness_final)

        if resultado["mejor_individuo"]["exito"]:
            exitos = exitos + 1
            if resultado["evaluaciones_hasta_exito"] is not None:
                pex_exitos.append(resultado["evaluaciones_hasta_exito"])

        if mejor_global is None:
            mejor_global = resultado["mejor_individuo"]
            mejor_expresion = resultado["expresion"]
        else:
            if resultado["mejor_individuo"]["fitness"] < mejor_global["fitness"]:
                mejor_global = resultado["mejor_individuo"]
                mejor_expresion = resultado["expresion"]

        guardar_historial(nombre_historial, problema_id, config_id,
                          ejecucion + 1, resultado["historial"], resultado["historial_pm"])

        print("Problema:", problema_id,
              "| Config:", config_id,
              "| Ejecución:", ejecucion + 1,
              "| Fitness:", round(fitness_final, 6),
              "| Éxito:", resultado["mejor_individuo"]["exito"])

        ejecucion = ejecucion + 1

    TE = exitos / numero_ejecuciones
    VAMM = statistics.mean(fitness_finales)
    DE_VAMM = desviacion_tipica(fitness_finales)

    if len(pex_exitos) > 0:
        PEX = statistics.mean(pex_exitos)
        DE_PEX = desviacion_tipica(pex_exitos)
    else:
        PEX = math.nan
        DE_PEX = math.nan

    tiempo_medio = statistics.mean(tiempos)

    guardar_resumen(nombre_resumen, problema_id, config_id, numero_ejecuciones,
                    TE, VAMM, DE_VAMM, PEX, DE_PEX, tiempo_medio,
                    mejor_expresion)

    return {
        "problema": problema_id,
        "configuracion": config_id,
        "TE": TE,
        "VAMM": VAMM,
        "DE_VAMM": DE_VAMM,
        "PEX": PEX,
        "DE_PEX": DE_PEX,
        "tiempo_medio": tiempo_medio,
        "mejor_expresion": mejor_expresion,
        "mejor_individuo": mejor_global,
        "muestras_x": muestras_x,
        "muestras_y": muestras_y
    }


def generar_grid_tuning():
    grid = []

    poblaciones = [50, 100]
    cruces = [0.8, 0.9]
    mutaciones = [0.15, 0.25]
    profundidades = [4, 5]

    i = 0
    while i < len(poblaciones):
        j = 0
        while j < len(cruces):
            k = 0
            while k < len(mutaciones):
                l = 0
                while l < len(profundidades):
                    config = configuracion_base()
                    config["tamano_poblacion"] = poblaciones[i]
                    config["probabilidad_cruce"] = cruces[j]
                    config["probabilidad_mutacion"] = mutaciones[k]
                    config["profundidad_inicial_maxima"] = profundidades[l]

                    config["max_generaciones"] = 150
                    config["profundidad_maxima"] = 8
                    config["profundidad_subarbol_mutacion"] = 4
                    config["tamano_objetivo"] = 35
                    config["intensidad_busqueda_local"] = 8
                    config["paso_busqueda_local"] = 0.2

                    config["pm_min"] = mutaciones[k] / 2.0
                    if config["pm_min"] < 0.02:
                        config["pm_min"] = 0.02
                    config["pm_max"] = min(0.5, mutaciones[k] * 2.0)
                    grid.append(config)
                    l = l + 1
                k = k + 1
            j = j + 1
        i = i + 1

    return grid


def seleccionar_mejor_configuracion(resultados):
    mejor = None
    i = 0
    while i < len(resultados):
        actual = resultados[i]
        if mejor is None:
            mejor = actual
        else:
            if actual["TE"] > mejor["TE"]:
                mejor = actual
            elif actual["TE"] == mejor["TE"]:
                if actual["VAMM"] < mejor["VAMM"]:
                    mejor = actual
                elif actual["VAMM"] == mejor["VAMM"]:
                    actual_pex = actual["PEX"]
                    mejor_pex = mejor["PEX"]

                    if math.isnan(mejor_pex) and not math.isnan(actual_pex):
                        mejor = actual
                    elif not math.isnan(actual_pex) and not math.isnan(mejor_pex):
                        if actual_pex < mejor_pex:
                            mejor = actual
        i = i + 1
    return mejor


def main():
    escribir_cabeceras_si_hace_falta()

    print("INICIO TUNING PROBLEMA 1")
    grid = generar_grid_tuning()
    resultados_tuning = []

    i = 0
    while i < len(grid):
        print("\nProbando configuración", i + 1, "de", len(grid))
        resultado = evaluar_configuracion(
            "problema_1",
            grid[i],
            5,
            "pg_historial_tuning.csv",
            "pg_resumen_tuning.csv"
        )
        resultados_tuning.append(resultado)
        i = i + 1

    mejor_configuracion = seleccionar_mejor_configuracion(resultados_tuning)
    print("\nMEJOR CONFIGURACIÓN EN PROBLEMA 1")
    print(mejor_configuracion["configuracion"])
    print("TE =", mejor_configuracion["TE"])
    print("VAMM =", mejor_configuracion["VAMM"])
    print("PEX =", mejor_configuracion["PEX"])

    config_final = None
    i = 0
    while i < len(grid):
        if construir_id_configuracion(grid[i]) == mejor_configuracion["configuracion"]:
            config_final = grid[i]
        i = i + 1

    print("\nINICIO EVALUACIÓN FINAL")
    problemas = ["problema_1", "problema_2",
                 "problema_3", "problema_4", "problema_5"]

    i = 0
    while i < len(problemas):
        problema_id = problemas[i]
        print("\nEvaluando", problema_id)

        if problema_id == "problema_1":
            config_problema = dict(config_final)
            config_problema["tamano_poblacion"] = 100
            config_problema["max_generaciones"] = 250
            config_problema["probabilidad_cruce"] = 0.8
            config_problema["probabilidad_mutacion"] = 0.15
            config_problema["profundidad_inicial_maxima"] = 5
            config_problema["profundidad_maxima"] = 8
            config_problema["profundidad_subarbol_mutacion"] = 4
            config_problema["tamano_objetivo"] = 35
            config_problema["intensidad_busqueda_local"] = 8
            config_problema["paso_busqueda_local"] = 0.2
            config_problema["pm_min"] = 0.05
            config_problema["pm_max"] = 0.30

        elif problema_id == "problema_4":
            config_problema = dict(config_final)
            config_problema["probabilidad_mutacion"] = 0.25
            config_problema["pm_min"] = 0.05
            config_problema["pm_max"] = 0.5

        elif problema_id == "problema_5":
            config_problema = dict(config_final)
            config_problema["tamano_poblacion"] = 100
            config_problema["max_generaciones"] = 400
            config_problema["probabilidad_cruce"] = 0.9
            config_problema["probabilidad_mutacion"] = 0.30
            config_problema["profundidad_inicial_maxima"] = 6
            config_problema["profundidad_maxima"] = 10
            config_problema["profundidad_subarbol_mutacion"] = 5
            config_problema["tamano_objetivo"] = 55
            config_problema["intensidad_busqueda_local"] = 12
            config_problema["paso_busqueda_local"] = 0.15
            config_problema["paciencia_mutacion"] = 4
            config_problema["factor_subida_mutacion"] = 1.3
            config_problema["factor_bajada_mutacion"] = 0.97
            config_problema["pm_min"] = 0.08
            config_problema["pm_max"] = 0.50

        else:
            config_problema = dict(config_final)

        resultado = evaluar_configuracion(
            problema_id,
            config_problema,
            10,
            "pg_historial_final.csv",
            "pg_resumen_final.csv"
        )

        guardar_predicciones(
            "pg_predicciones_final.csv",
            problema_id,
            resultado["configuracion"],
            resultado["muestras_x"],
            resultado["muestras_y"],
            resultado["mejor_individuo"]["arbol"]
        )

        i = i + 1


if __name__ == "__main__":
    main()
