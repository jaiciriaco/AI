import csv
import time

from ee_de_base import funcion_potencias_desplazada, funcion_ackley, ee_ejecutar, de_ejecutar


def calcular_te_vamm_pex(resultados, epsilon_exito):
    n = len(resultados)

    exitos = 0
    suma_mejores = 0.0
    suma_eval_exito = 0.0
    exitos_con_eval = 0

    i = 0
    while i < n:
        suma_mejores = suma_mejores + resultados[i]["mejor_fitness_final"]

        if resultados[i]["mejor_fitness_final"] <= epsilon_exito:
            exitos = exitos + 1

        if resultados[i]["evaluaciones_exito"] is not None:
            suma_eval_exito = suma_eval_exito + \
                resultados[i]["evaluaciones_exito"]
            exitos_con_eval = exitos_con_eval + 1

        i = i + 1

    te = exitos / n
    vamm = suma_mejores / n

    if exitos_con_eval > 0:
        pex = suma_eval_exito / exitos_con_eval
    else:
        pex = None

    return te, vamm, pex


def ejecutar_experimento_ee(funcion, n, lim_inf, lim_sup, theta,
                            mu, lambd, ng,
                            modo_sigma, recomb, supervivencia,
                            sigma_inicial, epsilon_sigma,
                            epsilon_exito, ejecuciones, semilla_base,
                            csv_historial):
    resultados = []

    with open(csv_historial, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algoritmo", "funcion", "ejecucion", "generacion", "mejor_fitness", "evaluaciones",
                         "mu", "lambda", "modo_sigma", "recomb", "supervivencia", "sigma_inicial"])

        e = 0
        while e < ejecuciones:
            semilla = semilla_base + e

            if funcion == "potencias":
                def funcion_objetivo(
                    x): return funcion_potencias_desplazada(x, theta)
            else:
                funcion_objetivo = funcion_ackley

            mejor, historial, evals = ee_ejecutar(
                funcion_objetivo=funcion_objetivo,
                n=n,
                limite_inferior=lim_inf,
                limite_superior=lim_sup,
                mu=mu,
                lambd=lambd,
                max_generaciones=ng,
                modo_sigma=modo_sigma,
                tipo_recombinacion=recomb,
                tipo_supervivencia=supervivencia,
                sigma_inicial=sigma_inicial,
                epsilon_sigma=epsilon_sigma,
                semilla=semilla
            )

            eval_exito = None
            i = 0
            while i < len(historial):
                if historial[i]["mejor_fitness"] <= epsilon_exito:
                    eval_exito = historial[i]["evaluaciones"]
                    break
                i = i + 1

            resultados.append({
                "mejor_fitness_final": mejor["fitness"],
                "evaluaciones_exito": eval_exito
            })

            i = 0
            while i < len(historial):
                writer.writerow(["EE", funcion, e,
                                 historial[i]["generacion"], historial[i]["mejor_fitness"], historial[i]["evaluaciones"],
                                 mu, lambd, modo_sigma, recomb, supervivencia, sigma_inicial])
                i = i + 1

            e = e + 1

    return resultados


def ejecutar_experimento_de(funcion, n, lim_inf, lim_sup, theta,
                            np, ng,
                            F, CR, variante,
                            epsilon_exito, ejecuciones, semilla_base,
                            csv_historial):
    resultados = []

    with open(csv_historial, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algoritmo", "funcion", "ejecucion", "generacion", "mejor_fitness", "evaluaciones",
                         "np", "F", "CR", "variante"])

        e = 0
        while e < ejecuciones:
            semilla = semilla_base + e

            if funcion == "potencias":
                def funcion_objetivo(
                    x): return funcion_potencias_desplazada(x, theta)
            else:
                funcion_objetivo = funcion_ackley

            mejor, historial, evals = de_ejecutar(
                funcion_objetivo=funcion_objetivo,
                n=n,
                limite_inferior=lim_inf,
                limite_superior=lim_sup,
                np=np,
                max_generaciones=ng,
                F=F,
                CR=CR,
                variante=variante,
                epsilon_exito=epsilon_exito,
                semilla=semilla
            )

            eval_exito = None
            i = 0
            while i < len(historial):
                if historial[i]["mejor_fitness"] <= epsilon_exito:
                    eval_exito = historial[i]["evaluaciones"]
                    break
                i = i + 1

            resultados.append({
                "mejor_fitness_final": mejor["fitness"],
                "evaluaciones_exito": eval_exito
            })

            i = 0
            while i < len(historial):
                writer.writerow(["DE", funcion, e,
                                 historial[i]["generacion"], historial[i]["mejor_fitness"], historial[i]["evaluaciones"],
                                 np, F, CR, variante])
                i = i + 1

            e = e + 1

    return resultados


def guardar_resumen_ee(csv_resumen, funcion, filas_resumen):
    with open(csv_resumen, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "mu", "lambda", "ng", "ejecuciones",
                         "modo_sigma", "recomb", "supervivencia", "sigma_inicial",
                         "epsilon_exito", "TE", "VAMM", "PEX", "tiempo_seg"])
        i = 0
        while i < len(filas_resumen):
            r = filas_resumen[i]
            writer.writerow([funcion,
                             r["mu"], r["lambda"], r["ng"], r["ejecuciones"],
                             r["modo_sigma"], r["recomb"], r["supervivencia"], r["sigma_inicial"],
                             r["epsilon_exito"], r["TE"], r["VAMM"], r["PEX"], r["tiempo_seg"]])
            i = i + 1


def guardar_resumen_de(csv_resumen, funcion, filas_resumen):
    with open(csv_resumen, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["funcion", "np", "ng", "ejecuciones",
                         "variante", "F", "CR",
                         "epsilon_exito", "TE", "VAMM", "PEX", "tiempo_seg"])
        i = 0
        while i < len(filas_resumen):
            r = filas_resumen[i]
            writer.writerow([funcion,
                             r["np"], r["ng"], r["ejecuciones"],
                             r["variante"], r["F"], r["CR"],
                             r["epsilon_exito"], r["TE"], r["VAMM"], r["PEX"], r["tiempo_seg"]])
            i = i + 1


def barrido_ee(funcion, n, lim_inf, lim_sup, theta,
               mu, lambd, ng, ejecuciones,
               sigma_inicial, epsilon_sigma,
               epsilon_exito, semilla_base,
               prefijo_historial, csv_resumen):
    configuraciones = []

    configuraciones.append(
        {"modo_sigma": "1", "recomb": "discreta_global",   "supervivencia": "(mu,lambda)"})
    configuraciones.append(
        {"modo_sigma": "1", "recomb": "intermedia_global", "supervivencia": "(mu+lambda)"})
    configuraciones.append(
        {"modo_sigma": "n", "recomb": "discreta_global",   "supervivencia": "(mu,lambda)"})
    configuraciones.append(
        {"modo_sigma": "n", "recomb": "intermedia_global", "supervivencia": "(mu+lambda)"})

    filas_resumen = []

    c = 0
    while c < len(configuraciones):
        modo_sigma = configuraciones[c]["modo_sigma"]
        recomb = configuraciones[c]["recomb"]
        supervivencia = configuraciones[c]["supervivencia"]

        nombre_csv_hist = prefijo_historial + "_ms" + modo_sigma + "_" + recomb + "_" + \
            supervivencia.replace("(", "").replace(")", "").replace(
                ",", "").replace("+", "plus") + ".csv"

        t0 = time.time()

        resultados = ejecutar_experimento_ee(
            funcion=funcion,
            n=n, lim_inf=lim_inf, lim_sup=lim_sup, theta=theta,
            mu=mu, lambd=lambd, ng=ng,
            modo_sigma=modo_sigma, recomb=recomb, supervivencia=supervivencia,
            sigma_inicial=sigma_inicial,
            epsilon_sigma=epsilon_sigma,
            epsilon_exito=epsilon_exito,
            ejecuciones=ejecuciones,
            semilla_base=semilla_base + 1000 * c,
            csv_historial=nombre_csv_hist
        )

        te, vamm, pex = calcular_te_vamm_pex(resultados, epsilon_exito)

        t1 = time.time()

        filas_resumen.append({
            "mu": mu, "lambda": lambd, "ng": ng, "ejecuciones": ejecuciones,
            "modo_sigma": modo_sigma, "recomb": recomb, "supervivencia": supervivencia,
            "sigma_inicial": sigma_inicial,
            "epsilon_exito": epsilon_exito,
            "TE": te, "VAMM": vamm, "PEX": pex,
            "tiempo_seg": (t1 - t0)
        })

        print("EE", funcion, "-> modo_sigma:", modo_sigma, "recomb:", recomb, "sup:", supervivencia,
              "TE:", te, "VAMM:", vamm, "PEX:", pex)

        c = c + 1

    guardar_resumen_ee(csv_resumen, funcion, filas_resumen)


def barrido_de(funcion, n, lim_inf, lim_sup, theta,
               np, ng, ejecuciones,
               epsilon_exito, semilla_base,
               prefijo_historial, csv_resumen):
    variantes = ["rand", "best"]
    Fs = [0.5, 0.8]
    CRs = [0.2, 0.9]

    filas_resumen = []

    iv = 0
    while iv < len(variantes):
        variante = variantes[iv]

        iF = 0
        while iF < len(Fs):
            F = Fs[iF]

            iCR = 0
            while iCR < len(CRs):
                CR = CRs[iCR]

                nombre_csv_hist = prefijo_historial + "_" + variante + "_F" + \
                    str(F).replace(".", "_") + "_CR" + \
                    str(CR).replace(".", "_") + ".csv"

                t0 = time.time()

                resultados = ejecutar_experimento_de(
                    funcion=funcion,
                    n=n, lim_inf=lim_inf, lim_sup=lim_sup, theta=theta,
                    np=np, ng=ng,
                    F=F, CR=CR, variante=variante,
                    epsilon_exito=epsilon_exito,
                    ejecuciones=ejecuciones,
                    semilla_base=semilla_base + 10000 * iv + 1000 * iF + 100 * iCR,
                    csv_historial=nombre_csv_hist
                )

                te, vamm, pex = calcular_te_vamm_pex(resultados, epsilon_exito)

                t1 = time.time()

                filas_resumen.append({
                    "np": np, "ng": ng, "ejecuciones": ejecuciones,
                    "variante": variante, "F": F, "CR": CR,
                    "epsilon_exito": epsilon_exito,
                    "TE": te, "VAMM": vamm, "PEX": pex,
                    "tiempo_seg": (t1 - t0)
                })

                print("DE", funcion, "-> variante:", variante, "F:", F, "CR:", CR,
                      "TE:", te, "VAMM:", vamm, "PEX:", pex)

                iCR = iCR + 1
            iF = iF + 1
        iv = iv + 1

    guardar_resumen_de(csv_resumen, funcion, filas_resumen)


def main():
    n = 10
    ng = 1200
    ejecuciones = 5

    epsilon_exito = 1e-6

    theta = 5
    lim_inf = -10.0
    lim_sup = 10.0

    mu = 30
    lambd = 200
    sigma_inicial_pot = 0.3
    epsilon_sigma = 1e-12
    np = 30

    print("BARRIDO EE (Potencias)")
    barrido_ee(
        funcion="potencias",
        n=n, lim_inf=lim_inf, lim_sup=lim_sup, theta=theta,
        mu=mu, lambd=lambd, ng=ng, ejecuciones=ejecuciones,
        sigma_inicial=sigma_inicial_pot, epsilon_sigma=epsilon_sigma,
        epsilon_exito=epsilon_exito, semilla_base=1,
        prefijo_historial="historial_ee_potencias",
        csv_resumen="resumen_ee_potencias.csv"
    )

    print("BARRIDO DE (Potencias)")
    barrido_de(
        funcion="potencias",
        n=n, lim_inf=lim_inf, lim_sup=lim_sup, theta=theta,
        np=np, ng=ng, ejecuciones=ejecuciones,
        epsilon_exito=epsilon_exito, semilla_base=100,
        prefijo_historial="historial_de_potencias",
        csv_resumen="resumen_de_potencias.csv"
    )

    lim_inf = -32.77
    lim_sup = 32.77

    sigma_inicial_ack = 1.0

    print("BARRIDO EE (Ackley)")
    barrido_ee(
        funcion="ackley",
        n=n, lim_inf=lim_inf, lim_sup=lim_sup, theta=theta,
        mu=mu, lambd=lambd, ng=ng, ejecuciones=ejecuciones,
        sigma_inicial=sigma_inicial_ack, epsilon_sigma=epsilon_sigma,
        epsilon_exito=epsilon_exito, semilla_base=200,
        prefijo_historial="historial_ee_ackley",
        csv_resumen="resumen_ee_ackley.csv"
    )

    print("BARRIDO DE (Ackley)")
    barrido_de(
        funcion="ackley",
        n=n, lim_inf=lim_inf, lim_sup=lim_sup, theta=theta,
        np=np, ng=ng, ejecuciones=ejecuciones,
        epsilon_exito=epsilon_exito, semilla_base=300,
        prefijo_historial="historial_de_ackley",
        csv_resumen="resumen_de_ackley.csv"
    )


if __name__ == "__main__":
    main()
