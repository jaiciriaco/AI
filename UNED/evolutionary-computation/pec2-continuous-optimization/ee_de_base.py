import random
import math


def funcion_potencias_desplazada(x, theta):
    n = len(x)
    i = 0
    suma = 0.0
    while i < n:
        potencia = i + 2
        suma = suma + (abs(x[i] - theta) ** potencia)
        i = i + 1
    return suma


def funcion_ackley(x):
    n = len(x)

    i = 0
    suma_cuadrados = 0.0
    suma_cos = 0.0
    while i < n:
        suma_cuadrados = suma_cuadrados + (x[i] * x[i])
        suma_cos = suma_cos + math.cos(2.0 * math.pi * x[i])
        i = i + 1

    termino1 = -20.0 * math.exp(-0.2 * math.sqrt(suma_cuadrados / n))
    termino2 = -1.0 * math.exp(suma_cos / n)
    return termino1 + termino2 + 20.0 + math.e


def recortar_a_dominio(x, limite_inferior, limite_superior):
    i = 0
    while i < len(x):
        if x[i] < limite_inferior:
            x[i] = limite_inferior
        if x[i] > limite_superior:
            x[i] = limite_superior
        i = i + 1
    return x


def copiar_lista(x):
    copia = []
    i = 0
    while i < len(x):
        copia.append(x[i])
        i = i + 1
    return copia


def argmin_fitness(poblacion):
    i = 0
    mejor_i = 0
    mejor_f = poblacion[0]["fitness"]
    while i < len(poblacion):
        if poblacion[i]["fitness"] < mejor_f:
            mejor_f = poblacion[i]["fitness"]
            mejor_i = i
        i = i + 1
    return mejor_i


def ordenar_por_fitness(poblacion):
    poblacion.sort(key=lambda ind: ind["fitness"])
    return poblacion


def inicializar_poblacion_ee(mu, n, limite_inferior, limite_superior, sigma_inicial, modo_sigma):
    poblacion = []
    i = 0
    while i < mu:
        x = []
        j = 0
        while j < n:
            x.append(random.uniform(limite_inferior, limite_superior))
            j = j + 1

        if modo_sigma == "1":
            sigma = sigma_inicial
        else:
            sigma = []
            j = 0
            while j < n:
                sigma.append(sigma_inicial)
                j = j + 1

        ind = {"x": x, "sigma": sigma, "fitness": None}
        poblacion.append(ind)
        i = i + 1
    return poblacion


def evaluar_poblacion(poblacion, funcion_objetivo):
    i = 0
    while i < len(poblacion):
        poblacion[i]["fitness"] = funcion_objetivo(poblacion[i]["x"])
        i = i + 1
    return poblacion


def recombinacion_discreta_global(padres, modo_sigma):
    n = len(padres[0]["x"])
    x_hijo = []
    i = 0
    while i < n:
        p = random.randint(0, len(padres) - 1)
        x_hijo.append(padres[p]["x"][i])
        i = i + 1

    if modo_sigma == "1":
        p = random.randint(0, len(padres) - 1)
        sigma_hijo = padres[p]["sigma"]
    else:
        sigma_hijo = []
        i = 0
        while i < n:
            p = random.randint(0, len(padres) - 1)
            sigma_hijo.append(padres[p]["sigma"][i])
            i = i + 1

    return {"x": x_hijo, "sigma": sigma_hijo, "fitness": None}


def recombinacion_intermedia_promediada_global(padres, modo_sigma):
    n = len(padres[0]["x"])
    x_hijo = []
    i = 0
    while i < n:
        suma = 0.0
        j = 0
        while j < len(padres):
            suma = suma + padres[j]["x"][i]
            j = j + 1
        x_hijo.append(suma / len(padres))
        i = i + 1

    if modo_sigma == "1":
        suma = 0.0
        j = 0
        while j < len(padres):
            suma = suma + padres[j]["sigma"]
            j = j + 1
        sigma_hijo = suma / len(padres)
    else:
        sigma_hijo = []
        i = 0
        while i < n:
            suma = 0.0
            j = 0
            while j < len(padres):
                suma = suma + padres[j]["sigma"][i]
                j = j + 1
            sigma_hijo.append(suma / len(padres))
            i = i + 1

    return {"x": x_hijo, "sigma": sigma_hijo, "fitness": None}


def mutacion_no_correlacionada(ind, tau, tau_prima, epsilon0, modo_sigma):
    n = len(ind["x"])

    if modo_sigma == "1":
        r = random.gauss(0.0, 1.0)
        sigma_nuevo = ind["sigma"] * math.exp(tau * r)
        if sigma_nuevo < epsilon0:
            sigma_nuevo = epsilon0
        ind["sigma"] = sigma_nuevo

        i = 0
        while i < n:
            ind["x"][i] = ind["x"][i] + ind["sigma"] * random.gauss(0.0, 1.0)
            i = i + 1

    else:
        r_global = random.gauss(0.0, 1.0)
        i = 0
        while i < n:
            r_i = random.gauss(0.0, 1.0)
            sigma_nuevo = ind["sigma"][i] * \
                math.exp(tau_prima * r_global + tau * r_i)
            if sigma_nuevo < epsilon0:
                sigma_nuevo = epsilon0
            ind["sigma"][i] = sigma_nuevo
            i = i + 1

        i = 0
        while i < n:
            ind["x"][i] = ind["x"][i] + \
                ind["sigma"][i] * random.gauss(0.0, 1.0)
            i = i + 1

    return ind


def ee_ejecutar(funcion_objetivo, n, limite_inferior, limite_superior,
                mu, lambd, max_generaciones,
                modo_sigma, tipo_recombinacion, tipo_supervivencia,
                sigma_inicial, epsilon_sigma, semilla):
    random.seed(semilla)

    tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
    tau_prima = 1.0 / math.sqrt(2.0 * n)

    poblacion = inicializar_poblacion_ee(
        mu, n, limite_inferior, limite_superior, sigma_inicial, modo_sigma)
    poblacion = evaluar_poblacion(poblacion, funcion_objetivo)
    evaluaciones = mu

    historial = []

    g = 0
    while g < max_generaciones:
        descendientes = []
        i = 0
        while i < lambd:
            padres = poblacion

            if tipo_recombinacion == "discreta_global":
                hijo = recombinacion_discreta_global(padres, modo_sigma)
            else:
                hijo = recombinacion_intermedia_promediada_global(
                    padres, modo_sigma)

            hijo = mutacion_no_correlacionada(
                hijo, tau, tau_prima, epsilon_sigma, modo_sigma)
            hijo["x"] = recortar_a_dominio(
                hijo["x"], limite_inferior, limite_superior)

            hijo["fitness"] = funcion_objetivo(hijo["x"])
            evaluaciones = evaluaciones + 1

            descendientes.append(hijo)
            i = i + 1

        if tipo_supervivencia == "(mu,lambda)":
            candidatos = descendientes
        else:
            candidatos = []
            i = 0
            while i < len(poblacion):
                candidatos.append(poblacion[i])
                i = i + 1
            i = 0
            while i < len(descendientes):
                candidatos.append(descendientes[i])
                i = i + 1

        candidatos = ordenar_por_fitness(candidatos)

        nueva_poblacion = []
        i = 0
        while i < mu:
            nueva_poblacion.append(candidatos[i])
            i = i + 1

        poblacion = nueva_poblacion

        mejor_i = argmin_fitness(poblacion)
        mejor_f = poblacion[mejor_i]["fitness"]
        historial.append(
            {"generacion": g, "mejor_fitness": mejor_f, "evaluaciones": evaluaciones})

        g = g + 1

    mejor_i = argmin_fitness(poblacion)
    return poblacion[mejor_i], historial, evaluaciones


def inicializar_poblacion_de(np, n, limite_inferior, limite_superior):
    poblacion = []
    i = 0
    while i < np:
        x = []
        j = 0
        while j < n:
            x.append(random.uniform(limite_inferior, limite_superior))
            j = j + 1
        poblacion.append({"x": x, "fitness": None})
        i = i + 1
    return poblacion


def de_mutacion(poblacion, i_objetivo, F, variante):
    np = len(poblacion)
    n = len(poblacion[0]["x"])

    indices = []
    j = 0
    while j < np:
        if j != i_objetivo:
            indices.append(j)
        j = j + 1

    random.shuffle(indices)

    if variante == "rand":
        r1 = indices[0]
        r2 = indices[1]
        r3 = indices[2]
        v = []
        k = 0
        while k < n:
            v.append(poblacion[r1]["x"][k] + F *
                     (poblacion[r2]["x"][k] - poblacion[r3]["x"][k]))
            k = k + 1
    else:
        mejor_i = argmin_fitness(poblacion)
        r1 = indices[0]
        r2 = indices[1]
        v = []
        k = 0
        while k < n:
            v.append(poblacion[mejor_i]["x"][k] + F *
                     (poblacion[r1]["x"][k] - poblacion[r2]["x"][k]))
            k = k + 1

    return v


def de_cruce_binomial(x_objetivo, v_mutante, CR):
    n = len(x_objetivo)
    u = []
    j_rand = random.randint(0, n - 1)

    j = 0
    while j < n:
        r = random.random()
        if (r < CR) or (j == j_rand):
            u.append(v_mutante[j])
        else:
            u.append(x_objetivo[j])
        j = j + 1

    return u


def de_ejecutar(funcion_objetivo, n, limite_inferior, limite_superior,
                np, max_generaciones,
                F, CR, variante, epsilon_exito, semilla):
    random.seed(semilla)

    poblacion = inicializar_poblacion_de(
        np, n, limite_inferior, limite_superior)
    i = 0
    while i < np:
        poblacion[i]["fitness"] = funcion_objetivo(poblacion[i]["x"])
        i = i + 1
    evaluaciones = np

    historial = []
    g = 0
    while g < max_generaciones:
        i = 0
        while i < np:
            v = de_mutacion(poblacion, i, F, variante)
            u = de_cruce_binomial(poblacion[i]["x"], v, CR)
            u = recortar_a_dominio(u, limite_inferior, limite_superior)

            fit_u = funcion_objetivo(u)
            evaluaciones = evaluaciones + 1

            if fit_u <= poblacion[i]["fitness"]:
                poblacion[i]["x"] = u
                poblacion[i]["fitness"] = fit_u

            i = i + 1

        mejor_i = argmin_fitness(poblacion)
        mejor_f = poblacion[mejor_i]["fitness"]
        historial.append(
            {"generacion": g, "mejor_fitness": mejor_f, "evaluaciones": evaluaciones})

        if mejor_f <= epsilon_exito:
            pass

        g = g + 1

    mejor_i = argmin_fitness(poblacion)
    return poblacion[mejor_i], historial, evaluaciones
