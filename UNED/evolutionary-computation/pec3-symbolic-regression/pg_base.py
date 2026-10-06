import random
import math


def problema_1(x):
    return 2.0 * math.exp(-2.0 * ((x - 1.0) ** 2)) - math.exp(-1.0 * ((x - 1.0) ** 2))


def problema_2(x):
    return math.sqrt(x)


def problema_3(x):
    return math.exp(-x) * math.sin(2.0 * x)


def problema_4(x):
    return math.log(math.log(x))


def problema_5(x):
    return 6.0 * math.exp(-2.0 * x) + 2.0 * math.sin(x) - math.cos(x)


def division_protegida(a, b):
    if abs(b) < 1e-12:
        return a
    return a / b


def log_protegido(a):
    if a <= 1e-12:
        return 0.0
    return math.log(a)


def sqrt_protegida(a):
    if a < 0.0:
        return 0.0
    return math.sqrt(a)


def exp_protegida(a):
    if a > 20.0:
        a = 20.0
    if a < -20.0:
        a = -20.0
    return math.exp(a)


def valor_finito(x):
    if math.isnan(x):
        return False
    if math.isinf(x):
        return False
    return True


TERMINALES = ["x", "constante"]
NO_TERMINALES = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"]


def es_operador_unario(op):
    if op == "sin":
        return True
    if op == "cos":
        return True
    if op == "exp":
        return True
    if op == "log":
        return True
    if op == "sqrt":
        return True
    return False


def crear_terminal_variable():
    return {"tipo": "variable", "valor": "x"}


def crear_terminal_constante():
    valor = random.uniform(-10.0, 10.0)
    return {"tipo": "constante", "valor": valor}


def crear_terminal_aleatoria():
    if random.random() < 0.5:
        return crear_terminal_variable()
    return crear_terminal_constante()


def copiar_arbol(arbol):
    if arbol["tipo"] == "variable":
        return {"tipo": "variable", "valor": arbol["valor"]}

    if arbol["tipo"] == "constante":
        return {"tipo": "constante", "valor": arbol["valor"]}

    if es_operador_unario(arbol["valor"]):
        return {
            "tipo": "operador",
            "valor": arbol["valor"],
            "hijo": copiar_arbol(arbol["hijo"])
        }

    return {
        "tipo": "operador",
        "valor": arbol["valor"],
        "izq": copiar_arbol(arbol["izq"]),
        "der": copiar_arbol(arbol["der"])
    }


def profundidad_arbol(arbol):
    if arbol["tipo"] == "variable" or arbol["tipo"] == "constante":
        return 1

    if es_operador_unario(arbol["valor"]):
        return 1 + profundidad_arbol(arbol["hijo"])

    profundidad_izq = profundidad_arbol(arbol["izq"])
    profundidad_der = profundidad_arbol(arbol["der"])

    if profundidad_izq > profundidad_der:
        return 1 + profundidad_izq
    return 1 + profundidad_der


def numero_nodos(arbol):
    if arbol["tipo"] == "variable" or arbol["tipo"] == "constante":
        return 1

    if es_operador_unario(arbol["valor"]):
        return 1 + numero_nodos(arbol["hijo"])

    return 1 + numero_nodos(arbol["izq"]) + numero_nodos(arbol["der"])


def crear_operador_aleatorio():
    return random.choice(NO_TERMINALES)


def generar_arbol_aleatorio(profundidad_actual, profundidad_maxima, modo):
    if profundidad_actual >= profundidad_maxima:
        return crear_terminal_aleatoria()

    if modo == "full":
        op = crear_operador_aleatorio()
        if es_operador_unario(op):
            return {
                "tipo": "operador",
                "valor": op,
                "hijo": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo)
            }
        return {
            "tipo": "operador",
            "valor": op,
            "izq": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo),
            "der": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo)
        }

    if profundidad_actual == 0:
        op = crear_operador_aleatorio()
        if es_operador_unario(op):
            return {
                "tipo": "operador",
                "valor": op,
                "hijo": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo)
            }
        return {
            "tipo": "operador",
            "valor": op,
            "izq": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo),
            "der": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo)
        }

    if random.random() < 0.5:
        return crear_terminal_aleatoria()

    op = crear_operador_aleatorio()
    if es_operador_unario(op):
        return {
            "tipo": "operador",
            "valor": op,
            "hijo": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo)
        }
    return {
        "tipo": "operador",
        "valor": op,
        "izq": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo),
        "der": generar_arbol_aleatorio(profundidad_actual + 1, profundidad_maxima, modo)
    }


def generar_poblacion_inicial(tamano_poblacion, profundidad_inicial_maxima):
    poblacion = []

    profundidades = []
    p = 2
    while p <= profundidad_inicial_maxima:
        profundidades.append(p)
        p = p + 1

    i = 0
    while i < tamano_poblacion:
        profundidad = random.choice(profundidades)

        if i % 2 == 0:
            modo = "full"
        else:
            modo = "grow"

        arbol = generar_arbol_aleatorio(0, profundidad - 1, modo)
        individuo = {
            "arbol": arbol,
            "fitness": None,
            "hits": None,
            "exito": False,
            "tamano": numero_nodos(arbol),
            "profundidad": profundidad_arbol(arbol)
        }
        poblacion.append(individuo)
        i = i + 1

    return poblacion


def evaluar_arbol(arbol, x):
    if arbol["tipo"] == "variable":
        return x

    if arbol["tipo"] == "constante":
        return arbol["valor"]

    op = arbol["valor"]

    if es_operador_unario(op):
        valor = evaluar_arbol(arbol["hijo"], x)

        if op == "sin":
            return math.sin(valor)
        if op == "cos":
            return math.cos(valor)
        if op == "exp":
            return exp_protegida(valor)
        if op == "log":
            return log_protegido(valor)
        if op == "sqrt":
            return sqrt_protegida(valor)

    izq = evaluar_arbol(arbol["izq"], x)
    der = evaluar_arbol(arbol["der"], x)

    if op == "+":
        return izq + der
    if op == "-":
        return izq - der
    if op == "*":
        return izq * der
    if op == "/":
        return division_protegida(izq, der)

    return 0.0


def evaluar_individuo(individuo, muestras_x, muestras_y, U, K0, K1, penalizacion_no_finito, penalizacion_tamano, tamano_objetivo):
    suma_error = 0.0
    hits = 0

    i = 0
    while i < len(muestras_x):
        y_estimada = evaluar_arbol(individuo["arbol"], muestras_x[i])

        if not valor_finito(y_estimada):
            error = penalizacion_no_finito
        else:
            error = abs(muestras_y[i] - y_estimada)

        if error <= U:
            peso = K0
            hits = hits + 1
        else:
            peso = K1

        suma_error = suma_error + (peso * error)
        i = i + 1

    fitness = suma_error / len(muestras_x)

    tamano = numero_nodos(individuo["arbol"])
    profundidad = profundidad_arbol(individuo["arbol"])

    if tamano > tamano_objetivo:
        fitness = fitness + penalizacion_tamano * (tamano - tamano_objetivo)

    individuo["fitness"] = fitness
    individuo["hits"] = hits
    individuo["exito"] = (hits == len(muestras_x))
    individuo["tamano"] = tamano
    individuo["profundidad"] = profundidad

    return individuo


def evaluar_poblacion(poblacion, muestras_x, muestras_y, U, K0, K1, penalizacion_no_finito, penalizacion_tamano, tamano_objetivo):
    i = 0
    while i < len(poblacion):
        evaluar_individuo(poblacion[i], muestras_x, muestras_y, U, K0, K1,
                          penalizacion_no_finito, penalizacion_tamano, tamano_objetivo)
        i = i + 1
    return poblacion


def mejor_individuo(poblacion):
    mejor = None
    i = 0
    while i < len(poblacion):
        if mejor is None:
            mejor = poblacion[i]
        else:
            if poblacion[i]["fitness"] < mejor["fitness"]:
                mejor = poblacion[i]
        i = i + 1
    return mejor


def peor_indice(poblacion):
    peor_i = 0
    peor_f = poblacion[0]["fitness"]

    i = 1
    while i < len(poblacion):
        if poblacion[i]["fitness"] > peor_f:
            peor_f = poblacion[i]["fitness"]
            peor_i = i
        i = i + 1

    return peor_i


def seleccion_torneo(poblacion, tamano_torneo):
    mejor = None
    i = 0
    while i < tamano_torneo:
        candidato = random.choice(poblacion)
        if mejor is None:
            mejor = candidato
        else:
            if candidato["fitness"] < mejor["fitness"]:
                mejor = candidato
        i = i + 1
    return mejor


def obtener_rutas_nodos(arbol, ruta_actual, rutas):
    rutas.append(ruta_actual)

    if arbol["tipo"] == "variable" or arbol["tipo"] == "constante":
        return rutas

    if es_operador_unario(arbol["valor"]):
        nueva_ruta = []
        i = 0
        while i < len(ruta_actual):
            nueva_ruta.append(ruta_actual[i])
            i = i + 1
        nueva_ruta.append("hijo")
        obtener_rutas_nodos(arbol["hijo"], nueva_ruta, rutas)
        return rutas

    ruta_izq = []
    i = 0
    while i < len(ruta_actual):
        ruta_izq.append(ruta_actual[i])
        i = i + 1
    ruta_izq.append("izq")
    obtener_rutas_nodos(arbol["izq"], ruta_izq, rutas)

    ruta_der = []
    i = 0
    while i < len(ruta_actual):
        ruta_der.append(ruta_actual[i])
        i = i + 1
    ruta_der.append("der")
    obtener_rutas_nodos(arbol["der"], ruta_der, rutas)

    return rutas


def listar_rutas_nodos(arbol):
    return obtener_rutas_nodos(arbol, [], [])


def obtener_subarbol(arbol, ruta):
    actual = arbol
    i = 0
    while i < len(ruta):
        actual = actual[ruta[i]]
        i = i + 1
    return actual


def reemplazar_subarbol(arbol, ruta, nuevo_subarbol):
    if len(ruta) == 0:
        return copiar_arbol(nuevo_subarbol)

    copia = copiar_arbol(arbol)
    actual = copia
    i = 0
    while i < len(ruta) - 1:
        actual = actual[ruta[i]]
        i = i + 1
    actual[ruta[-1]] = copiar_arbol(nuevo_subarbol)
    return copia


def podar_a_profundidad(arbol, profundidad_actual, profundidad_maxima):
    if profundidad_actual >= profundidad_maxima:
        return crear_terminal_aleatoria()

    if arbol["tipo"] == "variable" or arbol["tipo"] == "constante":
        return copiar_arbol(arbol)

    if es_operador_unario(arbol["valor"]):
        return {
            "tipo": "operador",
            "valor": arbol["valor"],
            "hijo": podar_a_profundidad(arbol["hijo"], profundidad_actual + 1, profundidad_maxima)
        }

    return {
        "tipo": "operador",
        "valor": arbol["valor"],
        "izq": podar_a_profundidad(arbol["izq"], profundidad_actual + 1, profundidad_maxima),
        "der": podar_a_profundidad(arbol["der"], profundidad_actual + 1, profundidad_maxima)
    }


def cruce_subarbol(padre_1, padre_2, probabilidad_cruce, profundidad_maxima):
    arbol_1 = copiar_arbol(padre_1["arbol"])
    arbol_2 = copiar_arbol(padre_2["arbol"])

    if random.random() >= probabilidad_cruce:
        return {
            "arbol": arbol_1,
            "fitness": None,
            "hits": None,
            "exito": False,
            "tamano": numero_nodos(arbol_1),
            "profundidad": profundidad_arbol(arbol_1)
        }, {
            "arbol": arbol_2,
            "fitness": None,
            "hits": None,
            "exito": False,
            "tamano": numero_nodos(arbol_2),
            "profundidad": profundidad_arbol(arbol_2)
        }

    rutas_1 = listar_rutas_nodos(arbol_1)
    rutas_2 = listar_rutas_nodos(arbol_2)

    ruta_1 = random.choice(rutas_1)
    ruta_2 = random.choice(rutas_2)

    sub_1 = copiar_arbol(obtener_subarbol(arbol_1, ruta_1))
    sub_2 = copiar_arbol(obtener_subarbol(arbol_2, ruta_2))

    hijo_1_arbol = reemplazar_subarbol(arbol_1, ruta_1, sub_2)
    hijo_2_arbol = reemplazar_subarbol(arbol_2, ruta_2, sub_1)

    if profundidad_arbol(hijo_1_arbol) > profundidad_maxima:
        hijo_1_arbol = podar_a_profundidad(hijo_1_arbol, 1, profundidad_maxima)
    if profundidad_arbol(hijo_2_arbol) > profundidad_maxima:
        hijo_2_arbol = podar_a_profundidad(hijo_2_arbol, 1, profundidad_maxima)

    hijo_1 = {
        "arbol": hijo_1_arbol,
        "fitness": None,
        "hits": None,
        "exito": False,
        "tamano": numero_nodos(hijo_1_arbol),
        "profundidad": profundidad_arbol(hijo_1_arbol)
    }

    hijo_2 = {
        "arbol": hijo_2_arbol,
        "fitness": None,
        "hits": None,
        "exito": False,
        "tamano": numero_nodos(hijo_2_arbol),
        "profundidad": profundidad_arbol(hijo_2_arbol)
    }

    return hijo_1, hijo_2


def mutacion_subarbol(individuo, probabilidad_mutacion, profundidad_maxima, profundidad_subarbol_mutacion):
    arbol = copiar_arbol(individuo["arbol"])

    if random.random() >= probabilidad_mutacion:
        return {
            "arbol": arbol,
            "fitness": None,
            "hits": None,
            "exito": False,
            "tamano": numero_nodos(arbol),
            "profundidad": profundidad_arbol(arbol)
        }

    rutas = listar_rutas_nodos(arbol)
    ruta = random.choice(rutas)

    nivel = len(ruta) + 1
    profundidad_disponible = profundidad_maxima - nivel + 1

    if profundidad_disponible <= 1:
        nuevo_subarbol = crear_terminal_aleatoria()
    else:
        profundidad_sub = profundidad_subarbol_mutacion
        if profundidad_sub > profundidad_disponible:
            profundidad_sub = profundidad_disponible

        if random.random() < 0.5:
            modo = "full"
        else:
            modo = "grow"

        nuevo_subarbol = generar_arbol_aleatorio(0, profundidad_sub - 1, modo)

    arbol_mutado = reemplazar_subarbol(arbol, ruta, nuevo_subarbol)

    if profundidad_arbol(arbol_mutado) > profundidad_maxima:
        arbol_mutado = podar_a_profundidad(arbol_mutado, 1, profundidad_maxima)

    return {
        "arbol": arbol_mutado,
        "fitness": None,
        "hits": None,
        "exito": False,
        "tamano": numero_nodos(arbol_mutado),
        "profundidad": profundidad_arbol(arbol_mutado)
    }


def listar_rutas_constantes(arbol, ruta_actual, rutas):
    if arbol["tipo"] == "constante":
        rutas.append(ruta_actual)
        return rutas

    if arbol["tipo"] == "variable":
        return rutas

    if es_operador_unario(arbol["valor"]):
        nueva_ruta = []
        i = 0
        while i < len(ruta_actual):
            nueva_ruta.append(ruta_actual[i])
            i = i + 1
        nueva_ruta.append("hijo")
        listar_rutas_constantes(arbol["hijo"], nueva_ruta, rutas)
        return rutas

    ruta_izq = []
    i = 0
    while i < len(ruta_actual):
        ruta_izq.append(ruta_actual[i])
        i = i + 1
    ruta_izq.append("izq")
    listar_rutas_constantes(arbol["izq"], ruta_izq, rutas)

    ruta_der = []
    i = 0
    while i < len(ruta_actual):
        ruta_der.append(ruta_actual[i])
        i = i + 1
    ruta_der.append("der")
    listar_rutas_constantes(arbol["der"], ruta_der, rutas)

    return rutas


def obtener_constantes(arbol):
    return listar_rutas_constantes(arbol, [], [])


def cambiar_constante_en_ruta(arbol, ruta, nuevo_valor):
    copia = copiar_arbol(arbol)
    actual = copia
    i = 0
    while i < len(ruta):
        actual = actual[ruta[i]]
        i = i + 1
    actual["valor"] = nuevo_valor
    return copia


def busqueda_local_constantes(individuo, muestras_x, muestras_y, U, K0, K1,
                              penalizacion_no_finito, penalizacion_tamano,
                              tamano_objetivo, intensidad_busqueda_local,
                              paso_busqueda_local):
    mejor = {
        "arbol": copiar_arbol(individuo["arbol"]),
        "fitness": individuo["fitness"],
        "hits": individuo["hits"],
        "exito": individuo["exito"],
        "tamano": individuo["tamano"],
        "profundidad": individuo["profundidad"]
    }

    rutas_constantes = obtener_constantes(mejor["arbol"])

    if len(rutas_constantes) == 0:
        return mejor, 0

    evaluaciones_extra = 0
    intentos = 0

    while intentos < intensidad_busqueda_local:
        ruta = random.choice(rutas_constantes)
        nodo_constante = obtener_subarbol(mejor["arbol"], ruta)
        valor_actual = nodo_constante["valor"]
        nuevo_valor = valor_actual + random.gauss(0.0, paso_busqueda_local)

        nuevo_arbol = cambiar_constante_en_ruta(
            mejor["arbol"], ruta, nuevo_valor)
        candidato = {
            "arbol": nuevo_arbol,
            "fitness": None,
            "hits": None,
            "exito": False,
            "tamano": numero_nodos(nuevo_arbol),
            "profundidad": profundidad_arbol(nuevo_arbol)
        }

        evaluar_individuo(candidato, muestras_x, muestras_y, U, K0, K1,
                          penalizacion_no_finito, penalizacion_tamano, tamano_objetivo)
        evaluaciones_extra = evaluaciones_extra + 1

        if candidato["fitness"] < mejor["fitness"]:
            mejor = candidato
            rutas_constantes = obtener_constantes(mejor["arbol"])

        intentos = intentos + 1

    return mejor, evaluaciones_extra


def valor_constante_a_texto(v):
    return f"{v:.4f}"


def arbol_a_texto(arbol):
    if arbol["tipo"] == "variable":
        return "x"

    if arbol["tipo"] == "constante":
        return valor_constante_a_texto(arbol["valor"])

    op = arbol["valor"]

    if es_operador_unario(op):
        return op + "(" + arbol_a_texto(arbol["hijo"]) + ")"

    texto_izq = arbol_a_texto(arbol["izq"])
    texto_der = arbol_a_texto(arbol["der"])
    return "(" + texto_izq + " " + op + " " + texto_der + ")"


def generar_muestras(funcion_real, a, b, M):
    xs = []
    ys = []

    delta = (b - a) / (M - 1)
    i = 0
    while i < M:
        x = a + i * delta
        y = funcion_real(x)
        xs.append(x)
        ys.append(y)
        i = i + 1

    return xs, ys


def ejecutar_pg(muestras_x, muestras_y, configuracion, semilla, mostrar_progreso):
    random.seed(semilla)

    tamano_poblacion = configuracion["tamano_poblacion"]
    max_generaciones = configuracion["max_generaciones"]
    probabilidad_cruce = configuracion["probabilidad_cruce"]
    probabilidad_mutacion_inicial = configuracion["probabilidad_mutacion"]
    profundidad_inicial_maxima = configuracion["profundidad_inicial_maxima"]
    profundidad_maxima = configuracion["profundidad_maxima"]
    tamano_torneo = configuracion["tamano_torneo"]
    elitismo = configuracion["elitismo"]
    U = configuracion["U"]
    K0 = configuracion["K0"]
    K1 = configuracion["K1"]
    penalizacion_no_finito = configuracion["penalizacion_no_finito"]
    penalizacion_tamano = configuracion["penalizacion_tamano"]
    tamano_objetivo = configuracion["tamano_objetivo"]
    profundidad_subarbol_mutacion = configuracion["profundidad_subarbol_mutacion"]
    usar_busqueda_local = configuracion["usar_busqueda_local"]
    intensidad_busqueda_local = configuracion["intensidad_busqueda_local"]
    paso_busqueda_local = configuracion["paso_busqueda_local"]
    paciencia_mutacion = configuracion["paciencia_mutacion"]
    factor_subida_mutacion = configuracion["factor_subida_mutacion"]
    factor_bajada_mutacion = configuracion["factor_bajada_mutacion"]
    pm_min = configuracion["pm_min"]
    pm_max = configuracion["pm_max"]

    probabilidad_mutacion_actual = probabilidad_mutacion_inicial
    sin_mejora = 0

    poblacion = generar_poblacion_inicial(
        tamano_poblacion, profundidad_inicial_maxima)
    evaluar_poblacion(poblacion, muestras_x, muestras_y, U, K0, K1,
                      penalizacion_no_finito, penalizacion_tamano, tamano_objetivo)

    evaluaciones = len(poblacion)
    historial = []
    historial_pm = []

    mejor_global = mejor_individuo(poblacion)
    mejor_global = {
        "arbol": copiar_arbol(mejor_global["arbol"]),
        "fitness": mejor_global["fitness"],
        "hits": mejor_global["hits"],
        "exito": mejor_global["exito"],
        "tamano": mejor_global["tamano"],
        "profundidad": mejor_global["profundidad"]
    }

    generacion_exito = None
    evaluaciones_hasta_exito = None

    g = 0
    while g < max_generaciones:
        mejor_actual = mejor_individuo(poblacion)
        historial.append(mejor_actual["fitness"])
        historial_pm.append(probabilidad_mutacion_actual)

        if mostrar_progreso:
            print("Generación:", g,
                  "| fitness:", round(mejor_actual["fitness"], 6),
                  "| hits:", mejor_actual["hits"],
                  "| pm:", round(probabilidad_mutacion_actual, 4))

        if mejor_actual["fitness"] < mejor_global["fitness"]:
            mejor_global = {
                "arbol": copiar_arbol(mejor_actual["arbol"]),
                "fitness": mejor_actual["fitness"],
                "hits": mejor_actual["hits"],
                "exito": mejor_actual["exito"],
                "tamano": mejor_actual["tamano"],
                "profundidad": mejor_actual["profundidad"]
            }
            sin_mejora = 0
            probabilidad_mutacion_actual = probabilidad_mutacion_actual * factor_bajada_mutacion
            if probabilidad_mutacion_actual < pm_min:
                probabilidad_mutacion_actual = pm_min
        else:
            sin_mejora = sin_mejora + 1

        if mejor_actual["exito"] and generacion_exito is None:
            generacion_exito = g
            evaluaciones_hasta_exito = evaluaciones

        if mejor_actual["exito"]:
            break

        if sin_mejora >= paciencia_mutacion:
            probabilidad_mutacion_actual = probabilidad_mutacion_actual * factor_subida_mutacion
            if probabilidad_mutacion_actual > pm_max:
                probabilidad_mutacion_actual = pm_max
            sin_mejora = 0

        nueva_poblacion = []

        while len(nueva_poblacion) < tamano_poblacion:
            padre_1 = seleccion_torneo(poblacion, tamano_torneo)
            padre_2 = seleccion_torneo(poblacion, tamano_torneo)

            hijo_1, hijo_2 = cruce_subarbol(padre_1, padre_2,
                                            probabilidad_cruce, profundidad_maxima)
            hijo_1 = mutacion_subarbol(hijo_1, probabilidad_mutacion_actual,
                                       profundidad_maxima, profundidad_subarbol_mutacion)
            hijo_2 = mutacion_subarbol(hijo_2, probabilidad_mutacion_actual,
                                       profundidad_maxima, profundidad_subarbol_mutacion)

            evaluar_individuo(hijo_1, muestras_x, muestras_y, U, K0, K1,
                              penalizacion_no_finito, penalizacion_tamano, tamano_objetivo)
            evaluaciones = evaluaciones + 1
            nueva_poblacion.append(hijo_1)

            if len(nueva_poblacion) < tamano_poblacion:
                evaluar_individuo(hijo_2, muestras_x, muestras_y, U, K0, K1,
                                  penalizacion_no_finito, penalizacion_tamano, tamano_objetivo)
                evaluaciones = evaluaciones + 1
                nueva_poblacion.append(hijo_2)

        if usar_busqueda_local:
            mejor_nuevo = mejor_individuo(nueva_poblacion)
            mejor_local, evaluaciones_extra = busqueda_local_constantes(
                mejor_nuevo, muestras_x, muestras_y, U, K0, K1,
                penalizacion_no_finito, penalizacion_tamano,
                tamano_objetivo, intensidad_busqueda_local,
                paso_busqueda_local)
            evaluaciones = evaluaciones + evaluaciones_extra

            if mejor_local["fitness"] < mejor_nuevo["fitness"]:
                indice_peor = peor_indice(nueva_poblacion)
                nueva_poblacion[indice_peor] = mejor_local

        if elitismo:
            mejor_nuevo = mejor_individuo(nueva_poblacion)
            if mejor_global["fitness"] < mejor_nuevo["fitness"]:
                indice_peor = peor_indice(nueva_poblacion)
                nueva_poblacion[indice_peor] = {
                    "arbol": copiar_arbol(mejor_global["arbol"]),
                    "fitness": mejor_global["fitness"],
                    "hits": mejor_global["hits"],
                    "exito": mejor_global["exito"],
                    "tamano": mejor_global["tamano"],
                    "profundidad": mejor_global["profundidad"]
                }

        poblacion = nueva_poblacion
        g = g + 1

    mejor_final = mejor_individuo(poblacion)
    if mejor_final["fitness"] < mejor_global["fitness"]:
        mejor_global = {
            "arbol": copiar_arbol(mejor_final["arbol"]),
            "fitness": mejor_final["fitness"],
            "hits": mejor_final["hits"],
            "exito": mejor_final["exito"],
            "tamano": mejor_final["tamano"],
            "profundidad": mejor_final["profundidad"]
        }

    return {
        "mejor_individuo": mejor_global,
        "historial": historial,
        "historial_pm": historial_pm,
        "generacion_exito": generacion_exito,
        "evaluaciones_hasta_exito": evaluaciones_hasta_exito,
        "evaluaciones_totales": evaluaciones,
        "expresion": arbol_a_texto(mejor_global["arbol"])
    }
