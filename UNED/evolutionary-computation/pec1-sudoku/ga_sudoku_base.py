import random

# Sudoku inicial (0 representa celdas vacías)

sudoku = [
    [0, 6, 0, 1, 0, 4, 0, 5, 0],
    [0, 0, 8, 3, 0, 5, 6, 0, 0],
    [2, 0, 0, 0, 0, 0, 0, 0, 1],
    [8, 0, 0, 4, 0, 7, 0, 0, 6],
    [0, 0, 6, 0, 0, 0, 3, 0, 0],
    [7, 0, 0, 9, 0, 1, 0, 0, 4],
    [5, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 0, 7, 2, 0, 6, 9, 0, 0],
    [0, 4, 0, 5, 0, 8, 0, 7, 0],
]

# Parametros del algoritmo genético base

tamaño_poblacion = 100
max_generaciones = 1000
tamaño_torneo = 4
probabilidad_cruce = 0.90
probabilidad_mutacion = 0.03
semilla = 1


# region Funciones auxiliares para Sudokus

# Saco la fila y columna a partir del índice lineal (0-80)
def indice_a_fila_columna(indice):
    fila = indice // 9
    columna = indice % 9
    return fila, columna


# Saco el índice lineal (0-80) a partir de la fila y columna
def fila_columna_a_indice(fila, columna):
    return fila * 9 + columna


# Saco las celdas de un subcuadro 3x3 dado una fila y columna
def celdas_subcuadro(fila, columna):
    inicio_fila = (fila // 3) * 3
    inicio_columna = (columna // 3) * 3

    celdas = []
    f = inicio_fila
    while f < inicio_fila + 3:
        c = inicio_columna
        while c < inicio_columna + 3:
            celdas.append((f, c))
            c += 1
        f += 1

    return celdas


# Defino las celdas fijas del Sudoku
def mascara_celdas_fijas(tablero):
    mascara = []

    fila = 0
    while fila < 9:
        columna = 0
        while columna < 9:
            if tablero[fila][columna] != 0:
                mascara.append(True)
            else:
                mascara.append(False)
            columna += 1
        fila += 1

    return mascara

# endregion


# region Precalculo de valores permitidos

# Precalculo los valores permitidos iniciales para cada celda
def valores_permitidos_iniciales(tablero, fila, columna):

    usados = set()  # Aquí guardo los números ya usados

    c = 0
    while c < 9:
        if tablero[fila][c] != 0:
            usados.add(tablero[fila][c])
        c += 1

    f = 0
    while f < 9:
        if tablero[f][columna] != 0:
            usados.add(tablero[f][columna])
        f += 1

    subcuadro = celdas_subcuadro(fila, columna)
    i = 0
    while i < len(subcuadro):
        f, c = subcuadro[i]
        if tablero[f][c] != 0:
            usados.add(tablero[f][c])
        i += 1

    permitidos = []
    valor = 1
    while valor <= 9:
        if valor not in usados:
            permitidos.append(valor)  # Si no está usado, es permitido
        valor += 1

    if len(permitidos) == 0:  # Si no hay permitidos, que es un caso de error, devuelvo todos
        permitidos = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    return permitidos


# Precalculo los valores permitidos para todas las celdas iterando sobre el tablero
def precalcular_valores_permitidos(tablero):
    listas = []

    i = 0
    while i < 81:
        fila, columna = indice_a_fila_columna(i)

        if tablero[fila][columna] != 0:
            listas.append([tablero[fila][columna]])
        else:
            listas.append(valores_permitidos_iniciales(tablero, fila, columna))

        i += 1

    return listas

# endregion


# region Operaciones del Algoritmo Genético

def crear_individuo(tablero, valores_permitidos):
    # Crear un individuo rellenando las celdas vacías con valores permitidos aleatorios
    individuo = []

    i = 0
    while i < 81:
        fila, columna = indice_a_fila_columna(i)

        if tablero[fila][columna] != 0:
            individuo.append(tablero[fila][columna])
        else:
            individuo.append(random.choice(valores_permitidos[i]))

        i += 1

    return individuo


def crear_poblacion(tablero, tamaño, valores_permitidos):
    # Crear una población inicial de individuos
    poblacion = []

    i = 0
    while i < tamaño:
        poblacion.append(crear_individuo(tablero, valores_permitidos))
        i += 1

    return poblacion


def fitness(individuo):
    suma = 0  # Sumo los conflictos

    i = 0
    while i < 81:
        fila, columna = indice_a_fila_columna(i)  # Saco su posición
        valor = individuo[i]  # Saco su valor

        # Calculo los conflictos en fila, columna y subcuadro y se los añado a la suma total
        conflictos_fila = 0
        c = 0
        while c < 9:
            if c != columna:
                j = fila_columna_a_indice(fila, c)
                if individuo[j] == valor:
                    conflictos_fila += 1
            c += 1

        conflictos_columna = 0
        f = 0
        while f < 9:
            if f != fila:
                j = fila_columna_a_indice(f, columna)
                if individuo[j] == valor:
                    conflictos_columna += 1
            f += 1

        conflictos_subcuadro = 0
        sub = celdas_subcuadro(fila, columna)
        k = 0
        while k < len(sub):
            f, c = sub[k]
            if not (f == fila and c == columna):
                j = fila_columna_a_indice(f, c)
                if individuo[j] == valor:
                    conflictos_subcuadro += 1
            k += 1

        suma = suma + conflictos_fila + conflictos_columna + conflictos_subcuadro
        i += 1

    return suma / 2


def seleccion_torneo(poblacion, tamaño):
    mejor = None
    mejor_fitness = None

    i = 0
    while i < tamaño:
        # Cojo un candidato(gen) aleatorio
        candidato = random.choice(poblacion)
        f = fitness(candidato)  # Calculo su fitness

        if mejor is None or f < mejor_fitness:  # Si es el mejor hasta ahora, lo guardo
            mejor = candidato
            mejor_fitness = f

        i += 1

    return mejor


def cruce_un_punto(padre1, padre2, celdas_fijas):
    hijo1 = []
    hijo2 = []

    # Si no se cruza, devuelvo copias exactas
    if random.random() >= probabilidad_cruce:
        i = 0
        while i < 81:
            hijo1.append(padre1[i])
            hijo2.append(padre2[i])
            i += 1
        return hijo1, hijo2

    punto = random.randint(1, 80)

    # Si se cruza, hago el cruce en el punto seleccionado
    i = 0
    while i < 81:
        if i < punto:
            hijo1.append(padre1[i])
            hijo2.append(padre2[i])
        else:
            hijo1.append(padre2[i])
            hijo2.append(padre1[i])
        i += 1

    # Con este bloque aseguro que las celdas fijas se mantienen iguales
    i = 0
    while i < 81:
        if celdas_fijas[i]:
            fila, columna = indice_a_fila_columna(i)
            valor = sudoku[fila][columna]
            hijo1[i] = valor
            hijo2[i] = valor
        i += 1

    return hijo1, hijo2


def mutacion(individuo, celdas_fijas, valores_permitidos):
    i = 0
    while i < 81:
        if not celdas_fijas[i]:  # Solo muto si no es una celda fija
            if random.random() < probabilidad_mutacion:  # probabilidad de mutación por gen
                opciones = valores_permitidos[i]
                valor_actual = individuo[i]

                # Si solo hay una opción, la pongo directamente
                if len(opciones) == 1:
                    individuo[i] = opciones[0]
                else:
                    # Si hay varias opciones, elijo una distinta a la actual
                    nuevo_valor = valor_actual
                    intentos = 0
                    while nuevo_valor == valor_actual and intentos < 10:
                        nuevo_valor = random.choice(opciones)
                        intentos += 1

                    individuo[i] = nuevo_valor
        i += 1

    return individuo


def mejor_individuo(poblacion):
    mejor = None
    mejor_fitness = None

    i = 0
    while i < len(poblacion):
        # Recorro la población buscando el mejor individuo de la misma
        f = fitness(poblacion[i])
        if mejor is None or f < mejor_fitness:
            mejor = poblacion[i]
            mejor_fitness = f
        i += 1

    return mejor, mejor_fitness

# endregion


# Algoritmo Genético Sudoku (esquema algoritmo evolutivo canónico)
def algoritmo_genetico(sudoku, mostrar_progreso=True):

    if semilla is not None:
        random.seed(semilla)

    historial = []  # Aquí guardo el historial de fitnesss
    gen_exito = None  # Generación en la que se encontró la solución

    celdas_fijas = mascara_celdas_fijas(sudoku)
    valores_permitidos = precalcular_valores_permitidos(sudoku)

    # Inicializo la población
    poblacion = crear_poblacion(sudoku, tamaño_poblacion, valores_permitidos)
    generacion = 0

    # Bucle principal del algoritmo genético (condición de terminación)
    while generacion < max_generaciones:
        # Selecciono el mejor individuo de la generación actual
        mejor_actual, fitness_actual = mejor_individuo(poblacion)
        historial.append(fitness_actual)
        if mostrar_progreso:
            print("Generación:", generacion, "| Mejor fitness:", fitness_actual)

        # Si encuentro la solución, la devuelvo
        if fitness_actual == 0:
            print("Solución encontrada")
            gen_exito = generacion
            return mejor_actual, fitness_actual, historial, gen_exito

        nueva_poblacion = []

        # Creo la nueva población (mediante selección, cruce y mutación)
        while len(nueva_poblacion) < tamaño_poblacion:
            padre1 = seleccion_torneo(poblacion, tamaño_torneo)
            padre2 = seleccion_torneo(poblacion, tamaño_torneo)

            hijo1, hijo2 = cruce_un_punto(padre1, padre2, celdas_fijas)

            hijo1 = mutacion(hijo1, celdas_fijas, valores_permitidos)
            hijo2 = mutacion(hijo2, celdas_fijas, valores_permitidos)

            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tamaño_poblacion:
                nueva_poblacion.append(hijo2)

        # Si la nueva población es peor, meto el mejor de la generación anterior (elitismo)
        _, fitness_nuevo = mejor_individuo(nueva_poblacion)

        if fitness_actual < fitness_nuevo:
            peor_indice = 0
            peor_fitness = fitness(nueva_poblacion[0])

            j = 1
            while j < len(nueva_poblacion):
                f = fitness(nueva_poblacion[j])
                if f > peor_fitness:
                    peor_fitness = f
                    peor_indice = j
                j += 1

            copia_mejor = []
            k = 0
            while k < 81:
                copia_mejor.append(mejor_actual[k])
                k += 1

            nueva_poblacion[peor_indice] = copia_mejor

        # Avanzo a la siguiente generación (Selecciono supervivientes)
        poblacion = nueva_poblacion
        generacion += 1

    mejor_final, fitness_final = mejor_individuo(poblacion)
    return mejor_final, fitness_final, historial, gen_exito


# region Ejecución del algoritmo genético con el Sudoku base

if __name__ == "__main__":
    solucion, fitness_final, _, _ = algoritmo_genetico(sudoku)
    print("Fitness final:", fitness_final)
    print("Solución:")

    i = 0
    while i < 81:
        print(solucion[i:i+9])
        i += 9

# endregion
