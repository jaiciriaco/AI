import ga_sudoku_base  # Traemos la lógica del algoritmo genético base
import csv  # Para guardar historial en ficheros CSV

sudoku_facil = [
    [0, 0, 0, 0, 3, 0, 0, 0, 4],
    [0, 9, 0, 4, 0, 6, 0, 7, 0],
    [0, 5, 0, 0, 0, 0, 3, 8, 0],
    [0, 0, 0, 0, 7, 8, 0, 0, 3],
    [3, 0, 0, 0, 0, 0, 6, 9, 0],
    [5, 4, 0, 6, 0, 0, 0, 2, 0],
    [7, 0, 5, 0, 2, 4, 0, 0, 0],
    [9, 8, 4, 0, 6, 5, 2, 0, 0],
    [0, 2, 6, 0, 8, 0, 0, 0, 9],
]


sudoku_dificil = [
    [0, 3, 8, 0, 0, 0, 0, 6, 0],
    [5, 4, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 1, 0, 9, 0, 3, 0, 0],
    [8, 6, 0, 0, 4, 0, 0, 7, 0],
    [4, 0, 0, 0, 8, 3, 5, 0, 0],
    [0, 0, 7, 0, 0, 0, 0, 4, 0],
    [3, 8, 6, 0, 0, 9, 0, 0, 4],
    [0, 2, 0, 4, 6, 0, 0, 3, 0],
    [0, 0, 0, 0, 0, 7, 0, 8, 0],
]

num_historial_sudoku = 0

# Selección del sudoku y ajuste de parámetros


def menu_selección_sudoku():
    print("Seleccione el sudoku a resolver:")
    print("1 - Sudoku fácil")
    print("2 - Sudoku difícil")
    print("3 - Evaluación por mutación (capítulo 9)")

    opcion = input("Introduzca 1 o 2 o 3: ")

    if opcion == "1":
        sudoku_elegido = sudoku_facil
        ga_sudoku_base.tamaño_poblacion = 10
        # Ponemos maximo de generaciones mayor para sudoku fácil porque aunque llegaría a la solución, tarda demasiado
        ga_sudoku_base.max_generaciones = 3000
        print("\nResolviendo Sudoku 'FÁCIL'...\n")

        solucion, fitness_final, _, _ = ga_sudoku_base.algoritmo_genetico(
            sudoku_elegido, mostrar_progreso=True)

        print("\nFitness final:", fitness_final)
        print("Solución encontrada:\n")

        i = 0
        while i < 81:
            print(solucion[i:i+9])
            i += 9

    elif opcion == "2":
        sudoku_elegido = sudoku_dificil
        ga_sudoku_base.tamaño_poblacion = 100
        ga_sudoku_base.max_generaciones = 1500
        print("\nResolviendo Sudoku 'DIFÍCIL'...\n")

        solucion, fitness_final, _, _ = ga_sudoku_base.algoritmo_genetico(
            sudoku_elegido, mostrar_progreso=True)

        print("\nFitness final:", fitness_final)
        print("Solución encontrada:\n")

        i = 0
        while i < 81:
            print(solucion[i:i+9])
            i += 9

    elif opcion == "3":
        print("\nEvaluación por mutación seleccionada.\n")
        print("Seleccione el sudoku a resolver:")
        print("1 - Sudoku fácil")
        print("2 - Sudoku difícil")

        opcion = input("Introduzca 1 o 2: ")

        if opcion == "1":
            sudoku_elegido = sudoku_facil
            ga_sudoku_base.tamaño_poblacion = 10
            ga_sudoku_base.max_generaciones = 2000
            ga_sudoku_base.semilla = None  # Para resultados diferentes en cada ejecución
            print("\nResolviendo Sudoku 'FÁCIL'...\n")

            evaluacion_por_mutacion(sudoku_elegido)

        elif opcion == "2":
            sudoku_elegido = sudoku_dificil
            ga_sudoku_base.tamaño_poblacion = 100
            ga_sudoku_base.max_generaciones = 1500
            ga_sudoku_base.semilla = None
            print("\nResolviendo Sudoku 'DIFÍCIL'...\n")

            evaluacion_por_mutacion(sudoku_elegido)

    else:
        print("Opción no válida")
        exit()


# Parámetros de evaluación
lista_mutaciones = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
numero_ejecuciones = 20


def evaluacion_por_mutacion(sudoku_elegido):
    global num_historial_sudoku

    archivo = open(
        f"historial_sudoku{num_historial_sudoku}.csv", mode="w", newline="")
    num_historial_sudoku += 1
    escritor = csv.writer(archivo)
    escritor.writerow(["mutacion", "ejecucion", "generacion", "fitness"])

    m = 0
    while m < len(lista_mutaciones):  # para cada probabilidad de mutación
        valor_mutacion = lista_mutaciones[m]
        # ajustamos el valor de mutación
        ga_sudoku_base.probabilidad_mutacion = valor_mutacion

        print("Probabilidad mutación =", valor_mutacion)

        exitos = 0
        suma_fitness_final = 0.0
        suma_generaciones_exito = 0.0
        cuenta_generaciones_exito = 0

        # Para curva de progreso media:
        suma_curva = []   # aquí iremos sumando fitness por generación
        contador_curvas = 0

        e = 0
        while e < numero_ejecuciones:  # para cada ejecución

            _, fitness_final, historial, gen_exito = ga_sudoku_base.algoritmo_genetico(
                # ejecutamos el AG sin mostrar progreso de cada generación
                sudoku_elegido, mostrar_progreso=False)

            # Guardar historial en CSV
            g = 0
            while g < len(historial):
                if g == 0:
                    escritor.writerow([valor_mutacion, e + 1, g, historial[g]])
                else:
                    if historial[g] != historial[g - 1]:
                        escritor.writerow(
                            [valor_mutacion, e + 1, g, historial[g]])
                g += 1
            archivo.flush()

            suma_fitness_final = suma_fitness_final + fitness_final

            if fitness_final == 0:
                exitos = exitos + 1

            if gen_exito is not None:
                suma_generaciones_exito = suma_generaciones_exito + gen_exito
                cuenta_generaciones_exito = cuenta_generaciones_exito + 1

            # sumar curva al acumulador
            if contador_curvas == 0:
                # primera vez: copiamos tamaño
                i = 0
                while i < len(historial):
                    suma_curva.append(historial[i])
                    i += 1
            else:
                # siguientes: sumamos hasta donde exista
                i = 0
                while i < len(historial) and i < len(suma_curva):
                    suma_curva[i] = suma_curva[i] + historial[i]
                    i += 1

            contador_curvas = contador_curvas + 1

            print("Ejecución", e + 1, "-> fitness_final =",
                  fitness_final, "| gen_exito =", gen_exito)
            e += 1

        # RESULTADOS (capítulo 9)
        TE = exitos / numero_ejecuciones
        VAMM = suma_fitness_final / numero_ejecuciones

        if cuenta_generaciones_exito > 0:
            gen_media_exito = suma_generaciones_exito / cuenta_generaciones_exito
        else:
            gen_media_exito = None

        print("\nResumen para mutación =", valor_mutacion)
        print("TE (tasa éxito) =", TE)
        print("Media fitness final (tipo VAMM) =", VAMM)

        if gen_media_exito is None:
            print("Generaciones medias hasta éxito = (ningún éxito)")
        else:
            print("Generaciones medias hasta éxito =", gen_media_exito)

        print("")
        m += 1

    archivo.close()


if __name__ == "__main__":

    menu_selección_sudoku()
