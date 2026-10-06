# Textos

print("Hola mundo ")

print('*' * 10)


# Variables

booleano = True

numero = 23

n = 34.675


muchoTexto = """
kjsdfgsdfjbgjksdfbg

muy buen tarde,

sdlkfsljhgldjkfhg


"""

print(muchoTexto)

texto = "elefante"


# Seleccion de caracteres y subcadenas

print(len(texto))

print(texto[0])
print(texto[-1])
print(texto[0:3])
print(texto[3:])
print(texto[:3])

print(texto)


# Caracteres de escape

texto = "elefante \n djfhsdjkh"

print(texto)


# Formateo de cadenas

first = "Jaime"
last = "gomez"
full = f"{first} {last}"
print(full)
print(full.upper())
print(full.lower())
print(full.title())
print(full.find("a"))
print(full.replace("a", "5"))


# Tipos de datos y operadores

x = 10
y = 3.2
z = 4 + 7j
print(type(x), type(y), type(z))
print(isinstance(x, int))
print(isinstance(y, float))
print(isinstance(z, complex))

print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x // y)  # División suelo; con operandos float devuelve float
print(x ** y)  # x^y
print(x % y)  # Módulo
print(abs(-x))  # valores absolutos
print(round(y))  # redondeo
print(pow(x, y))  # Potencia
print(divmod(x, y))  # Cociente y resto


# Control de flujo

if x > y:
    print("x > y")
elif x == y:
    print("x = y")
else:
    print("x < y")


age = 88

if age >= 18 and age < 65:
    print("eres adulto")
elif age < 18:
    print("eres un niño")
else:
    print("carcamal")


# Bucles

for number in range(4):
    print(number)

for number in range(1, 6):
    print(number)

for number in range(1, 11, 3):
    print(number)

contador = 0
for x in range(5):
    for y in range(5):
        print(f"({x}, {y})")
        contador += 1

print(contador)


number = 100
while number > 0:
    print(number)
    number //= 2

command = ""
while command.lower() != "salir":
    command = input("Escribe 'salir' para salir: ")
print("Has salido")


command = ""
while True:
    command = input("Escribe 'caracol' para salir: ")
    if command.lower() == "caracol":
        break
print("Has salido")


# Funciones

def greet():
    print("  Hola como estas")

    x = 12 ** 2

    print(f"\n y tengo {x} años")


greet()


def greet_with_name(name):
    print(f"Hola {name}")


greet_with_name("jaime")


def greet_first_and_last_name(first_name, last_name=None):
    if last_name:
        print(f"Hola, me llamo {first_name} {last_name}")
    else:
        print(f"Hola, me llamo {first_name}")

    print("y tu como te llamas?")


greet_first_and_last_name("Jaime", "Ciriaco")
greet_first_and_last_name("Michael")


def add(n1, n2):
    return n1 + n2


result = add(3, 5)
print(result)


def multiply(*nums):  # el * permite N argumentos
    total = 1
    print("multiplying")
    for num in nums:
        total *= num
        print(num, " ")
    return total


result = multiply(23, 43, 6, 7, 54)

print("= ", result)
