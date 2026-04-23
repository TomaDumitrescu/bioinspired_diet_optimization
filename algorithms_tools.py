from constantes import *
from database import *
from auxiliary_functions import *
from random import choice

class AlgorithmTools:
    def __init__(self, type, user_age):
        self.type = type
        self.user_age = user_age

    # Codification
    def get_codification_structure(self):
        return [0] * 77

    def get_food_type(self, idx: int):
        idx %= 11
        if idx == 0:
            return "bebida_desayuno"
        if idx in [1, 2]:
            return "desayuno"
        if idx in [3, 7]:
            return "snacks"
        if idx in [4, 8]:
            return "bebidas"

        return "almuerzo_cena"

    def get_random_solution(self):
        rand_sol = []
        for i in range(77):
            food_type = self.get_food_type(i)
            food_bd = comida_basedatos()
            filtered_food = filtrar_comida(food_bd, food_type, self.user_age)
            rand_sol.append(choice(filtered_food))

        return rand_sol

    def get_day(self, idx: int):
        return DIAS_SEMANA[int(idx / 11)]

    # Fitness function

# Usage
# algorithm_tools = AlgorithmTools("genetic", 23)
# sol = algorithm_tools.get_random_solution()
# print(traducir_solucion(sol, comida_basedatos()))
