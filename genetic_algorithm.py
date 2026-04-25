import random
import inspyred

from constantes import SEEDS
from database import comida_basedatos, sujetos_basedatos
from auxiliary_functions import filtrar_comida
from algorithms_tools import AlgorithmTools

#generate_population
#evaluation (fitness) --> stopping criteria
#selection
#crossover
#mutation
#build new population (replacement)




class GeneticAlgorithm:
    def __init__(
        self,
        user_profile,
        food_db,
        pop_size=100,
        max_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.05,
        tournament_size=3,
        num_elites=1,
    ):
        self.user_profile = user_profile
        self.food_db = food_db
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.num_elites = num_elites


        self.tools = AlgorithmTools("genetic", user_profile["edad"])

        #for mutation later
        self.food_pools = {}
        for food_type in ("bebida_desayuno", "desayuno", "snacks", "bebidas", "almuerzo_cena"):
            self.food_pools[food_type] = filtrar_comida(food_db, food_type, user_profile["edad"])

    def generator (self, random, args):
        return self.tools.get_random_solution(random)

    def evaluator(self, candidates, args):
        fitness_values = []
        for candidate in candidates:
            score = self.tools.calculate_fitness(
                candidate,
                self.food_db,
                self.user_profile["calorias"],
                self.user_profile.get("alergias", []),
                self.user_profile.get("gustos", []),
                self.user_profile.get("disgustos", []),
            )
            fitness_values.append(score)
        return fitness_values


    def run(self, seed):
        prng = random.Random()
        prng.seed(seed)


        
