from constantes import SEEDS
from database import comida_basedatos, sujetos_basedatos
from auxiliary_functions import filtrar_comida
from algorithms_tools import AlgorithmTools

from random import Random
from inspyred import ec, benchmarks

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

    #do 2 point crossover (having everything as days)
    def day_two_point_crossover(self, random, candidates, args):
        offspring = []

        for i in range(0, len(candidates) - 1, 2):
            parent1 = candidates[i]
            parent2 = candidates[i + 1]

            child1 = parent1[:]
            child2 = parent2[:]

            if random.random() <= self.crossover_rate:

                #crossover points: pick 2 different days to define the swapped segment
                day1 = random.randint(0, 5)
                day2 = random.randint(day1 + 1, 6)

                #convert day indices to gene positions (each day = 11 genes)
                gene_start = day1 * 11
                gene_end = (day2 + 1) * 11

                #swap the days between the two cut points
                child1[gene_start:gene_end] = parent2[gene_start:gene_end]
                child2[gene_start:gene_end] = parent1[gene_start:gene_end]

            offspring.append(child1)
            offspring.append(child2)

        return offspring

    def swap_mutation(self, random, candidates, args):
        mutated = []

        return mutated




    def run(self, seed):
        prng = Random()
        prng.seed(seed)

        ga = ec.GA(prng)
        ga.selector = ec.selectors.tournament_selection
        ga.variator = [self.day_two_point_crossover, self.swap_mutation]


        
