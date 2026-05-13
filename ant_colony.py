# ant_colony.py

import random

from algorithms_tools import AlgorithmTools
from constantes import NUM_ALIMENTOS_DIARIO, NUM_DIAS
from auxiliary_functions import calculo_macronutrientes, filtrar_comida, comida_basedatos
from copy import deepcopy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from typing import List, TextIO
from itertools import accumulate
import time
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class Ant:
    def __init__(self, user_profile, food_db, rng, ant_id, grouped_food, pheromone_parallel=[]):
        """
        Parameters:
        - user_profile: dict with keys: 'calorias', 'edad', 'gustos', 'disgustos', 'alergias'
        - food_db: list of food dicts from comida_basedatos()
        - rng: random.Random object for reproducible randomness
        """
        self.ant_id = ant_id
        self.user_profile = user_profile
        self.food_db = food_db
        self.rng = rng
        
        # Ants' journey
        self.path = []           # List of food IDs (indices into food_db)
        self.current_position = 0  # 0 to 76 (which meal slot we're filling)
        self.complete = False
        self.fitness = None

        self.pheromone_parallel = pheromone_parallel
        
        # Tools for validation and fitness
        self.tools = AlgorithmTools("ant", user_profile["edad"])

        # Filtered food
        self.grouped_food = {}
        allergies = self.user_profile.get("alergias", [])
        
        for food_type in ["bebida_desayuno", "desayuno", "snacks", "bebidas", "almuerzo_cena"]:
            allowed = filtrar_comida(self.food_db, food_type, self.user_profile["edad"])
            
            safe_allowed = [
                idx for idx in allowed 
                if not any(self.food_db[idx]["grupo"].startswith(a) for a in allergies)
            ]
            self.grouped_food[food_type] = safe_allowed

        self.reset()

    def reset(self):
        """Reset the ant for a new journey"""
        self.path = []
        self.current_position = 0
        self.complete = False
        self.fitness = None
        
        # Tracking values for current day
        self.current_day_cals = 0
        self.current_day_prot = 0
        self.current_day_carbs = 0
        self.current_day_fats = 0

    def get_current_food_type(self):
        """
        Returns what type of food we need at current position.
        Uses AlgorithmTools.get_food_type() which already handles the 11-per-day pattern.
        """
        return self.tools.get_food_type(self.current_position)

    def get_current_day(self):
        """Returns which day we're on (0-6: Monday to Sunday)"""
        return self.current_position // NUM_ALIMENTOS_DIARIO

    def get_current_meal_index(self):
        """Returns which meal within the day (0-4)"""
        return (self.current_position % NUM_ALIMENTOS_DIARIO) // 3  # Approximate, refine if needed

    def is_position_valid_for_food(self, food_index):
        """
        Check if a food can be placed at current position.
        
        Rules from problem:
        - Breakfast drink (position 0, 11, 22...): only bebida_desayuno
        - Breakfast food (positions 1-2, 12-13...): only desayuno items
        - Snacks (positions 3, 7, 14, 18...): only snacks (fruits or sugar items)
        - Lunch/Dinner drinks (positions 4, 8, 15, 19...): only bebidas
        - Lunch/Dinner food (positions 5-6, 9-10...): only almuerzo_cena
        
        Also checks allergies and age restrictions.
        """
        food_item = self.food_db[food_index]
        food_group = food_item["grupo"]
        food_type_needed = self.get_current_food_type()
        
        # Check allergy (hard constraint) - using grupo codes from constantes.py
        allergies = self.user_profile.get("alergias", [])
        if any(food_group.startswith(allergy) for allergy in allergies):
            return False
        
        # Get valid foods list from auxiliary_functions.py
        # This function already filters by type and age
        valid_foods = filtrar_comida(self.food_db, food_type_needed, self.user_profile["edad"])
        
        # Check if this food is in the valid list
        return food_index in valid_foods

    def add_food(self, food_index):
        """Add a food to the ant's path, advance position, and update macros"""
        food_item = self.food_db[food_index]

        self.current_day_cals += food_item["calorias"]
        self.current_day_prot += food_item["proteinas"]
        self.current_day_carbs += food_item["carbohidratos"]
        self.current_day_fats += food_item["grasas"]

        self.path.append(food_index)
        self.current_position += 1
        
        # Reset daily values when a new day starts
        if self.current_position % NUM_ALIMENTOS_DIARIO == 0:
            self.current_day_cals = 0
            self.current_day_prot = 0
            self.current_day_carbs = 0
            self.current_day_fats = 0
        
        if self.current_position >= NUM_DIAS * NUM_ALIMENTOS_DIARIO:  # 77
            self.complete = True
            self.calculate_fitness()

    def can_add_food(self, food_index):
        """Check if food can be added at current position"""
        if self.complete:
            return False
        return self.is_position_valid_for_food(food_index)

    def calculate_fitness(self):
        """
        Calculate fitness using AlgorithmTools.calculate_fitness()
        LOWER fitness = BETTER solution (minimization problem)
        """
        if len(self.path) != NUM_DIAS * NUM_ALIMENTOS_DIARIO:
            self.fitness = float('inf')
            return self.fitness
        
        # self.fitness = self.tools.calculate_fitness(
        #     self.path,
        #     self.food_db,
        #     self.user_profile["calorias"],
        #     self.user_profile.get("alergias", []),
        #     self.user_profile.get("gustos", []),
        #     self.user_profile.get("disgustos", [])
        # )

        self.fitness = self.tools.calculate_fitness_v3(
            self.path,
            self.food_db,
            self.user_profile["calorias"],
            self.user_profile.get("alergias", []),
            self.user_profile.get("gustos", []),
            self.user_profile.get("disgustos", [])
        )

        return self.fitness

    def get_path_copy(self):
        """Return a copy of the current path"""
        return self.path.copy()

    def build_solution(self, pheromone_matrix, alpha, beta):
        while len(self.path) < 77:
            food_type = self.get_current_food_type()
            filtered_food = self.grouped_food[food_type]
            next_food = self.get_next_point(filtered_food, pheromone_matrix, alpha, beta, self.rng)

            if next_food is None:
                break

            self.add_food(next_food)

    def get_next_point(self, allowed_food_indices, pheromone_matrix, alpha, beta, rng):
        """
        Probabilistically select the next food item using DYNAMIC ACO formula.
        """
        if not allowed_food_indices:
            return None

        # POPRAWKA: Zamieniliśmy 'ant.' na 'self.' bo jesteśmy wewnątrz klasy
        target_calories = self.user_profile["calorias"]
        current_meal_of_day = self.current_position % NUM_ALIMENTOS_DIARIO
        meals_left_today = NUM_ALIMENTOS_DIARIO - current_meal_of_day
        
        probabilities = []
        
        for next_food in allowed_food_indices:
            food_item = self.food_db[next_food]

            # Tu już bezpiecznie odczytujemy feromony
            pheromone = pheromone_matrix[self.current_position].get(next_food, 0.1)
            pheromone = pheromone ** alpha

            dynamic_eta = self.calculate_heuristic(
                food_item=food_item,
                current_day_cals=self.current_day_cals,
                current_day_prot=self.current_day_prot,
                current_day_carbs=self.current_day_carbs,
                current_day_fats=self.current_day_fats,
                target_calories=target_calories,
                remaining_slots=meals_left_today 
            )
            heuristic = dynamic_eta ** beta
            
            prob = pheromone * heuristic
            probabilities.append(prob)
        
        # Roulette wheel selection
        total = sum(probabilities)
        if total == 0:
            return rng.choice(allowed_food_indices)
        
        normalized_probs = [p / total for p in probabilities]
        return rng.choices(allowed_food_indices, weights=normalized_probs, k=1)[0]

    def calculate_heuristic(self, food_item, current_day_cals, current_day_prot, current_day_carbs, current_day_fats, target_calories, remaining_slots):
        # POPRAWKA: Dynamiczne liczenie Gustów zamiast zera!
        base_score = 1.0
        group = food_item["grupo"]
        
        if any(group.startswith(l) for l in self.user_profile.get("gustos", [])):
            base_score = 2.0
        elif any(group.startswith(d) for d in self.user_profile.get("disgustos", [])):
            base_score = 0.1

        food_cal = food_item["calorias"]
        new_total_cals = current_day_cals + food_cal

        if new_total_cals > target_calories * 1.2:
            return base_score * 0.001 
            
        missing_calories = target_calories - current_day_cals
        
        safe_remaining = max(1, remaining_slots) 
        ideal_cals_for_this_meal = missing_calories / safe_remaining
        calorie_diff = abs(ideal_cals_for_this_meal - food_cal)
        
        cal_multiplier = 1000.0 / (50.0 + calorie_diff)

        new_prot = current_day_prot + food_item["proteinas"]
        new_carbs = current_day_carbs + food_item["carbohidratos"]
        new_fats = current_day_fats + food_item["grasas"]
        
        macro_multiplier = 1.0
        
        if new_total_cals > 0:
            perc_prot, perc_carbs, perc_fats = calculo_macronutrientes(new_prot, new_carbs, new_fats)
            
            if 45 <= perc_carbs <= 65: macro_multiplier += 0.5
            else:
                if perc_carbs < 45:
                    macro_multiplier -= 0.2 + 0.2 * (45 - perc_carbs) / 45.0
                else:
                    macro_multiplier -= 0.2 + 0.2 * (perc_carbs - 65) / 35.0
                
            if 20 <= perc_fats <= 35: macro_multiplier += 0.5
            else:
                if perc_fats < 20:
                    macro_multiplier -= 0.2 + 0.2 * (20 - perc_fats) / 20.0
                else:
                    macro_multiplier -= 0.2 + 0.2 * (perc_fats - 35) / 65.0

            if 10 <= perc_prot <= 35: macro_multiplier += 0.5
            else:
                if perc_prot < 10:
                    macro_multiplier -= 0.2 + 0.2 * (10 - perc_prot) / 10.0
                else:
                    macro_multiplier -= 0.2 + 0.2 * (perc_prot - 35) / 65.0

        if macro_multiplier < 0.05:
            macro_multiplier = 0.05

        return base_score * cal_multiplier * macro_multiplier  

    def total_cost_function(self, ant_path, tools, food_db, user_profile):
        """
        Calculate total cost (fitness) for a complete ant path.
        This is a wrapper for AlgorithmTools.calculate_fitness()
        """
        if len(ant_path) != NUM_DIAS * NUM_ALIMENTOS_DIARIO:
            return float('inf')
        
        fitness = tools.calculate_fitness(
            ant_path,
            food_db,
            user_profile["calorias"],
            user_profile.get("alergias", []),
            user_profile.get("gustos", []),
            user_profile.get("disgustos", [])
        )
        return fitness

class ACO(AlgorithmTools):
    def __init__(self, user_profile, food_db, rng, num_ants):
        """
        Parameters:
        - user_profile: dict with keys: 'calorias', 'edad', 'gustos', 'disgustos', 'alergias'
        - food_db: list of food dicts from comida_basedatos()
        - rng: random.Random object for reproducible randomness
        """
        self.user_profile = user_profile
        self.food_db = food_db
        self.rng = rng
        self.num_ants = num_ants

        # Tools for validation and fitness
        self.tools = AlgorithmTools("ant", user_profile["edad"])

        # Pheromone initialization
        self.pheromone = None

        self.best_solution = None
        self.best_fitness = float("inf")

        self.evaporation_rate = 0.05
        self.pheromone_strength = 50000.0

        # Filtered food
        self.grouped_food = {}
        for food_type in ["bebida_desayuno", "desayuno", "snacks", "bebidas", "almuerzo_cena"]:
            self.grouped_food[food_type] = filtrar_comida(self.food_db, food_type, self.user_profile["edad"])

        # Dynamic ACO
        self.alpha = 1.0
        self.beta = 2.0

        self.ants = []
        for i in range(num_ants):
            self.ants.append(Ant(user_profile, food_db, random.Random(self.rng.randint(1, 10000)), i, self.grouped_food))

        # Plotting
        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def initialize_pheromones(self):
        self.pheromone = []

        for i in range(77):
            self.pheromone.append({})
            food_type = self.get_food_type(i)
            
            safe_allowed = self.grouped_food[food_type]
            
            for allowed in safe_allowed:
                self.pheromone[i][allowed] = 1.0

    def evaporate_pheromone(self):
        for idx in range(len(self.pheromone)):
            for food in self.pheromone[idx]:
                self.pheromone[idx][food] *= (1 - self.evaporation_rate)

    def deposit_pheromone(self, solution, fitness):
        amount = self.pheromone_strength * (1.0 / (1 + max(fitness, 0.000001)))

        for index in range(len(solution)):
            self.pheromone[index][solution[index]] += amount

    def update_pheromone(self, tfitnesses, tsolutions):
        self.pheromone_history.append(self.pheromone.copy())

        self.evaporate_pheromone()

        best_index = 0
        min_fitness = float("inf")
        for i in range(len(tfitnesses)):
            if tfitnesses[i] < min_fitness:
                min_fitness = tfitnesses[i]
                best_index = i

        best_solution = tsolutions[best_index]
        best_fitness = tfitnesses[best_index]

        self.deposit_pheromone(best_solution, best_fitness)

        if best_fitness < self.best_fitness:
            self.best_fitness = best_fitness
            self.best_solution = best_solution.copy()

    def aco(self, max_iterations = 20):
        self.initialize_pheromones()

        iterations = 0

        while iterations < max_iterations:
            tsolutions = []
            tfitnesses = []

            trails = []

            results = []
            args = [
                (
                    self.ants[i],
                    self.pheromone,
                    self.alpha,
                    self.beta,
                    self.tools,
                    self.food_db,
                    self.user_profile
                )
                for i in range(self.num_ants)
            ]

            with ProcessPoolExecutor(max_workers=16) as executor:
                results = list(
                    executor.map(
                        run_ant,
                        args,
                        chunksize=3
                    )
                )

            for result in results:
                solution, fitness = result
                tsolutions.append(solution)
                tfitnesses.append(fitness)
                trails.append((solution, fitness))

                if fitness < self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness

            self.update_pheromone(tfitnesses, tsolutions)
            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)

            iterations += 1
            print(f"\r  [In progress] Iteration {iterations}/{max_iterations}... (Best fitness: {round(self.best_fitness, 2)})", end="", flush=True)

        return self.best_solution    

def run_ant(args):
    ant, pheromone, alpha, beta, tools, food_db, user_profile = args
    ant.reset()
    ant.build_solution(pheromone, alpha, beta)
    solution = ant.get_path_copy()
    fitness = ant.total_cost_function(solution, tools, food_db, user_profile)

    return (solution, fitness)

# Testing part - Chat GPT generated
USER_PROFILES = [
    {
        "id": 1,
        "peso": 86,
        "altura": 180,
        "edad": 30,
        "sexo": "H",
        "actividad": "Alta",
        "calorias": 2700.00,
        "calorias_min": 2160.00,
        "calorias_max": 3240.00,
        "alergias": [],
        "gustos": [],
        "disgustos": [],
    },
    {
        "id": 2,
        "peso": 60,
        "altura": 170,
        "edad": 30,
        "sexo": "M",
        "actividad": "Muy Alto",
        "calorias": 2567.85,
        "calorias_min": 2054.28,
        "calorias_max": 3081.42,
        "alergias": ["A", "AB", "AC", "AD", "AE", "AF", "AG", "AI", "AK", "AM", "AN", "AO", "AP", "AS", "AT"],
        "gustos": ["BAE", "FC", "FE"],
        "disgustos": ["C", "CA", "CD", "CDE", "CDH"],
    },
    {
        "id": 3,
        "peso": 90,
        "altura": 175,
        "edad": 40,
        "sexo": "H",
        "actividad": "Alto",
        "calorias": 3102.84,
        "calorias_min": 2482.27,
        "calorias_max": 3723.41,
        "alergias": ["PAC", "PCA", "SNC"],
        "gustos": ["DAP", "MAC", "SEA"],
        "disgustos": ["BH", "BJS", "MIG"],
    },
    {
        "id": 4,
        "peso": 68,
        "altura": 160,
        "edad": 55,
        "sexo": "M",
        "actividad": "Ligero",
        "calorias": 1710.50,
        "calorias_min": 1368.40,
        "calorias_max": 2052.60,
        "alergias": ["F", "FA", "FC", "FE"],
        "gustos": ["AF", "BNH"],
        "disgustos": ["MB", "QA", "QC"],
    },
    {
        "id": 5,
        "peso": 72,
        "altura": 155,
        "edad": 72,
        "sexo": "M",
        "actividad": "Sedentario",
        "calorias": 1401.30,
        "calorias_min": 1121.04,
        "calorias_max": 1681.56,
        "alergias": ["BA", "BAB", "BAE", "BAH", "BAK", "BAR", "BH"],
        "gustos": ["AM", "JC"],
        "disgustos": ["MG", "MR"]
    }
]

food_db = comida_basedatos()

def get(x, key):
    return x[key] if isinstance(x, dict) else getattr(x, key)

def match(grupo, grupos):
    return any(grupo.startswith(g) for g in grupos)

def check_quality(sol, comida_bd, sujeto, fileptr: TextIO):
    assert len(sol) == 77

    total_error = 0
    cal_bad = macro_bad = allergy_bad = likes = dislikes = 0

    for dia in range(7):
        alimentos = [comida_bd[int(i)] for i in sol[dia * 11:(dia + 1) * 11]]

        cal = sum(get(a, "calorias") for a in alimentos)
        p = sum(get(a, "proteinas") for a in alimentos)
        c = sum(get(a, "carbohidratos") for a in alimentos)
        f = sum(get(a, "grasas") for a in alimentos)

        kcal_macros = 4 * p + 4 * c + 9 * f

        if kcal_macros == 0:
            p_pct = c_pct = f_pct = 0
        else:
            p_pct = 100 * 4 * p / kcal_macros
            c_pct = 100 * 4 * c / kcal_macros
            f_pct = 100 * 9 * f / kcal_macros

        total_error += abs(cal - sujeto["calorias"])

        cal_bad += not (sujeto["calorias_min"] <= cal <= sujeto["calorias_max"])
        macro_bad += not (45 <= c_pct <= 65)
        macro_bad += not (20 <= f_pct <= 35)
        macro_bad += not (10 <= p_pct <= 35)

        for a in alimentos:
            grupo = get(a, "grupo")
            allergy_bad += match(grupo, sujeto["alergias"])
            likes += match(grupo, sujeto["gustos"])
            dislikes += match(grupo, sujeto["disgustos"])

        fileptr.write(
            f"Día {dia + 1}: " + "\n" +
            f"cal={cal:.1f}, " + "\n" +
            f"C={c_pct:.1f}%, " + "\n" +
            f"G={f_pct:.1f}%, " + "\n" +
            f"P={p_pct:.1f}%" + "\n"
        )

    avg_error = total_error / 7
    hard_violations = cal_bad + macro_bad + allergy_bad

    fileptr.write("\nQUALITY SUMMARY"+ "\n")
    fileptr.write("---------------"+ "\n")
    fileptr.write(f"Average daily calorie error: {round(avg_error, 2)}"+ "\n")
    fileptr.write(f"Bad calorie days: {cal_bad}"+ "\n")
    fileptr.write(f"Macro violations: {macro_bad}"+ "\n")
    fileptr.write(f"Allergy violations: {allergy_bad}"+ "\n")
    fileptr.write(f"Liked foods used: {likes}"+ "\n")
    fileptr.write(f"Disliked foods used: {dislikes}"+ "\n")

    if allergy_bad > 0:
        verdict = "BAD: allergy violation"
    elif hard_violations > 0:
        verdict = "INVALID: hard constraints failed"
    elif avg_error < 150:
        verdict = "VERY GOOD"
    elif avg_error < 300:
        verdict = "GOOD"
    else:
        verdict = "VALID, but calories are weak"

    fileptr.write(f"Verdict: {verdict}"+ "\n")

    return {
        "avg_calorie_error": avg_error,
        "bad_calorie_days": cal_bad,
        "macro_violations": macro_bad,
        "allergy_violations": allergy_bad,
        "likes": likes,
        "dislikes": dislikes,
        "verdict": verdict,
    }

def stringify_individual(individual: List[int]) -> str:
    return ''.join([str(int(i)) for i in individual])

def plot_aco_run(aco, index):
    fitness_plot_fname = f"./output_aco/user_{index + 1}_fitness.png"
    diversity_plot_fname = f"./output_aco/user_{index + 1}_diversity.png"
    evolution_plot_fname = f"./output_aco/user_{index + 1}_evolution.png"

    # Fitness
    sns.set_style('darkgrid')

    fitness = np.array([[ant[1] for ant in trails] for trails in aco.trails_history ])
    best_fitness = np.array(aco.best_fitness_history)

    _, axs = plt.subplots(figsize=(5,5))
    axs.set_title('Fitness evolution')
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Fitness')

    axs.plot(best_fitness, label='best_high')

    median = np.median(fitness, axis=1)
    min = np.min(fitness, axis=1)
    max = np.max(fitness, axis=1)
    axs.plot(median, label='iterations_high')
    axs.fill_between(np.arange(len(median)), min, max, alpha=0.3, color='orange')

    axs.legend()

    plt.savefig(fitness_plot_fname, dpi=300, bbox_inches="tight")
    plt.close()

    # Population diversity
    population = np.array([[np.array(ant[0]) for ant in trails] for trails in aco.trails_history ])

    diversity = np.sum(np.std(population, axis=1), axis=1)
    _, axs = plt.subplots(figsize=(5,5))
    axs.set_title('Diversity evolution')
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Diversity')
    axs.plot(diversity, color='orange')

    axs.legend()

    plt.savefig(diversity_plot_fname, dpi=300, bbox_inches="tight")
    plt.close()

    # Number of different solutions
    population = np.array([[np.array(ant[0]) for ant in trails] for trails in aco.trails_history ])

    a = np.apply_along_axis(stringify_individual, 2, population)
    a = list(accumulate(a, lambda x, y: x.union(set(y)), initial=set()))

    _, axs = plt.subplots(figsize=(5,5))
    axs.plot([len(x) for x in a], color='orange')
    axs.set_title('Unique solutions evolution')
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Unique solutions')

    plt.savefig(evolution_plot_fname, dpi=300, bbox_inches="tight")
    plt.close()

def test_and_plot_aco():
    results = []
    acos = []
    run_times = []

    for user_profile in USER_PROFILES:
        start = time.perf_counter()
        aco = ACO(user_profile, food_db, random.Random(42), 50)
        solution = aco.aco()
        best_fitness = aco.best_fitness

        results.append((solution, best_fitness))
        acos.append(aco)
        end = time.perf_counter()

        total = end - start
        run_times.append(total)

    index = 0
    for solution, best_fitness in results:
        with open(f"./output_aco/user_{index + 1}.general", "w") as f:
            f.write(f"\n{'='*40}\nTEST FOR USER NR {index + 1} (Cel: {USER_PROFILES[index]['calorias']} kcal)\n{'='*40}\n")
            f.write("Solution: " + str(solution) + "\n")
            f.write("Best fitness: " + str(best_fitness) + "\n")
            f.write("Quality check:" + "\n")
            check_quality(solution, food_db, USER_PROFILES[index], f)

            f.write(f"Performance: {run_times[index]:.6f}" + "s\n")

            aco = acos[index]
            plot_aco_run(aco, index)

        index += 1

test_and_plot_aco()
