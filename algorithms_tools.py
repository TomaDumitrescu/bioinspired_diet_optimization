from constantes import *
from database import *
from auxiliary_functions import *

class AlgorithmTools:
    def __init__(self, type, user_age):
        self.type = type
        self.user_age = user_age
        self._cached_penalties_key_v1 = None
        self._cached_penalties_v1 = None
        self._cached_bonuses_v1 = None
        self._cached_penalties_key_v3 = None
        self._cached_penalties_v3 = None

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

    def get_random_solution(self, rng):
        food_bd = comida_basedatos()
        rand_sol = []
        for i in range(77):
            food_type = self.get_food_type(i)
            filtered_food = filtrar_comida(food_bd, food_type, self.user_age)
            rand_sol.append(rng.choice(filtered_food))

        return rand_sol

    def get_day(self, idx: int):
        return DIAS_SEMANA[int(idx / 11)]

    # Fitness function
    def calculate_fitness(self, solution, food_db, target_calories, allergies=None, likes=None, dislikes=None):
        """
        Calculates Fitness based on the official requirements from the document.
        GOAL: Minimize the difference between actual and target calories.
        RESULT: The closer to 0, the better the diet.
        """
        if allergies is None: allergies = []
        if likes is None: likes = []
        if dislikes is None: dislikes = []
        
        cache_key = (tuple(allergies), tuple(likes), tuple(dislikes))
        if self._cached_penalties_key_v1 != cache_key:
            self._cached_penalties_key_v1 = cache_key
            self._cached_penalties_v1 = []
            self._cached_bonuses_v1 = []
            for item in food_db:
                group = item["grupo"]
                penalty = 0
                bonus = 0
                if any(group.startswith(a) for a in allergies):
                    penalty += 10000
                if any(group.startswith(l) for l in likes):
                    bonus += 50
                if any(group.startswith(d) for d in dislikes):
                    penalty += 50
                self._cached_penalties_v1.append(penalty)
                self._cached_bonuses_v1.append(bonus)

        total_fitness = 0

        for day_idx in range(NUM_DIAS):
            # 1. Extract meals for the given day using original constant names
            start_idx = day_idx * NUM_ALIMENTOS_DIARIO
            end_idx = start_idx + NUM_ALIMENTOS_DIARIO
            daily_genes = solution[start_idx:end_idx]

            daily_calories = 0
            daily_proteins = 0
            daily_carbs = 0
            daily_fats = 0
            
            daily_penalties = 0
            daily_bonuses = 0

            # 2. Sum up values for the whole day and check specific products
            for gene in daily_genes:
                food_item = food_db[gene]
                
                daily_calories += food_item["calorias"]
                daily_proteins += food_item["proteinas"]
                daily_carbs += food_item["carbohidratos"]
                daily_fats += food_item["grasas"]

                daily_penalties += self._cached_penalties_v1[gene]
                daily_bonuses += self._cached_bonuses_v1[gene]

            # --- WHOLE DAY EVALUATION ---

            calorie_difference = abs(target_calories - daily_calories)
            total_fitness += calorie_difference

            # RULE 1: Hard calorie limit (80% - 120%)
            min_calories = target_calories * 0.8
            max_calories = target_calories * 1.2
            if daily_calories < min_calories or daily_calories > max_calories:
                daily_penalties += 5000

            # RULE 2, 3, 4: Macronutrient Limits (Proportional penalties)
            if daily_calories > 0:
                perc_protein, perc_carbs, perc_fats = calculo_macronutrientes(
                    daily_proteins, daily_carbs, daily_fats
                )

                # Carbohydrates (45 - 65)
                if perc_carbs < 45:
                    daily_penalties += (45 - perc_carbs) * 100
                elif perc_carbs > 65:
                    daily_penalties += (perc_carbs - 65) * 100

                # Fats (20 - 35)
                if perc_fats < 20:
                    daily_penalties += (20 - perc_fats) * 100
                elif perc_fats > 35:
                    daily_penalties += (perc_fats - 35) * 100

                # Protein (10 - 35)
                if perc_protein < 10:
                    daily_penalties += (10 - perc_protein) * 100
                elif perc_protein > 35:
                    daily_penalties += (perc_protein - 35) * 100

            # Summarize scoring for this day
            total_fitness += daily_penalties
            total_fitness -= daily_bonuses

        return max(0, total_fitness)


    def calculate_fitness_v3(self, solution, food_db, target_calories, allergies=None, likes=None, dislikes=None):
        """
        Same structure as v1 but with positive-baseline preference scoring.
        Liked = 0, Neutral = +50, Disliked = +100.
        Preferences are always >= 0 so they can never mask constraint violations.
        """
        if allergies is None: allergies = []
        if likes is None: likes = []
        if dislikes is None: dislikes = []

        cache_key = (tuple(allergies), tuple(likes), tuple(dislikes))
        if self._cached_penalties_key_v3 != cache_key:
            self._cached_penalties_key_v3 = cache_key
            self._cached_penalties_v3 = []
            for item in food_db:
                group = item["grupo"]
                penalty = 0
                if any(group.startswith(a) for a in allergies):
                    penalty += 10000
                if any(group.startswith(l) for l in likes):
                    pass
                elif any(group.startswith(d) for d in dislikes):
                    penalty += 100
                else:
                    penalty += 50
                self._cached_penalties_v3.append(penalty)

        total_fitness = 0

        for day_idx in range(NUM_DIAS):
            start_idx = day_idx * NUM_ALIMENTOS_DIARIO
            end_idx = start_idx + NUM_ALIMENTOS_DIARIO
            daily_genes = solution[start_idx:end_idx]

            daily_calories = 0
            daily_proteins = 0
            daily_carbs = 0
            daily_fats = 0
            daily_penalties = 0

            for gene in daily_genes:
                food_item = food_db[gene]

                daily_calories += food_item["calorias"]
                daily_proteins += food_item["proteinas"]
                daily_carbs += food_item["carbohidratos"]
                daily_fats += food_item["grasas"]

                daily_penalties += self._cached_penalties_v3[gene]

            calorie_difference = abs(target_calories - daily_calories)
            total_fitness += calorie_difference

            # RULE 1: Hard calorie limit (80% - 120%)
            min_calories = target_calories * 0.8
            max_calories = target_calories * 1.2
            if daily_calories < min_calories or daily_calories > max_calories:
                daily_penalties += 5000

            # RULE 2, 3, 4: Macronutrient Limits (Proportional penalties)
            if daily_calories > 0:
                perc_protein, perc_carbs, perc_fats = calculo_macronutrientes(
                    daily_proteins, daily_carbs, daily_fats
                )

                if perc_carbs < 45:
                    daily_penalties += (45 - perc_carbs) * 100
                elif perc_carbs > 65:
                    daily_penalties += (perc_carbs - 65) * 100

                if perc_fats < 20:
                    daily_penalties += (20 - perc_fats) * 100
                elif perc_fats > 35:
                    daily_penalties += (perc_fats - 35) * 100

                if perc_protein < 10:
                    daily_penalties += (10 - perc_protein) * 100
                elif perc_protein > 35:
                    daily_penalties += (perc_protein - 35) * 100

            total_fitness += daily_penalties

        return total_fitness

# Usage
# algorithm_tools = AlgorithmTools("genetic", 23)
# sol = algorithm_tools.get_random_solution(rng) --> rng = random.Random()      rng.seed(seed_number)
# print(traducir_solucion(sol, comida_basedatos()))
