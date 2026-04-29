# ant_colony.py

import random
from algorithms_tools import AlgorithmTools
from constantes import NUM_ALIMENTOS_DIARIO, NUM_DIAS, DIAS_SEMANA
from auxiliary_functions import calculo_macronutrientes, filtrar_comida


class Ant:
    """
    Ant class for ACO to solve the Diet Optimization Problem.
    Each ant builds a complete weekly meal plan (77 food items).
    """
    
    def __init__(self, ant_id, user_profile, food_db, rng):
        """
        Parameters:
        - ant_id: unique identifier for this ant
        - user_profile: dict with keys: 'calorias', 'edad', 'gustos', 'disgustos', 'alergias'
        - food_db: list of food dicts from comida_basedatos()
        - rng: random.Random object for reproducible randomness
        """
        self.ant_id = ant_id
        self.user_profile = user_profile
        self.food_db = food_db
        self.rng = rng
        
        # Ant's journey
        self.path = []           # List of food IDs (indices into food_db)
        self.current_position = 0  # 0 to 76 (which meal slot we're filling)
        self.complete = False
        self.fitness = None
        
        # Tools for validation and fitness
        self.tools = AlgorithmTools("ant", user_profile["edad"])

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
        
        self.fitness = self.tools.calculate_fitness(
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


def get_next_point(ant, allowed_food_indices, pheromone_matrix, alpha, beta, rng):
    """
    Probabilistically select the next food item using DYNAMIC ACO formula.
    """
    if not allowed_food_indices:
        return None
    
    current_food_index = ant.path[-1] if ant.path else None
    target_calories = ant.user_profile["calorias"]
    current_meal_of_day = ant.current_position % NUM_ALIMENTOS_DIARIO
    meals_left_today = NUM_ALIMENTOS_DIARIO - current_meal_of_day
    
    probabilities = []
    
    for next_food in allowed_food_indices:
        food_item = ant.food_db[next_food]
        
        pheromone = pheromone_matrix.get(current_food_index, {}).get(next_food, 0.1)
        pheromone = pheromone ** alpha
        
        dynamic_eta = calculate_heuristic(
            food_item=food_item,
            current_day_cals=ant.current_day_cals,
            current_day_prot=ant.current_day_prot,
            current_day_carbs=ant.current_day_carbs,
            current_day_fats=ant.current_day_fats,
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


def total_cost_function(ant_path, tools, food_db, user_profile):
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

def calculate_heuristic(food_item, current_day_cals, current_day_prot, current_day_carbs, current_day_fats, target_calories, remaining_slots):
    base_score = food_item["base_score"]

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
        else: macro_multiplier -= 0.2
            
        if 20 <= perc_fats <= 35: macro_multiplier += 0.5
        else: macro_multiplier -= 0.2
            
        if 10 <= perc_prot <= 35: macro_multiplier += 0.5
        else: macro_multiplier -= 0.2

    if macro_multiplier < 0.1:
        macro_multiplier = 0.1

    return base_score * cal_multiplier * macro_multiplier   
