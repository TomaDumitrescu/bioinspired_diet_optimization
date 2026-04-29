# ant_colony.py

import random
from algorithms_tools import AlgorithmTools
from constantes import NUM_ALIMENTOS_DIARIO, NUM_DIAS, DIAS_SEMANA


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
    
    def reset(self):
        """Reset the ant for a new journey"""
        self.path = []
        self.current_position = 0
        self.complete = False
        self.fitness = None
    
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
        from auxiliary_functions import filtrar_comida
        valid_foods = filtrar_comida(self.food_db, food_type_needed, self.user_profile["edad"])
        
        # Check if this food is in the valid list
        return food_index in valid_foods
    
    def add_food(self, food_index):
        """Add a food to the ant's path and advance position"""
        self.path.append(food_index)
        self.current_position += 1
        
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


def get_next_point(current_food_index, allowed_food_indices, pheromone_matrix, heuristic_matrix, alpha, beta, rng):
    """
    Probabilistically select the next food item using ACO formula.
    
    Parameters:
    - current_food_index: the food ID just placed (or None if first choice)
    - allowed_food_indices: list of food indices valid for next position
    - pheromone_matrix: 2D dict where pheromone[from_food][to_food] = tau
    - heuristic_matrix: 2D dict where heuristic[from_food][to_food] = eta
    - alpha: pheromone influence weight (1-3 typical)
    - beta: heuristic influence weight (2-5 typical)
    - rng: random number generator
    
    Returns: selected food index
    """
    if not allowed_food_indices:
        return None
    
    # For the first choice, no previous food exists
    if current_food_index is None:
        # Uniform probability for first selection
        return rng.choice(allowed_food_indices)
    
    probabilities = []
    
    for next_food in allowed_food_indices:
        # Get pheromone level (default 0.1 if not found)
        pheromone = pheromone_matrix.get(current_food_index, {}).get(next_food, 0.1)
        pheromone = pheromone ** alpha
        
        # Get heuristic value (default 1.0 if not found)
        heuristic = heuristic_matrix.get(current_food_index, {}).get(next_food, 1.0)
        heuristic = heuristic ** beta
        
        prob = pheromone * heuristic
        probabilities.append(prob)
    
    # Roulette wheel selection
    total = sum(probabilities)
    if total == 0:
        return rng.choice(allowed_food_indices)
    
    # Normalize and select
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