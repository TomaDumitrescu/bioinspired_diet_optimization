from constantes import NUM_DIAS, NUM_ALIMENTOS_DIARIO
from auxiliary_functions import calculo_macronutrientes

def calculate_fitness(solution, food_db, target_calories, allergies=None, likes=None, dislikes=None):
    """
    Calculates Fitness based on the official requirements from the document.
    GOAL: Minimize the difference between actual and target calories.
    RESULT: The closer to 0, the better the diet.
    """
    if allergies is None: allergies = []
    if likes is None: likes = []
    if dislikes is None: dislikes = []

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
            
            group = food_item["grupo"]
            
            daily_calories += food_item["calorias"]
            daily_proteins += food_item["proteinas"]
            daily_carbs += food_item["carbohidratos"]
            daily_fats += food_item["grasas"]

            # RULE 5: Allergies (Hard constraint -> Massive penalty)
            if any(group.startswith(a) for a in allergies):
                daily_penalties += 10000

            # RULE 6: User preferences
            if any(group.startswith(l) for l in likes):
                daily_bonuses += 50  # Reward (decreases score)
            if any(group.startswith(d) for d in dislikes):
                daily_penalties += 50  # Penalty (increases score)

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