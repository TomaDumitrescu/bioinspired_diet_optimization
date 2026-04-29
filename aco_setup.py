# aco_setup.py
from database import comida_basedatos

def setup_aco_data():
    print("Preparing data")
    food_db = comida_basedatos()
    
    user_profile = {
        "edad": 25,
        "calorias": 2000,
        "alergias": ["BA"],  # Allergy to cow milk
        "gustos": ["F"],     # Likes fruits
        "disgustos": ["J"]   # Dislikes fish
    }
    
    return food_db, user_profile

def prepare_food_db_with_scores(food_db, user_profile):
    likes = user_profile.get("gustos", [])
    dislikes = user_profile.get("disgustos", [])
    
    for food in food_db:
        group = food["grupo"]
        food["base_score"] = 1.0  
        
        if any(group.startswith(l) for l in likes):
            food["base_score"] = 2.0
        elif any(group.startswith(d) for d in dislikes):
            food["base_score"] = 0.1
            
    return food_db    