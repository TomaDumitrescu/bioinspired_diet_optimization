from database import comida_basedatos
from genetic_algorithm import GeneticAlgorithm
from auxiliary_functions import traducir_solucion
from constantes import SEEDS
import matplotlib.pyplot as plt
import database
import os

OUTPUT_DIR = "output_genetic"


def setup_genetic_data():
    food_db = comida_basedatos()

    user_profile = {
        "edad": 25,
        "calorias": 2000,
        "alergias": ["BA"],
        "gustos": ["F"],
        "disgustos": ["J"]
    }

    return food_db, user_profile

#gets all the users and food
def setup_genetic_data_database():
    food_db = comida_basedatos()
    user_profiles = sujetos_basedatos()

    return food_db, user_profiles

def run():
    food_db = comida_basedatos()
    user_profiles = database.sujetos_basedatos()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = passed = 0
    liked_sum = disliked_sum = 0

    for i, user in enumerate(user_profiles):
        fitness_historics = []
        diversity_historics = []

        for seed in SEEDS:
            ga = GeneticAlgorithm(user, food_db)
            best, fitness, best_fitness_historic, diversity_historic = ga.run(seed=seed)

            print_solution(best, fitness, food_db)

            fitness_historics.append(best_fitness_historic)
            diversity_historics.append(diversity_historic)

            solution_passed, liked_pct, disliked_pct = evaluate_solution(best, food_db, user)
            total += 1
            if solution_passed:
                passed += 1
            liked_sum += liked_pct
            disliked_sum += disliked_pct

        plot_average(fitness_historics, diversity_historics, f"user{i}_average")

    summary = (
        f"\n=== Summary ===\n"
        f"Passed constraints: {passed}/{total} ({100*passed/total:.1f}%)\n"
        f"Failed constraints: {total-passed}/{total} ({100*(total-passed)/total:.1f}%)\n"
        f"Avg liked foods:    {liked_sum/total:.1f}%\n"
        f"Avg disliked foods: {disliked_sum/total:.1f}%\n"
    )
    print(summary)
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)


def evaluate_solution(best, food_db, user):
    _, datos_dia = traducir_solucion(best, food_db)

    target_cal = user["calorias"]
    gustos = set(user["gustos"])
    disgustos = set(user["disgustos"])
    alergias = set(user["alergias"])

    constraints_ok = True
    for macros in datos_dia.values():
        cal = macros["calorias"]
        carbs = macros["porcentaje_carbohidratos"]
        fat = macros["porcentaje_grasas"]
        protein = macros["porcentaje_proteinas"]
        if not (0.8 * target_cal <= cal <= 1.2 * target_cal):
            constraints_ok = False
        if not (45 <= carbs <= 65):
            constraints_ok = False
        if not (20 <= fat <= 35):
            constraints_ok = False
        if not (10 <= protein <= 35):
            constraints_ok = False

    liked = neutral = disliked = 0
    total = len(best)
    for idx in best:
        grupo = food_db[int(idx)]["grupo"]
        if grupo in alergias:
            constraints_ok = False
        if grupo in gustos:
            liked += 1
        elif grupo in disgustos:
            disliked += 1
        else:
            neutral += 1

    liked_pct = round(100 * liked / total, 1) if total else 0
    disliked_pct = round(100 * disliked / total, 1) if total else 0

    return constraints_ok, liked_pct, disliked_pct


def plot_average(fitness_historics, diversity_historics, file_name):
    n = min(len(h) for h in fitness_historics)
    avg_fitness = [sum(h[g] for h in fitness_historics) / len(fitness_historics) for g in range(n)]
    avg_diversity = [sum(h[g] for h in diversity_historics) / len(diversity_historics) for g in range(n)]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(avg_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Avg Best Fitness over Generations')

    plt.subplot(1, 2, 2)
    plt.plot(avg_diversity)
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Avg Diversity over Generations')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{file_name}.png"))
    plt.show()

def print_solution(best, fitness, food_db):
    print(f"Best fitness: {fitness:.2f}")
    menu, datos_dia = traducir_solucion(best, food_db)

    for dia, comidas in menu.items():
        print(f"\n=== {dia} ===")
        for comida, (alimentos, calorias) in comidas.items():
            print(f"  {comida} ({calorias:.0f} kcal):")
            for alimento in alimentos:
                print(f"    {alimento}")
        macros = datos_dia[dia]
        print(f"  Total: {macros['calorias']:.0f} kcal | "
              f"P: {macros['porcentaje_proteinas']:.1f}% | "
              f"C: {macros['porcentaje_carbohidratos']:.1f}% | "
              f"G: {macros['porcentaje_grasas']:.1f}%")


def run2():
    food_db, user_profile = setup_genetic_data()

    ga = GeneticAlgorithm(user_profile, food_db)
    best, fitness, best_fitness_historic, diversity_historic = ga.run(seed=SEEDS[0])

    print(f"Best fitness: {fitness:.2f}")
    menu, datos_dia = traducir_solucion(best, food_db)

    for dia, comidas in menu.items():
        print(f"\n=== {dia} ===")
        for comida, (alimentos, calorias) in comidas.items():
            print(f"  {comida} ({calorias:.0f} kcal):")
            for alimento in alimentos:
                print(f"    {alimento}")
        macros = datos_dia[dia]
        print(f"  Total: {macros['calorias']:.0f} kcal | "
              f"P: {macros['porcentaje_proteinas']:.1f}% | "
              f"C: {macros['porcentaje_carbohidratos']:.1f}% | "
              f"G: {macros['porcentaje_grasas']:.1f}%")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(best_fitness_historic)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness over Generations')

    plt.subplot(1, 2, 2)
    plt.plot(diversity_historic)
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Diversity over Generations')

    plt.tight_layout()
    plt.savefig('ga_results.png')
    plt.show()

if __name__ == "__main__":
    run()
