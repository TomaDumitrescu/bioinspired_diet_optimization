from database import comida_basedatos
from genetic_algorithm import GeneticAlgorithm
from auxiliary_functions import traducir_solucion
from constantes import SEEDS


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


if __name__ == "__main__":
    food_db, user_profile = setup_genetic_data()

    ga = GeneticAlgorithm(user_profile, food_db)
    best = ga.run(seed=SEEDS[0])

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
