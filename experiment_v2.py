import time
import random
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from database import comida_basedatos
from genetic_algorithm import GeneticAlgorithm
from ant_colony import ACO

# ─── Config ─────────────────────────────────────────────────────────────────

NUM_RUNS       = 30
GA_POP         = 100
GA_GENERATIONS = 200
ACO_ANTS       = 50
ACO_ITERATIONS = 100

# random.seed(42)  # For reproducibility (can be uncommented if needed)

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
        "disgustos": ["MG", "MR"],
    }
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_section(text):
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")

def analyze_constraints(solution, food_db, profile):
    """Calculates diet details, checking calorie AND MACRO limits strictly day-by-day."""
    
    allergy_violations = 0
    dislike_violations = 0
    likes_included = 0
    
    allergy_groups = set(profile.get("alergias", []))
    dislike_groups = set(profile.get("disgustos", []))
    like_groups = set(profile.get("gustos", []))

    for food_id in solution:
        f_group = food_db[food_id].get('grupo', '')
        if any(f_group.startswith(a) for a in allergy_groups):
            allergy_violations += 1
        if any(f_group.startswith(d) for d in dislike_groups):
            dislike_violations += 1
        if any(f_group.startswith(l) for l in like_groups):
            likes_included += 1

    MEALS_PER_DAY = 11
    DAYS = 7
    
    target_cal = profile.get('calorias', 0)

    cal_min = target_cal * 0.8
    cal_max = target_cal * 1.2

    total_cal_penalty = 0.0
    daily_calories_list = []
    
    days_cal_in_range = 0
    days_macro_in_range = 0

    for day in range(DAYS):
        c_today = 0.0
        p_today = 0.0
        f_today = 0.0
        cb_today = 0.0
        
        meals_today = solution[day * MEALS_PER_DAY : (day + 1) * MEALS_PER_DAY]
        
        for food_id in meals_today:
            food = food_db[food_id]
            c_today += float(food.get('calorias', 0))
            p_today += float(food.get('proteinas', 0))
            f_today += float(food.get('grasas', 0))
            cb_today += float(food.get('carbohidratos', 0))
            
        daily_calories_list.append(c_today)
        
        if c_today < cal_min:
            total_cal_penalty += (cal_min - c_today)
        elif c_today > cal_max:
            total_cal_penalty += (c_today - cal_max)
        else:
            days_cal_in_range += 1  

        cals_prot = p_today * 4.0
        cals_fat = f_today * 9.0
        cals_carb = cb_today * 4.0
        macro_cals_today = cals_prot + cals_fat + cals_carb
        
        if macro_cals_today > 0:
            pct_p = (cals_prot / macro_cals_today) * 100
            pct_f = (cals_fat / macro_cals_today) * 100
            pct_cb = (cals_carb / macro_cals_today) * 100
            
            p_ok = 10 <= pct_p <= 35
            f_ok = 20 <= pct_f <= 35
            cb_ok = 45 <= pct_cb <= 65
            
            if p_ok and f_ok and cb_ok:
                days_macro_in_range += 1

    achieved_cal_avg = sum(daily_calories_list) / DAYS

    return {
        "target_cal": round(target_cal, 2),
        "achieved_cal": round(achieved_cal_avg, 2),
        "cal_penalty": round(total_cal_penalty, 2),
        "days_cal_ok": days_cal_in_range,           
        "days_macro_ok": days_macro_in_range,       
        "hard_constraints_ok": allergy_violations == 0,
        "allergy_violations": allergy_violations,
        "dislike_violations": dislike_violations,
        "likes_included": likes_included
    }

# ─── ACO wrapper with progress ───────────────────────────────────────────────

def aco_run_patched(aco_instance, max_iterations, run_label):
    aco_instance.initialize_pheromones()
    fitness_history = []

    for iteration in range(1, max_iterations + 1):
        tsolutions = []
        tfitnesses = []

        for ant in aco_instance.ants:
            ant.reset()
            ant.build_solution(aco_instance.pheromone, aco_instance.alpha, aco_instance.beta)
            solution = ant.get_path_copy()
            fitness = ant.fitness if ant.fitness is not None else float('inf')
            tsolutions.append(solution)
            tfitnesses.append(fitness)

            if fitness < aco_instance.best_fitness:
                aco_instance.best_solution = solution.copy()
                aco_instance.best_fitness = fitness

        aco_instance.update_pheromone(tfitnesses, tsolutions)
        fitness_history.append(aco_instance.best_fitness)

        print(
            f"\r  Now running {run_label}, Iteration {iteration}/{max_iterations}"
            f" | Best fitness: {round(aco_instance.best_fitness, 2)}   ",
            end="", flush=True
        )

    print()
    return aco_instance.best_solution, fitness_history

# ─── Main experiment ─────────────────────────────────────────────────────────

def run_experiment():
    random.seed(42)
    food_db = comida_basedatos()

    all_results     = []
    all_convergence = []

    print_header("STARTING FULL EXPERIMENT (5 PROFILES)")

    for profile in USER_PROFILES:
        pid = profile["id"]
        print_section(f"Profile {pid} | Target: {profile['calorias']} kcal | {NUM_RUNS} runs each")

        seeds = [random.randint(1000, 99999) for _ in range(NUM_RUNS)]

        # ── GA runs ──────────────────────────────────────────────────────────
        for run_idx in range(NUM_RUNS):
            seed = seeds[run_idx]
            t0 = time.time()
            ga = GeneticAlgorithm(
                user_profile=profile,
                food_db=food_db,
                pop_size=GA_POP,
                max_generations=GA_GENERATIONS,
            )
            # Retrieve best solution and fitness
            best_sol_ga, fitness_ga, fitness_hist_ga, _ = ga.run(seed=seed)
            elapsed = time.time() - t0
            
            # Diet analysis
            constraints_ga = analyze_constraints(best_sol_ga, food_db, profile)

            all_results.append({
                "profile_id": pid,
                "algorithm":  "GA",
                "run":        run_idx + 1,
                "fitness":    fitness_ga,
                "time_s":     elapsed,
                **constraints_ga
            })

            for step, f in enumerate(fitness_hist_ga, start=1):
                all_convergence.append({
                    "profile_id": pid, "algorithm": "GA", "run": run_idx + 1,
                    "step": step, "fitness": f,
                })

            print(f"  GA  {run_idx+1}/{NUM_RUNS} | Fitness: {fitness_ga:.2f} | Valid: {constraints_ga['hard_constraints_ok']} | Time: {elapsed:.2f}s")

        # ── ACO runs ─────────────────────────────────────────────────────────
        for run_idx in range(NUM_RUNS):
            seed = seeds[run_idx]
            run_label = f"ACO {run_idx+1}/{NUM_RUNS} (Prof {pid})"
            
            t0 = time.time()
            aco = ACO(
                user_profile=profile,
                food_db=food_db,
                rng=random.Random(seed),
                num_ants=ACO_ANTS,
            )
            best_sol_aco, fitness_hist_aco = aco_run_patched(aco, ACO_ITERATIONS, run_label)
            elapsed = time.time() - t0
            fitness_aco = aco.best_fitness
            
            # Diet analysis
            constraints_aco = analyze_constraints(best_sol_aco, food_db, profile)

            all_results.append({
                "profile_id": pid,
                "algorithm":  "ACO",
                "run":        run_idx + 1,
                "fitness":    fitness_aco,
                "time_s":     elapsed,
                **constraints_aco
            })

            for step, f in enumerate(fitness_hist_aco, start=1):
                all_convergence.append({
                    "profile_id": pid, "algorithm": "ACO", "run": run_idx + 1,
                    "step": step, "fitness": f,
                })

    # ── Save results ──────────────────────────────────────────────────────────
    results_df     = pd.DataFrame(all_results)
    convergence_df = pd.DataFrame(all_convergence)

    results_df.to_csv("experiment_results_full.csv", index=False)
    convergence_df.to_csv("convergence_data_full.csv", index=False)

    print_header("Experiment complete! Saved to _full.csv files.")
    print_section("STATISTICAL ANALYSIS (Paired Design & Holm Correction)")

    # Initialize text report list
    report_lines = []
    report_lines.append("=== STATISTICAL REPORT (Wilcoxon Test + Holm Correction) ===")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    ga_medians  = []
    aco_medians = []
    p_values_raw = []
    profile_ids = [p["id"] for p in USER_PROFILES]

    # 1. Calculate Wilcoxon tests for each profile (raw p-values)
    for pid in profile_ids:
        sub = results_df[results_df["profile_id"] == pid].copy()
        
        # Pairing with identical seeds
        ga_sub = sub[sub["algorithm"] == "GA"].sort_values("run")["fitness"].values
        aco_sub = sub[sub["algorithm"] == "ACO"].sort_values("run")["fitness"].values
        
        ga_medians.append(np.median(ga_sub))
        aco_medians.append(np.median(aco_sub))
        
        try:
            _, p_val = stats.wilcoxon(aco_sub, ga_sub, alternative='two-sided')
        except ValueError:
            p_val = 1.0 # In case of identical results
            
        p_values_raw.append(p_val)

    # 2. Holm correction for multiple testing
    reject, p_corrected, _, _ = multipletests(p_values_raw, alpha=0.05, method='holm')

    wins_ga = 0
    wins_aco = 0

    alpha_info = "  Target Alpha: 0.05 (Family-wise Error Rate controlled by Holm method)\n"
    print(alpha_info)
    report_lines.append(alpha_info)

    for i, pid in enumerate(profile_ids):
        g_med = ga_medians[i]
        a_med = aco_medians[i]
        
        if reject[i]: # If difference is statistically significant after correction
            if a_med < g_med:
                wins_aco += 1
                sig = "WINNER: ACO"
            else:
                wins_ga += 1
                sig = "WINNER: GA "
        else:
            sig = "DRAW (No sig. diff)"

        # Print line and add to text report
        line = f"  Profile {pid} | GA Med: {g_med:.1f} | ACO Med: {a_med:.1f} | p-corr: {p_corrected[i]:.5f} -> {sig}"
        print(line)
        report_lines.append(line)

    # 3. Global Test (Sign Test based on victories per profile)
    global_lines = [
        "\n  --- Global Conclusion (Sign Test across Profiles) ---",
        f"  GA won on:  {wins_ga}/5 profiles",
        f"  ACO won on: {wins_aco}/5 profiles"
    ]
    
    draws = 5 - (wins_ga + wins_aco)
    if draws > 0:
        global_lines.append(f"  Draws:      {draws}/5 profiles")

    if wins_aco > wins_ga:
        global_lines.append("  VERDICT: ACO is the superior algorithm globally 🏆")
    elif wins_ga > wins_aco:
        global_lines.append("  VERDICT: GA is the superior algorithm globally 🏆")
    else:
        global_lines.append("  VERDICT: Both algorithms perform equally globally (Draw)")

    # Print global conclusions and add to text report
    for gl in global_lines:
        print(gl)
        report_lines.append(gl)

    # Save full report to text file
    with open("statistical_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("\n  [INFO] Statistics successfully saved to 'statistical_summary.txt'")

if __name__ == "__main__":
    run_experiment()