import os
import time
import random
import collections
import itertools
import re
from multiprocessing import Pool, cpu_count, freeze_support
from functools import partial

# ==============================================================================
# 1. Main Parameters
# ==============================================================================
FILE_PATH = "e_100000.txt"
PROCESSES = max(1, cpu_count() - 1) if cpu_count() and cpu_count() > 1 else 1
R_VALUES_TO_TEST = [3, 4, 5]
TRAINING_DATA_RATIO = 0.75
TRAINING_ITERATIONS = 75000

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def setup_data_file(filepath: str, num_digits: int = 100000):
    """Generates a sample numerical data file for analysis if it does not exist."""
    if not os.path.exists(filepath):
        print(f"Info: '{filepath}' not found. Generating sample data.")
        random.seed(int(time.time()))
        try:
            with open(filepath, 'w') as f:
                f.write(''.join([str(random.randint(0, 9)) for _ in range(num_digits)]))
            print(f"Sample data generated in '{filepath}'.")
        except IOError as e:
            print(f"Error: Failed to create sample file: {e}")

def load_full_data(filepath: str) -> list[int]:
    """Reads numerical data from a file and returns it as a list of integers."""
    try:
        with open(filepath, 'r') as f:
            raw_data = f.read().strip()
        if not raw_data.isdigit():
            raise ValueError("File contains non-numeric characters.")
        return [int(char) for char in raw_data]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return []

def get_canonical_form(digits: list[int]) -> int:
    """Converts a 3-digit list into its canonical form (a number sorted in ascending order)."""
    s = sorted(digits)
    return s[0] * 100 + s[1] * 10 + s[2]

def get_permutation_form(digits: list[int]) -> int:
    """Converts a 3-digit list into a number while maintaining its order."""
    return digits[0] * 100 + digits[1] * 10 + digits[2]
    
def parse_filter_range(filter_str: str) -> tuple[int, int]:
    """Parses a string like ' 45 - 49 items' into a tuple (45, 49)."""
    numbers = re.findall(r'\d+', filter_str)
    if len(numbers) == 2:
        return int(numbers[0]), int(numbers[1])
    return 0, 0

# ==============================================================================
# 3. Analysis Pipeline (Refactored for speed)
# ==============================================================================

def build_lookup_worker(chunk_with_args):
    """[New] Worker function to build a part of the lookup table in parallel."""
    chunk, r = chunk_with_args
    local_lookup = collections.defaultdict(collections.Counter)
    scan_limit = len(chunk) - (r + 3) + 1
    if scan_limit <= 0:
        return local_lookup

    for i in range(scan_limit):
        current_r_digits = chunk[i:i+r]
        if r < 3: continue
        
        future_3_digits_val = get_permutation_form(chunk[i+r : i+r+3])
        
        for p in itertools.permutations(current_r_digits, 3):
            fetch = get_permutation_form(list(p))
            local_lookup[fetch][future_3_digits_val] += 1
            
    return local_lookup

def build_pattern_lookup(data: list[int], r: int) -> dict:
    """[New] Scans the entire dataset once to build a pattern lookup table."""
    if r < 3:
        print(f"Analysis skipped for r={r} because it is less than 3.")
        return {}

    print("Building pattern lookup table...", end="", flush=True)
    start_time = time.time()
    
    lookup_table = collections.defaultdict(collections.Counter)
    
    chunk_size = len(data) // PROCESSES
    overlap = r + 2
    tasks = []
    for i in range(PROCESSES):
        start = i * chunk_size
        end = (i + 1) * chunk_size + overlap if i < PROCESSES - 1 else len(data)
        if start < end:
            tasks.append((data[start:end], r))
            
    with Pool(processes=PROCESSES) as pool:
        results = pool.map(build_lookup_worker, tasks)
        for local_lookup in results:
            for fetch, counter in local_lookup.items():
                lookup_table[fetch].update(counter)
                
    end_time = time.time()
    print(f" Done ({end_time - start_time:.2f}s)")
    return lookup_table

def get_aggregated_counts_from_lookup(recent_numbers: list[int], r: int, lookup_table: dict) -> collections.Counter:
    """[New] Quickly retrieves subsequent pattern counters using the lookup table."""
    if r < 3: return collections.Counter()
    
    aggregated_counter = collections.Counter()
    recent_permutations = {get_permutation_form(list(p)) for p in itertools.permutations(recent_numbers, 3)}
    
    for fetch in recent_permutations:
        if fetch in lookup_table:
            aggregated_counter.update(lookup_table[fetch])
            
    return aggregated_counter

def generate_prediction_list(n3_counter: collections.Counter) -> list[int]:
    """Generates a prediction list based on the aggregated counter."""
    if not n3_counter: return []
    y_counters = [collections.Counter() for _ in range(3)]
    for num, count in n3_counter.items():
        s_num = str(num).zfill(3)
        y_counters[0][int(s_num[0])] += count
        y_counters[1][int(s_num[1])] += count
        y_counters[2][int(s_num[2])] += count
    top_groups, bottom_groups = [], []
    for y_counter in y_counters:
        if len(y_counter) < 6: return []
        sorted_digits = [item[0] for item in y_counter.most_common(10)]
        top_groups.append(sorted_digits[:3])
        bottom_groups.append(sorted_digits[-3:])
    top_combinations = {(d1, d2, d3) for d1 in top_groups[0] for d2 in top_groups[1] for d3 in top_groups[2]}
    bottom_combinations = {(d1, d2, d3) for d1 in bottom_groups[0] for d2 in bottom_groups[1] for d3 in bottom_groups[2]}
    normalized_candidates = set()
    for top_combo in top_combinations:
        for bottom_combo in bottom_combinations:
            combined_digits = list(set(top_combo + bottom_combo))
            if len(combined_digits) < 3: continue
            for combo in itertools.combinations(combined_digits, 3):
                if combo[0] == combo[1] == combo[2]: continue
                normalized_candidates.add(get_canonical_form(list(combo)))
    return sorted(list(normalized_candidates))

def analyze_binned_results(results_log: list, r_value: int) -> list[dict]:
    """Analyzes the backtest log by binning results."""
    if not results_log: return []
    step = 5
    binned_results = collections.defaultdict(lambda: {'trials': 0, 'hits': 0})
    for result in results_log:
        count = result['count']
        if count >= 3000: continue
        bin_key = (count // step) * step
        binned_results[bin_key]['trials'] += 1
        binned_results[bin_key]['hits'] += result['hit']
    analyzed_spots = []
    min_trials_for_reliability = len(results_log) * 0.05 if len(results_log) > 0 else 0
    for lower_bound in sorted(binned_results.keys()):
        stats = binned_results[lower_bound]
        trials = stats['trials']
        if trials < min_trials_for_reliability: continue
        hits = stats['hits']
        hit_rate = hits / trials if trials > 0 else 0.0
        spot_data = {'r': r_value, 'filter': f"{lower_bound:>4} - {lower_bound + step - 1:<4} items", 'trials': trials, 'hits': hits, 'hit_rate': hit_rate}
        analyzed_spots.append(spot_data)
    analyzed_spots.sort(key=lambda x: x['hit_rate'], reverse=True)
    return analyzed_spots

# ==============================================================================
# 5. Main Execution Block (Modified for high-speed logic)
# ==============================================================================
def main():
    main_start_time = time.time()
    setup_data_file(FILE_PATH)
    full_data = load_full_data(FILE_PATH)
    if len(full_data) < 100:
        print("Error: Data is too short for analysis.")
        return

    split_point = int(len(full_data) * TRAINING_DATA_RATIO)
    training_data = full_data[:split_point]
    testing_data = full_data[split_point:]
    print(f"Data split: Training = {len(training_data)} digits, Testing = {len(testing_data)} digits")

    for r_value in R_VALUES_TO_TEST:
        print("\n" + "="*30 + f" Starting analysis for r = {r_value} " + "="*30)
        
        # [New] Build the lookup table once from the training data
        pattern_lookup = build_pattern_lookup(training_data, r_value)

        # --- Stage 1: Discover the most effective model (filter) from training data ---
        print("\n--- Stage 1: Discovering model from training data ---")
        min_len_required = r_value + 3
        if len(training_data) < min_len_required + 3:
            print(f"Training data is too short for r={r_value}. Skipping.")
            continue
        
        max_iterations = min(TRAINING_ITERATIONS, (len(training_data) - min_len_required) // 3)
        results_log = []
        
        for i in range(max_iterations):
            end_index = len(training_data) - i * 3
            recent_numbers = training_data[end_index - 3 - r_value : end_index - 3]
            future_value_canonical = get_canonical_form(training_data[end_index - 3 : end_index])
            
            progress = (i + 1) / max_iterations
            print(f"\rTraining... [{i+1}/{max_iterations}] ({progress:.1%})", end="")

            # [!] Call the fast lookup function
            n3_counter = get_aggregated_counts_from_lookup(recent_numbers, r_value, pattern_lookup)
            prediction_list = generate_prediction_list(n3_counter)
            
            is_hit = 1 if future_value_canonical in prediction_list else 0
            results_log.append({'count': len(prediction_list), 'hit': is_hit})

        print("\nTraining complete. Most effective model found:")
        all_spots = analyze_binned_results(results_log, r_value)
        if not all_spots:
            print("No significant models found from training data."); continue
        
        best_model = all_spots[0]
        lower_bound, upper_bound = parse_filter_range(best_model['filter'])
        print(f"  Optimal Filter: {best_model['filter']}, Hit Rate: {best_model['hit_rate']:.2%}")

        # --- Stage 2: Verify the model's reproducibility with unseen test data ---
        print("\n--- Stage 2: Verifying reproducibility with test data ---")
        test_trials = 0
        test_hits = 0
        num_test_points = (len(testing_data) - 3) // 3
        if num_test_points <= 0:
            print("Test data is too short, skipping Stage 2."); continue

        for i in range(num_test_points):
            current_pos = i * 3
            recent_numbers_for_test = full_data[split_point + current_pos - r_value : split_point + current_pos]
            future_value_list = testing_data[current_pos : current_pos + 3]
            if len(future_value_list) < 3: continue

            progress = (i + 1) / num_test_points
            print(f"\rTesting... [{i+1}/{num_test_points}] ({progress:.1%})", end="")

            # [!] Make predictions using the pre-built lookup table
            n3_counter = get_aggregated_counts_from_lookup(recent_numbers_for_test, r_value, pattern_lookup)
            prediction_list = generate_prediction_list(n3_counter)
            prediction_count = len(prediction_list)

            if lower_bound <= prediction_count <= upper_bound:
                test_trials += 1
                future_value_canonical = get_canonical_form(future_value_list)
                if future_value_canonical in prediction_list:
                    test_hits += 1
        
        print("\nTesting complete.")
        test_hit_rate = test_hits / test_trials if test_trials > 0 else 0.0

        # --- Final Results Comparison ---
        print("\n" + "-"*25 + f" Final Results for r = {r_value} " + "-"*25)
        print("| Stage    | Filter Range        | Trials   | Hits   | Hit Rate |")
        print("|:---------|:--------------------|---------:|-------:|---------:|")
        print(f"| Training | {best_model['filter']} | {best_model['trials']:>8} | {best_model['hits']:>6} | {best_model['hit_rate']:>8.2%} |")
        print(f"| Testing  | {best_model['filter']} | {test_trials:>8} | {test_hits:>6} | {test_hit_rate:>8.2%} |")
        
        if test_trials < 30:
            print("\n[Conclusion] Cannot draw a statistically reliable conclusion because the number of test trials is less than 30.")
        elif test_hit_rate > best_model['hit_rate'] * 0.8 :
             print("\n[Conclusion] Reproducibility confirmed! The model's performance was maintained on unseen data.")
        elif test_hit_rate > 0.05:
             print("\n[Conclusion] Reproducibility is limited. Performance degraded, but a signal stronger than random may still exist.")
        else:
             print("\n[Conclusion] No reproducibility. The model did not work on unseen data and was likely a phantom (overfitting).")

    main_end_time = time.time()
    print(f"\n\nTotal processing time: {main_end_time - main_start_time:.2f} seconds")
    print("=" * 72)

if __name__ == '__main__':
    freeze_support()
    main()
