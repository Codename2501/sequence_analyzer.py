import os
import time
import random
import collections
import itertools
import re
from multiprocessing import Pool, cpu_count, freeze_support

# ==============================================================================
# 2. Main Parameters
# ==============================================================================
# Input file containing the numerical data for analysis
FILE_PATH = "PI_20000.txt"
# Number of CPU cores to use for parallel processing
PROCESSES = max(1, cpu_count() - 1) if cpu_count() and cpu_count() > 1 else 1
# List of r-values to use for comparative analysis
R_VALUES_TO_TEST = [3,4,5]

# Split ratio for training and testing data
TRAINING_DATA_RATIO = 0.75

# Maximum number of backtest iterations to run in the learning phase
TRAINING_ITERATIONS = 15000

# ==============================================================================
# Helper Functions (No changes)
# ==============================================================================

def setup_data_file(filepath: str, num_digits: int = 20000):
    """Generates a numerical data file for analysis if it does not exist."""
    if not os.path.exists(filepath):
        print(f"Info: '{filepath}' not found. Generating sample data.")
        random.seed(int(time.time()))
        try:
            with open(filepath, 'w') as f:
                f.write(''.join([str(random.randint(0, 9)) for _ in range(num_digits)]))
            print(f"Sample data generated in '{filepath}'.")
        except IOError as e:
            print(f"Error: Failed to create the sample file: {e}")

def load_full_data(filepath: str) -> list[int]:
    """Reads numerical data from a file and returns it as a list of integers."""
    try:
        with open(filepath, 'r') as f:
            raw_data = f.read().strip()
        if not raw_data.isdigit():
            raise ValueError("The file contains non-numeric characters.")
        return [int(char) for char in raw_data]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return []

def get_canonical_form(digits: list[int]) -> int:
    """Converts a list of 3 digits into its canonical form (an integer sorted in ascending order)."""
    s = sorted(digits)
    return s[0] * 100 + s[1] * 10 + s[2]

def parse_filter_range(filter_str: str) -> tuple[int, int]:
    """Parses a string like ' 45 - 49 items' and returns a tuple (45, 49)."""
    numbers = re.findall(r'\d+', filter_str)
    if len(numbers) == 2:
        return int(numbers[0]), int(numbers[1])
    return 0, 0

# ==============================================================================
# 3. Analysis Pipeline (Aggregation logic modified)
# ==============================================================================
def worker_task(args):
    chunk, r, recent_set = args
    local_n3_counter = collections.Counter()
    scan_limit = len(chunk) - (r + 3) + 1
    if scan_limit <= 0: return local_n3_counter
    for i in range(scan_limit):
        current_r_digits = chunk[i:i+r]
        if r < 3: continue
        for p in itertools.permutations(current_r_digits, 3):
            fetch = p[0] * 100 + p[1] * 10 + p[2]
            if fetch in recent_set:
                future_3_digits = chunk[i+r : i+r+3]
                fetch2 = future_3_digits[0] * 100 + future_3_digits[1] * 10 + future_3_digits[2]
                local_n3_counter[fetch2] += 1
    return local_n3_counter

def run_pattern_aggregation(analysis_data: list[int], r: int, recent_numbers: list[int]) -> collections.Counter:
    if r < 3 or len(analysis_data) < r: return collections.Counter()
    # Create a matching pattern set (recent_set) from the r digits immediately preceding the prediction
    recent_set = {p[0] * 100 + p[1] * 10 + p[2] for p in itertools.permutations(recent_numbers, 3)}

    chunk_size = len(analysis_data) // PROCESSES
    overlap = r + 2
    tasks = []
    for i in range(PROCESSES):
        start = i * chunk_size
        end = (i + 1) * chunk_size + overlap if i < PROCESSES - 1 else len(analysis_data)
        if start < end:
            tasks.append((analysis_data[start:end], r, recent_set))

    aggregated_counter = collections.Counter()
    with Pool(processes=PROCESSES) as pool:
        results = pool.map(worker_task, tasks)
        for counter in results:
            aggregated_counter.update(counter)
    return aggregated_counter

def generate_prediction_list(n3_counter: collections.Counter) -> list[int]:
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
    min_trials_for_reliability = len(results_log) * 0.05
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
# 5. Main Execution Block (★★★★★ [REVISED LOGIC] ★★★★★)
# ==============================================================================
def main():
    """Main execution function: verifies reproducibility in two stages (learning and testing)."""
    main_start_time = time.time()
    setup_data_file(FILE_PATH)
    full_data = load_full_data(FILE_PATH)
    if len(full_data) < 100:
        print("Error: Data is too short for analysis.")
        return

    split_point = int(len(full_data) * TRAINING_DATA_RATIO)
    training_data = full_data[:split_point]
    testing_data = full_data[split_point:]
    print(f"Data split: Training set = {len(training_data)} digits, Test set = {len(testing_data)} digits")

    for r_value in R_VALUES_TO_TEST:
        print("\n" + "="*30 + f" Starting analysis for r = {r_value} " + "="*30)
        
        # --- Stage 1: Discover the most effective model (filter) in the training data ---
        print("\n--- Stage 1: Discovering model in training data ---")
        min_len_required = r_value + 3
        if len(training_data) < min_len_required + 3:
            print(f"Training data is too short for r={r_value}. Skipping.")
            continue
        
        max_iterations = min(TRAINING_ITERATIONS, (len(training_data) - min_len_required) // 3)
        results_log = []
        
        # Backtesting in the learning phase
        for i in range(max_iterations):
            end_index = len(training_data) - i * 3
            analysis_data_for_training = training_data[:end_index-3]
            # The r digits immediately preceding the prediction, used as a trigger
            recent_numbers_for_training = training_data[end_index-3-r_value:end_index-3]
            future_value_list = training_data[end_index-3:end_index]
            
            future_value_canonical = get_canonical_form(future_value_list)
            progress = (i + 1) / max_iterations
            print(f"\rLearning... [{i+1}/{max_iterations}] ({progress:.1%})", end="")
            
            n3_counter = run_pattern_aggregation(analysis_data_for_training, r_value, recent_numbers_for_training)
            prediction_list = generate_prediction_list(n3_counter)
            is_hit = 1 if future_value_canonical in prediction_list else 0
            results_log.append({'count': len(prediction_list), 'hit': is_hit})

        print("\nLearning complete. The most effective model found:")
        all_spots = analyze_binned_results(results_log, r_value)
        
        if not all_spots:
            print("No significant model could be found from the training data.")
            continue
        
        best_model = all_spots[0]
        lower_bound, upper_bound = parse_filter_range(best_model['filter'])
        print(f"  Optimal Filter: {best_model['filter']}, Hit Rate: {best_model['hit_rate']:.2%}")

        # --- Stage 2: Verify the model's reproducibility on unseen test data ---
        print("\n--- Stage 2: Verifying reproducibility on test data (revised logic) ---")
        test_trials = 0
        test_hits = 0
        
        # ★★★ [MODIFICATION] The data used for pattern analysis is fixed to the 'training_data' ★★★
        test_analysis_base_data = training_data 
        
        # Calculate the number of testable points
        num_test_points = (len(full_data) - split_point) // 3
        if num_test_points <= 0:
            print("Not enough test data to proceed. Skipping Stage 2.")
            continue

        for i in range(num_test_points):
            # Index of the point to be predicted
            current_pos = split_point + (i * 3)
            
            # Get the r digits immediately preceding the prediction
            recent_numbers_for_test = full_data[current_pos - r_value : current_pos]

            # The 3 digits to be predicted
            future_value_list = full_data[current_pos : current_pos + 3]
            if len(future_value_list) < 3: continue

            progress = (i + 1) / num_test_points
            print(f"\rTesting... [{i+1}/{num_test_points}] ({progress:.1%})", end="")

            # ★★★ [MODIFICATION] Analysis is always performed on the training data, passing only the recent r-digits info ★★★
            n3_counter = run_pattern_aggregation(test_analysis_base_data, r_value, recent_numbers_for_test)
            prediction_list = generate_prediction_list(n3_counter)
            prediction_count = len(prediction_list)

            # Check if it matches the filter range from the discovered model
            if lower_bound <= prediction_count <= upper_bound:
                test_trials += 1
                future_value_canonical = get_canonical_form(future_value_list)
                if future_value_canonical in prediction_list:
                    test_hits += 1
        
        print("\nTesting complete.")
        test_hit_rate = test_hits / test_trials if test_trials > 0 else 0.0

        # --- Comparison of Final Results ---
        print("\n" + "-"*25 + f" Final Results for r = {r_value} " + "-"*25)
        print("| Stage    | Filter Range        | Trials   | Hits   | Hit Rate |")
        print("|:---------|:--------------------|---------:|-------:|---------:|")
        print(f"| Training | {best_model['filter']} | {best_model['trials']:>8} | {best_model['hits']:>6} | {best_model['hit_rate']:>8.2%} |")
        print(f"| Testing  | {best_model['filter']} | {test_trials:>8} | {test_hits:>6} | {test_hit_rate:>8.2%} |")
        
        if test_trials < 30:
            print("\n[Conclusion] The number of test trials is less than 30, so no statistically reliable conclusion can be drawn.")
        elif test_hit_rate > best_model['hit_rate'] * 0.8 :
             print("\n[Conclusion] Reproducibility confirmed! The model's performance was maintained on unseen data.")
        elif test_hit_rate > 0.05:
             print("\n[Conclusion] Reproducibility is limited. Performance decreased, but a signal stronger than random may still exist.")
        else:
             print("\n[Conclusion] No reproducibility. The model did not work on unseen data and was likely a 'phantom' (overfitting).")


    main_end_time = time.time()
    print(f"\n\nTotal processing time: {main_end_time - main_start_time:.2f} seconds")
    print("=" * 72)

if __name__ == '__main__':
    freeze_support()
    main()
