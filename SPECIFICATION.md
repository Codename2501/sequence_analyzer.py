# Algorithm Specification: Discovery of a Predictable Structure in Mathematical Constants

## 1. Objective

This document defines the technical details of the algorithm implemented in the Python script `PIanalysis.py`. The objective of this algorithm is to construct a reproducible model that predicts the combination of the next three digits from the digit sequences of Pi (π) and Napier's number (e).

## 2. Definition of Prediction Target

This algorithm predicts the **"combination" of three digits, where the order does not matter.**

* **Implementation:** `get_canonical_form(digits: list[int])` function.
* **Specification:** Takes a list of three integers `[d1, d2, d3]` as input, sorts them in ascending order, and converts them into a single integer value `s1*100 + s2*10 + s3`. This converted value is referred to as the **"Canonical Form"**.
* **Example:** An input of `[8, 0, 5]` is sorted to `[0, 5, 8]` and treated as the canonical form `58`. Similarly, `[5, 8, 0]` is also treated as `58`.

## 3. Analysis Pipeline Details

The analysis is executed in two stages as defined in the `main()` function.

### 3.1. Stage 1: Model Discovery (Learning Phase)

**Objective:** To discover the most effective prediction condition (the "filter") by conducting backtests within the training dataset.

**Process:**
1.  **Data Slicing:** The process loops backward from the end of the training data for the number of times specified by `TRAINING_ITERATIONS`. In each loop, the target three digits and their preceding `r` digits are sliced.
2.  **Pattern Aggregation (`run_pattern_aggregation`):**
    * Generates all permutations of three digits from the sliced "preceding `r` digits" and defines this as the "matching pattern set".
    * Scans earlier parts of the training data in parallel (`worker_task`) to find occurrences that match the "matching pattern set".
    * Converts the three digits immediately following each match into their canonical form and aggregates their frequency of occurrence (`n3_counter`).
3.  **Prediction List Generation (`generate_prediction_list`):**
    * Generates a list of candidate canonical forms based on the aggregated `n3_counter`.
4.  **Result Logging:**
    * Records a pair of values for each iteration: the number of candidates in the generated prediction list ("count") and whether the actual future value was included in the list ("hit" or "miss"). This is stored in `results_log`.
5.  **Optimal Model Identification (`analyze_binned_results`):**
    * The `results_log` is grouped (binned) by the number of candidates in the prediction list.
    * The hit rate for each group is calculated, and the group with the highest hit rate is identified as the "optimal filter".

### 3.2. Stage 2: Reproducibility Verification (Testing Phase)

**Objective:** To verify that the "optimal filter" discovered in Stage 1 is also effective on unseen test data.

**Process:**
1.  **Data Slicing:** The process loops forward through the test dataset, slicing it into three-digit segments.
2.  **Pattern Aggregation (`run_pattern_aggregation`):**
    * Generates a "matching pattern set" using the "preceding `r` digits" sliced from the test data.
    * **The dataset used for analysis is always fixed to the entire training dataset.** The full training data is scanned to find occurrences matching the pattern set derived from the test data, and the subsequent three digits are aggregated.
3.  **Prediction List Generation (`generate_prediction_list`):**
    * Generates a prediction list from the aggregated results.
4.  **Verification:**
    * An iteration is counted as a "trial" only if the number of candidates in the generated prediction list falls within the range of the "optimal filter" found in Stage 1.
    * A "trial" is counted as a "hit" if the actual future value is included in the prediction list.
5.  **Final Evaluation:**
    * The hit rates from the learning phase and the testing phase are compared to evaluate the model's reproducibility.

## 4. Key Parameters

* `FILE_PATH`: The filename of the data file to be analyzed.
* `R_VALUES_TO_TEST`: A list of `r` values (the number of preceding digits) to be used for prediction.
* `TRAINING_DATA_RATIO`: The ratio used to split the full dataset into training and testing sets.
* `TRAINING_ITERATIONS`: The maximum number of backtest iterations to run during the learning phase.
