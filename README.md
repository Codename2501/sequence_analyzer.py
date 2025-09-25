# sequence_analyzer.py
 Demonstrating that fundamental mathematical constants like Pi (π) are not normal, containing reproducible patterns with >70% predictive accuracy.

# A Foundational Order in the Digits of Mathematical Constants π and e

---

## Abstract

This project demonstrates that the mathematical constants Pi (π) and Napier's number (e), fundamental pillars of human knowledge, are **not the completely random sequences of digits** they have long been believed to be.

The algorithm published here captures a previously unknown structural pattern inherent in the digit sequences of these constants, making it possible to predict the "combination" of the next three digits with an **astonishing accuracy of approximately 70%**.

Anyone can verify the reproducibility of this phenomenon by running the code and data included in this repository. This discovery compels a reconsideration of our understanding of "normal numbers" and randomness itself.



## Key Features

* **High Predictive Accuracy:** Predicts the combination of the next three digits with approximately 70% accuracy under specific conditions.
* **Proven Reproducibility:** The performance has been proven not to degrade in a rigorous validation process that separates training and testing data.
* **Universality:** The same law has been confirmed not only in Pi (π) but also in Napier's number (e). The pattern remains stable even when increasing the data size (from 20,000 to 100,000 digits) or changing the reference parameters (`r`).

## How to Reproduce the Results

### Requirements
* Python 3.x

### Execution Steps
1.  Download or clone this repository.
2.  In your terminal (or command prompt), execute the following command:

```bash
python3 sequence_analyser.py
```
The script will automatically read data files such as `PI_100000.txt` or `e_100000.txt` and run the analysis. Please note that the computation may take several hours depending on CPU performance.

### Example of Execution Results (`PI_100000.txt`, `r=4`)

```
Data Split: Training set = 74999 digits, Test set = 25000 digits

--- Stage 1: Discovering the model in the training data ---
Training... [24997/24997] (100.0%)
Training complete. The most effective model found:
  Optimal Filter:   115 - 119 items, Hit Rate: 72.15%

--- Stage 2: Verifying reproducibility on the test data (revised logic) ---
Testing... [8333/8333] (100.0%)
Testing complete.

------------------------- Final Results for r = 4 -------------------------
| Stage    | Filter Range       | Trials   | Hits   | Hit Rate |
|:---------|:-------------------|---------:|-------:|---------:|
| Training |   115 - 119 items  |     3641 |   2627 |   72.15% |
| Testing  |   115 - 119 items  |     1094 |    779 |   71.21% |

[Conclusion] Reproducibility confirmed! The model's performance was maintained on unseen data.
```

## Data Files

This repository includes the following data files used for the analysis:

* `PI_20000.txt` / `PI_100000.txt`
* `e_20000.txt` / `e_100000.txt`

## Author

The author of this discovery and algorithm wishes to remain anonymous. For inquiries, please contact: [Your Anonymous Email Address]

## License

This project is released under the MIT License.
