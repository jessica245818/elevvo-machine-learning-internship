# Task 9: Industrial Predictive Maintenance

This project predicts industrial machine failures from sensor readings while
explicitly minimizing unnecessary factory-stop alarms.

## Dataset

The analysis uses the
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
from the UCI Machine Learning Repository. It contains 10,000 synthetic
operations with product type, air and process temperatures, rotational speed,
torque, tool wear, an overall machine-failure flag, and five failure modes:

- Tool Wear Failure (TWF)
- Heat Dissipation Failure (HDF)
- Power Failure (PWF)
- Overstrain Failure (OSF)
- Random Failure (RNF)

The script downloads the official UCI archive automatically.

## Industrial evaluation design

Accuracy is misleading because failures are rare. The project therefore:

1. Creates independent stratified training, validation, and test partitions.
2. Uses cost-sensitive Logistic Regression, Random Forest, and XGBoost models.
3. Engineers temperature difference, mechanical power, tool stress, and a
   torque-speed ratio.
4. Tunes each alarm threshold **only on validation data**.
5. Minimizes False Discovery Rate (`FP / (TP + FP)`) while requiring at least
   50% validation recall.
6. Evaluates the selected operating point once on the held-out test set.
7. Trains separate multi-label Random Forest diagnostic models for the five
   failure types.

The recall constraint prevents a technically perfect-looking false-discovery
rate obtained by suppressing almost every alarm.

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python task9_predictive_maintenance/src/predictive_maintenance.py
```

Use a stricter failure-recall requirement:

```bash
python task9_predictive_maintenance/src/predictive_maintenance.py \
  --minimum-recall 0.70
```

## Results

The dataset contains 339 failures among 10,000 operations (3.39%). Random
Forest was selected from validation performance at a probability threshold of
0.81. On the untouched 2,000-row test set:

| Metric | Result |
| --- | ---: |
| False Discovery Rate | **4.76%** |
| Precision | **95.24%** |
| Recall | 58.82% |
| F1 | 72.73% |
| Accuracy | 98.50% |
| ROC AUC | 98.11% |
| Average precision | 85.40% |
| False alarms | **2** |
| Correct failure alarms | 40 |
| Missed failures | 28 |

The tuned model issued 42 alarms, of which only two were false. For comparison,
cost-sensitive Logistic Regression produced 30 false alarms and a 50% FDR at
its validation-selected threshold. XGBoost happened to produce no false alarms
on the test set, but Random Forest remains the selected model because it had the
lowest validation FDR; choosing XGBoost after viewing the test results would be
test leakage.

### Failure-type diagnosis

| Failure type | Test cases | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Heat dissipation | 29 | 90.62% | 100.00% | 95.08% |
| Power | 13 | 100.00% | 100.00% | 100.00% |
| Overstrain | 16 | 84.21% | 100.00% | 91.43% |
| Tool wear | 10 | 2.94% | 10.00% | 4.55% |
| Random | 4 | 0.00% | 0.00% | 0.00% |

The rare Tool Wear and Random classes are not reliable with this sample size;
the project reports these failures directly rather than presenting only the
strong diagnostic classes.

### Sensor associations

The engineered torque-speed ratio has the strongest absolute association with
failure (`r = 0.206`). Among raw sensors, torque is strongest (`r = 0.191`),
followed by tool wear (`r = 0.105`) and air temperature (`r = 0.083`).
Correlation identifies association, not a temporal “lead” signal: the dataset
has no time axis from which lead/lag behavior can be established.

## Time-to-failure bonus

The project does not invent a remaining-useful-life target. AI4I rows represent
independent synthetic products rather than timestamped histories for individual
machines. Treating row order as a machine timeline would create misleading
time-to-failure labels. A valid RUL model requires asset IDs, timestamps, and
run-to-failure sequences.

## Generated artifacts

- Low-FDR model comparison and selected-model classification report
- Held-out probabilities and alarm decisions
- Per-failure-type diagnostic metrics
- Sensor/failure correlation analysis
- Class-balance, FDR, precision-recall, and correlation figures
- Serialized alarm and failure-type models
