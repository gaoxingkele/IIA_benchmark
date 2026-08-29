# Experiments

No paper-score reproduction is asserted by collection generation alone. Add frozen dataset, grouped split, seeds, hyperparameters, and target metrics here after execution.

## Local FCC engineering validation (E2/P1)

- Dataset/split: official FCC activation sequences, complete-run 60/20/20 split for all 16 scenarios.
- Method: maximum-entropy next-alarm model with time-distance features.
- Result: 3,550 evaluated transitions, vocabulary coverage 0.919927, top-1 accuracy 0.187324, top-3 accuracy 0.412958, NLL 3.155777, and Brier score 0.017883.
- Evidence: `configs/experiments/fcc_max_entropy_next_alarm_validation.json` and `experiments/reports/fcc_max_entropy_next_alarm_validation.json`.
- Boundary: simultaneous alarms require a declared deterministic ordering; the source paper event log, exact constraints, and paper score remain gated.

## Table 5.15 and three-dataset grouped validation (E3/P1)

- The exact single-constraint solver returns multipliers -1.0414/2.0794/-2.2513 and analytical `P(x4)=0.8`, within 0.0001 of the printed 0.7999.
- Mean Top-1 on TEP/NPP/FCC is 0.1109/0.0881/0.2209. NPP is far below the global-frequency baseline 0.4976; FCC exceeds its 0.0918 control.
- Macro-F1 eta surrogates are 0.0528/0.0376/0.0742, so all nine units fail the book's effectiveness gate 0.8 despite nonuniform mechanism output.
- Evidence: `experiments/reports/book_ch5_multidataset_validation.json`.
- Boundary: the 26 historical TEP floods, Tables 5.18-5.22, and the 100-run Monte Carlo sample-size curves remain unavailable.
