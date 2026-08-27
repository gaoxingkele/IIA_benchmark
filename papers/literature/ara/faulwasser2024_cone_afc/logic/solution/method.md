# Method

ConE-AFC wraps online alarm-flood classifiers with conformal calibration and returns prediction sets at controlled error levels.

For every expanding window and class, Algorithm 1 sorts held-out true-class probability scores in descending order (or distance scores ascending) and applies the finite-sample thresholds in Eqs. (2)-(3). Algorithm 2 includes a class when its online score passes the corresponding class/window threshold, as specified by Eqs. (4)-(5). Eq. (1) reports expected finite-calibration coverage from alpha, delta, and epsilon.
