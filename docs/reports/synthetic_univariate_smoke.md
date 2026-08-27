# synthetic_univariate_smoke

Generated engineering-validation record. Synthetic smoke results are not leaderboard evidence.

```json
{
  "experiment_id": "synthetic_univariate_smoke",
  "task": "univariate_alarm_design",
  "result": {
    "parameters": {
      "threshold": 0.5,
      "delay": 8,
      "deadband": 0.25,
      "direction": "high"
    },
    "design_loss": 0.2157142857142857,
    "metrics": {
      "precision": 0.9940119760479041,
      "recall": 0.996,
      "f1": 0.995004995004995,
      "false_alarm_rate": 0.004285714285714286,
      "missed_alarm_rate": 0.004,
      "average_alarm_delay": 2.0
    },
    "samples": 1200,
    "warning": "Smoke experiment tunes and evaluates on one synthetic run; it is not a leaderboard result."
  },
  "config": "configs/experiments/synthetic_univariate_smoke.json"
}
```
