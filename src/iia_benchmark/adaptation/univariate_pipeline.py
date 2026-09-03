"""End-to-end calibrated univariate transfer for a registered dataset."""

from __future__ import annotations

import numpy as np

from ..data import UnivariateTransferBundle
from ..evaluation import (
    ApplicabilityThresholds,
    alarm_event_metrics,
    assess_univariate_calibration,
    audit_univariate_partitions,
    binary_alarm_metrics,
    block_event_rate_posterior,
    block_bootstrap_alarm_metrics,
)
from ..models import BlockCalibratedECDFAlarm, EmpiricalCDFAlarm


def run_univariate_transfer(
    bundle: UnivariateTransferBundle,
    adaptation: dict[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    """Audit, route, fit, and evaluate one registered univariate dataset."""

    gate_values = adaptation["applicability_thresholds"]
    thresholds = ApplicabilityThresholds(
        normal_ks_adaptation=float(gate_values["normal_ks_adaptation"]),
        normal_median_shift_sd_adaptation=float(
            gate_values["normal_median_shift_sd_adaptation"]
        ),
        autocorrelation_block_calibration=float(
            gate_values["autocorrelation_block_calibration"]
        ),
        minimum_block_auc=float(gate_values["minimum_block_auc"]),
        minimum_direction_consistency=float(
            gate_values["minimum_direction_consistency"]
        ),
        chronological_blocks=int(gate_values["chronological_blocks"]),
    )
    applicability = assess_univariate_calibration(
        bundle.normal_train,
        bundle.abnormal_calibration,
        thresholds=thresholds,
    )
    posthoc = audit_univariate_partitions(
        bundle.normal_train,
        bundle.normal_evaluation,
        bundle.abnormal_calibration,
        bundle.abnormal_evaluation,
        direction=applicability.direction,
    )
    result: dict[str, object] = {
        "dataset_id": bundle.dataset_id,
        "feature_name": bundle.feature_name,
        "sample_period_seconds": bundle.sample_period_seconds,
        "leaderboard_eligible": bundle.leaderboard_eligible,
        "citation": bundle.citation,
        "partition_sources": bundle.partition_sources,
        "calibration_applicability": applicability.as_dict(),
        "held_out_posthoc_audit": posthoc.as_dict(),
    }
    if applicability.status == "reject_univariate":
        result.update(
            {
                "status": "denied_univariate",
                "selected_model": None,
                "empirical": None,
                "event_metrics": None,
                "block_bootstrap": None,
                "block_event_rate_posterior": None,
            }
        )
        return result

    model_values = adaptation["model"]
    if applicability.status == "adapt":
        model = BlockCalibratedECDFAlarm(
            tail_probability=float(model_values["tail_probability"]),
            tail=applicability.direction,
            reference_windows=tuple(
                int(value) for value in model_values["reference_windows"]
            ),
            delays=tuple(int(value) for value in model_values["delays"]),
            validation_fraction=float(model_values["validation_fraction"]),
            block_size=int(model_values["block_size"]),
            target_point_false_alarm_rate=float(
                model_values["target_point_false_alarm_rate"]
            ),
            target_block_alarm_rate=float(model_values["target_block_alarm_rate"]),
            block_weight=float(model_values["block_weight"]),
        ).fit(bundle.normal_train)
        selected_model = "block_calibrated_ecdf"
        model_parameters = {
            "reference_window": model.selected_reference_window_,
            "delay": model.selected_delay_,
        }
    else:
        model = EmpiricalCDFAlarm(
            tail_probability=float(model_values["tail_probability"]),
            tail=applicability.direction,
            delay=int(model_values["static_delay"]),
        ).fit(bundle.normal_train)
        selected_model = "static_ecdf"
        model_parameters = {"delay": int(model_values["static_delay"])}
    normal_alarm = model.predict(bundle.normal_evaluation)
    abnormal_alarm = model.predict(bundle.abnormal_evaluation)
    truth = np.r_[
        np.zeros(len(normal_alarm), dtype=bool),
        np.ones(len(abnormal_alarm), dtype=bool),
    ]
    uncertainty = adaptation["uncertainty"]
    event_posterior = {
        "normal_false_alarm_blocks": block_event_rate_posterior(
            normal_alarm,
            block_size=int(uncertainty["block_size"]),
            confidence=float(uncertainty["confidence"]),
            prior_alpha=float(uncertainty.get("event_prior_alpha", 1.0)),
            prior_beta=float(uncertainty.get("event_prior_beta", 1.0)),
        ).as_dict(),
        "abnormal_detection_blocks": block_event_rate_posterior(
            abnormal_alarm,
            block_size=int(uncertainty["block_size"]),
            confidence=float(uncertainty["confidence"]),
            prior_alpha=float(uncertainty.get("event_prior_alpha", 1.0)),
            prior_beta=float(uncertainty.get("event_prior_beta", 1.0)),
        ).as_dict(),
    }
    result.update(
        {
            "status": "scored",
            "selected_model": selected_model,
            "model_parameters": model_parameters,
            "empirical": binary_alarm_metrics(
                truth, np.r_[normal_alarm, abnormal_alarm]
            ),
            "event_metrics": alarm_event_metrics(
                normal_alarm,
                abnormal_alarm,
                sample_period_seconds=bundle.sample_period_seconds,
            ).as_dict(),
            "block_bootstrap": block_bootstrap_alarm_metrics(
                normal_alarm,
                abnormal_alarm,
                block_size=int(uncertainty["block_size"]),
                draws=int(uncertainty["draws"]),
                confidence=float(uncertainty["confidence"]),
                seed=seed,
            ).as_dict(),
            "block_event_rate_posterior": event_posterior,
        }
    )
    return result
