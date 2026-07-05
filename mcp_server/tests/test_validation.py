from core.validation import REPORT_REQUIRED_PARAMS, validate_report_params


def test_similarity_report_does_not_require_tabsyndex_target_type() -> None:
    """Similarity report should not be blocked by TabSynDex-only params."""
    assert REPORT_REQUIRED_PARAMS.get("Similarity") in (None, {})
    assert validate_report_params("Similarity", params={}) is None


def test_privacy_report_does_not_require_overfitting_validation_data() -> None:
    """Privacy report should not be blocked by DCR_Overfitting-only params."""
    required = REPORT_REQUIRED_PARAMS.get("Privacy", {})
    assert "real_validation_data" not in required.get("params", [])
    assert validate_report_params("Privacy", params={}) is None


def test_ml_efficacy_report_still_requires_targets_override() -> None:
    """Report override should still enforce report-level targets requirement."""
    required = REPORT_REQUIRED_PARAMS.get("Machine_Learning_Efficacy", {})
    assert required.get("params") == ["targets"]

    error = validate_report_params("Machine_Learning_Efficacy", params={})
    assert error is not None
    assert error["error"] == "MISSING_REQUIRED_PARAMETERS"
    assert error["missing_parameters"][0]["parameter"] == "targets"
