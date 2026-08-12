import numpy as np
import pytest

from core.data_pipeline.intermediary import Example
from core.data_pipeline.preprocessing import MelBand
from core.data_pipeline.preprocessing.transform import MEL_FILTER_COUNT


def test_melband_produces_log_mel_features_and_preserves_example_data() -> None:
    example = Example(
        value=np.ones(1_600, dtype=np.float32),
        label=3,
        metadata={"patient_id": "patient-1"},
    )

    transformed_example = MelBand().transform([example])[0]

    assert transformed_example.value.ndim == 2
    assert transformed_example.value.shape[1] == MEL_FILTER_COUNT
    assert np.isfinite(transformed_example.value).all()
    assert transformed_example.label == example.label
    assert transformed_example.metadata == example.metadata


def test_melband_rejects_empty_waveforms() -> None:
    example = Example(value=np.array([], dtype=np.float32), label=0, metadata={})

    with pytest.raises(ValueError, match="MelBand requires at least one audio sample"):
        MelBand().transform([example])
