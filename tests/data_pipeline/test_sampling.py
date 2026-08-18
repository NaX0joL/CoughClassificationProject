
import numpy as np
import pytest

from core.data_pipeline.intermediary import Example
from core.data_pipeline.preprocessing import Resampler


def _make_example(length:int=100, n_features:int=10) -> Example:
    return Example(
        value=np.random.rand(length, n_features).astype(np.float32),
        label=1,
        metadata={"patient_id": "p1"},
    )


class TestResampler:

    def test_resizes_to_target_length(self) -> None:
        example = _make_example(length=100)
        resampler = Resampler(target_length=50)

        result = resampler.transform([example])[0]

        assert result.value.shape[0] == 50
        assert result.value.shape[1] == example.value.shape[1]

    def test_preserves_label_and_metadata(self) -> None:
        example = _make_example(length=100)
        resampler = Resampler(target_length=50)

        result = resampler.transform([example])[0]

        assert result.label == example.label
        assert result.metadata == example.metadata

    def test_no_change_when_target_equals_source(self) -> None:
        example = _make_example(length=100)
        resampler = Resampler(target_length=100)

        result = resampler.transform([example])[0]

        np.testing.assert_array_equal(result.value, example.value)

    def test_linear_interpolation_preserves_values(self) -> None:
        source = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
        example = Example(value=source, label=0, metadata={})
        resampler = Resampler(target_length=5, method="linear")

        result = resampler.transform([example])[0]

        assert result.value.shape[0] == 5
        assert result.value[0, 0] == pytest.approx(0.0)
        assert result.value[2, 0] == pytest.approx(1.0)
        assert result.value[4, 0] == pytest.approx(0.0)

    def test_nearest_interpolation(self) -> None:
        source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        example = Example(value=source, label=0, metadata={})
        resampler = Resampler(target_length=6, method="nearest")

        result = resampler.transform([example])[0]

        assert result.value.shape[0] == 6
        for val in result.value:
            assert val[0] in (0.0, 1.0, 2.0)

    def test_rejects_empty_array(self) -> None:
        example = Example(value=np.array([]).reshape(0, 5), label=0, metadata={})
        resampler = Resampler(target_length=10)

        with pytest.raises(ValueError, match="Resampler requires a non-empty array"):
            resampler.transform([example])

    def test_rejects_non_positive_target_length(self) -> None:
        with pytest.raises(ValueError, match="target_length must be positive"):
            Resampler(target_length=0)

        with pytest.raises(ValueError, match="target_length must be positive"):
            Resampler(target_length=-1)

    def test_rejects_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            Resampler(target_length=10, method="cubic")

    def test_multiple_examples(self) -> None:
        examples = [_make_example(length=100) for _ in range(3)]
        resampler = Resampler(target_length=50)

        results = resampler.transform(examples)

        assert len(results) == 3
        for result in results:
            assert result.value.shape[0] == 50
