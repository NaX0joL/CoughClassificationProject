import numpy as np
import pytest
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from core.data_pipeline.dataset import ExampleDataset
from core.data_pipeline.intermediary import Example
from core.gallery import ModelOutput
from core.persistence.processes.model_output_persistence import _save_output_page


@pytest.mark.parametrize("values", [np.arange(10), np.arange(20).reshape(10, 2)])
def test_model_output_page_supports_univariate_and_multivariate_features(tmp_path, values) -> None:
    dataset = ExampleDataset([Example(value=values, label=0, metadata={})])
    path = tmp_path / "output.pdf"

    with PdfPages(path) as pdf:
        _save_output_page(
            pdf=pdf,
            dataset=dataset,
            sample_index=0,
            output=ModelOutput(prediction=0, confidence=0.9, grad_cam=None, legrad=None),
            class_names={0: "healthy"},
            color_normalization=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20),
            feature_colormap="inferno",
            x_axis_label="Time step",
            y_axis_label="Amplitude",
            colorbar_label="Intensity",
        )

    assert path.stat().st_size > 0
