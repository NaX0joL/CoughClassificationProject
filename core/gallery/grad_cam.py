import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from ..model import FullModel


class GradCam:

    def supports(self, model:FullModel) -> bool:
        return self._get_target_layer(model) is not None

    def create(
        self,
        model:FullModel,
        values:Tensor,
        target_class:int,
    ) -> np.ndarray:
        target_layer = self._get_target_layer(model)
        if target_layer is None:
            raise ValueError("Grad-CAM requires a model with a Conv1d layer")

        activations:Tensor|None = None
        gradients:Tensor|None = None

        def save_activations(
            module:nn.Module,
            inputs:tuple[Tensor, ...],
            output:Tensor,
        ) -> None:
            nonlocal activations
            activations = output

        def save_gradients(
            module:nn.Module,
            gradient_inputs:tuple[Tensor|None, ...],
            gradient_outputs:tuple[Tensor|None, ...],
        ) -> None:
            nonlocal gradients
            gradients = gradient_outputs[0]

        forward_handle = target_layer.register_forward_hook(save_activations)
        backward_handle = target_layer.register_full_backward_hook(save_gradients)

        try:
            model.zero_grad(set_to_none=True)
            logits = model(values)
            logits[:, target_class].sum().backward()
        finally:
            forward_handle.remove()
            backward_handle.remove()

        if activations is None or gradients is None:
            raise ValueError("Grad-CAM could not capture layer outputs")

        channel_weights = gradients.mean(dim=2, keepdim=True)
        heatmap = torch.relu((channel_weights * activations).sum(dim=1))
        heatmap = functional.interpolate(
            heatmap.unsqueeze(1),
            size=values.shape[1],
            mode="linear",
            align_corners=False,
        )[0, 0]
        return self._normalize(heatmap)

    @staticmethod
    def _get_target_layer(model:FullModel) -> nn.Conv1d|None:
        convolution_layers = [
            module
            for module in model.modules()
            if isinstance(module, nn.Conv1d)
        ]
        return convolution_layers[-1] if convolution_layers else None

    @staticmethod
    def _normalize(heatmap:Tensor) -> np.ndarray:
        heatmap = heatmap.detach().cpu()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap = heatmap / maximum
        return heatmap.numpy()
