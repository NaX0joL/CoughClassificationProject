import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from ..model import FullModel
from ..model.architectures.PatchTST import PatchTST
from ..model.architectures.PatchTST.patchtst.attention_mechanism.scaled_dot_product_attention import (
    _ScaledDotProductAttention,
)


class LeGrad:
    """Layer-gradient attribution over PatchTST attention maps."""

    def supports(self, model:FullModel) -> bool:
        return isinstance(model.architecture, PatchTST) and bool(self._attention_layers(model))

    def create(
        self,
        model:FullModel,
        values:Tensor,
        target_class:int,
    ) -> np.ndarray:
        attention_layers = self._attention_layers(model)
        model.zero_grad(set_to_none=True)
        logits = model(values)
        target_score = logits[:, target_class].sum()
        attention_maps = [
            layer.attention_maps
            for layer in attention_layers
            if layer.attention_maps is not None
        ]
        if not attention_maps:
            raise ValueError("LeGrad could not capture attention maps")

        gradients = torch.autograd.grad(
            target_score,
            attention_maps,
            allow_unused=True,
        )
        layer_scores = [
            torch.relu(gradient).mean(dim=1).mean(dim=1)
            for gradient in gradients
            if gradient is not None
        ]
        if not layer_scores:
            raise ValueError("LeGrad could not capture attention gradients")

        token_scores = torch.stack(layer_scores).mean(dim=0)
        token_scores = token_scores.reshape(
            values.shape[0],
            -1,
            token_scores.shape[-1],
        ).mean(dim=1)
        heatmap = functional.interpolate(
            token_scores.unsqueeze(1),
            size=values.shape[1],
            mode="linear",
            align_corners=False,
        )[0, 0]
        return self._normalize(heatmap)

    @staticmethod
    def _attention_layers(model:FullModel) -> list[_ScaledDotProductAttention]:
        return [
            module
            for module in model.modules()
            if isinstance(module, _ScaledDotProductAttention)
        ]

    @staticmethod
    def _normalize(heatmap:Tensor) -> np.ndarray:
        heatmap = heatmap.detach().cpu()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap = heatmap / maximum
        return heatmap.numpy()
