from dataclasses import dataclass

import numpy as np
import torch

from modules.resolve_pytorch_device import get_model_device

from ..data_pipeline.intermediary import Example
from ..model import FullModel
from .grad_cam import GradCam
from .legrad import LeGrad


@dataclass
class ModelOutput:
    prediction:int
    confidence:float
    grad_cam:np.ndarray|None
    legrad:np.ndarray|None


def create_model_output(
    model:FullModel,
    example:Example,
    include_grad_cam:bool,
    include_legrad:bool,
) -> ModelOutput:
    device = get_model_device(model)
    values = torch.as_tensor(example.value, dtype=torch.float32).unsqueeze(0)
    values = values.to(device)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            probabilities = model.predict_probabilities(values)[0]
            prediction = int(probabilities.argmax().item())
            confidence = float(probabilities[prediction].item())

        grad_cam = None
        legrad = None
        grad_cam_creator = GradCam()
        if include_grad_cam and grad_cam_creator.supports(model):
            grad_cam = grad_cam_creator.create(model, values, prediction)
        legrad_creator = LeGrad()
        if include_legrad and legrad_creator.supports(model):
            legrad = legrad_creator.create(model, values, prediction)
    finally:
        model.train(was_training)

    return ModelOutput(
        prediction=prediction,
        confidence=confidence,
        grad_cam=grad_cam,
        legrad=legrad,
    )
