from dataclasses import dataclass

from .abstract import ModelArchitecture, ModelBehavior


@dataclass
class ModelConfig:
    architecture:ModelArchitecture
    behavior:ModelBehavior
