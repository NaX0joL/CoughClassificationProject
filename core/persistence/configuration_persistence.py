import pickle
import random
from dataclasses import fields, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

from torch import Tensor, nn


def save_configuration(run_directory:Path, config:dict[str, Any]) -> None:
    with (run_directory / "config.pkl").open("wb") as config_file:
        pickle.dump(config, config_file)

    with (run_directory / "config.txt").open("w", encoding="utf-8") as config_file:
        config_file.write(_format_configuration(config))
        config_file.write("\n")
    return


def _format_configuration(config:dict[str, Any]) -> str:
    return "\n\n".join(
        f"{name} = {_format_value(value, indentation=0)}"
        for name, value in config.items()
    )


def _format_value(value:Any, indentation:int) -> str:
    if isinstance(value, Path):
        return pformat(str(value))

    if isinstance(value, (str, int, float, bool, type(None))):
        return pformat(value)

    if isinstance(value, dict):
        return _format_dictionary(value, indentation)

    if isinstance(value, (list, tuple)):
        return _format_sequence(value, indentation)

    parameters = _get_display_parameters(value)
    return _format_object(value.__class__.__name__, parameters, indentation)


def _format_dictionary(value:dict[Any, Any], indentation:int) -> str:
    if not value:
        return "{}"

    nested_indentation = indentation + 1
    prefix = "    " * nested_indentation
    entries = [
        f"{prefix}{key!r}: {_format_value(item, nested_indentation)}"
        for key, item in value.items()
    ]
    closing_prefix = "    " * indentation
    return "{\n" + ",\n".join(entries) + f"\n{closing_prefix}}}"


def _format_sequence(value:list[Any]|tuple[Any, ...], indentation:int) -> str:
    if not value:
        return "[]"

    if all(_is_scalar(item) for item in value):
        return pformat(value, sort_dicts=False)

    nested_indentation = indentation + 1
    prefix = "    " * nested_indentation
    entries = [
        f"{prefix}{_format_value(item, nested_indentation)}"
        for item in value
    ]
    closing_prefix = "    " * indentation
    return "[\n" + ",\n".join(entries) + f"\n{closing_prefix}]"


def _format_object(
    class_name:str,
    parameters:dict[str, Any],
    indentation:int,
) -> str:
    if not parameters:
        return f"{class_name}()"

    nested_indentation = indentation + 1
    prefix = "    " * nested_indentation
    entries = [
        f"{prefix}{name}={_format_value(parameter, nested_indentation)}"
        for name, parameter in parameters.items()
    ]
    closing_prefix = "    " * indentation
    return f"{class_name}(\n" + ",\n".join(entries) + f"\n{closing_prefix})"


def _is_scalar(value:Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), Path))


def _get_display_parameters(value:Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in fields(value)
        }
    return _get_public_parameters(value)


def _get_public_parameters(value:Any) -> dict[str, Any]:
    return {
        name: parameter
        for name, parameter in vars(value).items()
        if _is_display_parameter(name, parameter)
    }


def _is_display_parameter(name:str, parameter:Any) -> bool:
    if name.startswith("_") or name == "training":
        return False
    if isinstance(parameter, (nn.Module, Tensor, random.Random)):
        return False
    return True
