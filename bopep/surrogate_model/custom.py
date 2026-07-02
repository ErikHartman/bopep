import inspect
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


PredictionValue = Union[
    Tuple[float, float],
    Dict[str, Tuple[float, float]],
]


class CustomSurrogateModel:
    """
    Adapter for user-provided surrogate models.

    The wrapped model must expose predict_dict(input_dict, **kwargs) or be
    directly callable with input_dict. fit_dict(input_dict, objective_dict,
    **kwargs) is optional and may be a no-op for pretrained predictors.
    """

    def __init__(self, model: Any, default_std: float = 0.0):
        if model is None:
            raise ValueError("A custom surrogate model instance must be provided.")
        predict_method = getattr(model, "predict_dict", None)
        if not callable(predict_method) and not callable(model):
            raise TypeError(
                "Custom surrogate models must expose predict_dict(input_dict, **kwargs) "
                "or be callable with input_dict."
            )

        self.model = model
        self.default_std = float(default_std)
        if self.default_std < 0:
            raise ValueError("default_std for a custom surrogate must be non-negative.")

    def fit_dict(
        self,
        embedding_dict: Dict[str, Any],
        objective_dict: Dict[str, Any],
        val_embedding_dict: Optional[Dict[str, Any]] = None,
        val_objective_dict: Optional[Dict[str, Any]] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        device: Optional[Any] = None,
        **kwargs: Any,
    ) -> float:
        fit_method = getattr(self.model, "fit_dict", None)
        if fit_method is None:
            return 0.0

        call_kwargs = {
            "val_embedding_dict": val_embedding_dict,
            "val_objective_dict": val_objective_dict,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            **kwargs,
        }
        result = self._call_with_supported_kwargs(
            fit_method,
            embedding_dict,
            objective_dict,
            **{key: value for key, value in call_kwargs.items() if value is not None},
        )
        return 0.0 if result is None else float(result)

    def predict_dict(
        self,
        embedding_dict: Dict[str, Any],
        device: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, PredictionValue]:
        predict_method = getattr(self.model, "predict_dict", None)
        if not callable(predict_method):
            raw_predictions = self._call_with_supported_kwargs(
                self.model,
                embedding_dict,
                device=device,
                **kwargs,
            )
        else:
            raw_predictions = self._call_with_supported_kwargs(
                predict_method,
                embedding_dict,
                device=device,
                **kwargs,
            )

        if not isinstance(raw_predictions, dict):
            raise TypeError(
                "Custom surrogate predict_dict must return a dict keyed by sequence."
            )

        missing = set(embedding_dict.keys()) - set(raw_predictions.keys())
        if missing:
            preview = ", ".join(sorted(missing)[:5])
            raise ValueError(
                f"Custom surrogate predictions are missing {len(missing)} sequence(s): {preview}"
            )

        return {
            sequence: self._normalize_prediction(raw_predictions[sequence])
            for sequence in embedding_dict.keys()
        }

    def to(self, device: Any) -> "CustomSurrogateModel":
        to_method = getattr(self.model, "to", None)
        if callable(to_method):
            to_method(device)
        return self

    def cpu(self) -> "CustomSurrogateModel":
        cpu_method = getattr(self.model, "cpu", None)
        if callable(cpu_method):
            cpu_method()
        return self

    def _call_with_supported_kwargs(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return method(*args, **kwargs)

        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return method(*args, **kwargs)

        filtered_kwargs = {
            key: value for key, value in kwargs.items()
            if key in signature.parameters
        }
        return method(*args, **filtered_kwargs)

    def _normalize_prediction(self, value: Any) -> PredictionValue:
        if isinstance(value, dict):
            return {
                objective_name: self._normalize_single_prediction(objective_value)
                for objective_name, objective_value in value.items()
            }

        return self._normalize_single_prediction(value)

    def _normalize_single_prediction(self, value: Any) -> Tuple[float, float]:
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return self._with_default_std(value.reshape(-1)[0])
            if value.size >= 2:
                return self._mean_std_tuple(value.reshape(-1)[0], value.reshape(-1)[1])

        if isinstance(value, (tuple, list)):
            if len(value) == 1:
                return self._with_default_std(value[0])
            if len(value) >= 2:
                return self._mean_std_tuple(value[0], value[1])

        return self._with_default_std(value)

    def _with_default_std(self, mean: Any) -> Tuple[float, float]:
        return self._mean_std_tuple(mean, self.default_std)

    def _mean_std_tuple(self, mean: Any, std: Any) -> Tuple[float, float]:
        mean_float = self._as_float(mean, "mean")
        std_float = self._as_float(std, "std")
        if std_float < 0:
            raise ValueError("Custom surrogate prediction std must be non-negative.")
        return mean_float, std_float

    def _as_float(self, value: Any, field_name: str) -> float:
        if isinstance(value, Number):
            return float(value)

        array = np.asarray(value)
        if array.size == 1:
            return float(array.reshape(-1)[0])

        raise TypeError(
            f"Custom surrogate prediction {field_name} values must be scalar numbers."
        )
