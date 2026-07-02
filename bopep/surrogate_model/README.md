# Surrogate modelling

## Architectures

## Bayesian modalities


## Multi-objective vs single-objective

Single objective

```py
obj = {sequence: val, sequence: val} etc.
```

Multi objective

```py
obj = {sequence: {name:val, name:val}, sequence: {name:val, name:val}} etc.
```

Outputs replace the vals with tuples of mean,std

## Custom surrogate models

BoPep can use a user-provided surrogate by passing it in
`surrogate_model_kwargs`:

```py
surrogate_model_kwargs = {
    "custom_model": my_predictor,
    "default_std": 0.1,  # used if the predictor returns only means
}
```

The custom model must implement:

```py
predict_dict(input_dict, **kwargs) -> dict
```

where `input_dict` is keyed by sequence. Return either
`{sequence: (mean, std)}` or `{sequence: mean}` for single-objective models.
For multi-objective models, return
`{sequence: {objective_name: (mean, std)}}`.

`fit_dict(input_dict, objective_dict, **kwargs)` is optional. If it is not
implemented, BoPep treats the custom surrogate as a pretrained predictor.

For BoGA, raw sequence inputs can be passed directly to a custom surrogate:

```py
boga = BoGA(
    initial_sequences=["ACDEFG", "HIKLMN"],
    mode="sequence",
    surrogate_model_kwargs={"custom_model": amp_predictor, "default_std": 0.1},
    embed_method=None,
)
```

With `embed_method=None`, the surrogate input is `{sequence: sequence}`.
