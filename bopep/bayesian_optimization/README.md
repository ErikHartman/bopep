# Bayesian Optimization module

Should contain modules for:

1. Models (ensemble, RF, BNN)
```py
model = EnsembleNN()
```
2. Accompanying training functions
```py
model.train(docked_peptides)
means, stds = model.predict(not_docked_peptides)
```
2.5 Hyperparameter Optimization
```py
model = hparam_opt(model)
```
3. Acquisition functions
```py
scores = expected_improvement(means, stds)
```
4. Selection (using acquisition functions)
```py
peptides_to_dock = select(scores)
```
5. Finally, the main run loop that the user interacts with.
```py
bopep = BoPep()

bopep.run(peptides)
```