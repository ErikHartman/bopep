# Bayesian Optimization module

Should contain modules for:

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