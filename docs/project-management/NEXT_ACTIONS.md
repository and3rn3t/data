# Fix ML Configuration Issues - Action Plan

## Files to Update

### 1. examples/demo_modern_toolchain.py

- Lines 190, 292, 351: Add min_samples_leaf and max_features to RandomForestClassifier

### 2. runners/level_7_challenge_1_runner.py

- Lines 148, 183, 248: Add missing hyperparameters
- Lines 38-40, 67-69, 96-98: Replace np.random.\* with modern Generator API

### 3. sandbox/core/game_engine.py

- Refactor \_is_challenge_completed() (line 147) - complexity 19->15
- Refactor get_available_challenges() (line 285) - complexity 29->15

## Quick Fixes

```python
# Replace this:
RandomForestClassifier(n_estimators=100, random_state=42)

# With this:
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=1,
    max_features='sqrt'
)

# Replace this:
np.random.randn(1000)

# With this:
rng = np.random.default_rng(42)
rng.standard_normal(1000)
```
