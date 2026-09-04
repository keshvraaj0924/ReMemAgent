# verl training value contract

The dependency-free verl integration validates the numeric values that cross the token-level training boundary.

- `VerlTrajectory.reward` must be a real, finite number.
- `VerlTrainingBatch.advantages` must contain one real, finite value per trajectory.
- Boolean values are rejected even though Python treats `bool` as an `int` subclass.
- NaN and positive/negative infinity are rejected before framework-specific collation.

These checks complement the GRPO layer, which already validates finite rewards and advantages. The duplicated boundary is intentional: `VerlTrajectory` and `VerlTrainingBatch` are public construction points and must remain safe when callers bypass the GRPO helpers.

The validation is an integrity guard, not a training policy. ReMemAgent does not normalize, clip, replace, or otherwise alter rewards or advantages. Any such transformation belongs to the caller-owned training configuration and should be recorded as part of experiment provenance.

No claim is made about training effectiveness until a real model checkpoint, tokenizer, verl runtime, and benchmark workload have been executed under a recorded configuration.
