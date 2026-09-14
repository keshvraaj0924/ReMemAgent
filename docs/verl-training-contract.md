# verl training value contract

The dependency-free verl integration validates the numeric values and mutable research state that cross the token-level training boundary.

- `VerlTrajectory.reward` must be a real, finite number and is canonicalized to a plain Python `float`.
- `VerlTrajectory.response_logprobs`, when present, must align one-to-one with `response_ids`; every value must be real and finite and is canonicalized to `float`.
- `validate_agent_loop_output(...).response_logprobs`, when present, enforces the same alignment and numeric contract for externally generated agent-loop outputs: finite real scalars are canonicalized to `float`, while booleans, non-real values, NaN, and infinity are rejected.
- `VerlTrajectory.prompt_ids` and `VerlTrajectory.response_ids` are validated and canonicalized to owned immutable tuples at construction time, so caller-owned mutable token sequences cannot change token identity after validation.
- `VerlTrajectory.response_mask` and externally validated agent-loop response masks must align one-to-one with `response_ids`, contain only binary integers, and contain at least one active token.
- `VerlTrainingBatch.trajectories` is validated to contain only `VerlTrajectory` values and canonicalized to an owned immutable tuple, so direct construction cannot retain a mutable caller-owned batch sequence.
- `VerlTrainingBatch.advantages` must contain one real, finite value per trajectory; each value is canonicalized to `float`.
- `AgentLoopRequest.reward` and `adapt_agent_loop_output(..., reward=...)` enforce the same finite-real contract at their own public boundaries and canonicalize compatible real scalar implementations to plain Python `float` values.
- Boolean values are rejected even though Python treats `bool` as an `int` subclass.
- NaN and positive/negative infinity are rejected before framework-specific collation.

The active-token invariant is enforced at both public construction boundaries. External agent-loop output is rejected immediately when every response token is masked out, because such a rollout contains no trainable response token. `VerlTrajectory` repeats the invariant intentionally so direct construction cannot bypass the safety check.

These checks complement the GRPO layer, which validates and canonicalizes finite rewards and advantages. The duplicated boundary is intentional: `VerlTrajectory`, `VerlTrainingBatch`, `AgentLoopRequest`, `validate_agent_loop_output`, and the agent-loop output adapter are public construction points and must remain safe when callers bypass the GRPO helpers.

Mutable provenance is detached at the same boundaries. External verl `extra_fields` are deep-copied during validation and again when serialized, so later mutations to rollout-owned nested lists or dictionaries cannot rewrite an already validated record. `VerlTrajectory` requires metadata to be a mapping, deep-copies nested metadata during construction, and returns detached nested metadata from `to_dict()`. Its prompt and response token sequences are also copied into immutable tuples, keeping validated token identity stable even if a caller supplied mutable sequence objects at runtime. `VerlTrainingBatch` likewise owns its ordered trajectory tuple, preventing later mutation of an input list from changing trajectory/advantage alignment. This keeps trainer-side collation, row mutation, or caller-side container mutation from altering the stored training record.

`AgentLoopRequest` is also validated at construction time. Sampling parameters, dataset keyword arguments, and research metadata are deep-copied and then placed behind immutable top-level mapping proxies. This makes queued concurrent requests stable even when the caller later mutates the dictionaries or nested lists/dictionaries originally supplied to the request. The request reward is required to be a real, finite number and is canonicalized before the external agent loop can be scheduled.

Canonicalizing token and batch containers and numeric scalar representations is not reward shaping or token transformation. ReMemAgent preserves token values, trajectory order, and does not clip, rescale, replace, or otherwise change the numeric value of rewards, log probabilities, or advantages at this boundary. Likewise, deep-copying provenance is an ownership guarantee, not a semantic transformation of the research record. Training-policy transformations remain caller-owned and should be recorded as part of experiment provenance.

No claim is made about training effectiveness until a real model checkpoint, tokenizer, verl runtime, and benchmark workload have been executed under a recorded configuration.
