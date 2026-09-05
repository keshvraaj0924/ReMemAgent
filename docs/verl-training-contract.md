# verl training value contract

The dependency-free verl integration validates the numeric values that cross the token-level training boundary.

- `VerlTrajectory.reward` must be a real, finite number.
- `VerlTrajectory.response_mask` must align one-to-one with `response_ids`, contain only binary integers, and contain at least one active token.
- `VerlTrainingBatch.advantages` must contain one real, finite value per trajectory.
- `AgentLoopRequest.reward` and `adapt_agent_loop_output(..., reward=...)` enforce the same real-number contract.
- Boolean values are rejected even though Python treats `bool` as an `int` subclass.
- NaN and positive/negative infinity are rejected before framework-specific collation.

The active-token invariant is deliberately enforced on the framework-owned trajectory type rather than the external output validator. An external agent-loop output can describe a response mask exactly as emitted by verl; once it becomes a training trajectory, an all-zero mask would provide no learnable response tokens and therefore fails closed before collation.

These checks complement the GRPO layer, which already validates finite rewards and advantages. The duplicated boundary is intentional: `VerlTrajectory`, `VerlTrainingBatch`, `AgentLoopRequest`, and the agent-loop output adapter are public construction points and must remain safe when callers bypass the GRPO helpers.

`AgentLoopRequest` is also validated at construction time. Sampling parameters, dataset keyword arguments, and research metadata are copied into immutable mapping proxies. This makes queued concurrent requests stable even when the caller later mutates the dictionaries originally supplied to the request. The request reward is required to be a real, finite number before the external agent loop can be scheduled.

These request-level checks are lifecycle guarantees, not training policies. ReMemAgent does not normalize, clip, replace, or otherwise alter rewards or advantages. Any such transformation belongs to the caller-owned training configuration and should be recorded as part of experiment provenance.

No claim is made about training effectiveness until a real model checkpoint, tokenizer, verl runtime, and benchmark workload have been executed under a recorded configuration.
