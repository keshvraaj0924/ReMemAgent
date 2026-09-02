# GRPO data contract

ReMemAgent treats GRPO records as a boundary between research episode execution and an external trainer. The framework owns the structural record; the trainer owns tokenization, tensors, optimization, and distributed execution.

## Validation boundary

`GrpoSample` rejects records with:

- an empty prompt, completion, or group identifier;
- a non-finite reward;
- an empty memory identifier.

`GrpoBatch` additionally requires:

- at least one sample;
- exactly one advantage per sample;
- finite advantages.

These checks happen before JSONL persistence or dispatch to an external training consumer.

## Group semantics

`build_grpo_batch()` still rejects singleton groups because a single candidate cannot provide a comparative group-relative advantage. Callers must therefore provide a `group_id_builder` that places multiple comparable completions in the same group when constructing training data.

The default `build_grpo_samples()` grouping remains `episode-{index}` intentionally: it is safe for representing independent episodes, but it is not a claim that those episodes form a valid GRPO comparison group. This distinction prevents the integration layer from inventing task equivalence merely to make a batch constructible.

## Scientific boundary

Finite-record validation is an engineering invariant, not evidence that a training run is valid or effective. The framework does not infer model quality, statistical significance, or training convergence from these checks.
