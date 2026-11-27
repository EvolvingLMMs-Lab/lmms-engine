# LMMs-Engine + lmms-eval RL MVP Design

## Goal

Build a lightweight RL path that reuses lmms-eval's existing agentic loop and
keeps LMMs-Engine responsible for distributed orchestration, buffering, FSDP2
training, and policy weight synchronization.

The MVP should match the design diagram:

- Ray manages outer rollout parallelism.
- lmms-eval keeps a synchronous LoopWorker as one atomic episode.
- vLLM runs as a separate OpenAI-compatible policy server.
- DataBuffer is independent from TrainingEngine.
- TrainingEngine is FSDP2-only for the first pass.
- Backpressure pauses/resumes rollout based on DataBuffer state.

## Existing Code To Reuse

lmms-eval already has the key agentic pieces:

- `lmms_eval.agentic.types`: proto-friendly `ContentBlock`, `AgentInput`,
  `AgentOutput`, `EnvState`, `EpisodeResult`.
- `lmms_eval.agentic.loop.worker.simple.SimpleLoopWorker`: synchronous
  single-agent episode loop.
- `lmms_eval.agentic.loop.manager.LoopManager`: batches ready model requests
  across loop sessions.
- `lmms_eval.agentic.model_server.openai.OpenAIModelServer`: OpenAI-compatible
  model boundary, suitable for vLLM.
- `lmms_eval.tasks.vizdoom_agentic`: reference task using
  `output_type: generate_until_game`, `game_env`, `observation_parser`, and
  `action_parser`.

Engine already has:

- FSDP2 trainer/checkpoint flow in `src/lmms_engine/train/fsdp2/fsdp2_trainer.py`.
- Async eval-server polling in `src/lmms_engine/eval/backends.py`.
- FSDP2 checkpoint merge utilities in `src/lmms_engine/merger`.

## Ownership Boundary

### lmms-eval

Owns environment and episode semantics:

- Reset/step environment through `EnvManager`.
- Convert observations into `AgentInput`.
- Parse model output into actions.
- Return one `EpisodeResult` per atomic episode.

New agentic rollout shim:

- `lmms_eval.agentic.rollout.RolloutEpisodeSpec`
- `lmms_eval.agentic.rollout.EpisodeComponentBuilder`
- `lmms_eval.agentic.rollout.SyncEpisodeRolloutWorker`

This shim deliberately wraps the existing agentic loop instead of introducing a
second agent framework.

### LMMs-Engine

Owns RL system orchestration:

- Ray actor pool for rollout workers.
- Independent producer-consumer `DataBuffer`.
- `RewardedTrajectory` and `TrainBatch` protocol.
- vLLM weight reload client.

Four visible system modules:

- Rollout Manager: `lmms_engine.rl.rollout_manager`
- LMMS Eval bridge: `lmms_engine.rl.lmms_eval`
- Data Buffer: `lmms_engine.rl.data_buffer`
- Training Engine boundary: `lmms_engine.rl.training_engine`

Shared RL infrastructure:

- `lmms_engine.rl.config`
- `lmms_engine.rl.protocol`
- `lmms_engine.rl.core.interfaces`
- `lmms_engine.rl.core.registry`
- `lmms_engine.rl.core.factory`
- `lmms_engine.rl.core.orchestrator`

Trainer-side RL algorithms live under `lmms_engine.train.rl`:

- `lmms_engine.train.rl.batch`
- `lmms_engine.train.rl.bridge`
- `lmms_engine.train.rl.grpo`

Because the RL layer is still in design, compatibility aliases are intentionally
not kept. New code should import through the four visible system modules or the
shared `core` package.

## Runtime Flow

The concrete loop should depend on `RLOrchestrator`, not concrete backends.

1. Trainer publishes initial model version to vLLM.
2. `RLOrchestrator` builds components from `RLComponentFactory`.
3. RolloutManager creates Ray actors. Each actor owns one
   `SyncEpisodeRolloutWorker`.
4. RolloutManager submits `RolloutTask(payload=RolloutEpisodeSpec(...))`.
5. lmms-eval runs one synchronous atomic episode and returns `EpisodeResult`.
6. Engine adapter converts `EpisodeResult` to `RewardedTrajectory`.
7. DataBuffer stores complete trajectories with the producing `ModelVersion`.
8. BatchBuilder pops fixed global batches from DataBuffer.
9. TrainBatchAdapter tensorizes samples for the RL algorithm.
10. FSDP2 trainer consumes one train batch and emits a new model version.
11. vLLMWeightSyncClient asks vLLM to reload/update policy weights.
12. Backpressure pauses rollout when DataBuffer reaches high watermark and
    resumes once it drains below low watermark.

## Extension Points

The MVP has named component boundaries:

- `TrajectoryAdapter`: converts lmms-eval `EpisodeResult` into engine
  `RewardedTrajectory`.
- `DataBuffer`: hides local memory, Ray actor, Redis, or durable queue choices.
- `RolloutManager`: hides Ray today and can support another scheduler later.
- `BatchBuilder`: defines global batch construction from buffered trajectories
  and lives in `lmms_engine.train.rl`.
- `TrainBatchAdapter`: owns PPO/GRPO/etc. tensorization and lives in
  `lmms_engine.train.rl`.
- `TrainerBridge`: hides FSDP2 trainer-specific step APIs and lives in
  `lmms_engine.train.rl`.
- `WeightSyncClient`: hides vLLM reload, file checkpoint, or future direct
  weight-transfer protocols.

`RLComponentFactory` builds the default components by name:

- `data_buffer.backend: in_memory`
- `rollout.backend: ray`
- `training.batch_builder: fixed_global`
- `vllm.backend: vllm_http`

New implementations should register a backend instead of changing the
orchestrator.

The default `GRPOBatchAdapter` lives at `lmms_engine.train.rl.grpo`. It currently
defines the GRPO boundary payload and config; concrete tensorization and loss
computation should be implemented there or in a trainer subclass, not in
`lmms_engine.rl`.

## Concurrency Model

The atomic episode stays synchronous inside lmms-eval. Concurrency is outside:

- Ray actors run many episodes in parallel.
- Each actor can use lmms-eval's existing model-server batching where useful.
- Engine manager APIs provide sync and async entry points.
- DataBuffer exposes sync and async push/pop methods.

This avoids mixing environment state machines with distributed scheduling.

## Data Contracts

### RolloutTask

Engine-owned scheduling unit:

- `task_id`
- `payload`: usually `lmms_eval.agentic.rollout.RolloutEpisodeSpec`
- `model_version`
- `seed`
- `metadata`

### RewardedTrajectory

Engine-owned complete training sample:

- `trajectory_id`
- `task_id`
- `model_version`
- `steps`
- `metrics`
- `final_state`
- `metadata`

Each step carries observation, request, response, parsed action, reward, done,
and info. Non-text payloads are intentionally typed as `Any` so JEPA-like,
latent, embedding, tensor, video, and image blocks can pass through.

### TrainBatch

The fixed global batch consumed by training:

- `batch_id`
- `model_version`
- `trajectories`
- `global_batch_size`
- `payload`
- `metadata`

`payload` is left algorithm-specific and is produced by `TrainBatchAdapter`.

## VizDoom As Reference

The first end-to-end smoke path should use:

- task: `vizdoom`
- YAML: `src/lmms-eval/lmms_eval/tasks/vizdoom_agentic/vizdoom.yaml`
- env factory: `utils.vizdoom_env_manager`
- observation parser: `{name: vizdoom, human_view: true, video: true}`
- action parser: `{name: vizdoom}`

For RL rollout, the resolved equivalent is a `RolloutEpisodeSpec` with the same
doc and component specs. No VizDoom-specific code belongs in engine.

## MVP Non-Goals

- No PPO/GRPO loss details in this pass.
- No custom vLLM weight-transfer implementation yet.
- No distributed durable DataBuffer yet.
- No changes to existing SFT trainer behavior.
- No replacement of lmms-eval agentic APIs.

## Next Implementation Steps

1. Add a config loader that can resolve lmms-eval task docs into
   `RolloutEpisodeSpec`.
2. Add a Ray actor smoke test using VizDoom with debug/fixed action model
   server first, then OpenAI/vLLM.
3. Add a minimal `train_batch(...)` method to an FSDP2 RL trainer subclass.
4. Implement old-logprob computation against vLLM or trainer-side policy.
5. Add real weight reload/update endpoint contract for vLLM.
6. Add checkpoint/resume state for RolloutManager and DataBuffer.
