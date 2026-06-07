# Change Log

## Unreleased — RL formulation redesign (P0)
  - **Fixed the degenerate MDP (root cause that the SAC NaN merely symptomized).** Previously the agent could not learn: `environment.rs::extract_with_params` ignored both the selected node and the continuous params and always extracted from the *whole document*, so every action produced identical text and identical reward. Node selection and params now actually drive extraction (`environment::extract_selected` + `CandidateContent::extract`), so different actions yield different rewards. New test `environment::action_choice_changes_reward` guards this.
  - **Real state representation.** `build_state()` no longer returns a near-constant placeholder vector (hardcoded `0.5`s). It now encodes real per-candidate DOM features (word count, link density, stopword ratio, tag type, depth, Readability-style class/id signals) via the new `node_features` module, plus global document and selection state.
  - **Ground-truth reward.** Training now threads the labelled article text (`text` field of the paired JSON) into the environment via the new `TrainingSample` type, and the reward is token F1 against the ground truth (`TextUtils::token_f1`), consistent with `GroundTruthEvaluator`. Falls back to a quality proxy when no ground truth is present. The dead, self-referential `ImprovedRewardCalculator` path (whose `improvement_reward` compared a value to itself and was always 0) was removed from `train_with_improvements`.
  - **Hybrid supervised + RL.** Added `node_classifier` module: a supervised MLP that picks the content node (labels derived for free via `label_from_f1` = argmax-F1 candidate), with RL left to tune extraction params. `HybridExtractor` wires classifier → node selection → param-driven extraction, with a Readability-style heuristic fallback when untrained.
  - **Wired the extractor into the inference path.** Added a shared public `extract_article(html, url, config, agent)` entry point: it runs a trained RL agent greedily through the environment when one is supplied, otherwise uses the hybrid/heuristic node selector, with baseline fallback. The CLI `extract`/`extract-batch` commands and the Python `extract`/`extract_batch` bindings now use it (previously they loaded the agent but always returned the plain baseline). `--model` is now optional for `extract`.
  - **Classifier serialization + training command.** `NodeClassifier` can now `save`/`load` its weights (safetensors via `VarMap`). Added `train_classifier` + `build_classifier_dataset` (pointwise `(features, label)` examples; labels from `label_from_f1`), a `train-classifier` CLI subcommand (loads ground-truth samples, trains, saves a `.safetensors`), and a `--classifier` option on `extract`/`extract-batch` (and `extract_single`/`extract_batch`) so a trained classifier drives node selection via `HybridExtractor`.
  - **Docs.** Rewrote the README usage sections (Rust library, CLI, and Python-via-wheel) to match the actual APIs, and documented building/installing the `content_extractor_rl_rs` wheel with maturin.
  - **SAC gradient clipping.** SAC lacked the global-norm gradient clipping the DQN already had; combined with over-large learning rates this let actor/critic gradients explode to NaN and permanently corrupt the weights. Added `clip_grad_norm` (max-norm 1.0) before every SAC optimizer step. Regression test `test_sac_stable_under_high_learning_rate` trains at lr=5e-3 for 100 steps and asserts the loss/weights stay finite.
  - **Hyperparameter search stability.** Learning-rate search is now sampled log-uniformly and capped at 3e-3 (the failing run used a uniformly-sampled ~5.9e-3); dropped the 5e-3 grid entry.
  - **Decorrelated sampling.** Training samples a random page each episode instead of deterministic `episode % len` cycling.
  - Build: added `target-cpu=native` rustflags for `aarch64-unknown-linux-gnu` so candle's `gemm-f16` FP16 SIMD compiles on the Cortex-A76 host.

## Version 0.1.3:
  - Upgraded versions of all dependencies to the latest supported versions
  - Added trained ONNX model files: DuelingDQN.onnx (0.826 quality), PPO.onnx (0.845 quality), SAC.onnx
  - Added best hyperparameter JSON files and training plots for all three algorithms
  - Fixed SAC NaN loss: `alpha_loss` was computed from a constant `Tensor::from_vec` not connected to the `log_alpha` `Var`, so the entropy temperature received zero gradient and never updated. Changed to derive `alpha_loss` via `self.log_alpha.as_tensor().broadcast_as(...)` so gradients flow correctly
  - Fixed DuelingDQN model save: `save_to_onnx_with_metadata` called `File::create` twice, truncating and rewriting metadata redundantly; removed the duplicate block
  - Fixed DuelingDQN save warnings: LayerNorm bias tensors (`ln1.bias`, `ln2.bias`, `ln3.bias`) are initialized to zero by convention and correctly suppressed from the "100% zeros" warning
  - Fixed SAC target network never updating: `soft_update_linear` and `soft_update_layernorm` were empty no-ops (TODO stubs), so the target Q-network retained its random initialization forever. Replaced with a real VarMap-based soft update using `Var::set()` that interpolates `target = tau * online + (1 - tau) * target` every train step. Also initializes target weights as a hard copy of the critic at agent construction.
  - Fixed SAC NaN from log(0): `action_probs.log()` was called without clamping, producing `-inf` for near-zero probabilities which propagated as NaN via `-inf * 0`. Added `clamp(1e-8, 1.0)` before the log.
  - Fixed SAC `gaussian_log_prob` numerical instability: clamped `std` to `[1e-6, 1e6]` before computing variance and log_std to prevent division by near-zero and log(0) when action distribution becomes very sharp.
  - Fixed DQN/PPO broken computation graph: both `train_step` (DQN) and `ppo_update` (PPO) were converting the combined loss to an f32 scalar and then re-wrapping it in a new `Tensor::from_vec`/`Tensor::new`, which has no computation history; `backward()` therefore returned an empty `GradStore` and the neural networks never received any gradient updates. Fixed by combining losses with tensor arithmetic (`+`, `*`) before calling `backward()`.
  - Fixed DQN target network update: `impl RLAgent for DQNAgent::update_target_network` was calling the old broken `copy_network_weights` helper (which delegated to a no-op `copy_weights_from`); now calls `do_target_network_update` which performs a proper VarMap-based hard copy via `Var::set()`.
  - Fixed DQN `load_with_device`: `target_varmap` field was omitted from the returned `Self`, causing a compilation error when the struct gained the new field.
  - Fixed DQN/PPO `param_logstd` not in optimizer: `DuelingDQN::param_logstd` and `ActorCriticNetwork::actor_param_logstd` are created via `Var::from_tensor` and never registered in the `VarMap`, so `varmap.all_vars()` excluded them and they received no gradient updates. Added `param_logstd_var()` / `logstd_var()` accessors and pushed the `Var` into the trainable-vars list before constructing the optimizer.
  - Fixed PPO `save_with_metadata`: previously only saved to the custom binary format; now also calls `save_to_safetensors` so both formats are written on every save.
  - Fixed PPO `get_info`: `num_parameters` was hard-coded to 0 and `state_dim` to 0; now computed from the actual architecture dimensions.

## Version 0.1.2:
  - Fixed minor compilation errors, missing package metadata and cargo publishing errors

## Version 0.1.1:
  - Minor bug fixes and fixed cargo publishing errors

## Version 0.1.0:
  - Initial version
