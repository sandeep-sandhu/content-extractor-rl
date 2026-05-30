# Change Log

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
