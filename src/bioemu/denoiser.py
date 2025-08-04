# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import cast

import numpy as np
import torch
from torch_geometric.data.batch import Batch
from omegaconf import OmegaConf

from .chemgraph import ChemGraph
from .sde_lib import SDE, CosineVPSDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat

TwoBatches = tuple[Batch, Batch]


class EulerMaruyamaPredictor:
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        *,
        corruption: SDE,
        noise_weight: float = 1.0,
        marginal_concentration_factor: float = 1.0,
    ):
        """
        Args:
            noise_weight: A scalar factor applied to the noise during each update. The parameter controls the stochasticity of the integrator. A value of 1.0 is the
            standard Euler Maruyama integration scheme whilst a value of 0.0 is the probability flow ODE.
            marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. The sampler targets p(x)^{MCF} where p(x)
            is the data distribution. A value of 1.0 is the standard Euler Maruyama / probability flow ODE integration.

            See feynman/projects/diffusion/sampling/samplers_readme.md for more details.

        """
        self.corruption = corruption
        self.noise_weight = noise_weight
        self.marginal_concentration_factor = marginal_concentration_factor

    def reverse_drift_and_diffusion(
        self, *, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor, score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = 0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        return drift, diffusion

    def update_given_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> TwoBatches:
        z = torch.randn_like(drift)

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z,
                tol=self.corruption.tol,
            )
        else:
            mean = x + drift * dt
            sample = mean + self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z
        return sample, mean

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> TwoBatches:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )

    def forward_sde_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> TwoBatches:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)


def get_score(
    batch: ChemGraph,
    sdes: dict[str, SDE],
    score_model: torch.nn.Module,
    t: torch.Tensor,
    mlcv: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Calculate predicted score for the batch.

    Args:
        batch: Batch of corrupted data.
        sdes: SDEs.
        score_model: Score model.  The score model is parametrized to predict a multiple of the score.
          This function converts the score model output to a score.
        t: Diffusion timestep. Shape [batch_size,]
    """
    tmp = score_model(
        batch,
        t,
        mlcv=mlcv,
    )
    # Score is in axis angle representation [N,3] (vector is along axis of rotation, vector length
    # is rotation angle in radians).
    assert isinstance(sdes["node_orientations"], SO3SDE)
    node_orientations_score = (
        tmp["node_orientations"]
        * sdes["node_orientations"].get_score_scaling(t, batch_idx=batch.batch)[:, None]
    )

    # Score model is trained to predict score * std, so divide by std to get the score.
    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(tmp["pos"]),
        t=t,
        batch_idx=batch.batch,
    )
    pos_score = tmp["pos"] / pos_std

    return {"node_orientations": node_orientations_score, "pos": pos_score}


def heun_denoiser(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
            )

        for field in fields:
            batch[field] = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,
            )[0]

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score = get_score(batch=batch, t=t_next, score_model=score_model, sdes=sdes)

            drifts = {}
            avg_drift = {}
            for field in fields:
                drifts[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field], t=t_next, batch_idx=batch.batch, score=score[field]
                )

                avg_drift[field] = (drifts[field] + drift_hat[field]) / 2
            for field in fields:
                batch[field] = (
                    0.0
                    + predictors[field].update_given_drift_and_diffusion(
                        x=batch_hat[field],
                        dt=(t_next - t_hat)[0],
                        drift=avg_drift[field],
                        diffusion=0.0,
                    )[0]
                )

    return batch


def _t_from_lambda(sde: CosineVPSDE, lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Used for DPMsolver. https://arxiv.org/abs/2206.00927 Appendix Section D.4
    """
    f_lambda = -1 / 2 * torch.log(torch.exp(-2 * lambda_t) + 1)
    exponent = f_lambda + torch.log(torch.cos(torch.tensor(np.pi * sde.s / 2 / (1 + sde.s))))
    t_lambda = 2 * (1 + sde.s) / np.pi * torch.acos(torch.exp(exponent)) - sde.s
    return t_lambda


def dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    mlcv: torch.Tensor,
    record_grad_steps: set[int] = set(),
    condition_mode: str = "none",
    cfg: OmegaConf = None,
) -> ChemGraph:

    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.
    DPM solver is used only for positions, not node orientations.
    """
    grad_is_enabled = torch.is_grad_enabled()
    assert isinstance(batch, ChemGraph)
    assert max_t < 1.0

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)  # help out mypy/linter

    # NOTE: MLCV condition on final representation
    num_graphs = batch.num_graphs
    if condition_mode == "backbone-both" and mlcv is not None:
        mlcv_expanded = mlcv.repeat_interleave(int(batch.batch.shape[0] / num_graphs), dim=0)
        batch.pos = batch.pos + score_model.model_nn.get_submodule("zero_conv_mlp_pos")(torch.cat([batch.pos, mlcv_expanded], dim=1))
        
        # PROPER ROTATION CONDITIONING: Use Lie algebra (so(3)) approach
        # Generate conditioning in the tangent space (axis-angle representation)
        
        # Extract rotation-invariant features for conditioning
        # Use trace and Frobenius norm as rotation-invariant descriptors
        batch_orient_trace = torch.diagonal(batch.node_orientations, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)  # [N, 1]
        batch_orient_frob = torch.norm(batch.node_orientations.view(batch.node_orientations.shape[0], -1), dim=-1, keepdim=True)  # [N, 1]
        rotation_features = torch.cat([batch_orient_trace, batch_orient_frob], dim=-1)  # [N, 2]
        
        axis_angle_conditioning = score_model.model_nn.get_submodule("zero_conv_mlp_orient")(torch.cat([rotation_features, mlcv_expanded], dim=1))  # Shape: [N, 3]
        
        # Convert axis-angle to skew-symmetric matrix
        def axis_angle_to_skew_matrix(axis_angle):
            """Convert axis-angle vector to skew-symmetric matrix"""
            x, y, z = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
            zeros = torch.zeros_like(x)
            skew = torch.stack([
                torch.stack([zeros, -z, y], dim=-1),
                torch.stack([z, zeros, -x], dim=-1),
                torch.stack([-y, x, zeros], dim=-1)
            ], dim=-2)
            return skew
        
        # Convert axis-angle conditioning to rotation matrix via matrix exponential
        def axis_angle_to_rotation_matrix(axis_angle, eps=1e-6):
            """Convert axis-angle to rotation matrix using Rodrigues' formula"""
            angle = torch.norm(axis_angle, dim=-1, keepdim=True)
            
            # Handle small angles for numerical stability
            small_angle_mask = (angle < eps).squeeze(-1)
            angle = torch.where(angle < eps, eps, angle)
            
            axis = axis_angle / angle
            skew = axis_angle_to_skew_matrix(axis)
            
            # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
            cos_angle = torch.cos(angle).unsqueeze(-1)
            sin_angle = torch.sin(angle).unsqueeze(-1)
            
            I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(skew)
            R = I + sin_angle * skew + (1 - cos_angle) * torch.bmm(skew, skew)
            
            # For very small angles, approximate as identity + skew
            R_small = I + axis_angle_to_skew_matrix(axis_angle)
            R = torch.where(small_angle_mask.unsqueeze(-1).unsqueeze(-1), R_small, R)
            
            return R
        
        # Apply conditioning by composition: R_new = R_conditioning @ R_original
        R_conditioning = axis_angle_to_rotation_matrix(axis_angle_conditioning)
        
        # Debug: Check rotation conditioning
        if hasattr(score_model, '_debug_conditioning') and score_model._debug_conditioning:
            print(f"        [Rotation Conditioning] axis_angle range: [{axis_angle_conditioning.min().item():.6f}, {axis_angle_conditioning.max().item():.6f}]")
            print(f"        [Rotation Conditioning] R_conditioning det: {torch.det(R_conditioning).mean().item():.6f} (should be ~1.0)")
            print(f"        [Rotation Conditioning] R_conditioning orthogonal error: {torch.norm(torch.bmm(R_conditioning.transpose(-1,-2), R_conditioning) - torch.eye(3, device=R_conditioning.device)).item():.6f}")
        
        batch.node_orientations = torch.bmm(R_conditioning, batch.node_orientations)
        
        # ALTERNATIVE SIMPLER APPROACH (comment out above and uncomment below):
        # from .so3_sde import rotvec_to_rotmat
        # R_conditioning = rotvec_to_rotmat(axis_angle_conditioning)
        # batch.node_orientations = torch.bmm(R_conditioning, batch.node_orientations)
        
        # ALTERNATIVE QUATERNION APPROACH (most numerically stable):
        # # Convert MLP output to quaternion (4D)
        # zero_linear_orient = nn.Linear(2 + cfg.model.mlcv_model.mlcv_dim, 4)  # -> quaternion
        # quat_conditioning = torch.tanh(axis_angle_conditioning)  # Bound the output
        # quat_conditioning = quat_conditioning / torch.norm(quat_conditioning, dim=-1, keepdim=True)  # Normalize
        # from .openfold.utils.rigid_utils import quat_to_rot
        # R_conditioning = quat_to_rot(quat_conditioning)
        # batch.node_orientations = torch.bmm(R_conditioning, batch.node_orientations)
        
    elif condition_mode == "backbone" and mlcv is not None:
        mlcv_expanded = mlcv.repeat_interleave(int(batch.batch.shape[0] / num_graphs), dim=0)
        batch.pos = batch.pos + score_model.model_nn.get_submodule("zero_conv_mlp")(torch.cat([batch.pos, mlcv_expanded], dim=1))
    
    elif condition_mode == "input" and mlcv is not None:
        # For input mode, conditioning is applied inside the score model during get_score() calls
        # No need to modify the batch here, just ensure MLCV is passed to get_score()
        # Debug: Verify input conditioning will work
        if cfg and cfg.log.debug_mlcv and hasattr(score_model, 'model_nn') and hasattr(score_model.model_nn, 'zero_conv_mlp'):
            print(f"        [dpm_solver] Input conditioning: MLCV will be applied via score model")
            print(f"        [dpm_solver] MLCV shape: {mlcv.shape}, range: [{mlcv.min().item():.3f}, {mlcv.max().item():.3f}]")
    
    elif condition_mode in ["latent", "input-control"] and mlcv is not None:
        # These modes should also work through the score model, not direct batch modification
        if cfg.log.debug_mlcv:
            print(f"        [dpm_solver] {condition_mode} conditioning: MLCV will be applied via score model")
    
    elif condition_mode != "none" and mlcv is not None:
        if cfg.log.debug_mlcv:
            print(f"        [dpm_solver] WARNING: Unknown condition_mode '{condition_mode}' - no conditioning applied!")
    
    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)

    for i in range(N - 1):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)

        # Evaluate score
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            # Debug: Check MLCV before score computation
            if cfg.log.debug_mlcv and i == 0 and condition_mode != "none" and mlcv is not None:
                print(f"        [dpm_solver] Step {i}: Passing MLCV to get_score")
                print(f"        [dmp_solver] MLCV shape: {mlcv.shape}, values: {mlcv.flatten()[:3]}")
            
            score = get_score(
                batch=batch,
                t=t,
                score_model=score_model,
                sdes=sdes,
                mlcv=mlcv,
            )
            
        # t_{i-1} in the algorithm is the current t
        batch_idx = batch.batch
        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch.pos, t=t, batch_idx=batch_idx)
        lambda_t = torch.log(alpha_t / sigma_t)
        alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t + dt, batch_idx=batch_idx
        )
        lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

        # t+dt < t, lambad_t_next > lambda_t
        h_t = lambda_t_next - lambda_t

        # For a given noise schedule (cosine is what we use), compute the intermediate t_lambda
        lambda_t_middle = (lambda_t + lambda_t_next) / 2
        t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)

        # t_lambda has all the same components
        t_lambda = torch.full((batch.num_graphs,), t_lambda[0][0], device=device)

        alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t_lambda, batch_idx=batch_idx
        )
        # Note in the paper the algorithm uses noise instead of score, but we use score.
        # So the formulation is slightly different in the prefactor.
        u = (
            alpha_t_lambda / alpha_t * batch.pos
            + sigma_t_lambda * sigma_t * (torch.exp(h_t / 2) - 1) * score["pos"]
        )

        # Update positions to the intermediate timestep t_lambda
        batch_u = batch.replace(pos=u)

        # Get node orientation at t_lambda

        # Denoise from t to t_lambda
        assert score["node_orientations"].shape == (u.shape[0], 3)
        assert batch.node_orientations.shape == (u.shape[0], 3, 3)
        so3_predictor = EulerMaruyamaPredictor(
            corruption=so3_sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch.node_orientations,
            score=score["node_orientations"],
            t=t,
            batch_idx=batch_idx,
        )
        sample, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=t_lambda[0] - t[0],
        )  # dt is negative, diffusion is 0
        assert sample.shape == (u.shape[0], 3, 3)
        batch_u = batch_u.replace(node_orientations=sample)

        # Correction step
        # Evaluate score at updated pos and node orientations
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            if cfg.log.debug_mlcv and i == 0 and condition_mode != "none" and mlcv is not None:
                print(f"        [dpm_solver] Correction step {i}: Passing MLCV to get_score")
            
            score_u = get_score(
                batch=batch_u,
                t=t_lambda,
                sdes=sdes,
                score_model=score_model,
                mlcv=mlcv,
            )

        pos_next = (
            alpha_t_next / alpha_t * batch.pos
            + sigma_t_next * sigma_t_lambda * (torch.exp(h_t) - 1) * score_u["pos"]
        )

        batch_next = batch.replace(pos=pos_next)

        assert score_u["node_orientations"].shape == (u.shape[0], 3)

        # Try a 2nd order correction
        node_score = (
            score_u["node_orientations"]
            + 0.5
            * (score_u["node_orientations"] - score["node_orientations"])
            / (t_lambda[0] - t[0])
            * dt
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch_u.node_orientations,
            score=node_score,
            t=t_lambda,
            batch_idx=batch_idx,
        )
        sample, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=dt,
        )  # dt is negative, diffusion is 0
        batch = batch_next.replace(node_orientations=sample)

    return batch


def training_rollout_denoiser(
    *,
    sdes: dict[str, SDE],
    batch: Batch,
    score_model: torch.nn.Module,
    device: torch.device,
    eps_t: float = 0.001,  # Final target (unused, for compatibility) 
    mid_t: float = 0.786,  # Stop point for dpm_solver (matches training)
    max_t: float = 0.99,
    N: int = 7,           # Match training N_rollout
    mlcv: torch.Tensor = None,
    condition_mode: str = "none",
    cfg: OmegaConf = None,
) -> ChemGraph:
    """
    Denoiser that exactly replicates the training rollout process:
    1. dpm_solver from max_t to mid_t (partial denoising)
    2. Direct jump from mid_t to clean x0 using score prediction
    
    This matches the training process in bioemu/src/bioemu/training/loss.py:_rollout
    """
    print(f"        [training_rollout_denoiser] Using training-matched rollout process")
    print(f"        [training_rollout_denoiser] Step 1: dmp_solver {max_t} → {mid_t} ({N} steps)")
    
    batch_size = batch.num_graphs

    # Step 1: Perform partial denoising (same as training)
    x_mid: ChemGraph = dpm_solver(
        sdes=sdes,
        batch=batch,
        eps_t=mid_t,        # Stop at mid_t (0.786)
        max_t=max_t,        # Start from max_t (0.99)
        N=N,                # Use training N_rollout (7)
        device=device,
        score_model=score_model,
        mlcv=mlcv,          # Pass MLCV for conditioning
        condition_mode=condition_mode,
        record_grad_steps=set(),  # No gradients during sampling,
        cfg=cfg,
    )

    print(f"        [training_rollout_denoiser] Step 2: Direct jump {mid_t} → clean x0")
    
    # Step 2: Direct jump to clean x0 (same as training)
    mid_t_expanded = torch.full((batch_size,), mid_t, device=device)
    score_mid_t = get_score(
        batch=x_mid, 
        sdes=sdes, 
        t=mid_t_expanded, 
        score_model=score_model,
        mlcv=mlcv,  # Pass MLCV for conditioning
    )["pos"]

    # Compute clean positions using the same formula as training
    x0_pos = _get_x0_given_xt_and_score_sampling(
        sde=sdes["pos"],
        x=x_mid.pos,
        t=torch.full((batch_size,), mid_t, device=device),
        batch_idx=x_mid.batch,
        score=score_mid_t,
    )

    print(f"        [training_rollout_denoiser] Rollout complete: generated clean structures")
    
    # Return clean sample (for orientations, use the mid_t result)
    return x_mid.replace(pos=x0_pos)


def _get_x0_given_xt_and_score_sampling(
    sde: SDE,
    x: torch.Tensor,
    t: torch.Tensor,
    batch_idx: torch.LongTensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """
    Compute x_0 given x_t and score.
    Copied from bioemu/src/bioemu/training/loss.py for sampling compatibility.
    """
    from .so3_sde import SO3SDE  # Import here to avoid circular imports
    assert not isinstance(sde, SO3SDE)

    alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx)

    return (x + sigma_t**2 * score) / alpha_t
