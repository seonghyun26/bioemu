import torch
import lightning
from typing import Optional

from mlcolvar.cvs import BaseCV, VariationalAutoEncoderCV
from mlcolvar.core import FeedForward, Normalization
from mlcolvar.core.loss.elbo import elbo_gaussians_loss
from mlcolvar.core.transform import Transform


class DIM_NORMALIZATION(Transform):
    def __init__(
        self,
        feature_dim = 1,
        normalization_factor = 10,
    ):
        super().__init__(in_features=feature_dim, out_features=feature_dim)
        self.register_buffer("feature_dim", torch.tensor(feature_dim))
        self.normalization_factor = normalization_factor
        
    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=-1) * self.normalization_factor
        return x
    

class MLCV(BaseCV, lightning.LightningModule):
    BLOCKS = ["norm_in", "encoder",]

    def __init__(
        self,
        mlcv_dim: int,
        dim_normalization: bool,
        encoder_layers: list,
        normalization_factor: float = 1.0,
        options: dict = None,
        **kwargs,
    ):
        super().__init__(in_features=encoder_layers[0], out_features=encoder_layers[-1], **kwargs)
        # ======= OPTIONS =======
        options = self.parse_options(options)
        
        # ======= BLOCKS =======
        # initialize norm_in
        o = "norm_in"
        if (options[o] is not False) and (options[o] is not None):
            self.norm_in = Normalization(self.in_features, **options[o])

        # initialize encoder
        o = "encoder"
        self.encoder = FeedForward(encoder_layers, **options[o])
        if dim_normalization:
            self.postprocessing = DIM_NORMALIZATION(
                feature_dim=mlcv_dim,
                normalization_factor=normalization_factor,
            )


class VDELoss(torch.nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        output: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        z_t: torch.Tensor,
        z_t_tau: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        elbo_loss = elbo_gaussians_loss(target, output, mean, log_variance, weights)
        auto_correlation_loss = 0
        
        z_t_mean = z_t.mean(dim=0)
        z_t_tau_mean = z_t_tau.mean(dim=0)
        z_t_centered = z_t - z_t_mean.repeat(z_t.shape[0], 1)
        z_t_tau_centered = z_t_tau - z_t_tau_mean.repeat(z_t_tau.shape[0], 1)
        
        # auto_correlation_loss = - (z_t_centered @ z_t_tau_centered.T)[torch.eye(z_t.shape[0], dtype=torch.bool, device = z_t.device)].mean()
        # auto_correlation_loss = auto_correlation_loss / (z_t.std(dim=0).T @ z_t_tau.std(dim=0))
        ac_num = z_t_centered.reshape(1, -1) @ z_t_tau_centered.reshape(-1, 1)
        ac_den = z_t_centered.norm(2) * z_t_tau_centered.norm(2)
        auto_correlation_loss = - ac_num / ac_den
        
        return elbo_loss, auto_correlation_loss
        
        
class VariationalDynamicsEncoder(VariationalAutoEncoderCV):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # =======   LOSS  =======
        # ELBO loss function when latent space and reconstruction distributions are Gaussians.
        self.loss_fn = VDELoss()
        self.optimizer = Adam(self.parameters(), lr=1e-4)
    
    def backward(self, loss):
        loss.backward(retain_graph=True)
    
    def training_step(
        self,
        train_batch, 
        batch_idx
    ):
        x = train_batch["data"]
        input = x
        loss_kwargs = {}
        if "weights" in train_batch:
            loss_kwargs["weights"] = train_batch["weights"]

        # Encode/decode.
        mean, log_variance, x_hat = self.encode_decode(x)

        # Reference output (compare with a 'target' key if any, otherwise with input 'data')
        if "target" in train_batch:
            x_ref = train_batch["target"]
        else:
            x_ref = x
        
        # Values for autocorrealtion loss
        if self.norm_in is not None:
            input_normalized = self.norm_in(input)
            x_ref_normalized = self.norm_in(x_ref)
        z_t = self.encoder(input_normalized)
        z_t_tau = self.encoder(x_ref_normalized)
        
        # Loss function.
        elbo_loss, auto_correlation_loss = self.loss_fn(
            x_ref, x_hat, mean, log_variance,
            z_t, z_t_tau,
            **loss_kwargs
        )

        # Log.
        name = "train" if self.training else "valid"
        self.log(f"{name}_elbo_loss", elbo_loss, on_epoch=True)
        self.log(f"{name}_auto_correlation_loss", auto_correlation_loss, on_epoch=True)
        self.log(f"{name}_loss", elbo_loss + auto_correlation_loss, on_epoch=True)

        return elbo_loss + auto_correlation_loss


def load_ours(
    mlcv_dim: int = 2,
    dim_normalization: bool = False,
    normalization_factor: float = 1.0,
    **kwargs,
):
    encoder_layers = [45, 100, 100, mlcv_dim]
    options = {
        "encoder": {
            "activation": "tanh",
            "dropout": [0.1, 0.1, 0.1]
        },
        "norm_in": {
        },
    }
    mlcv_model = MLCV(
        mlcv_dim = mlcv_dim,
        dim_normalization = dim_normalization,
        normalization_factor = normalization_factor,
        options = options,
        encoder_layers = encoder_layers,
    )
    mlcv_model.train()
    print(mlcv_model)
    
    return mlcv_model

def load_tda():
    model = torch.jit.load("/home/shpark/prj-mlcv/lib/bioemu/model/tda-jit.pt")
    model.eval()
    print(model)
    return model

def load_vde():
    model = torch.jit.load("/home/shpark/prj-mlcv/lib/bioemu/model/vde-jit.pt")
    model.eval()
    print(model)
    return model
