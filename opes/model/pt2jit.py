import torch
import lightning

from mlcolvar.cvs import BaseCV
from mlcolvar.core import FeedForward, Normalization
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



def convert_pt_to_jit(pt_file_path):
    """
    Convert a saved .pt file with key 'mlcv_model' to a JIT model.
    
    Args:
        pt_file_path (str): Path to the input .pt file
        output_path (str): Path for the output JIT model (default: 'mlcv-jit.pt')
    """
    pt_file_path = f"./{date}.pt"
    print(f"Loading model from: {pt_file_path}")
    checkpoint = torch.load(pt_file_path, map_location='cpu')
    options = {
      "encoder": {
          "activation": "tanh",
          "dropout": [0.1, 0.1, 0.1]
      },
      "norm_in": {
      },
    }
    mlcv_dim = 2
    mlcv_model = MLCV(
      mlcv_dim = mlcv_dim,
      dim_normalization = False,
      normalization_factor = 1.0,
      options = options,
      encoder_layers = [45, 100, 100, mlcv_dim],
    )
    mlcv_model.load_state_dict(checkpoint['mlcv_state_dict'])
    mlcv_model.eval()
    
    print("Converting model to JIT...")
    try:
        mlcv_model.trainer = lightning.Trainer(logger=None, enable_checkpointing=False, enable_model_summary=False)
        sample_input = torch.randn(1, 45).to(mlcv_model.device)
        jit_model = torch.jit.trace(mlcv_model, sample_input)
        output_path = f"./{date}-jit.pt"
        print(f"Saving JIT model to: {output_path}")
        torch.jit.save(jit_model, output_path)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Failed to convert using torch.jit.trace: {e}")
    


if __name__ == "__main__":
    date = "0729_081523"
    convert_pt_to_jit(date)


