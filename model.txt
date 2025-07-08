score_model
DiGConditionalScoreModel(
  (model_nn): DistributionalGraphormer(
    (step_emb): SinusoidalPositionEmbedder()
    (x1d_proj): Sequential(
      (0): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      (1): Linear(in_features=384, out_features=512, bias=False)
    )
    (x2d_proj): Sequential(
      (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (1): Linear(in_features=128, out_features=256, bias=False)
    )
    (rp_proj): RelativePositionBias(
      (relative_attention_bias): Embedding(64, 256)
    )
    (st_module): StructureModule(
      (encoder): SAEncoder(
        (layers): ModuleList(
          (0-7): 8 x SAEncoderLayer(
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (attn): SAAttention(
              (scalar_query): Linear(in_features=512, out_features=512, bias=False)
              (scalar_key): Linear(in_features=512, out_features=512, bias=False)
              (scalar_value): Linear(in_features=512, out_features=512, bias=False)
              (pair_bias): Linear(in_features=256, out_features=32, bias=False)
              (point_query): Linear(in_features=512, out_features=384, bias=False)
              (point_key): Linear(in_features=512, out_features=384, bias=False)
              (point_value): Linear(in_features=512, out_features=768, bias=False)
              (pair_value): Linear(in_features=256, out_features=512, bias=False)
              (fc_out): Linear(in_features=2048, out_features=512, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (ffn): FeedForward(
              (ff): Sequential(
                (0): Linear(in_features=512, out_features=1024, bias=True)
                (1): GELU(approximate='none')
                (2): Dropout(p=0.1, inplace=False)
                (3): Linear(in_features=1024, out_features=512, bias=True)
                (4): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (diff_head): DiffHead(
        (fc_t): Sequential(
          (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=512, out_features=512, bias=True)
          (2): ReLU()
          (3): Linear(in_features=512, out_features=3, bias=True)
        )
        (fc_eps): Sequential(
          (0): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (1): Linear(in_features=512, out_features=512, bias=True)
          (2): ReLU()
          (3): Linear(in_features=512, out_features=3, bias=True)
        )
      )
    )
  )
)
