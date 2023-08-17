import torch.nn as nn
import os



class CrossViewTransformer(nn.Module):   # cvt模型
    def __init__(
        self,
        encoder, #_target_: cross_view_transformer.model.encoder.Encoder
        decoder, # _target_: cross_view_transformer.model.decoder.Decoder
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}

    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),        # 归一化
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))

    def forward(self, batch):
        x = self.encoder(batch)  # batch
        y = self.decoder(x)
        z = self.to_logits(y)

        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}
