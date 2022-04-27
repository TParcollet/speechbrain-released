"""This lobes proposes different models based on the MLP Mixer architecture

source:

Authors
 * Titouan Parcollet 2020
"""
import torch
import random
import numpy as np
from torch import nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
from speechbrain.lobes.models.convolution import PositionalConvEmbedding
from speechbrain.dataio.dataio import length_to_mask


class MixAndMLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_blocks: int,
        location_mixers: list,
        dropout_rate: float,
        activation=torch.nn.GELU(),
        positional_encoding=None,
        feature_mixing=True,
        max_length=3000,
        pooling_layers=None,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_mixing = feature_mixing
        self.max_length = max_length
        self.activation = activation
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.positional_encoding = positional_encoding
        assert self.positional_encoding in ["fixed_abs_sine", None]
        if self.positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(
                hidden_size, self.max_length
            )
        elif self.positional_encoding == "relative_conv":
            self.positional_encoding = PositionalConvEmbedding(hidden_size)
        elif positional_encoding is None:
            pass
            # no positional encodings

        self.input_projection = nn.Linear(input_size, hidden_size)  # TOTEST
        self.lms = nn.ModuleList(location_mixers)
        self.pooling_layers = nn.ModuleList(pooling_layers)

        self.mlps = nn.ModuleList(
            [MLP(hidden_size, hidden_size) for _ in range(num_blocks)]
        )

        self.ln1s = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_blocks)]
        )
        self.ln2s = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, wav_lens, pad_idx):

        # reshpae the src vector to [Batch, Time, Fea] is a 4d vector is given
        if x.ndim == 4:
            bz, t, ch1, ch2 = x.shape
            x = x.view(bz, t, ch1 * ch2)

        # Computing padding masks
        pad_masks = self.make_mask(x, wav_lens, pad_idx).unsqueeze(-1)

        # (B, T, F)
        out = self.input_projection(x)

        # add pos embeddings
        if self.positional_encoding is not None:
            out = out + self.positional_encoding(out)

        masked_input = out * pad_masks

        for ln1, lm, ln2, mlp, pool in zip(
            self.ln1s, self.lms, self.ln2s, self.mlps, self.pooling_layers
        ):

            out = ln1(masked_input)
            out = self.dropout(out)
            out = out.transpose(1, 2)
            # (B, F, T)
            out = lm(out, pad_masks)
            out = out.transpose(1, 2)
            # (B, T, F)
            out = masked_input + out

            if self.feature_mixing:
                out = ln2(out)
                out = masked_input + mlp(out)

            # pool
            # out = pool(out)

        # (B, F, T)
        # out = out.permute(0, 2, 1)

        # if self.mode == "max_pooling":
        #    out = torch.max(out, 2)[0]
        # elif self.mode == "mean_pooling":
        #    out = out.transpose(1, 2) * attention_mask.unsqueeze(-1)
        #    length = attention_mask.sum(1)
        #    out = out.sum(1) / length.unsqueeze(-1)
        # else:
        #    out = out[:, :, 0]

        return out

    def make_mask(self, src, wav_len, pad_idx=0):
        """This method generates the masks for training the HyperMixer model.

        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        tgt : tensor
            The sequence to the decoder (required).
        pad_idx : int
            The index for <pad> token (default=0).
        """

        abs_len = torch.round(wav_len * src.shape[1])
        src_padding_mask = length_to_mask(abs_len).bool()

        return src_padding_mask


class HyperMixer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        hypernet_size,
        dropout_rate,
        num_blocks,
        activation=torch.nn.GELU(),
        positional_encoding=None,
        feature_mixing=True,
        max_length=3000,
        tied=False,
        inter_layer_pooling_size=None,
    ):
        super().__init__()
        location_mixers = [
            HyperMixerLayer(hidden_size, hidden_size, tied)
            for _ in range(num_blocks)
        ]

        pooling_layers = [
            sb.nnet.pooling.Pooling1d(
                pool_type="max", input_dims=3, kernel_size=size, pool_axis=1,
            )
            for size in inter_layer_pooling_size
        ]

        self.model = MixAndMLP(
            input_size,
            hidden_size,
            num_blocks,
            location_mixers,
            dropout_rate,
            activation,
            positional_encoding,
            feature_mixing,
            max_length,
            pooling_layers,
        )

    def forward(self, x, wav_len, pad_idx=0):

        return self.model(x, wav_len, pad_idx)


class HyperMixerLayer(nn.Module):
    def __init__(
        self, input_output_dim: int, hypernet_size: int, tied=False
    ) -> None:
        super().__init__()
        self.hyper = HyperNetwork(input_output_dim, hypernet_size, tied=tied)
        self.activation = nn.GELU()

    def forward(self, out, attention_mask):

        # add position embedding before passing to hypernetwork
        hyp_input = out.transpose(1, 2)
        W1, W2 = self.hyper(hyp_input)

        # we stick MLP1 together manually
        out = _mlp_pass_from_components(out, W1, W2, self.activation)
        return out


class HyperNetwork(nn.Module):
    def __init__(
        self, input_output_dim: int, hypernet_size: int, tied=False
    ) -> None:
        super().__init__()

        self.tied = tied
        self.w1_gen = MLP(input_output_dim, hypernet_size)
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = MLP(input_output_dim, hypernet_size)

    def forward(self, position_embeddings: torch.Tensor):
        """
        position embeddings : [batchsize, max_positions, d]
        The HyperNetwork is supposed to generate an MLP of the form W_2(GELU(W1 x)), where
        W1 : N -> k and W2 : k -> N, so it has to return W1 and W2
        """

        W1 = self.w1_gen(position_embeddings)
        W2 = self.w2_gen(position_embeddings)

        return W1, W2


def MLP(in_dim: int, h_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, h_dim, bias=False),
        nn.GELU(),
        nn.Linear(h_dim, in_dim, bias=False),
    )


def _mlp_pass_from_components(out, W1, W2, activation):
    # we stick MLP1 together manually
    out = torch.bmm(out, W1)
    out = activation(out)
    out = torch.bmm(out, W2.transpose(1, 2))
    return out


def compute_mask(shape, padding_mask, mask_prob, mask_length):
    bs, padded_sample_len = shape

    if padding_mask is not None:
        sample_lens = (padded_sample_len - torch.sum(padding_mask, -1)).type(
            torch.int32
        )
    else:
        sample_lens = torch.ones(bs, dtype=torch.int32) * padded_sample_len

    min_sample_len = sample_lens.min()
    # So we dont have ragged tensors number of masks is the same for each sample.
    num_mask = int(
        mask_prob * min_sample_len / float(mask_length) + random.random()
    )

    mask_idcs = []
    for i in range(bs):
        sample_len = sample_lens[i].item()
        mask_indices = np.random.choice(
            sample_len - mask_length, num_mask, replace=False
        )

        mask_indices = np.asarray(
            [
                mask_indices[j] + offset
                for j in range(len(mask_indices))
                for offset in range(mask_length)
            ]
        )
        mask_idcs.append(np.unique(mask_indices[mask_indices < sample_len]))

    mask = np.full((bs, padded_sample_len), False)
    # unique may have caused num masks to go down..
    num_mask_total = num_mask * mask_length
    for i, mask_idc in enumerate(mask_idcs):
        # ..so we dont have ragged tensors we need to make the number of masked elements the same for each sample in the batch
        if padding_mask is not None and len(mask_idc) < num_mask_total:
            num_mask_missing = num_mask_total - len(mask_idc)
            arange = np.arange(sample_lens[i].item())
            arange = np.delete(arange, mask_idc)
            extra_indcs = np.random.choice(
                arange, num_mask_missing, replace=False
            )
            mask[i, extra_indcs] = True
        mask[i, mask_idc] = True
    return mask
