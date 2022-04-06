"""This lobes proposes different models based on the MLP Mixer architecture

source:

Authors
 * Titouan Parcollet 2020
"""
import torch
from torch import nn
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
        elif self.positional_encoder == "relative_conv":
            self.positional_encoding = PositionalConvEmbedding(hidden_size)
        elif positional_encoding is None:
            pass
            # no positional encodings

        self.input_projection = nn.Linear(input_size, hidden_size)  # TOTEST
        self.lms = nn.ModuleList(location_mixers)
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

        # Computing padding masks
        pad_masks = self.make_mask(x, wav_lens, pad_idx)

        # (B, T, F)
        out = self.input_projection(x)

        # add pos embeddings
        if self.positional_encoding is not None:
            out = out + self.positional_encoding(out)

        for ln1, lm, ln2, mlp in zip(self.ln1s, self.lms, self.ln2s, self.mlps):

            masked_input = out * pad_masks
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
        src_padding_mask = ~length_to_mask(abs_len).bool()

        return src_padding_mask


def HyperMixer(
    input_size,
    hidden_size,
    dropout_rate,
    num_blocks,
    activation=torch.nn.GELU(),
    positional_encoding=None,
    feature_mixing=True,
    max_length=3000,
    tied=False,
):
    location_mixers = [
        HyperMixerLayer(hidden_size, hidden_size, tied)
        for _ in range(num_blocks)
    ]
    return MixAndMLP(
        input_size,
        hidden_size,
        num_blocks,
        location_mixers,
        dropout_rate,
        activation,
        positional_encoding,
        feature_mixing,
        max_length,
    )


class HyperMixerLayer(nn.Module):
    def __init__(
        self, embedding_dim: int, hidden_layer_size: int, tied=False
    ) -> None:
        super().__init__()
        self.hyper = HyperNetwork(embedding_dim, hidden_layer_size, tied=tied)
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
        self, embedding_dim: int, hidden_layer_size: int, tied=False
    ) -> None:
        super().__init__()

        self.tied = tied
        self.w1_gen = MLP(embedding_dim, hidden_layer_size)
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = MLP(embedding_dim, hidden_layer_size)

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
