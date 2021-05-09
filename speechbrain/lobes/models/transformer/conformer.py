"""Conformer implementaion in the SpeechBrain sytle.

Authors
* Jianyuan Zhong 2020
"""

import torch
import torch.nn as nn
from typing import Optional

from speechbrain.nnet.attention import (
    RelPosMHAXL,
    MultiheadAttention,
    PositionalwiseFeedForward,
)
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.nnet.activations import Swish


class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ----------
    input_size : int
        The expected size of the input embedding.
    dropout : int
        Dropout for the encoder (Optional).
    bias: bool
        Bias to convolution module.
    kernel_size: int
        Kernel size of convolution model.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = ConvolutionModule(512, 3)
    >>> output = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self, input_size, kernel_size, bias=True, activation=Swish, dropout=0.1
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_size)
        self.convolution_module = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
            # depthwise
            nn.Conv1d(
                input_size,
                input_size,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=input_size,
                bias=bias,
            ),
            nn.BatchNorm1d(input_size),
            activation(),
            # pointwise
            nn.Conv1d(
                input_size, input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.convolution_module(out)
        out = out.transpose(1, 2)
        return out


class ConformerEncoderLayer(nn.Module):
    """This is an implementation of Conformer encoder layer.

    Arguments
    ----------
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    nhead : int
        Number of attention heads.
    d_model : int
        The expected size of the input embedding.
    reshape : bool
        Whether to automatically shape 4-d input to 3-d.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : int
        Dropout for the encoder (Optional).
    bias : bool
        Bias to convolution module.
    kernel_size : int
        Kernel size of convolution model.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_embs = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoderLayer(d_ffn=512, nhead=8, d_model=512, kernel_size=3)
    >>> output = net(x, pos_embs=pos_embs)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.1,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )
        elif attention_type == "RelPosMHAXL":
            # transformerXL style positional encoding
            self.mha_layer = RelPosMHAXL(
                num_heads=nhead,
                embed_dim=d_model,
                dropout=dropout,
                mask_pos_future=causal,
            )

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)

        # muti-head attention module
        skip = x
        x = self.norm1(x)
        x, self_attn = self.mha_layer(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn


class ConformerEncoder(nn.Module):
    """This class implements the Conformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of Conformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    input_shape : tuple
        Expected shape of an example input.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module : torch class
        The module to process the source input feature to expected
        feature dimension (Optional).

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> pos_emb = torch.rand((1, 2*60-1, 512))
    >>> net = ConformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x, pos_embs=pos_emb)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation=Swish,
        kernel_size=31,
        bias=True,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if input_shape is None and d_model is None:
            raise ValueError("Expected one of input_shape or d_model")

        if input_shape is not None and d_model is None:
            if len(input_shape) == 3:
                msg = "Input shape of the Transformer must be (batch, time, fea). Please revise the forward function in TransformerInterface to handel arbitary shape of input."
                raise ValueError(msg)
            d_model = input_shape[-1]

        self.layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        attention_lst = []
        print(src.shape)
        for enc_layer in self.layers:
            output, attention = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
            )
            attention_lst.append(attention)
        output = self.norm(output)

        return output, attention_lst
