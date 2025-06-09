#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiHeadAttention, FeedForward, FeedForwardConformer, ConvolutionModule, RelativeMultiHeadAttention, RMSNorm, MultiHeadAttentionWeight, ConvolutionGated

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, rms_norm=False):
        super().__init__()
        if rms_norm:
            self.norm_1 = RMSNorm(d_model)
            self.norm_2 = RMSNorm(d_model)
            self.norm_3 = RMSNorm(d_model)
        else:
            self.norm_1 = nn.LayerNorm(d_model)
            self.norm_2 = nn.LayerNorm(d_model)
            self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, d_model_q=d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, d_model_q=d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x2, attn_dec_dec = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)
        x2, attn_dec_enc = self.attn_2(x2, e_outputs, e_outputs, src_mask)
        x = x + self.dropout_2(x2)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, attn_dec_dec, attn_dec_enc

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, d_model_q=d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x2, attn_enc_enc = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, attn_enc_enc

class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, batchnorm_momentum=0.1, rms_norm=False):
        super().__init__()
        self.ff_1 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)
        if rms_norm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(heads, d_model, dropout=dropout)
        self.conv_module = ConvolutionModule(d_model, dropout=dropout, batchnorm_momentum=batchnorm_momentum, rms_norm=rms_norm)
        self.ff_2 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout, rms_norm=rms_norm)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, pe, mask):
        x = x + 0.5 * self.ff_1(x)
        res = x
        x = self.norm(x)
        x, attn_enc_enc = self.attn(x,x,x,pe,mask)
        x = res + self.dropout_1(x)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_2(x)
        return x, attn_enc_enc

class ZipformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropput=0.1):
        # OK
        self.ff_1 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)

        self.attn_weight = MultiHeadAttentionWeight(heads, d_model, d_model_q=d_model, dropout=dropout)
        # non-linear attention is not used.
        #self.nla = NonLinearAttention()
        
    def forwardf(self, x, mask):
        res = x
        weight = self.attn_weight(x)
        x = x + self.ff_1(x)
        
        return x


class DenceformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model) 
        self.attn = RelativeMultiHeadAttention(heads, d_model, dropout=dropout)  
        self.ln_2 = nn.LayerNorm(d_model)
        #self.mlp = MLP(config)
        self.ff = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout, rms_norm=rms_norm)

    def forward(self, x, pe, mask):
        res = x
        x = self.ln_1(x)
        x = res + self.attn(x, x, x, pe, mask)
        x = x + self.ff(self.ln_2(x))
        return x


class EBranchFormerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, rms_norm=False, macaron_style=False):
        super().__init__()
        self.macaron_style = macaron_style

        if macaron_style:
            self.ln_macaron = nn.LayerNorm(d_model)
            self.ff_macaron = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        # def __init__(self, dim: int, hidden_dim: int, kernel_size: int, dropout: float=0.1):
        self.gated = ConvolutionGated(d_model, hidden_dim=d_model*4, kernel_size=31)
        self.attn = RelativeMultiHeadAttention(heads, d_model, dropout=dropout)  
        self.ln_2 = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(d_model*2, d_model*2, kernel_size=7, padding=3, groups=d_model*2, bias=True)
        #self.mlp = MLP(config)
        self.ff = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout, rms_norm=rms_norm)

        self.dropout = nn.Dropout(dropout)
        self.merge_proj = nn.Linear(d_model*2, d_model)
        self.ln_3 = nn.LayerNorm(d_model)
        self.ln_4 = nn.LayerNorm(d_model)

    def forward(self, x, pe, mask):

        if self.macaron_style:
            x_macaron = self.ln_macaron(x)
            x = x + 0.5 * self.dropout(self.ff_macaron(x_macaron))

        x1 = x
        x2 = x
        
        # multi-head attention 
        x1 = self.ln_1(x1)
        x1, attn_enc_enc = self.attn(x1, x1, x1, pe, mask)
        x1 = self.dropout(x1)

        x2 = self.ln_2(x2)
        x2 = self.gated(x2)
        x2 = self.dropout(x2)

        x_concat = torch.cat([x1, x2], dim=-1)
        x_concat = self.depthwise(x_concat.transpose(1,2)).transpose(1,2)
        x = x + self.dropout(self.merge_proj(x_concat))

        res = x
        x = self.ln_3(x)

        x = res + 0.5 * self.dropout(self.ff(x))

        x = self.ln_4(x)
        return x, attn_enc_enc


