''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False, num_primes=1):

        super().__init__()

        self.order_size = 7 + 1      # NKCYXRS + 1
        self.order_emb = nn.Embedding(self.order_size, d_word_vec, padding_idx=pad_idx)
        self.tile_size = 30 + 1      # 12+1(start)
        # self.tile2_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        # self.tile3_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        # self.tile5_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        # self.tile7_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        self.tile_emb_list = nn.ModuleList([nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
                                            for _ in range(num_primes)])
        self.parallel_size = 2 + 1  # 0, 1
        self.sp_tile2_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        # self.sp_tile3_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        # self.sp_tile7_emb = nn.Embedding(self.tile_size, d_word_vec, padding_idx=pad_idx)
        # self.parallel_emb = nn.Embedding(self.parallel_size, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.num_primes = num_primes

    def forward(self, trg_seq, trg_mask, return_attns=False):
        # TODO
        # trg_seq [batch,level*7,5]
        # trg_mask [batch, level*7]
        dec_slf_attn_list = []

        # -- Forward
        dec_output = self.order_emb(trg_seq[:, :, 0])
        for i in range(self.num_primes):
            dec_output += self.tile_emb_list[i](trg_seq[:, :, i + 1])
        dec_output += self.sp_tile2_emb(trg_seq[:, :, self.num_primes + 2])

        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, slf_attn_mask=trg_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []

        # if return_attns:
        #     return dec_output, dec_slf_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, scale_emb_or_prj='prj', num_primes=1):

        super().__init__()

        self.trg_pad_idx = None

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        self.num_primes = num_primes

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=self.trg_pad_idx, dropout=dropout, scale_emb=scale_emb, num_primes=num_primes)

        # self.order_prj = nn.Linear(d_model, self.decoder.order_size, bias=False)
        # self.tile2_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)
        # self.tile3_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)
        # self.tile5_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)
        # self.tile7_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)
        # self.parallel_prj = nn.Linear(d_model, self.decoder.parallel_size, bias=False)
        self.tile_prj_list = nn.ModuleList([nn.Linear(d_model, self.decoder.tile_size, bias=False)
                                            for _ in range(num_primes)])
        self.sp_tile2_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)
        # self.sp_tile3_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)
        # self.sp_tile7_prj = nn.Linear(d_model, self.decoder.tile_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            # self.order_prj.weight = self.decoder.order_emb.weight
            # self.tile2_prj.weight = self.decoder.tile2_emb.weight
            # self.tile3_prj.weight = self.decoder.tile3_emb.weight
            # self.tile5_prj.weight = self.decoder.tile5_emb.weight
            # self.tile7_prj.weight = self.decoder.tile7_emb.weight
            # self.parallel_prj.weight = self.decoder.parallel_emb.weight
            for i in range(num_primes):
                self.tile_prj_list[i].weight = self.decoder.tile_emb_list[i].weight
            self.sp_tile2_prj.weight = self.decoder.sp_tile2_emb.weight
            # self.sp_tile3_prj.weight = self.decoder.sp_tile3_emb.weight
            # self.sp_tile7_prj.weight = self.decoder.sp_tile7_emb.weight

    def forward(self, trg_seq):
        # trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        trg_mask = None
        dec_output = self.decoder(trg_seq, trg_mask)[:,-1,:]
        # order_logit = self.order_prj(dec_output)
        # tile2_logit = self.tile2_prj(dec_output)
        # tile3_logit = self.tile3_prj(dec_output)
        # tile5_logit = self.tile5_prj(dec_output)
        # tile7_logit = self.tile7_prj(dec_output)
        # parallel_logit = self.parallel_prj(dec_output)
        tile_logits = []
        for i in range(self.num_primes):
            tile_logits.append(self.tile_prj_list[i](dec_output))
        tile_logits = torch.stack(tile_logits, dim=1)
        sp_tile2_logit = self.sp_tile2_prj(dec_output)
        # sp_tile3_logit = self.sp_tile3_prj(dec_output)
        # sp_tile7_logit = self.sp_tile7_prj(dec_output)
        if self.scale_prj:
            # order_logit *= self.d_model ** -0.5
            # tile2_logit *= self.d_model ** -0.5
            # tile3_logit *= self.d_model ** -0.5
            # tile5_logit *= self.d_model ** -0.5
            # tile7_logit *= self.d_model ** -0.5
            # parallel_logit *= self.d_model ** -0.5
            tile_logits *= self.d_model ** -0.5
            sp_tile2_logit *= self.d_model ** -0.5
            # sp_tile3_logit *= self.d_model ** -0.5
            # sp_tile7_logit *= self.d_model ** -0.5

        return tile_logits, sp_tile2_logit
