import torch
import math
import torch.nn as nn
import numpy as np

"""
1-Attention 注意力机制
2-MultiAttention 多头注意力机制
3-MultiSelfAttention 多头自注意力机制
4-AtrousSelfAttention 空洞多头注意力机制
5-LocalSelfAttention 局部多头注意力机制
6-SparseSelfAttention 稀疏多头注意力机制
"""

def to_mask(x, mask, mode='mul'):
    """
    mask: (batch_size, seq_len) 或 (batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(x.dim() - mask.dim()):
            mask = torch.unsequeeze(mask, mask.dim())
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10


def extract_seq_patches(x, kernel_size, step):
    seq_dim = x.shape[-1]
    seq_len = x.shape[1]
    k_size = kernel_size + (step - 1) * (kernel_size - 1)

    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    pad_left = torch.zeros(x.shape[0], p_left, seq_dim)
    x = torch.cat((pad_left, x), dim=1)
    pad_rigth = torch.zeros(x.shape[0], p_right, seq_dim)
    x = torch.cat((x, pad_rigth), dim=1)

    xs = [x[:, i: i + seq_len] for i in range(0, k_size, step)]
    x = torch.cat(xs, dim=2)

    x = torch.reshape(x, (-1, seq_len, kernel_size, seq_dim))
    return x


class Attention(nn.Module):
    """
    1-Attention 注意力机制
    """
    def __init__(self, q_dim, k_dim, v_dim, hidden_dim=8, dropout=0.5):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.W_q = nn.Linear(q_dim, hidden_dim)
        self.W_k = nn.Linear(k_dim, hidden_dim)
        self.W_v = nn.Linear(v_dim, hidden_dim)

        self.add_W_v = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, q, k, v, seq_len, mode='mul'):
        if mode == 'add':
            # 加性
            q = self.W_q(q)
            k = self.W_k(k)

            add_attn = q.unsqueeze(2) + k.unsqueeze(1)
            add_attn = torch.tanh(add_attn)
            print(add_attn.shape)
            output = self.add_W_v(add_attn)
            output = output.squeeze(-1)

        elif mode == 'mul':
            q = self.W_q(q)
            k = self.W_k(k)
            v = self.W_v(v)
            #点积
            attn = torch.matmul(q, k.transpose(-1, -2))
            print(attn.shape)
            # 缩放
            scale = 1.0 / math.sqrt(self.v_dim)
            attn *= scale
            # 计算 softmax
            attn = torch.softmax(attn, dim=-1)

            output = torch.matmul(attn, v)
        else:
            print('Not yet implemented')
        return output


class MultiAttention(nn.Module):
    """
    2-MultiAttention 多头注意力机制
    """
    def __init__(self, heads, size_per_head, embedidng_dim, key_size=None, mask_right=False, **kwargs):
        super(MultiAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        self.out_dim = heads * size_per_head
        self.embedding_dim = embedidng_dim

        self.W_q = nn.Linear(embedidng_dim, self.key_size * self.heads, bias=False)
        self.W_k = nn.Linear(embedidng_dim, self.key_size * self.heads, bias=False)
        self.W_v = nn.Linear(embedidng_dim, self.out_dim, bias=False)

    def forward(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]

        # Linear transfomer
        q_w, k_w, v_w = self.W_q(q), self.W_k(k), self.W_v(v)
        # change shape
        q_w = torch.reshape(q_w, (-1, q_w.shape[1], self.heads, self.key_size))
        k_w = torch.reshape(k_w, (-1, k_w.shape[1], self.heads, self.key_size))
        v_w = torch.reshape(v_w, (-1, v_w.shape[1], self.heads, self.key_size))
        # adjust dimension
        q_w = torch.permute(q_w, (0, 2, 1, 3))
        k_w = torch.permute(k_w, (0, 2, 1, 3))
        v_w = torch.permute(v_w, (0, 2, 1, 3))
        # caculate attention weight
        a = torch.matmul(q_w, k_w.transpose(-1, -2)) / math.sqrt(self.key_size)
        # mask
        a = torch.permute(a, (0, 3, 2, 1))
        a = to_mask(a, v_mask, 'add')
        a = torch.permute(a, (0, 3, 2, 1))
        print('a>>', a.shape)
        if (self.mask_right is not False) or isinstance(self.mask_right, torch.Tensor):
            if self.mask_right is True:
                ones = torch.ones_like(a[:1, :1])
                mask = (ones - torch.tril(ones, diagonal=0)) * 1e10
                a = a - mask
            else:
                # input mask matrix, (q_len, k_len)
                print('mask_right>>', self.mask_right.shape)
                mask =(1 - self.mask_right) * 1e10
                print('mask>>', mask.shape)
                mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
                self.mask = mask
                a = a - mask
        a = torch.softmax(a, dim=-1)
        self.a = a
        # caculate attention weight ouput
        o = torch.matmul(a, v_w)
        o = torch.permute(o, (0, 2, 1, 3))
        o = torch.reshape(o, (-1, o.shape[1], self.out_dim))
        o = to_mask(o, q_mask, 'mul')
        return o


class MultiSelfAttention(nn.Module):
    """
    3-MultiSelfAttention 多头自注意力机制
        时间复杂度 O(n^2)
    """
    def __init__(self, heads, size_per_head, embedidng_dim, key_size=None, mask_right=False, **kwargs):
        super(MultiSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        self.out_dim = heads * size_per_head
        self.embedding_dim = embedidng_dim

        self.attention = MultiAttention(self.heads,
                                        self.size_per_head,
                                        self.embedding_dim,
                                        self.key_size,
                                        self.mask_right)

    def forward(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
            o = self.attention([x, x, x, x_mask, x_mask])
        else:
            x = inputs
            o = self.attention([x, x, x])
        return o


class AtrousSelfAttention(nn.Module):
    """
    4-AtrousSelfAttention 空洞多头注意力机制
        每个元素只跟大约 n/k 个元素计算相关性，时间复杂度 O(n^2/k)
    """
    def __init__(self, heads, size_per_head, embedidng_dim, step=2,
                 key_size=None, mask_right=False, **kwargs):
        super(AtrousSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        self.out_dim = heads * size_per_head
        self.embedding_dim = embedidng_dim
        self.step = step

        self.attention = MultiAttention(self.heads,
                                        self.size_per_head,
                                        self.embedding_dim,
                                        self.key_size,
                                        self.mask_right)

    def forward(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        print('x>>', x.shape)
        seq_dim = x.shape[-1]
        seq_len = x.shape[1]
        pad_len = self.step - seq_len % self.step
        pad_part = torch.zeros(x.shape[0], pad_len, seq_dim)
        print('pad_part>>', pad_part.shape)
        x = torch.cat((x, pad_part), dim=1)
        print('x>>', x.shape)

        if x_mask is not None:
            mask_pad_part = torch.zeros(x_mask[0], pad_len, x_mask[-1])
            x_mask = torch.cat((x_mask, mask_pad_part), dim=1)
        new_seq_len = x.shape[1]
        blocks = new_seq_len // self.step

        # change shape
        x = torch.reshape(x, (-1, blocks, self.step, seq_dim))
        print('x after reshape>>', x.shape)
        x = torch.permute(x, (0, 2, 1, 3))
        print('x after permute>>', x.shape)
        x = torch.reshape(x, (-1, blocks, seq_dim))
        print('x after reshape>>', x.shape)
        if x_mask is not None:
            x_mask = torch.reshape(x_mask, (-1, blocks, self.step, 1))
            x_mask = torch.permute(x_mask, (0, 2, 1, 3))
            x_mask = torch.reshape(x_mask, (-1, blocks, 1))

        # caculate attention weight
        if x_mask is not None:
            o = self.attention([x, x, x, x_mask, x_mask])
        else:
            o = self.attention([x, x, x])

        # recover shape
        print('o >>', o.shape)
        o = torch.reshape(o, (-1, self.step, blocks, self.out_dim))
        o = torch.permute(o, (0, 2, 1, 3))
        o = torch.reshape(o, (-1, new_seq_len, self.out_dim))
        o = o[:, :- pad_len]
        return o


class LocalSelfAttention(nn.Module):
    """
    5-LocalSelfAttention 局部多头注意力机制
    """
    def __init__(self, heads, size_per_head, embedidng_dim, neighbors=2, step=1,
                 key_size=None, mask_right=False, **kwargs):
        super(LocalSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        self.out_dim = heads * size_per_head
        self.embedding_dim = embedidng_dim
        self.step = step
        self.neighbors = neighbors

        if self.mask_right:
            mask_right = torch.ones((1, 1 + 2 * self.neighbors))
            mask_right[:, - self.neighbors:] = 0
        else:
            mask_right = self.mask_right
        self.attention = MultiAttention(self.heads,
                                        self.size_per_head,
                                        self.embedding_dim,
                                        self.key_size,
                                        mask_right)

    def forward(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        print('x>>', x.shape)
        # extract local features
        kernel_size = 1 + 2 * self.neighbors
        xp = extract_seq_patches(x, kernel_size, self.step)
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.step)
        # change shape
        seq_len = x.shape[1]
        seq_dim = x.shape[-1]
        x = torch.reshape(x, (-1, 1, seq_dim))
        xp = torch.reshape(xp, (-1, kernel_size, seq_dim))
        if x_mask is not None:
            xp_mask = torch.reshape(xp_mask, (-1, kernel_size, 1))
        if x_mask is not None:
            o = self.attention([x, xp, xp, xp_mask])
        else:
            o = self.attention([x, xp, xp])
        # recover shape
        o = torch.reshape(o, (-1, seq_len, self.out_dim))
        o = to_mask(o, x_mask, 'mul')
        return o


class SparseSelfAttention(nn.Module):
    """
    6-SparseSelfAttention 稀疏多头注意力机制
        每个元素只与相对距离小于等于 step 的元素、相对距离为 step 的倍数元素相关
    """
    def __init__(self, heads, size_per_head, embedidng_dim, step=3,
                 key_size=None, mask_right=False, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        self.out_dim = heads * size_per_head
        self.embedding_dim = embedidng_dim
        assert step != 1, print('if rate = 1, use SelfAttention directly')
        self.step = step
        self.neighbors = step - 1

        self.W_q = nn.Linear(embedidng_dim, self.key_size * self.heads, bias=False)
        self.W_k = nn.Linear(embedidng_dim, self.key_size * self.heads, bias=False)
        self.W_v = nn.Linear(embedidng_dim, self.out_dim, bias=False)

    def forward(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        print('x>>', x.shape)
        seq_dim = x.shape[-1]
        seq_len = x.shape[1]
        pad_len = self.step - seq_len % self.step
        pad_part = torch.zeros(x.shape[0], pad_len, seq_dim)
        print('pad_part>>', pad_part.shape)
        x = torch.cat((x, pad_part), dim=1)
        print('x>>', x.shape)

        if x_mask is not None:
            mask_pad_part = torch.zeros(x_mask[0], pad_len, x_mask[-1])
            x_mask = torch.cat((x_mask, mask_pad_part), dim=1)
        new_seq_len = x.shape[1]
        blocks = new_seq_len // self.step
        x = torch.reshape(x, (-1, new_seq_len, seq_dim))  # 经过padding后shape可能变为None，所以重新声明一下shape

        # Linear transfomer
        q_w, k_w, v_w = self.W_q(x), self.W_k(x), self.W_v(x)

        # extract local features
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(k_w, kernel_size, self.step)  # (None, seq_len, kernel_size, out_dim)
        vwp = extract_seq_patches(v_w, kernel_size, self.step)  # (None, seq_len, kernel_size, out_dim)
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.step)

        # change shape
        q_w = torch.reshape(q_w, (-1, blocks, self.step, self.heads, self.key_size))
        k_w = torch.reshape(k_w, (-1, blocks, self.step, self.heads, self.key_size))
        v_w = torch.reshape(v_w, (-1, blocks, self.step, self.heads, self.size_per_head))
        kwp = torch.reshape(kwp, (-1, blocks, self.step, kernel_size, self.heads, self.size_per_head))
        vwp = torch.reshape(vwp, (-1, blocks, self.step, kernel_size, self.heads, self.size_per_head))
        if x_mask is not None:
            x_mask = torch.reshape(x_mask, (-1, blocks, self.step, 1, 1))
            xp_mask = torch.reshape(xp_mask, (-1, blocks, self.step, kernel_size, 1, 1))
        # adjust dimension
        q_w = torch.permute(q_w, (0, 3, 2, 1, 4))  # (None, heads, step, block, size)
        k_w = torch.permute(k_w, (0, 3, 2, 1, 4))
        v_w = torch.permute(v_w, (0, 3, 2, 1, 4))
        qwp = torch.unsqueeze(q_w, 4)
        kwp = torch.permute(kwp, (0, 4, 2, 1, 3, 5))  # (None, heads, step, block, kernel_size, out_dim)
        vwp = torch.permute(vwp,  (0, 4, 2, 1, 3, 5))
        if x_mask is not None:
            x_mask = torch.permute(x_mask, (0, 3, 2, 1, 4))
            xp_mask = torch.permute(xp_mask, (0, 4, 2, 1, 3, 5))

        # caculate attention weight-1
        a = torch.matmul(q_w, k_w.transpose(-1, -2)) / math.sqrt(self.key_size)
        a = torch.permute(a, (0, 1, 2, 4, 3))
        a = to_mask(a, x_mask, 'add')
        a = torch.permute(a, (0, 1, 2, 4, 3))
        if self.mask_right:
            ones = torch.ones_like(a[:1, :1, :1])
            mask = (ones - torch.tril(ones, diagonal=0)) * 1e10
            a = a - mask
        # caculate attention weight-2
        ap = torch.matmul(qwp, kwp.transpose(-1, -2)) / math.sqrt(self.key_size)

        ap = torch.permute(ap, (0, 1, 2, 3, 5, 4))
        if x_mask is not None:
            ap = to_mask(ap, xp_mask, 'add')
        ap = torch.permute(ap, (0, 1, 2, 3, 5, 4))
        if self.mask_right:
            mask = torch.ones((1, kernel_size))
            mask[:, -self.neighbors:] = 0
            mask = (1 - mask) * 1e10
            ap = ap - mask
        ap = ap[..., 0, :]
        # Merge two attention
        A = torch.cat((a, ap), -1)
        A = torch.softmax(A, -1)
        a, ap = A[..., : a.shape[-1]], A[..., a.shape[-1]:]

        o1 = torch.matmul(a, v_w)

        ap = torch.unsqueeze(ap, -2)
        o2 = torch.matmul(ap, vwp)
        o2 = o2[..., 0, :]

        o = o1 + o2
        o = to_mask(o, x_mask, 'mul')
        o = torch.permute(o, (0, 3, 2, 1, 4))
        o = torch.reshape(o, (-1, new_seq_len, self.out_dim))
        o = o[:, : - pad_len]
        return o


if __name__ == '__main__':
    """
        1-Attention 注意力机制
    """
    # q, k, v = torch.randn(2, 1, 20), torch.randn(2, 10, 2), torch.randn(2, 10, 4)
    # model = Attention(q_dim=q.shape[-1], k_dim=k.shape[-1], v_dim=v.shape[-1], hidden_dim=8, dropout=0.5)
    # output_add = model(q, k, v, seq_len=5, mode='add')
    # print(f'output_add >> {output_add.shape}')
    # output_mul = model(q, k, v, seq_len=5, mode='mul')
    # print(f'output_mul >> {output_mul.shape}')

    """
        2-MultiAttention 多头注意力机制
    """
    # q, k, v = torch.randn(2, 5, 10), torch.randn(2, 7, 10), torch.randn(2, 7, 10)
    # multiAttention = MultiAttention(heads=4, size_per_head=4, embedidng_dim=q.shape[-1],
    #                                 key_size=None, mask_right=True)
    # output_multiAttention = multiAttention([q, k, v])
    # print(f'output_multiAttention >> {output_multiAttention.shape}')
    #
    # # give a mask matrix
    # multiAttention = MultiAttention(heads=4, size_per_head=4, embedidng_dim=q.shape[-1], key_size=None,
    #                                 mask_right=torch.tril(torch.ones(q.shape[1], k.shape[1]), diagonal=0))
    # output_multiAttention = multiAttention([q, k, v])
    # print(f'output_multiAttention >> {output_multiAttention}')

    """
        3-MultiSelfAttention 多头自注意力机制
    """
    # x = torch.randn(2, 5, 10)  # 自注意力机制 q = k = v
    # multiSelfAttention = MultiSelfAttention(heads=4, size_per_head=4, embedidng_dim=x.shape[-1],
    #                                         key_size=None, mask_right=True)
    # output_multiSelfAttention = multiSelfAttention(x)
    # print(f'output_multiSelfAttention >> {output_multiSelfAttention.shape}')

    """
        4-AtrousSelfAttention 空洞多头注意力机制
    """
    # x = torch.randn(2, 5, 10)  # 自注意力机制 q = k = v
    # atrousSelfAttention = AtrousSelfAttention(heads=4, size_per_head=4, embedidng_dim=x.shape[-1],step=2,
    #                                         key_size=None, mask_right=True)
    # output_atrousSelfAttention = atrousSelfAttention(x)
    # print(f'output_atrousSelfAttention >> {output_atrousSelfAttention.shape}')

    """
        5-LocalSelfAttention 局部多头注意力机制
    """
    # x = torch.randn(2, 5, 10)  # 自注意力机制 q = k = v
    # localSelfAttention = LocalSelfAttention(heads=4, size_per_head=4, embedidng_dim=x.shape[-1],neighbors=2,
    #                                           key_size=None, mask_right=True)
    # output_localSelfAttention = localSelfAttention(x)
    # print(f'output_localSelfAttention >> {output_localSelfAttention.shape}')

    """
        6-SparseSelfAttention 稀疏多头注意力机制
    """
    x = torch.randn(2, 5, 10)  # 自注意力机制 q = k = v
    sparseSelfAttention = SparseSelfAttention(heads=4, size_per_head=4, embedidng_dim=x.shape[-1], step=3,
                                            key_size=None, mask_right=True)
    output_sparseSelfAttention = sparseSelfAttention(x)
    print(f'output_sparseSelfAttention >> {output_sparseSelfAttention.shape}')
