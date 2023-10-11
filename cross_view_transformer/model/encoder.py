import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from einops import rearrange

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


#def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters #sh=2
    sw = w / w_meters #sw=2

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  #在图像送入网络训练之前，减去图片的均值，算是一种归一化操作。
                                                                                #图像其实是一种平稳的分布，减去数据对应维度的统计平均值，可以消除公共部分。
                                                                                # 以凸显个体之前的差异和特征。
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):   # 6,3,224,480
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):  # Embedding
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))  # 高
        w = bev_width // (2 ** len(decoder_blocks))   # 宽

        # bev coordinates BEV坐标
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        # print("qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq",q.size())
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras 做注意力机制
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k)
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [output_dim * output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        return out.view(x.size(0), x.size(1), self.output_dim, self.output_dim)






class DirectionCrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        print(q,'qqqqqqqqqqqq')
        print(q.size())
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')


        # Project with multiple heads
        q = self.to_q(q)                                # b n (H W) (heads dim_head)
        k = self.to_k(k)                                # b n (h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        q1 = q[:, 0:1, :, :]  # 第12维度的第一层数据作为替换数据
        q1[:, 1:, :, :] = q1  # 将替换数据广播到第12维度的其余层

        q2 = q[:, 1:2, :, :]  # 第12维度的第一层数据作为替换数据
        q2[:, 1:, :, :] = q2  # 将替换数据赋值给第12维度的第二层

        q3 = q[:, 2:3, :, :]  # 第12维度的第一层数据作为替换数据
        q3[:, 1:, :, :] = q3  # 将替换数据赋值给第12维度的第二层

        q4 = q[:, 3:4, :, :]  # 第12维度的第一层数据作为替换数据
        q4[:, 1:, :, :] = q4  # 将替换数据赋值给第12维度的第二层

        q5 = q[:, 4:5, :, :]  # 第12维度的第一层数据作为替换数据
        q5[:, 1:, :, :] = q5  # 将替换数据赋值给第12维度的第二层

        q6 = q[:, 5:6, :, :]  # 第12维度的第一层数据作为替换数据
        q6[:, 1:, :, :] = q6  # 将替换数据赋值给第12维度的第二层



        # Group the head dim with batch dim
        q1 = rearrange(q1, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q2 = rearrange(q2, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        q3 = rearrange(q3, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q4 = rearrange(q4, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q5 = rearrange(q5, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q6 = rearrange(q6, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot1 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q1, k)
        dot2 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q2, k)

        dot3 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q3, k)
        dot4 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q4, k)
        dot5 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q5, k)
        dot6 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q6, k)

        dotk1=dot1+dot2
        #dotk1=dot1+dot2+dot3+dot4+dot5+dot6



        #dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', dot1, dot2)
        dot = rearrange(dotk1, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (d level features)
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z



class TrackingCrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        print(q,'qqqqqqqqqqqq')
        print(q.size())
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b (n h w) d')
        print(k.size())
        print(v.size())

        # Project with multiple heads
        q = self.to_q(q)                                # b n (H W) (heads dim_head)
        k = self.to_k(k)                                # b n (h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        print(k.size())
        print(v.size())
        print(q.size())

        q1 = q[:, 0:1, :, :]  # 第12维度的第一层数据作为替换数据
        q1[:, 1:, :, :] = q1  # 将替换数据广播到第12维度的其余层

        q2 = q[:, 1:2, :, :]  # 第12维度的第一层数据作为替换数据
        q2[:, 1:, :, :] = q2  # 将替换数据赋值给第12维度的第二层
        q1 = q1.expand(1, 12, -1, -1)
        q2 = q2.expand(1, 12, -1, -1)
        print(q1.size())

        q3 = q[:, 2:3, :, :]  # 第12维度的第一层数据作为替换数据
        q3[:, 1:, :, :] = q3  # 将替换数据赋值给第12维度的第二层

        q4 = q[:, 3:4, :, :]  # 第12维度的第一层数据作为替换数据
        q4[:, 1:, :, :] = q4  # 将替换数据赋值给第12维度的第二层

        q5 = q[:, 4:5, :, :]  # 第12维度的第一层数据作为替换数据
        q5[:, 1:, :, :] = q5  # 将替换数据赋值给第12维度的第二层

        q6 = q[:, 5:6, :, :]  # 第12维度的第一层数据作为替换数据
        q6[:, 1:, :, :] = q6  # 将替换数据赋值给第12维度的第二层


        k1 = k[:, 0:1, :, :]  # 第12维度的第一层数据作为替换数据
        k1[:, 1:, :, :] = k1  # 将替换数据广播到第12维度的其余层
        k2 = k[:, 1:2, :, :]  # 第12维度的第一层数据作为替换数据
        k2[:, 1:, :, :] = k2  # 将替换数据赋值给第12维度的第二层

        k1 = k1.expand(1, 12, -1, -1)
        k2 = k2.expand(1, 12, -1, -1)
        # Group the head dim with batch dim
        q1 = rearrange(q1, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q2 = rearrange(q2, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        q3 = rearrange(q3, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q4 = rearrange(q4, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q5 = rearrange(q5, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        q6 = rearrange(q6, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        k1 = rearrange(k1, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k2 = rearrange(k2, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot1 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q1, k1)
        dot2 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q2, k2)

        dot3 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q3, k)
        dot4 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q4, k)
        dot5 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q5, k)
        dot6 = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q6, k)

        dotk1=dot1+dot2
        #dotk1=dot1+dot2+dot3+dot4+dot5+dot6



        #dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', dot1, dot2)
        dot = rearrange(dotk1, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1)

        # Combine values (d level features)
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,  #2，6，32，56，120；；2，12，32，56，120
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w
        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        #query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=, n=n)      # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W
        # print("ev_embed", bev_embed.size())
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)

class Direction(nn.Module):
    def __init__(
            self,
            feat_height: int,
            feat_width: int,
            feat_dim: int,
            dim: 1,
            image_height: int,
            image_width: int,
            qkv_bias: bool,
            heads: int = 4,
            dim_head: int = 32,
            no_image_features: bool = False,
            skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cross_attend = DirectionCrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
            self,
            x: torch.FloatTensor,
            bev: BEVEmbedding,
            feature: torch.FloatTensor,  # 2，6，32，56，120；；2，12，32，56，120
            I_inv: torch.FloatTensor,
            E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane  # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]  # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]  # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)  # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')  # 1 1 3 (h w)
        cam = I_inv @ pixel_flat  # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)  # b n 4 (h w)


        '''        hidden_dim = 512
        output_dim = 4
        num_layers = 4
        mlp = MLP(4 * 4, hidden_dim, output_dim, num_layers).to("cuda")
        input_data = E_inv             
        print("E_inv", E_inv.size())
        input_data = input_data.reshape(1, 6, 4 * 4) 
        output = mlp(input_data)
        print("out",output.shape) 
        #E_inv=output+E_inv
        #E_inv=output+E_inv'''


        
        d = E_inv @ cam  # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)  # (b n) 4 h w
        d_embed = self.img_embed(d_flat)  # (b n) d h w
        img_embed = d_embed - c_embed  # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d h w

        world = bev.grid[:2]  # 2 H W
        w_embed = self.bev_embed(world[None])  # 1 d H W
        bev_embed = w_embed - c_embed  # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)  # (b n) d H W
        # query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=, n=n)      # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)  # b n d H W
        # print("ev_embed", bev_embed.size())
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')  # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)  # (b n) d h w
        else:
            key_flat = img_embed  # (b n) d h w

        val_flat = self.feature_linear(feature_flat)  # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)








class Tracking(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = TrackingCrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,  #2，6，32，56，120；；2，12，32，56，120
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w
        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        world = bev.grid[:2]                                                    # 2 H W
        w_embed = self.bev_embed(world[None])                                   # 1 d H W
        bev_embed = w_embed - c_embed                                           # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
        #query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=, n=n)      # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W
        # print("ev_embed", bev_embed.size())
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        query = query_pos + x[:, None]                                          # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        return self.cross_attend(query, key, val, skip=x if self.skip else None)



class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()
        direction=list()
        tracking=list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

            dir = Direction(feat_height, feat_width, feat_dim, dim, **cross_view)
            direction.append(dir)
            tra = Tracking(feat_height, feat_width, feat_dim, dim, **cross_view)
            tracking.append(tra)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.direction = nn.ModuleList(direction)
        self.tracking = nn.ModuleList(tracking)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):####这个地方输入batch；image 2,6,3,224,480;;bev;;1,12,200,200
        b, n, _, _, _ = batch['image'].shape  #n:6;b:1

        image = batch['image'].flatten(0, 1)  #   6,3,224,480  # b n c h w
        I_inv = batch['intrinsics'].inverse() #  1,6,3,3      # b n 3 3 内参
        #print('iiiiiiiiiiiiiiiiiiiiiii',I_inv)
        E_inv = batch['extrinsics'].inverse() #   1,6,4,4    # b n 4 4 外参

        features = [self.down(y) for y in self.backbone(self.norm(image))]  # 0:6,32,56,120; 1:6,112,14,30

        x = self.bev_embedding.get_prior()     #128 25 25         # d H W
        x = repeat(x, '... -> b ...', b=b)      # 1,128,25,25         # b d H W

        for cross_view, feature, layer, direction,tracking in zip(self.cross_views, features, self.layers,self.direction,self.tracking):

            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            y = direction(x, self.bev_embedding, feature, I_inv, E_inv)
            z = tracking(x, self.bev_embedding, feature, I_inv, E_inv)
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
            y = layer(y)
            z = layer(z)
            x = layer((y+x+z)/3)

        return x
