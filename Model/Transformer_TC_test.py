


import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
import math

import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes=64, kernel_size=3, stride=1, padding=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class IR_ERA5_FPN(nn.Module):
    def __init__(self, IR_chancel, ERA5_chancel, block, layers):
        super(IR_ERA5_FPN, self).__init__()
        # self.inplanes = 64

        self.IR_conv1 = nn.Conv2d(IR_chancel, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.IR_bn1 = nn.BatchNorm2d(16)

        self.ERA5_conv1 = nn.Conv2d(ERA5_chancel, 128, kernel_size=7, stride=1, padding=3, bias=False)
        self.ERA5_bn1 = nn.BatchNorm2d(128)
        
        # Bottom-up layers
        self.IR_layer1 = self._make_layer(block, 16, 32, layers[0])
        self.IR_layer2 = self._make_layer(block, 64, 128, layers[1], stride=2, down_stride=2)
        self.IR_layer3 = self._make_layer(block, 256, 128, layers[2], stride=2, down_stride=2)
        
        
        self.ERA5_layer = self._make_layer(block, 128, 128, 1, kernel_size=6, stride=1, padding=0, downsample_kernel_size=6, down_stride=1)
        # Top layer
        # fussion
        self.fussion = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.lateral_layer1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer2 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer3 = nn.Conv2d( 64, 256, kernel_size=1, stride=1, padding=0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, planes, blocks, kernel_size=3, stride=1, padding=1, downsample_kernel_size=1, down_stride=1):
        downsample  = None
        if stride != 1 or inplanes != block.expansion * planes:
            downsample  = nn.Sequential(
                nn.Conv2d(inplanes, block.expansion * planes, kernel_size=downsample_kernel_size, stride=down_stride,bias=False),
                #nn.BatchNorm2d(block.expansion * planes)
            )
        layers = []
        layers.append(block(inplanes, planes, kernel_size, stride,  padding, downsample))

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, padding=padding))

        return nn.Sequential(*layers)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, IR, ERA5):

        IR = self.IR_conv1(IR)
        IR = self.IR_bn1(IR)
        ERA5 = self.ERA5_conv1(ERA5)
        ERA5 = self.ERA5_bn1(ERA5)

        # print("IR:{}".format(IR.shape))
        # print("ERA5:{}".format(ERA5.shape))
        IR_c1 = IR
        IR_c2 = self.IR_layer1(IR_c1)
        #print("IR_c2:{}".format(IR_c2.shape))
        IR_c3 = self.IR_layer2(IR_c2)
        #print("IR_c3:{}".format(IR_c3.shape))
        IR_c4 = self.IR_layer3(IR_c3)
        #print("IR_c4:{}".format(IR_c4.shape))

        ERA5_c = ERA5
        #print("ERA5_c:{}".format(ERA5_c.shape))
        ERA5_c = self.ERA5_layer(ERA5_c)
        #print("ERA5_c:{}".format(ERA5_c.shape))
        IR_c5 = torch.cat((IR_c4, ERA5_c), dim=-3)
        #print("IR_c5:{}".format(IR_c5.shape))
        IR_p5 = self.fussion(IR_c5)
        #print("IR_p5:{}".format(IR_p5.shape))
        IR_p4 = self._upsample_add(IR_p5, self.lateral_layer1(IR_c4))
        IR_p4 = self.smooth1(IR_p4)
        #print("IR_p4:{}".format(IR_p4.shape))
        IR_p3 = self._upsample_add(IR_p4, self.lateral_layer2(IR_c3))
        IR_p3 = self.smooth2(IR_p3)
        #print("IR_p3:{}".format(IR_p3.shape))
        IR_p2 = self._upsample_add(IR_p3, self.lateral_layer3(IR_c2))
        IR_p2 = self.smooth3(IR_p2)
        #print("IR_p2:{}".format(IR_p2.shape))
        
        
        
        return  IR_p2, IR_p3, IR_p4, IR_p5




def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)




class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=256, dropout=0.) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        
        project_out = not (heads ==1 and dim_head == dim)

        self.heads =heads
        self.scale = dim_head ** -0.5

        #self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.qlv_embedding = nn.Linear(dim, heads*dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    
    def forward(self, img_embedding, seq_embedding):
        # x = self.norm(x)
        q = self.qlv_embedding(seq_embedding)
        k = self.qlv_embedding(img_embedding)
        v = self.qlv_embedding(img_embedding)
        # qkv = self.to_qkv(x).chunk(3, dim=-1)#按最后一维分成三块

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
       
        return self.to_out(out)
class Cross_Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout),
                Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout),
            ]))
            
        self.last_att = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.last_feed = FeedForward(dim=dim, hid_dim=mlp_dim, dropout=dropout)
            
                
    def forward(self, img_embedding, seq_embedding):
        for att_seq, feed_seq, att_img, feed_img, in self.layers:
            seq_embedding = att_seq(img_embedding, seq_embedding) + seq_embedding
            seq_embedding = feed_seq(seq_embedding) + seq_embedding
            img_embedding = att_img(seq_embedding, img_embedding, ) + img_embedding
            img_embedding = feed_img(img_embedding) + img_embedding
            #print(seq_embedding.shape, img_embedding.shape)

        seq_embedding = self.last_att(img_embedding, seq_embedding) + seq_embedding
        seq_embedding = self.last_feed(seq_embedding) + seq_embedding 

        return self.norm(seq_embedding)

class ST_transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.) -> None:
        super().__init__()
        self.Cross_Transformer = Cross_Transformer(dim, depth, heads, dim, mlp_dim, dropout)
    
    def forward(self, img_embedding, seq_embedding):
        """
        img_embedding: [B, T, n, dim]
        seq_embedding: [B, T, 1, dim]
        """
        B, T, n, dim = img_embedding.shape
        
        fusion_embedding = torch.concat((img_embedding[:, 0, :, :], seq_embedding[:, 0, :, :]), dim=-2)
        # print(f"fusion_embedding: {fusion_embedding.shape}")
        att_score = self.Cross_Transformer(fusion_embedding, seq_embedding[:, 0, :, :])
        # print(f"att_score: {att_score.shape}")
        for t in range(1, T):
            for t_of_t in range(0, t+1):
                fusion_embedding = torch.concat((img_embedding[:, t_of_t, :, :], seq_embedding[:, t_of_t, :, :], seq_embedding[:, t, :, :]), dim=-2)
                # print(f"fusion_embedding: {fusion_embedding.shape}")
                att_score_t = self.Cross_Transformer(img_embedding[:, t_of_t, :, :], seq_embedding[:, t, :, :])
                att_score = torch.concat((att_score, att_score_t), dim=-2)
        # print(f"att_score: {att_score.shape}")
        return att_score
    
class Multiscale_ST_ViT_TC_IR(nn.Module):
    def __init__(self, TCIR_image_size=(140, 140), ERA5_image_size=(40, 40), patch_size=(5, 5), TCIR_channels=1, ERA5_channels=1, dim=1024, seq_dim=2, seq_len=4, pre_len=4,
                 depth=6, heads=8, mlp_dim=2, pool='cls',dropout=0., emb_dropout=0.) -> None:
        super().__init__()

        self.TCIR_channels = TCIR_channels
        img_H1, img_W1 = pair(TCIR_image_size)
        img_H2, img_W2 = pair(ERA5_image_size)
        patch_H, patch_W = pair(patch_size)

        self.patch_H = patch_H
        self.patch_W = patch_W

        #确保可以分割完全
        assert (img_H1 % patch_H==0) and (img_W1 % patch_W==0) and (img_H2 % patch_H==0) and (img_W2 % patch_W==0), 'Image dimensions must be divisible by the patch size.'

        num_patches1 = (img_H1 // patch_H) * (img_W1 // patch_W)
        num_patches2 = (img_H2 // patch_H) * (img_W2 // patch_W)

       
        self.IR_ERA5_FPN = IR_ERA5_FPN(IR_chancel=TCIR_channels, ERA5_chancel=ERA5_channels, block=Bottleneck, layers=[2,2,2])

       
        fussion_patch_dim = 256 * patch_H * patch_W
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pre_len = pre_len


        num_patches = (img_H1 // patch_H) * (img_W1 // patch_W) + (img_H1 // patch_H // 2) * (img_W1 // patch_W // 2) + 2 * (img_H1 // patch_H // 4) * (img_W1 // patch_W // 4)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches+1, dim))#随机生成size为(1, num_patches+1, dim )的位置编码
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
       

        self.dropout = nn.Dropout(emb_dropout)
        
        self.seq_embedding = nn.Linear(seq_dim, dim)

        self.ST_transformer = ST_transformer(dim, depth, heads, dim, mlp_dim, dropout)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool
        self.to_latent = nn.Identity()

        seq_att_num = int((1 + seq_len)*seq_len/2)
        self.mlp_head = nn.Linear(dim*seq_att_num, seq_dim*pre_len)

    def forward(self, new_GridSat_data, new_era5_data, inp_lable):
        
        B, T, _ = inp_lable.shape
        
        seq = inp_lable
        new_GridSat_data = torch.reshape(new_GridSat_data, (B*T, new_GridSat_data.shape[-3],  new_GridSat_data.shape[-2], new_GridSat_data.shape[-1]))
        new_era5_data = torch.reshape(new_era5_data, (B*T, new_era5_data.shape[-3],  new_era5_data.shape[-2], new_era5_data.shape[-1]))
        #print(new_era5_data.shape)
        IR_p2, IR_p3, IR_p4, IR_p5 = self.IR_ERA5_FPN(new_GridSat_data, new_era5_data,)
        IR_p2 = self.to_patch(IR_p2)
        IR_p3 = self.to_patch(IR_p3)
        IR_p4 = self.to_patch(IR_p4)
        IR_p5 = self.to_patch(IR_p5)
        # print(IR_p2.shape, IR_p3.shape, IR_p4.shape, IR_p5.shape)
        fussion_token = torch.concat((IR_p2, IR_p3, IR_p4, IR_p5), dim=1)
        # print(fussion_token.shape)

        fussion_token = torch.reshape(fussion_token, (B, T, fussion_token.shape[-2], fussion_token.shape[-1]))
        
        cls_tokens = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=B, t=T)
        fussion_token = torch.cat((cls_tokens, fussion_token), dim=-2)
        fussion_token = fussion_token + self.pos_embedding
        fussion_token = self.dropout(fussion_token) 
        
        x = fussion_token
        # print(x.shape)
        #(B, T, C) -> (B, T, dim)
        seq = self.seq_embedding(seq)
        seq = torch.unsqueeze(seq, dim=-2)
        #print(seq.shape)
        seq_pre = self.ST_transformer(x, seq)
        #print(seq_pre.shape)
        if self.pool == 'mean': 
            seq_pre = seq_pre.mean(dim = 1)
        elif self.pool == 'cls':
            seq_pre = torch.reshape(seq_pre, (seq_pre.shape[0], seq_pre.shape[1]*seq_pre.shape[2]))
        #print(seq_pre.shape)

        seq_pre = self.to_latent(seq_pre)
        pre_data =self.mlp_head(seq_pre)
        pre_data = rearrange(pre_data, 'b (t d) -> b t d', t=self.pre_len)
        return pre_data
    




class AR_decoder(nn.Module):
    def __init__(self,dim, depth, heads, mlp_dim, seq_len, pre_len, hid_dim , seq_dim=2, dropout=0.) -> None:
        super().__init__()
        self.Cross_Transformer = Cross_Transformer(dim, depth, heads, dim, mlp_dim, dropout)
        self.seq_att_num = int((1 + seq_len)*seq_len/2)
        self.pre_len = pre_len
        self.Pre_Head = nn.Sequential(
            nn.LayerNorm(dim*self.seq_att_num),
            nn.Linear(dim*self.seq_att_num, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, seq_dim),
        )
        self.seq_embedding = nn.Linear(seq_dim, dim)
    def forward(self, hid_state, final_seq, step):

        
        final_seq_embedding = self.seq_embedding(final_seq)
        final_seq_embedding = torch.unsqueeze(final_seq_embedding, dim=1)
        final_seq_embedding = torch.concat((final_seq_embedding, hid_state), dim=-2)
        #print(f"final_seq_embedding {final_seq_embedding.shape}")
        hid_state = self.Cross_Transformer(final_seq_embedding, hid_state)

        hid_state_pre = torch.reshape(hid_state, (hid_state.shape[0], -1))
        pre_seq = self.Pre_Head(hid_state_pre) + final_seq
        
        final_seq = pre_seq
        
        for _ in range(1, step):
            #save_fig(hid_state[0], f"hid_state{step}")
            final_seq_embedding = self.seq_embedding(final_seq)
            final_seq_embedding = torch.unsqueeze(final_seq_embedding, dim=1)
            final_seq_embedding = torch.concat((final_seq_embedding, hid_state), dim=-2)
            
            hid_state = self.Cross_Transformer(final_seq_embedding, hid_state)
            hid_state_pre = torch.reshape(hid_state, (hid_state.shape[0], -1))
            pre_seq_t = self.Pre_Head(hid_state_pre) + final_seq
            
            final_seq = pre_seq_t
        final_seq = torch.unsqueeze(final_seq, dim=1)
        return final_seq


    
class Multiscale_STAR_ViT_TC_IR_test(nn.Module):
    def __init__(self, TCIR_image_size=(140, 140), ERA5_image_size=(40, 40), patch_size=(5, 5), 
                 TCIR_channels=1, ERA5_channels=69, dim=128, seq_dim=2, seq_len=4, pre_len=4,
                 depth=3, heads=4, mlp_dim=32,dropout=0.1, emb_dropout=0.1, AR_hid_dim=256, is_use_lifetime_num=False) -> None:
        super().__init__()
        self.is_use_lifetime_num = is_use_lifetime_num
        self.TCIR_channels = TCIR_channels
        img_H1, img_W1 = pair(TCIR_image_size)
        img_H2, img_W2 = pair(ERA5_image_size)
        patch_H, patch_W = pair(patch_size)

        self.patch_H = patch_H
        self.patch_W = patch_W

       
        self.IR_ERA5_FPN = IR_ERA5_FPN(IR_chancel=TCIR_channels, ERA5_chancel=ERA5_channels, block=Bottleneck, layers=[2,2,2])

       
        fussion_patch_dim = 256 * patch_H * patch_W
        self.to_patch_p2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_p3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_p4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_p5 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_H, p2=patch_W),
            nn.LayerNorm(fussion_patch_dim), 
            nn.Linear(fussion_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pre_len = pre_len

        if is_use_lifetime_num:
            self.lifetime_embedding = nn.Embedding(300, dim)
        num_patches = (img_H1 // patch_H) * (img_W1 // patch_W) + (img_H1 // patch_H // 2) * (img_W1 // patch_W // 2) + 2 * (img_H1 // patch_H // 4) * (img_W1 // patch_W // 4)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches+1, dim))#随机生成size为(1, num_patches+1, dim )的位置编码
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
       

        self.dropout = nn.Dropout(emb_dropout)
        
        self.seq_embedding = nn.Linear(seq_dim, dim)

        self.ST_transformer = ST_transformer(dim, depth, heads, dim, mlp_dim, dropout)

        self.AR_decoder = AR_decoder(dim, depth, heads, mlp_dim, seq_len, pre_len, hid_dim=AR_hid_dim , seq_dim=seq_dim, dropout=dropout)
        

    def forward(self, new_GridSat_data, new_era5_data, inp_lable, step):
        if self.is_use_lifetime_num:
            life_num = inp_lable[:, :, -1].int()
            # print(life_num.shape)
            lifetime_embed = self.lifetime_embedding(life_num)
            # print(lifetime_embed.shape)
            inp_lable = inp_lable[:, :, :-1]
            
        B, T, _ = inp_lable.shape
        seq = inp_lable
        new_GridSat_data = torch.reshape(new_GridSat_data, (B*T, new_GridSat_data.shape[-3],  new_GridSat_data.shape[-2], new_GridSat_data.shape[-1]))
        new_era5_data = torch.reshape(new_era5_data, (B*T, new_era5_data.shape[-3],  new_era5_data.shape[-2], new_era5_data.shape[-1]))
        #print(new_era5_data.shape)
        IR_p2, IR_p3, IR_p4, IR_p5 = self.IR_ERA5_FPN(new_GridSat_data, new_era5_data,)
        
        IR_p2 = self.to_patch_p2(IR_p2)
        IR_p3 = self.to_patch_p3(IR_p3)
        IR_p4 = self.to_patch_p4(IR_p4)
        IR_p5 = self.to_patch_p5(IR_p5)


        fussion_token = torch.concat((IR_p2, IR_p3, IR_p4, IR_p5), dim=1)

        fussion_token = torch.reshape(fussion_token, (B, T, fussion_token.shape[-2], fussion_token.shape[-1]))
        #print(fussion_token.shape)
        #(B, T, C, H, W) -> (B, (HW)/(p1p2), dim)
  
        
        cls_tokens = repeat(self.cls_token, '1 1 1 d -> b t 1 d', b=B, t=T)
        fussion_token = torch.cat((cls_tokens, fussion_token), dim=-2)
        fussion_token = fussion_token + self.pos_embedding
        fussion_token = self.dropout(fussion_token) 
        
        
        x = fussion_token
        
        seq = self.seq_embedding(seq)
        if self.is_use_lifetime_num:
            seq = seq + lifetime_embed
        seq = torch.unsqueeze(seq, dim=-2)

        #print(f"seq.shape{seq.shape}")
        seq_pre = self.ST_transformer(x, seq)
        
        pre_data = self.AR_decoder(seq_pre, inp_lable[:, -1, :], step)
        
        return pre_data
    



if __name__ == "__main__":
    

    model = Multiscale_STAR_ViT_TC_IR_test(patch_size=7, AR_hid_dim=128, is_use_lifetime_num=False)
    for i in range(5):
        a = torch.randn((2, 4, 1, 140, 140))
        b = torch.randn((2, 4, 1, 40, 40))
        c = torch.randn((2, 4, 2))
        lifetime_num = torch.ones((2, 4, 1))
        c = torch.concat((c, lifetime_num), dim=-1)
        d = model(a, b, c, 1)
        print(d.shape)


