import os
from collections import OrderedDict
from typing import Dict, List, Optional, Union
from torch.nn import functional as F
import torch
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.runner.checkpoint import _load_checkpoint
from torch import nn
import ipdb
import ml_collections
from torch.nn import Parameter
import numpy as np

from torch import nn

from turtle import forward
from numpy import rint
from sklearn import datasets
import torch.nn.functional as F
import math
logger = MMLogger.get_current_instance()


class ECRL(nn.Module):
    def __init__(self, num_class=20):
        super().__init__()

        self.module1 = UniFormerV2(num_class=num_class) 
        self.module2 = UniFormerV2bg(num_class=num_class) 
        self.module3 = UniFormerV2obj(num_class=num_class) 
        self.frontIntervention = GLCA(module_dim = 768)
        self.backIntervention = BD(768, 768, strategy = 'dp_cause',num_class=num_class)
        
        self.bgTemporal = TemporalSelfAttention_bg(768,768,8)
        self.objTemporal= TemporalSelfAttention_obj(768,768,8)

        self.dense = nn.Linear(768*4,768)
        self.fc = nn.Linear(768,num_class)
        
        self.dense1 = nn.Linear(768*2,768)
        self.fc1 = nn.Linear(768,num_class)
        self.fc2 = nn.Linear(768,num_class)
        self.fc3 = nn.Linear(768,num_class)

        self.sigmoid = nn.Sigmoid()
        
        
        print(num_class)


    def forward(self, frames,obj, bg,confounder_dictionary,prior,global_feat_dict,batchsize):
        
        ipdb.set_trace()
        
        video_feature,video_featuret,logits1 = self.module1(frames)  
        local_feat = video_feature
        
        
        num_samples, sample_dim = global_feat_dict.shape
        num_requested_samples = batchsize
        sample_indices = np.random.choice(num_samples, num_requested_samples, replace=False)  
        global_feat = global_feat_dict[sample_indices]  
        
       
        local_feat = torch.cat([local_feat,local_feat],dim=1)
        global_feat = torch.cat([global_feat,global_feat],dim=1)
        causal_feature = self.frontIntervention(local_feat,global_feat)
        causal_feature = self.dense(causal_feature)
        lg_logits = self.fc(causal_feature)
        
        obj_feature,obj_featuret,logits3 = self.module3(obj) 
        bg_feature,bg_featuret,logits2 = self.module2(bg) 
        h_feature = torch.cat([obj_feature,bg_feature],dim=1) #bs,768*2
        joint_feature = self.dense1(h_feature) ##bs,768
        h_logits = self.fc1(joint_feature) #
        
        
        gz = self.backIntervention(joint_feature, confounder_dictionary, prior) # gz: bs,768
        gz_logits = self.fc2(gz) #
        
        #do,do_logits
        dox =joint_feature + gz
        dox_logits = self.fc3(dox) #cls
        
        logits = lg_logits + dox_logits
        return logits

# ==========================================================causal inference=======================================================
#=============================================================BD module=================================================================
class BD(nn.Module):
    def __init__(self, num_joint_feature, num_gz, strategy,num_class): #num_joint_feature d=1536 num_gz;768
        super(BD, self).__init__()
        self.num_joint_feature = num_joint_feature
        self.num_gz = num_gz
        if strategy == 'dp_cause':
              self.causal_intervention = dot_product_intervention(num_gz, num_joint_feature )  #768,1536
        elif strategy == 'ad_cause':
              self.causal_intervention = additive_intervention(num_gz, num_joint_feature )
        else:
              raise ValueError("Do Not Exist This Strategy.")
                
        self.w_h = Parameter(torch.Tensor(self.num_joint_feature, 768)) 
        self.w_g = Parameter(torch.Tensor(self.num_gz, 768)) 
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_h)
        nn.init.xavier_normal_(self.w_g)

    def forward(self, joint_feature, confounder_dictionary, prior):
#         ipdb.set_trace()
        g_z = self.causal_intervention(confounder_dictionary, joint_feature, prior)
        do_x = joint_feature + g_z
        proj_h = torch.matmul(joint_feature, self.w_h)
        proj_g_z = torch.matmul(g_z.to(torch.float32), self.w_g) 
        do_x = proj_h + proj_g_z
        return g_z,do_x
    
class dot_product_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super(dot_product_intervention, self).__init__()
        self.con_size = con_size 
        self.fuse_size = fuse_size  
        self.query = nn.Linear(self.fuse_size, 768, bias= False) 
        self.key = nn.Linear(self.con_size, 768, bias= False) 

    def forward(self, confounder_set, fuse_rep, probabilities):
#         ipdb.set_trace()
        query = self.query(fuse_rep) 
        key = self.key(confounder_set)  
        mid = torch.matmul(query, key.transpose(0,1)) / math.sqrt(self.con_size)  
        attention = F.softmax(mid, dim=-1) 
        attention = attention.unsqueeze(2)  
        fin = (attention*confounder_set*probabilities).sum(1)
        return fin


class additive_intervention(nn.Module):
    def __init__(self, con_size, fuse_size):
        super(additive_intervention,self).__init__()
        self.con_size = con_size
        self.fuse_size = fuse_size
        self.Tan = nn.Tanh()
        self.query = nn.Linear(self.fuse_size, 768, bias = False)
        self.key = nn.Linear(self.con_size, 768, bias = False)
        self.w_t = nn.Linear(768, 1, bias=False)

    def forward(self, confounder_set, fuse_rep, probabilities):
        query = self.query(fuse_rep)
        key = self.key(confounder_set)
        query_expand = query.unsqueeze(1)  
        fuse = query_expand + key 
        fuse = self.Tan(fuse)
        attention = self.w_t(fuse) 
        attention = F.softmax(attention, dim=1)
        fin = (attention*confounder_set*probabilities).sum(1)  

        return fin
        
#==================================================================GLCA module=====================================================================
        
class GLCA(nn.Module):
    def __init__(self, module_dim=768):
        super(GLCA, self).__init__()
        self.feature_LL = FeatureAggregation_LL(module_dim=768)
        self.feature_LG = FeatureAggregation_LG(module_dim=768)
        
    def forward(self,local_feature,global_feature):
        
        ll_feature = self.feature_LL(local_feature)
        lg_feature = self.feature_LG(local_feature,global_feature)
        
        causal_feature = torch.cat([ll_feature,lg_feature],dim=1)
        
        return causal_feature

class FeatureAggregation_LL(nn.Module):
    def __init__(self, module_dim=768):
        super(FeatureAggregation_LL, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self,local_feature):
#         ipdb.set_trace()
        local_feature = self.dropout(local_feature)
        q_proj = self.q_proj(local_feature)
        v_proj = self.v_proj(local_feature)

        v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)
        attn = F.softmax(attn, dim=1)

        v_distill = (attn * local_feature)

        return v_distill



class FeatureAggregation_LG(nn.Module):
    def __init__(self, module_dim=768):
        super(FeatureAggregation_LG, self).__init__()
        self.module_dim = module_dim

        self.q_proj = nn.Linear(2*module_dim, module_dim, bias=False)
        self.v_proj = nn.Linear(2*module_dim, module_dim, bias=False)

        self.cat = nn.Linear(2 * module_dim, module_dim)
        self.attn = nn.Linear(module_dim, 1)

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, local_feature,global_feature):
        global_feature = self.dropout(global_feature)
        q_proj = self.q_proj(local_feature)
        v_proj = self.v_proj(global_feature)

        v_q_cat = torch.cat((v_proj, q_proj * v_proj), dim=-1)
        v_q_cat = self.cat(v_q_cat)
        v_q_cat = self.activation(v_q_cat)

        attn = self.attn(v_q_cat)
        attn = F.softmax(attn, dim=1)

        v_distill = (attn * global_feature)

        return v_distill


# =====================================BCM/FCM Moudel=============================================================================

class TemporalSelfAttention_bg(nn.Module):
    def __init__(self, input_size, hidden_size, num_frames):
        super(TemporalSelfAttention_bg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.query_projection = nn.Linear(input_size, hidden_size)
        self.key_projection = nn.Linear(input_size, hidden_size)
        self.value_projection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        #  (batch_size, num_frames, input_size)
        queries = self.query_projection(x)  # (batch_size, num_frames, hidden_size)
        keys = self.key_projection(x)  # (batch_size, num_frames, hidden_size)
        values = self.value_projection(x)  # (batch_size, num_frames, hidden_size)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (
                self.hidden_size ** 0.5)  # (batch_size, num_frames, num_frames)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.bmm(attention_weights, values)  # (batch_size, num_frames, hidden_size)

        return weighted_sum


class TemporalSelfAttention_obj(nn.Module):
    def __init__(self, input_size, hidden_size, num_frames):
        super(TemporalSelfAttention_obj, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames

        self.query_projection = nn.Linear(input_size, hidden_size)
        self.key_projection = nn.Linear(input_size, hidden_size)
        self.value_projection = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # (batch_size, num_frames, input_size)

        queries = self.query_projection(x)  # (batch_size, num_frames, hidden_size)
        keys = self.key_projection(x)  # (batch_size, num_frames, hidden_size)
        values = self.value_projection(x)  # (batch_size, num_frames, hidden_size)

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (
                self.hidden_size ** 0.5)  # (batch_size, num_frames, num_frames)
        attention_weights = F.softmax(attention_scores, dim=-1)

        weighted_sum = torch.bmm(attention_weights, values)  # (batch_size, num_frames, hidden_size)

        return weighted_sum


#---------------------------------------------------------uniformerv2------------------------------------------------------------------------------
class QuickGELU(BaseModule):
    """Quick GELU function. Forked from https://github.com/openai/CLIP/blob/d50
    d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py.

    Args:
        x (torch.Tensor): The input features of shape :math:`(B, N, C)`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Local_MHRA(BaseModule):
    """Local MHRA.

    Args:
        d_model (int): Number of input channels.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        pos_kernel_size (int): Kernel size of local MHRA.
            Defaults to 3.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        dw_reduction: float = 1.5,
        pos_kernel_size: int = 3,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(
                re_d_model,
                re_d_model,
                kernel_size=(pos_kernel_size, 1, 1),
                stride=(1, 1, 1),
                padding=(padding, 0, 0),
                groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        logger.info('Init zero for Conv in pos_emb')
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_embed(x)


class ResidualAttentionBlock(BaseModule):
    """Local UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        dw_reduction (float): Downsample ratio of input channels.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA.
            Defaults to True.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        drop_path: float = 0.0,
        dw_reduction: float = 1.5,
        no_lmhra: bool = False,
        double_lmhra: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.n_head = n_head
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')

        self.no_lmhra = no_lmhra
        self.double_lmhra = double_lmhra
        logger.info(f'No L_MHRA: {no_lmhra}')
        logger.info(f'Double L_MHRA: {double_lmhra}')
        if not no_lmhra:
            self.lmhra1 = Local_MHRA(d_model, dw_reduction=dw_reduction)
            if double_lmhra:
                self.lmhra2 = Local_MHRA(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x: torch.Tensor, T: int = 8) -> torch.Tensor:
        # x: 1+HW, NT, C
        if not self.no_lmhra:
            # Local MHRA
            tmp_x = x[1:, :, :]
            L, NT, C = tmp_x.shape
            N = NT // T
            H = W = int(L**0.5)
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra1(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # MHSA
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        # Local MHRA
        if not self.no_lmhra and self.double_lmhra:
            tmp_x = x[1:, :, :]
            tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0,
                                                      1).contiguous()
            tmp_x = tmp_x + self.drop_path(self.lmhra2(tmp_x))
            tmp_x = tmp_x.view(N, C, T,
                               L).permute(3, 0, 2,
                                          1).contiguous().view(L, NT, C)
            x = torch.cat([x[:1, :, :], tmp_x], dim=0)
        # FFN
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Extractor(BaseModule):
    """Global UniBlock.

    Args:
        d_model (int): Number of input channels.
        n_head (int): Number of attention head.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers.
            Defaults to 4.0.
        drop_out (float): Stochastic dropout rate.
            Defaults to 0.0.
        drop_path (float): Stochastic depth rate.
            Defaults to 0.0.
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_factor: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        logger.info(f'Drop path rate: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        d_mlp = round(mlp_factor * d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_mlp)),
                         ('gelu', QuickGELU()),
                         ('dropout', nn.Dropout(dropout)),
                         ('c_proj', nn.Linear(d_mlp, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.ln_3 = nn.LayerNorm(d_model)

        # zero init
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.constant_(self.attn.out_proj.weight, 0.)
        nn.init.constant_(self.attn.out_proj.bias, 0.)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[-1].weight, 0.)
        nn.init.constant_(self.mlp[-1].bias, 0.)

    def attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d_model = self.ln_1.weight.size(0)
        q = (x @ self.attn.in_proj_weight[:d_model].T
             ) + self.attn.in_proj_bias[:d_model]

        k = (y @ self.attn.in_proj_weight[d_model:-d_model].T
             ) + self.attn.in_proj_bias[d_model:-d_model]
        v = (y @ self.attn.in_proj_weight[-d_model:].T
             ) + self.attn.in_proj_bias[-d_model:]
        Tx, Ty, N = q.size(0), k.size(0), q.size(1)
        q = q.view(Tx, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        k = k.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        v = v.view(Ty, N, self.attn.num_heads,
                   self.attn.head_dim).permute(1, 2, 0, 3)
        aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5))

        aff = aff.softmax(dim=-1)
        out = aff @ v
        out = out.permute(2, 0, 1, 3).flatten(2)
        out = self.attn.out_proj(out)
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.ln_1(x), self.ln_3(y)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(BaseModule):
    """Backbone:

    Args:
        width (int): Number of input channels in local UniBlock.
        layers (int): Number of layers of local UniBlock.
        heads (int): Number of attention head in local UniBlock.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA
            in local UniBlock. Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        init_cfg (dict, optional): The config of weight initialization.
            Defaults to None.
    """

    def __init__(
        self,
        width: int,
        layers: int=12,
        heads: int=12,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
        dw_reduction: float = 1.5,
        no_lmhra: bool = True,
        double_lmhra: bool = True,
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.T = t_size
        self.return_list = return_list
        # backbone
        b_dpr = [
            x.item()
            for x in torch.linspace(0, backbone_drop_path_rate, layers)
        ]
        self.resblocks = ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
                no_lmhra=no_lmhra,
                double_lmhra=double_lmhra,
            ) for i in range(layers)
        ])

        # global block
        assert n_layers == len(return_list)
        self.temporal_cls_token = nn.Parameter(torch.zeros(1, 1, n_dim))
        self.dpe = ModuleList([
            nn.Conv3d(
                n_dim,
                n_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=n_dim) for _ in range(n_layers)
        ])
        for m in self.dpe:
            nn.init.constant_(m.bias, 0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.dec = ModuleList([
            Extractor(
                n_dim,
                n_head,
                mlp_factor=mlp_factor,
                dropout=mlp_dropout[i],
                drop_path=dpr[i],
            ) for i in range(n_layers)
        ])
        # weight sum
        self.norm = nn.LayerNorm(n_dim)
        self.balance = nn.Parameter(torch.zeros((n_dim)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
#         ipdb.set_trace()
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down
        H = W = int((L - 1)**0.5)
        cls_token = self.temporal_cls_token.repeat(1, N, 1)

        j = -1
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T_down)
            if i in self.return_list:
                j += 1
                tmp_x = x.clone()
                tmp_x = tmp_x.view(L, N, T_down, C)
                # dpe
                _, tmp_feats = tmp_x[:1], tmp_x[1:]
                tmp_feats = tmp_feats.permute(1, 3, 2,
                                              0).reshape(N, C, T_down, H, W)
                tmp_feats = self.dpe[j](tmp_feats.clone()).view(
                    N, C, T_down, L - 1).permute(3, 0, 2, 1).contiguous()
                tmp_x[1:] = tmp_x[1:] + tmp_feats
                # global block
                tmp_x = tmp_x.permute(2, 0, 1, 3).flatten(0, 1)  # T * L, N, C
                cls_token = self.dec[j](cls_token, tmp_x)

        weight = self.sigmoid(self.balance)
        out1 = x.view(L, N, T_down, C)[0]  #bs,t,768
        residual = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C
        out = self.norm((1 - weight) * cls_token[0, :, :] + weight * residual)
        return out,out1


# @MODELS.register_module()
class UniFormerV2(BaseModule):
    """UniFormerV2:

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        input_resolution (int): Number of input resolution.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        width (int): Number of input channels in local UniBlock.
            Defaults to 768.
        layers (int): Number of layers of local UniBlock.
            Defaults to 12.
        heads (int): Number of attention head in local UniBlock.
            Defaults to 12.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        temporal_downsample (bool): Whether downsampling temporal dimentison.
            Defaults to False.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA in local UniBlock.
            Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        clip_pretrained (bool): Whether to load pretrained CLIP visual encoder.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        # backbone
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        num_class:int =20,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
#         t_size: int = 16,
        kernel_size: int = 3,
        dw_reduction: float = 1.5,
        temporal_downsample: bool = False,
        no_lmhra: bool = True,
        double_lmhra: bool = True,
        # global block
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        # pretrain
        clip_pretrained: bool = False,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.clip_pretrained = clip_pretrained
        self.input_resolution = input_resolution
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
        )
        self.classifer = nn.Linear(768,num_class)
#         self.classifier2 = nn.Linear(768,200)


    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.clip_pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x):
#         ipdb.set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        features,out1 = self.transformer(x)
        out = self.classifer(features)

        return features,out1,out

# @MODELS.register_module()
class UniFormerV2bg(BaseModule):
    """UniFormerV2:

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        input_resolution (int): Number of input resolution.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        width (int): Number of input channels in local UniBlock.
            Defaults to 768.
        layers (int): Number of layers of local UniBlock.
            Defaults to 12.
        heads (int): Number of attention head in local UniBlock.
            Defaults to 12.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        temporal_downsample (bool): Whether downsampling temporal dimentison.
            Defaults to False.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA in local UniBlock.
            Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        clip_pretrained (bool): Whether to load pretrained CLIP visual encoder.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        # backbone
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        num_class:int =20,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
#         t_size: int = 16,
        kernel_size: int = 3,
        dw_reduction: float = 1.5,
        temporal_downsample: bool = False,
        no_lmhra: bool = True,
        double_lmhra: bool = True,
        # global block
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        # pretrain
        clip_pretrained: bool = False,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.clip_pretrained = clip_pretrained
        self.input_resolution = input_resolution
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
        )
        self.classifer = nn.Linear(768,num_class)
#         self.classifier2 = nn.Linear(768,200)


    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.clip_pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x):
#         ipdb.set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        features,out1 = self.transformer(x)
        out = self.classifer(features)

        return features,out1,out

class UniFormerV2obj(BaseModule):
    """UniFormerV2:

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        input_resolution (int): Number of input resolution.
            Defaults to 224.
        patch_size (int): Number of patch size.
            Defaults to 16.
        width (int): Number of input channels in local UniBlock.
            Defaults to 768.
        layers (int): Number of layers of local UniBlock.
            Defaults to 12.
        heads (int): Number of attention head in local UniBlock.
            Defaults to 12.
        backbone_drop_path_rate (float): Stochastic depth rate
            in local UniBlock. Defaults to 0.0.
        t_size (int): Number of temporal dimension after patch embedding.
            Defaults to 8.
        temporal_downsample (bool): Whether downsampling temporal dimentison.
            Defaults to False.
        dw_reduction (float): Downsample ratio of input channels in local MHRA.
            Defaults to 1.5.
        no_lmhra (bool): Whether removing local MHRA in local UniBlock.
            Defaults to False.
        double_lmhra (bool): Whether using double local MHRA in local UniBlock.
            Defaults to True.
        return_list (List[int]): Layer index of input features
            for global UniBlock. Defaults to [8, 9, 10, 11].
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of layers of global UniBlock.
            Defaults to 4.
        n_dim (int): Number of input channels in global UniBlock.
            Defaults to 768.
        n_head (int): Number of attention head in global UniBlock.
            Defaults to 12.
        mlp_factor (float): Ratio of hidden dimensions in MLP layers
            in global UniBlock. Defaults to 4.0.
        drop_path_rate (float): Stochastic depth rate in global UniBlock.
            Defaults to 0.0.
        mlp_dropout (List[float]): Stochastic dropout rate in each MLP layer
            in global UniBlock. Defaults to [0.5, 0.5, 0.5, 0.5].
        clip_pretrained (bool): Whether to load pretrained CLIP visual encoder.
            Defaults to True.
        pretrained (str): Name of pretrained model.
            Defaults to None.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        # backbone
        input_resolution: int = 224,
        patch_size: int = 16,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        num_class:int =20,
        backbone_drop_path_rate: float = 0.,
        t_size: int = 8,
#         t_size: int = 16,
        kernel_size: int = 3,
        dw_reduction: float = 1.5,
        temporal_downsample: bool = False,
        no_lmhra: bool = True,
        double_lmhra: bool = True,
        # global block
        return_list: List[int] = [8, 9, 10, 11],
        n_layers: int = 4,
        n_dim: int = 768,
        n_head: int = 12,
        mlp_factor: float = 4.0,
        drop_path_rate: float = 0.,
        mlp_dropout: List[float] = [0.5, 0.5, 0.5, 0.5],
        # pretrain
        clip_pretrained: bool = False,
        pretrained: Optional[str] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.pretrained = pretrained
        self.clip_pretrained = clip_pretrained
        self.input_resolution = input_resolution
        padding = (kernel_size - 1) // 2
        if temporal_downsample:
            self.conv1 = nn.Conv3d(
                3,
                width, (kernel_size, patch_size, patch_size),
                (2, patch_size, patch_size), (padding, 0, 0),
                bias=False)
            t_size = t_size // 2
        else:
            self.conv1 = nn.Conv3d(
                3,
                width, (1, patch_size, patch_size),
                (1, patch_size, patch_size), (0, 0, 0),
                bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(
            width,
            layers,
            heads,
            dw_reduction=dw_reduction,
            backbone_drop_path_rate=backbone_drop_path_rate,
            t_size=t_size,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            return_list=return_list,
            n_layers=n_layers,
            n_dim=n_dim,
            n_head=n_head,
            mlp_factor=mlp_factor,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
        )
        self.classifer = nn.Linear(768,num_class)
#         self.classifier2 = nn.Linear(768,200)


    def _inflate_weight(self,
                        weight_2d: torch.Tensor,
                        time_dim: int,
                        center: bool = True) -> torch.Tensor:
        logger.info(f'Init center: {center}')
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim
        return weight_3d

    def init_weights(self):
        """Initialize the weights in backbone."""
        if self.clip_pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            self._load_pretrained(self.pretrained)
        else:
            if self.pretrained:
                self.init_cfg = dict(
                    type='Pretrained', checkpoint=self.pretrained)
            super().init_weights()

    def forward(self, x):
#         ipdb.set_trace()
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        features,out1 = self.transformer(x)
        out = self.classifer(features)

        return features,out1,out

# if __name__ == '__main__':
    
#     model = ECRL()
#     ipdb.set_trace()
#     frames = torch.rand(2,3,8,224, 224)  # [batch size, channl, frames, H, W
#     global_feat_dict = torch.rand(512,768)
#     bg=torch.rand(2,3,8,224, 224)
#     obj=torch.rand(2,3,8,224, 224)
#     confounder_dictionary= torch.rand(20,768)
#     prior=torch.rand(20,1)    
    
    
#     output = model(frames,obj, bg,confounder_dictionary,prior,global_feat_dict,2)
#     print(output.shape)

