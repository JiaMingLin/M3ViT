r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from fmoe.layers import FMoE, _fmoe_general_global_forward
from fmoe.linear import FMoELinear
from functools import partial
import tree
import torch
import torch.nn as nn

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import AllGather, Slice
from fmoe.gates import NaiveGate

from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE

from pdb import set_trace
import numpy as np

class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x

class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_gate=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        gate=NaiveGate,
        world_size=1,
        top_k=2,
        vmoe_noisy_std=1,
        gate_return_decoupled_activation=False,
        gate_task_specific_dim=-1,
        multi_gate=False,
        regu_experts_fromtask = False,
        num_experts_pertask = -1,
        num_tasks = -1,
        regu_sem = False,
        sem_force = False,
        regu_subimage = False,
        expert_prune = False,
        prune_threshold = 0.1,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        self.our_d_gate = d_gate
        self.our_d_model = d_model

        self.num_expert = num_expert
        self.regu_experts_fromtask = regu_experts_fromtask
        self.num_experts_pertask = num_experts_pertask
        self.num_tasks = num_tasks
        self.regu_sem = regu_sem
        self.sem_force = sem_force
        self.regu_subimage = regu_subimage
        self.expert_prune = expert_prune
        self.prune_threshold = prune_threshold
        if self.sem_force:
            self.force_id=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]
        if self.regu_experts_fromtask:
            self.start_experts_id=[]
            start_id = 0
            for i in range(self.num_tasks):
                start_id = start_id + int(i* (self.num_expert-self.num_experts_pertask)/(self.num_tasks-1))
                self.start_experts_id.append(start_id)
            print('self.start_experts_id',self.start_experts_id)

        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.gate_task_specific_dim = gate_task_specific_dim
        self.multi_gate=multi_gate
        if gate_task_specific_dim<0:
            d_gate = d_model
        else:
            d_gate = d_model+gate_task_specific_dim
        print('multi_gate',self.multi_gate)
        if gate == NoisyGate:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=gate_return_decoupled_activation, regu_experts_fromtask = self.regu_experts_fromtask,
                    num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=self.regu_sem,sem_force = self.sem_force)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=gate_return_decoupled_activation, regu_experts_fromtask = self.regu_experts_fromtask,
                num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=self.regu_sem,sem_force = self.sem_force)
        elif gate == NoisyGate_VMoE:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=gate_return_decoupled_activation,
                    noise_std=vmoe_noisy_std,regu_experts_fromtask = self.regu_experts_fromtask,
                    num_experts_pertask=self.num_experts_pertask, num_tasks=self.num_tasks,regu_sem=self.regu_sem,sem_force = self.sem_force, regu_subimage=self.regu_subimage)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=gate_return_decoupled_activation,
                noise_std=vmoe_noisy_std,regu_experts_fromtask = self.regu_experts_fromtask,
                num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks,regu_sem=self.regu_sem,sem_force = self.sem_force, regu_subimage=self.regu_subimage)

        else:
            raise ValueError("No such gating type")
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, gate_inp=None, task_id = None, task_specific_feature = None, sem=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        if gate_inp is None:
            gate_inp = inp
        

        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        gate_channel = gate_inp.shape[-1]
        gate_inp = gate_inp.reshape(-1, gate_channel)
        # print('task_id, task_specific_feature',task_id, task_specific_feature)
        if (task_id is not None) and (task_specific_feature is not None):
            assert self.multi_gate is False
            size = gate_inp.shape[0]
            gate_inp = torch.cat((gate_inp,task_specific_feature.repeat(size,1)),dim=-1)
        output = self.forward_moe(gate_inp=gate_inp, moe_inp=inp, task_id=task_id, sem=sem)
        return output.reshape(original_shape)


    def forward_moe(self, gate_inp, moe_inp, task_id=None, sem=None):
        r"""
        MoE（Mixture of Experts）前向傳播的核心方法
        
        流程說明：
        1. 透過 gate 模組計算 expert 選擇分數
        2. 根據 gate 分數進行 MoE 前向計算
        3. 將 gate 分數作為權重乘以各 expert 的輸出
        
        參數：
            gate_inp: 輸入給 gate 模組的資料，用於計算 expert 選擇
            moe_inp: 輸入給 expert 的資料，用於實際計算
            task_id: 任務 ID，用於多任務學習
            sem: 語義資訊，用於語義感知的 expert 選擇
        """
        # === 步驟 1：輸入驗證 ===
        # 檢查所有輸入張量的 batch size 是否一致
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        # === 步驟 2：分散式訓練設置 ===
        # 如果使用多個 GPU，確保張量在正確的通信群組中
        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
            tree.map_structure(ensure_comm_func, gate_inp)
            
        # === 步驟 3：數據切片處理 ===
        # 如果啟用數據切片，將輸入按切片大小分割
        if self.slice_size > 1:
            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)

        # === 步驟 4：Gate 計算 - 選擇 top-k experts ===
        if self.multi_gate:
            # 多 gate 模式：針對不同任務使用不同的 gate
            if task_id is not None:
                # 使用指定任務的 gate
                gate_top_k_idx, gate_score = self.gate[task_id](gate_inp)
            else:
                # 如果沒有指定任務 ID，使用第一個 gate
                gate_top_k_idx, gate_score = self.gate[0](gate_inp)
        else:
            # 單 gate 模式：根據 gate 類型傳遞不同參數
            if isinstance(self.gate, NoisyGate_VMoE):
                # NoisyGate_VMoE 支援 task_id 和 sem 參數
                gate_top_k_idx, gate_score = self.gate(gate_inp, task_id=task_id, sem=sem)
            else:
                # NoisyGate 只接受輸入參數
                gate_top_k_idx, gate_score = self.gate(gate_inp)

        # === 步驟 5：Expert 剪枝 ===
        # 如果啟用 expert 剪枝，將低於閾值的分數設為 0
        if self.expert_prune:
            gate_score = torch.where(gate_score>self.prune_threshold, gate_score, 0.)
            prune_prob = 1-torch.nonzero(gate_score).shape[0]/torch.cumprod(torch.tensor(gate_score.shape),dim=0)[-1]
            print('prune_prob', prune_prob)
            
        # === 步驟 6：語義強制設置 ===
        # 根據語義資訊強制選擇特定的 expert
        if self.sem_force and (sem is not None):
            batch = sem.shape[0]
            gate_top_k_idx = gate_top_k_idx.reshape(batch, -1, self.top_k)
            sem = sem.reshape(batch, -1)
            # 遍歷每個 batch 和每個位置，根據語義強制選擇 expert
            for k in range(batch):
                for i in range(sem.shape[-1]):
                    for j in range(len(self.force_id)):
                        if sem[k,i] in self.force_id[j]:
                            # 強制選擇特定的 expert 對
                            gate_top_k_idx[k,i+1,:] = [j*2, j*2+1]
            gate_top_k_idx = gate_top_k_idx.reshape(-1, self.top_k)
            # 設置均等的 gate 分數
            gate_score = torch.ones((gate_score.shape[0], self.top_k), device=gate_score.device) * 0.5

        # === 步驟 7：任務特定的 Expert 調整 ===
        # 如果啟用任務特定的 expert 分配，調整 expert 索引
        if self.regu_experts_fromtask and (task_id is not None):
            # 將 expert 索引偏移到該任務對應的 expert 範圍
            gate_top_k_idx = gate_top_k_idx + self.start_experts_id[task_id]

        # === 步驟 8：Gate Hook ===
        # 如果設置了 gate hook，執行額外的處理
        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # === 步驟 9：遮罩處理 ===
        # 刪除被遮罩的張量（用於處理變長序列）
        if self.mask is not None and self.mask_dict is not None:
            def delete_mask_func(tensor):
                # 只保留未被遮罩的元素
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]

        # === 步驟 10：MoE 前向計算核心 ===
        # 執行實際的 expert 計算
        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size
        )

        # === 步驟 11：遮罩恢復 ===
        # 恢復被刪除的張量，填入對應的值
        if self.mask is not None and self.mask_dict is not None:
            def recover_func(tensor):
                # 重塑為 (batch_size * seq_len, top_k, dim) 格式
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # 創建完整大小的零張量
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # 恢復未遮罩的位置
                x[mask == 0] = tensor
                # 填入遮罩位置的預設值
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x

            moe_outp = tree.map_structure(recover_func, fwd)
        else:
            # 如果沒有遮罩，直接重塑張量
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp = tree.map_structure(view_func, fwd)

        # === 步驟 12：Gate 分數加權 ===
        # 將 gate 分數作為權重應用到 expert 輸出上
        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            # 使用批次矩陣乘法將 gate 分數與 expert 輸出相乘
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        # === 步驟 13：數據切片聚合 ===
        # 如果使用了數據切片，將結果聚合回來
        if self.slice_size > 1:
            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)

        # === 步驟 14：輸出驗證 ===
        # 確保輸出張量的 batch size 一致
        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        
        return moe_outp
