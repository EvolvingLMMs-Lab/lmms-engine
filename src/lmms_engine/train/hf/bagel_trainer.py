import torch
import torch.distributed as dist
from .trainer import Trainer
from lmms_engine.train.registry import TRAINER_REGISTER

@TRAINER_REGISTER.register("bagel_trainer")
class BagelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算Bagel模型的损失，参考reference.py中的实现
        支持CE loss（视觉理解）和MSE loss（视觉生成）
        """
        # 继承父类的MFU计算逻辑
        if self.state.global_step == 0 or getattr(self, "cur_time", None) is None:
            self.cur_time = __import__('time').perf_counter()
            self.mfu = 0.0
            self.total_seq_len = []
        
        # 调用模型前向传播
        loss_dict = model(**inputs)
        
        total_loss = 0
        device = next(model.parameters()).device
        
        # 处理CE loss（视觉理解）
        ce = loss_dict.get("ce", None)
        if ce is not None:
            # 计算CE tokens总数
            ce_loss_indexes = inputs.get('ce_loss_indexes', None)
            if ce_loss_indexes is not None:
                total_ce_tokens = torch.tensor(ce_loss_indexes.sum().item(), device=device)
            else:
                # fallback: 使用实际ce loss的数量
                total_ce_tokens = torch.tensor(ce.numel() if ce.numel() > 0 else 1, device=device)
            
            # 分布式归约
            if dist.is_initialized():
                dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            
            # 处理CE loss权重重新加权（如果提供）
            ce_loss_weights = inputs.get('ce_loss_weights', None)
            if ce_loss_weights is not None and getattr(self.args, 'ce_loss_reweighting', False):
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                if dist.is_initialized():
                    dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce_loss = ce.sum() * (dist.get_world_size() if dist.is_initialized() else 1) / total_ce_loss_weights
            else:
                ce_loss = ce.sum() * (dist.get_world_size() if dist.is_initialized() else 1) / total_ce_tokens
            
            total_loss = total_loss + ce_loss * getattr(self.args, 'ce_weight', 1.0)
        else:
            ce_loss = torch.tensor(0.0, device=device)
        
        # 处理MSE loss（视觉生成）
        mse = loss_dict.get("mse", None)
        if mse is not None:
            # 计算MSE tokens总数
            mse_loss_indexes = inputs.get('mse_loss_indexes', None)
            if mse_loss_indexes is not None:
                total_mse_tokens = torch.tensor(mse_loss_indexes.sum().item(), device=device)
            else:
                # fallback: 使用实际mse loss的数量
                total_mse_tokens = torch.tensor(mse.numel() if mse.numel() > 0 else 1, device=device)
            
            # 分布式归约
            if dist.is_initialized():
                dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            
            # MSE loss归一化
            mse_loss = mse.mean(dim=-1).sum() * (dist.get_world_size() if dist.is_initialized() else 1) / total_mse_tokens
            
            total_loss = total_loss + mse_loss * getattr(self.args, 'mse_weight', 1.0)
        else:
            mse_loss = torch.tensor(0.0, device=device)
        
        # MFU计算逻辑（从父类继承）
        if self.state.global_step % 10 == 0 and self.state.global_step > 0:
            prev_time = self.cur_time
            self.cur_time = __import__('time').perf_counter()
            try:
                import lmms_engine.models.utils as model_utils
                import lmms_engine.parallel.process_group_manager as pgm
                from lmms_engine.parallel.sequence_parallel.ulysses import get_ulysses_sequence_parallel_world_size
                
                device_rank = self.args.local_rank
                flops, promised_flops = model_utils.flops_counter.estimate_flops(
                    self.total_seq_len, delta_time=self.cur_time - prev_time
                )
                flops_tensor = torch.tensor(flops, device=device_rank)
                if dist.is_initialized():
                    dist.all_reduce(flops_tensor, op=dist.ReduceOp.SUM)
                sp_size = pgm.process_group_manager.cp_world_size
                self.mfu = (
                    flops_tensor.item() / self.args.world_size / sp_size / promised_flops
                )
                self.log({"mfu": round(self.mfu, 2)})
                self.total_seq_len.clear()
            except Exception:
                # 如果MFU计算失败，则跳过
                pass
        
        # 应用序列并行缩放（从父类继承）
        try:
            from lmms_engine.parallel.sequence_parallel.ulysses import get_ulysses_sequence_parallel_world_size
            total_loss = total_loss * get_ulysses_sequence_parallel_world_size()
        except Exception:
            # 如果没有序列并行，则不进行缩放
            pass
        
        # 构建输出
        outputs = {
            'ce': ce_loss.detach() if ce is not None else torch.tensor(0.0, device=device),
            'mse': mse_loss.detach() if mse is not None else torch.tensor(0.0, device=device),
        }
        
        return (total_loss, outputs) if return_outputs else total_loss
