import torch
import torch.distributed as dist
import contextlib
from torch import nn
from torch.autograd import Variable
from typing import List
import weigou.parallel.manager as pgm

# --- Buckets ---

class Bucket:
    def __init__(self, params: List[torch.nn.Parameter], grad_data: torch.Tensor, process_group: torch.distributed.ProcessGroup) -> None:
        self.params = set(params)    
        self.params_with_grad_ready = set() 
        self.grad_data = grad_data  
        self.process_group = process_group  
        self.process_group_size = dist.get_world_size(group=self.process_group) 
        self.handle = None 
        self.reset()
    
    def sync_gradient(self) -> None:
        assert self.handle is None
        self.grad_data /= self.process_group_size
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)
    
    def reset(self) -> None:
        self.handle = None
        self.params_with_grad_ready.clear() 
        self.grad_data.zero_() 

    def wait(self) -> None:
        assert self.handle is not None, "You should launch an allreduce operation before waiting for it to finish"
        self.handle.wait() 

    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

class BucketManager:
    def __init__(self, params: List[torch.nn.Parameter], process_group: torch.distributed.ProcessGroup, bucket_size: int, grad_type: torch.dtype = torch.float32) -> None:
        self.params = list(params) 
        self.device = self.params[0].device if self.params[0].is_cuda else torch.device("cpu")
        self.buckets = [] 
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(group=self.process_group)
        self.params_to_bucket_location = {} 
        self.bucket_size = bucket_size
        self.grad_type = grad_type
        self.grad_data_list = []
        self._initialize_buckets()
    
    def _initialize_buckets(self) -> None:
        cur_bucket_size = 0 
        cur_bucket_idx = 0
        
        for param in self.params:
            if not param.requires_grad:
                continue
            
            if cur_bucket_size == 0:
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
                continue
            
            if cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                self.params_to_bucket_location[param] = (0, param.numel(), cur_bucket_idx)
                cur_bucket_size = param.numel()
            else:
                self.params_to_bucket_location[param] = (cur_bucket_size, cur_bucket_size + param.numel(), cur_bucket_idx)
                cur_bucket_size += param.numel()

        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)
        
        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(torch.zeros(bucket_sizes[i], dtype=self.grad_type, device=self.device))
            self.buckets.append(Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group))
        
        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.params_to_bucket_location[param]
            param.main_grad = self.grad_data_list[bucket_id][data_start_index:data_end_index].view(param.shape)

    def reset(self) -> None:
        for bucket in self.buckets:
            bucket.reset()
    
    def wait(self) -> None:
        for bucket in self.buckets:
            bucket.wait()
    
    def mark_param_as_ready(self, param: torch.nn.Parameter) -> None:
        bucket_idx = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)

# --- Data Parallel ---

class DataParallelNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True 
        self.register_backward_hook(self._allreduce_grads)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def register_backward_hook(self, hook):
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_post_accumulate_grad_hook(hook)
                
    def _allreduce_grads(self, grad):
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.process_group_manager.cp_dp_group)
            grad /= pgm.process_group_manager.cp_dp_world_size
        return grad 
    
    @contextlib.contextmanager
    def no_sync(self):
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True

class DataParallelBucket(nn.Module):
    def __init__(self, module, bucket_cap_mb=25, grad_type = torch.float32):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True 
        grad_size = 2 if grad_type == torch.bfloat16 else 4 
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size 
        self.bucket_manager = BucketManager(module.parameters(), pgm.process_group_manager.cp_dp_group, bucket_size, grad_type)
        self.register_backward_hook()
        self._post_backward_callback_set = False 
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        return self.module.backward(input_tensor, output_tensor, output_tensor_grad)
    
    def register_backward_hook(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                
    def _make_param_hook(self, param: torch.nn.Parameter,bucket_manager: BucketManager):
        def param_hook(*unused):
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data) 
                param.grad = None
                
                if self.require_backward_grad_sync:
                    if not self._post_backward_callback_set:
                        Variable._execution_engine.queue_callback(self._post_backward)
                        self._post_backward_callback_set = True
                        
                    bucket_manager.mark_param_as_ready(param) 
        return param_hook
    
    @contextlib.contextmanager
    def no_sync(self):
        self.require_backward_grad_sync = False
        yield
        self.require_backward_grad_sync = True
        
    def _post_backward(self):
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype) 

    def reset(self):
        self.bucket_manager.reset()
