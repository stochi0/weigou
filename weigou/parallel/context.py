import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Any, Optional, Tuple, List
import weigou.parallel.manager as pgm

# --- Communications ---

STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"

class ContextCommunicate:
    def __init__(self, msg: str = ""):
        global STEP
        global VERBOSE
        self._pending_operations: List[dist.P2POp] = []
        self._active_requests = None
        self.rank = pgm.process_group_manager.cp_rank
        self.world_size = pgm.process_group_manager.cp_world_size
        self.send_rank = pgm.process_group_manager.cp_send_rank
        self.recv_rank = pgm.process_group_manager.cp_recv_rank
        if VERBOSE: print(f"RingComm ({msg}) | initialized | RANK:{self.rank} | WORLD_SIZE:{self.world_size} | SEND_RANK:{self.send_rank} | RECV_RANK:{self.recv_rank}", flush=True)

    def send_recv(self, tensor_to_send, recv_tensor=None):
        if recv_tensor is None:
            result_tensor = torch.zeros_like(tensor_to_send)
        else:
            result_tensor = recv_tensor

        send_operation = dist.P2POp(dist.isend, tensor_to_send, self.send_rank, group=pgm.process_group_manager.cp_group)
        recv_operation = dist.P2POp(dist.irecv, result_tensor, self.recv_rank, group=pgm.process_group_manager.cp_group)
        
        self._pending_operations.extend([send_operation, recv_operation])

        if VERBOSE:
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | ACTION:sending | TO:{self.send_rank} | TENSOR:{tensor_to_send}", flush=True)
            print(f"RingComm | send_recv | STEP:{STEP} | RANK:{self.rank} | ACTION:receiving | FROM:{self.recv_rank} | TENSOR:{result_tensor}", flush=True)
        return result_tensor

    def commit(self):
        if self._active_requests is not None: raise RuntimeError("Commit called twice")
        self._active_requests = dist.batch_isend_irecv(self._pending_operations)
        if VERBOSE: print(f"RingComm | commit | STEP:{STEP} | RANK:{self.rank} | ACTION:committed | NUM_OPS:{len(self._pending_operations) // 2}", flush=True)

    def wait(self):
        if self._active_requests is None: raise RuntimeError("Wait called before commit")
        for i, request in enumerate(self._active_requests):
            request.wait()
            if VERBOSE:
                operation_type = "send" if i % 2 == 0 else "receive"
                peer_rank = self.send_rank if operation_type == "send" else self.recv_rank
                print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | ACTION:completed_{operation_type} | {'FROM' if operation_type == 'receive' else 'TO'}:{peer_rank}", flush=True)
        torch.cuda.synchronize()
        self._active_requests = None
        self._pending_operations = []
        if VERBOSE: print(f"RingComm | wait | STEP:{STEP} | RANK:{self.rank} | ACTION:all_operations_completed", flush=True)

# --- Attention Logic ---

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_: Optional[Any] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    def _update(current_out, current_lse):
        current_out = current_out - F.sigmoid(block_lse - current_lse) * (current_out - block_out)
        current_lse = current_lse - F.logsigmoid(current_lse - block_lse)
        return current_out, current_lse
    
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        return block_out, block_lse

    if slice_ is not None:
        out[slice_], lse[slice_] = _update(out[slice_], lse[slice_])
    else:
        out, lse = _update(out, lse)
        
    return out, lse

def ring_attention_forward(q, k, v, sm_scale, is_causal):
    batch_size, nheads, seqlen, d = q.shape
    S = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if is_causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, nheads, seqlen, seqlen)
        S.masked_fill_(causal_mask, float('-inf'))

    S_max = torch.max(S, dim=-1, keepdim=True)[0]
    exp_S = torch.exp(S - S_max)
    exp_sum = torch.sum(exp_S, dim=-1, keepdim=True)
    log_sum_exp = torch.log(exp_sum) + S_max
    P = exp_S / exp_sum
    O = torch.matmul(P, v)
    return O, log_sum_exp.squeeze(-1)

def ring_attention_backward(dO, Q, K, V, O, softmax_lse, sm_scale, is_causal):
    batch_size, nheads, seqlen, d = Q.shape
    
    S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if is_causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

    P = torch.exp(S - softmax_lse.unsqueeze(-1))
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    dS = P * (dP - D)
    if is_causal:
        dS = dS.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), 0)
    dQ = torch.matmul(dS, K) * sm_scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * sm_scale
    return dQ, dK, dV

class RingAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal):
        comm = ContextCommunicate("comm")
        k_og = k.clone()
        v_og = v.clone()
        out, lse = None, None
        next_k, next_v = None, None

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal or step <= comm.rank:
                block_out, block_lse  = ring_attention_forward(
                    q, k, v, sm_scale, is_causal and step == 0
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        kv_comm = ContextCommunicate("kv_comm")
        d_kv_comm = ContextCommunicate("d_kv_comm")
        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        
        block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        next_dk, next_dv = None, None
        next_k, next_v = None, None

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            if step <= kv_comm.rank or not is_causal:
                bwd_causal = is_causal and step == 0

                block_dq_buffer, block_dk_buffer, block_dv_buffer = ring_attention_backward(
                    dout, q, k, v, out, softmax_lse, sm_scale, bwd_causal
                )

                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_buffer.to(torch.float32)
                    dv = block_dv_buffer.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq, next_dk, next_dv, None, None

def ring_attention(q, k, v, sm_scale, is_causal):
    return RingAttentionFunc.apply(q, k, v, sm_scale, is_causal)

def update_rope_for_context_parallel(cos, sin):
    seq_len, _ = cos.size()
    cp_rank, cp_world_size = pgm.process_group_manager.cp_rank, pgm.process_group_manager.cp_world_size
    assert seq_len % cp_world_size == 0, f"Input sequence length ({seq_len}) must be divisible by cp_world_size ({cp_world_size})"
    size_per_partition = seq_len // cp_world_size
    start_idx, end_idx = cp_rank * size_per_partition, (cp_rank + 1) * size_per_partition
    return cos[start_idx:end_idx], sin[start_idx:end_idx]

def apply_context_parallel(model):
    os.environ["CONTEXT_PARALLEL"] = "1" if pgm.process_group_manager.cp_world_size > 1 else "0"
    return model
