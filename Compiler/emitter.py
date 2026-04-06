import tvm
import tvm.relay as relay
import numpy as np
from utils import get_tensor_shape, float_to_int32_bits
from analyzer import UniversalLayerAnalyzer

class NPUFullProgramEmitter(relay.ExprVisitor):
    def __init__(self, params_dict):
        super().__init__()
        self.params_dict = params_dict
        self.instructions = []
        self.pc_counter = 0 
        self.memory_map = {} 
        
        self.current_free_addr = 0x00500000 
        
        self.cpu_op_names = set()
        self.cpu_tasks = []
        
        # Weight Block
        self.weight_memory = bytearray(64 * 1024 * 1024)
        self.max_weight_offset = 0 

    def visit_call(self, call):
        super().visit_call(call)
        analyzer = UniversalLayerAnalyzer()
        
        if isinstance(call.op, relay.Function) and call.op.attrs is not None and "Primitive" in call.op.attrs:
            analyzer.visit(call.op.body)
        elif isinstance(call.op, tvm.ir.Op):
            analyzer.visit(call) 
        else: return

        last_inner_out_addr = None

        def get_real_addr(node):
            if isinstance(node, relay.Var) and hasattr(call.op, "params"):
                for p, a in zip(call.op.params, call.args):
                    if p == node: return get_real_addr(a)
            if isinstance(node, relay.TupleGetItem):
                return get_real_addr(node.tuple_value)
            return self.memory_map.get(node, 0x00000000)

        def get_w_np(node):
            if isinstance(node, relay.Var) and hasattr(call.op, "params"):
                for p, a in zip(call.op.params, call.args):
                    if p == node: return get_w_np(a)
            if isinstance(node, relay.Constant): return node.data.numpy()
            if isinstance(node, relay.Var) and node.name_hint in self.params_dict: 
                return self.params_dict[node.name_hint].numpy()
            return None

        for op_label, inner_call in analyzer.ops_to_emit:
            wgt_addr = 0x40000000 + self.max_weight_offset
            
            in_shape = get_tensor_shape(inner_call.args[0].checked_type) if hasattr(inner_call, "args") and len(inner_call.args) > 0 else [1, 1, 1, 1]
            out_shape = get_tensor_shape(inner_call.checked_type)
            
            in_c, in_h, in_w = 1, 1, 1
            if len(in_shape) >= 4: in_c, in_h, in_w = in_shape[1], in_shape[2], in_shape[3]
            elif len(in_shape) == 3: in_c, in_h, in_w = in_shape[0], in_shape[1], in_shape[2]
            elif len(in_shape) == 2: in_c, in_h, in_w = in_shape[1], 1, 1
                
            out_c = out_shape[1] if len(out_shape) >= 2 else (out_shape[0] if len(out_shape) == 1 else 1)
            layer_size = int(np.prod(out_shape)) * 4 
            
            in_addr = 0x0
            if len(inner_call.args) > 0:
                in_addr = get_real_addr(inner_call.args[0])

            layer_scale = 1.0 

            if op_label == "ADD" and len(inner_call.args) > 1:
                wgt_addr = get_real_addr(inner_call.args[1])
            elif op_label == "CONCAT":
                tup_node = inner_call.args[0]
                if isinstance(tup_node, relay.Var) and hasattr(call.op, "params"):
                    for p, a in zip(call.op.params, call.args):
                        if p == tup_node: tup_node = a
                if isinstance(tup_node, relay.Tuple):
                    in_addr = get_real_addr(tup_node.fields[0]) if len(tup_node.fields) > 0 else 0x0
                    wgt_addr = get_real_addr(tup_node.fields[1]) if len(tup_node.fields) > 1 else 0x0

            if op_label == "CONV" and len(inner_call.args) > 1:
                w_np = get_w_np(inner_call.args[1])
                if w_np is not None:
                    w_np = w_np.astype(np.float32)
                    max_val = np.max(np.abs(w_np))
                    layer_scale = max_val / 127.0 if max_val != 0 else 1.0
                    w_int8 = np.round(w_np / layer_scale).astype(np.int8)
                    w_int8 = np.clip(w_int8, -127, 127)
                    w_bytes = w_int8.tobytes()
                    
                    offset = wgt_addr - 0x40000000
                    self.weight_memory[offset : offset + len(w_bytes)] = w_bytes
                    self.max_weight_offset += (len(w_bytes) + 0xF) & ~0xF

            if op_label == "BIAS" and len(inner_call.args) > 1:
                b_np = get_w_np(inner_call.args[1])
                if b_np is not None:
                    b_np = b_np.astype(np.float32)
                    b_bytes = b_np.tobytes()
                    offset = wgt_addr - 0x40000000
                    self.weight_memory[offset : offset + len(b_bytes)] = b_bytes
                    self.max_weight_offset += (len(b_bytes) + 0xF) & ~0xF

            out_addr = self.current_free_addr
            self.memory_map[inner_call] = out_addr 
            last_inner_out_addr = out_addr
            self.current_free_addr = (out_addr + layer_size + 0x3F) & ~0x3F 

            scale_bits = float_to_int32_bits(layer_scale)
            stride = 1; pad = 0; kernel = 1
            if hasattr(inner_call, "attrs"):
                if hasattr(inner_call.attrs, "strides"): stride = int(inner_call.attrs.strides[0])
                if hasattr(inner_call.attrs, "padding"): pad = int(inner_call.attrs.padding[0])
                if hasattr(inner_call.attrs, "kernel_size"): kernel = int(inner_call.attrs.kernel_size[0])

            is_cpu = (in_h == 1 and in_w == 1) or in_w > 1000 or op_label == "OTHER"

            if is_cpu:
                op_name = inner_call.op.name if hasattr(inner_call, "op") else "unknown"
                self.cpu_op_names.add(op_name)
                task_code = f"    elif interrupt_pc == {self.pc_counter}:\n" \
                            f"        cpu_execute_{op_name.replace('.', '_')}(sram, 0x{in_addr:08X}, 0x{out_addr:08X}, {in_h}, {in_w}, {in_c})\n"
                self.cpu_tasks.append(task_code)
                self.instructions.append(f"OP:OTHER  | IN:0x{in_addr:08X} | WGT:0x{wgt_addr:08X} | OUT:0x{out_addr:08X} | FLAGS:0x0 | STRIDE:{stride} | PAD:0 | KERNEL:1")
                self.pc_counter += 1
            else:
                self.instructions.append(f"OP:CONFIG | IN_H:{in_h} | IN_W:{in_w} | IN_C:{in_c} | OUT_C:{out_c} | STRIDE:{stride} | SCALE:0x{scale_bits:08X}")
                self.pc_counter += 1
                self.instructions.append(f"OP:{op_label:6s} | IN:0x{in_addr:08X} | WGT:0x{wgt_addr:08X} | OUT:0x{out_addr:08X} | FLAGS:0x{analyzer.flags:X} | STRIDE:{stride} | PAD:{pad} | KERNEL:{kernel}")
                self.pc_counter += 1

        if last_inner_out_addr is not None:
            self.memory_map[call] = last_inner_out_addr