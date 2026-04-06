import tvm
import tvm.relay as relay

class UniversalLayerAnalyzer(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.ops_to_emit = [] 
        self.flags = 0

    def visit_call(self, call):
        super().visit_call(call)
        if isinstance(call.op, tvm.ir.Op):
            op_name = call.op.name
            if op_name == "sigmoid": self.flags |= 0x1
            elif op_name == "multiply": self.flags |= 0x2
            elif op_name in ["nn.relu", "relu"]: self.flags |= 0x4
            elif op_name == "nn.conv2d": self.ops_to_emit.append(("CONV", call))
            elif op_name == "nn.bias_add": self.ops_to_emit.append(("BIAS", call))
            elif op_name == "nn.max_pool2d": self.ops_to_emit.append(("POOL", call))
            elif op_name == "add": self.ops_to_emit.append(("ADD", call))
            elif "concatenate" in op_name: self.ops_to_emit.append(("CONCAT", call))
            elif op_name in ["image.resize2d", "split"]: self.ops_to_emit.append(("OTHER", call))
            else: self.ops_to_emit.append(("OTHER", call))