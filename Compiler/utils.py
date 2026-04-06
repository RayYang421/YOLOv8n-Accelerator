import struct

def get_tensor_shape(checked_type):
    if checked_type is None: return [1, 1, 1, 1]
    if hasattr(checked_type, "shape"): return [int(d) for d in checked_type.shape]
    elif hasattr(checked_type, "fields"): return [int(d) for d in checked_type.fields[0].shape]
    else: return [1, 1, 1, 1]

def float_to_int32_bits(f_val):
    return struct.unpack('<I', struct.pack('<f', f_val))[0]