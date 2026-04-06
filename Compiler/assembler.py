

def text_to_hex_full(input_file, output_file):
    OP_MAP = {"CONV": 0x1, "POOL": 0x2, "CONCAT": 0x3, "ADD": 0x4, "OTHER": 0x5, "CONFIG": 0x6, "BIAS": 0x7, "HALT": 0xF}
    with open(input_file, 'r') as f, open(output_file, 'w') as out:
        for line in f:
            if "|" not in line: continue
            parts = {p.split(':')[0].strip(): p.split(':')[1].strip() for p in line.split('|')}
            op_name = parts['OP'].strip()
            opcode = OP_MAP.get(op_name, 0x0)
            
            if op_name == "CONFIG":
                scale_val = int(parts.get('SCALE', '0x3F800000'), 16) 
                val = (opcode << 124) | (int(parts['IN_H']) << 108) | (int(parts['IN_W']) << 92) | \
                      (int(parts['IN_C']) << 76) | (int(parts['OUT_C']) << 60) | (int(parts['STRIDE']) << 56) | \
                      (scale_val)
            else:
                val = (opcode << 124) | \
                      ((int(parts['IN'], 16) & 0xFFFFFFFF) << 92) | \
                      ((int(parts['WGT'], 16) & 0xFFFFFFFF) << 60) | \
                      ((int(parts['OUT'], 16) & 0xFFFFFFFF) << 28) | \
                      (int(parts['FLAGS'], 16) << 16) | \
                      (int(parts['STRIDE']) << 12) | \
                      (int(parts['PAD']) << 8) | \
                      (int(parts['KERNEL']) << 4)
            out.write(f"{val:032x}\n")