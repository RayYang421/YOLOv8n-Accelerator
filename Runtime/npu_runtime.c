#include "npu_runtime.h"
#include <stdio.h>

#if HOST_SIMULATION
    uint8_t  mock_sram[1024 * 1024];
    uint32_t mock_regs[4] = {0, 2, 0, 0}; 
#endif

void npu_init(void) {
    printf("[System] Initializing NPU and loading data\n");

    volatile uint8_t* weight_mem = (volatile uint8_t*) SRAM_WEIGHT_ADDR;
    for (uint32_t i = 0; i < npu_weights_len; i++) {
        weight_mem[i] = npu_weights[i];
    }
    printf("[System] Loaded %u bytes of INT8 weights to 0x%08lX\n", npu_weights_len, (unsigned long)SRAM_WEIGHT_ADDR);

    REG_NPU_IMEM_BASE = (uint32_t)(uintptr_t) npu_instructions;
    printf("[System] Instruction Memory mapped to 0x%08X\n", REG_NPU_IMEM_BASE);
}

void cpu_fallback_handler(uint32_t halt_pc) {
    printf("CPU taking over at NPU PC = %u\n", halt_pc);

    for (int i = 0; i < task_count; i++) {
        if (task_table[i].halt_pc == halt_pc) {
            if (task_table[i].op_type == OP_SOFTMAX) {
                printf("   -> Executing CPU Softmax at 0x%08X\n", (uint32_t)task_table[i].in_addr);
            } else if (task_table[i].op_type == OP_RESIZE) {
                printf("   -> Executing CPU Resize at 0x%08X\n", (uint32_t)task_table[i].in_addr);
            }
            printf("CPU task finished.\n");
            return;
        }
    }

    printf("Panic: Unexpected halt at PC %u\n", halt_pc);
    while(1);
}

void npu_run_inference(void) {
    printf("[NPU] Starting inference\n");
    REG_NPU_CTRL |= NPU_CMD_START;

    while (1) {
        uint32_t status = REG_NPU_STATUS;

        if (status & NPU_STATUS_DONE) {
            printf("[NPU] Execution Completed, YOLOv8n Finished.\n");
            REG_NPU_CTRL |= NPU_CMD_CLEAR_INT;
            break;
        }

        if (status & NPU_STATUS_REQ_CPU) {
            uint32_t current_pc = REG_NPU_PC;
            cpu_fallback_handler(current_pc);

            REG_NPU_CTRL |= NPU_CMD_CLEAR_INT;
            REG_NPU_CTRL |= NPU_CMD_START;
        }
    }
}