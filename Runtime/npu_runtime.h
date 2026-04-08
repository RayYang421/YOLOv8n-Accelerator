#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdint.h>

// 0: on chip
// 1: simulator
#define HOST_SIMULATION 1

#if HOST_SIMULATION

    extern uint8_t  mock_sram[];
    extern uint32_t mock_regs[];

    #define SRAM_WEIGHT_ADDR  ((uintptr_t)&mock_sram[0])
    #define REG_NPU_CTRL      (mock_regs[0])
    #define REG_NPU_STATUS    (mock_regs[1])
    #define REG_NPU_PC        (mock_regs[2])
    #define REG_NPU_IMEM_BASE (mock_regs[3])

    typedef struct {
        uint32_t halt_pc;
        uint32_t op_type;
        uintptr_t in_addr;  
        uintptr_t out_addr;
    } TaskDescriptor;


    extern const TaskDescriptor task_table[];
    extern const int task_count;


    #define OP_SOFTMAX 1
    #define OP_RESIZE  2

#else

    // Bare-metal RISC-V

    #define SRAM_BASE         0x00000000
    #define SRAM_WEIGHT_ADDR  (SRAM_BASE + 0x40000000)
    #define NPU_CTRL_BASE     0x80000000
    #define REG_NPU_CTRL      (*((volatile uint32_t*)(NPU_CTRL_BASE + 0x00)))
    #define REG_NPU_STATUS    (*((volatile uint32_t*)(NPU_CTRL_BASE + 0x04)))
    #define REG_NPU_PC        (*((volatile uint32_t*)(NPU_CTRL_BASE + 0x08)))
    #define REG_NPU_IMEM_BASE (*((volatile uint32_t*)(NPU_CTRL_BASE + 0x0C)))
#endif


#define NPU_CMD_START       (1 << 0)
#define NPU_CMD_CLEAR_INT   (1 << 1)
#define NPU_STATUS_BUSY     (1 << 0)
#define NPU_STATUS_DONE     (1 << 1)
#define NPU_STATUS_REQ_CPU  (1 << 2)


extern const uint32_t npu_instructions[]; 
extern const uint32_t npu_instructions_len;
extern const uint8_t  npu_weights[];
extern const uint32_t npu_weights_len;

void npu_init(void);
void cpu_fallback_handler(uint32_t halt_pc);
void npu_run_inference(void);

#endif