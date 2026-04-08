#include <stdio.h>
#include "npu_runtime.h"

int main(void) {

    npu_init();

    printf("[System] Image loaded to Input Buffer.\n");

    npu_run_inference();

    printf("[System] End\n");

    return 0;
}