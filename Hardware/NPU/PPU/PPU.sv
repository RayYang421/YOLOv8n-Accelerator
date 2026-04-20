`include "src/PPU/PostQuant.sv"
`include "src/PPU/Maxpool_Qint8.sv"
`include "src/PPU/ReLU_Qint8.sv"
`include "define.svh"

module PPU (
    input clk,
    input rst,
    input [`DATA_BITS-1:0] data_in,
    input [5:0] scaling_factor,
    input maxpool_en,
    input maxpool_init,
    input relu_sel,
    input relu_en,
    output logic[7:0] data_out
);
/* TODO: Start writing your implementation here */

    logic [7:0] result_PTQ;
    logic [7:0] result_MaxPool;
    logic [7:0] result_ReLU;
    logic [7:0] out;

    PostQuant PTQ(
        .data_in(data_in),
        .scaling_factor(scaling_factor),
        .data_out(result_PTQ)
    );

    ReLU_Qint8 ReLU(
        .en(relu_en),
        .data_in(result_PTQ),
        .data_out(result_ReLU)
    );

    Maxpool_Qint8 MaxPool(
            .clk(clk),
            .rst(rst),
            .en(maxpool_en),
            .init(maxpool_init),
            .data_in(result_ReLU),
            .data_out(result_MaxPool)
    );

    always_comb begin
        if (relu_sel == 1) begin
            out = result_MaxPool;
        end else begin
            out = result_ReLU;
        end
    end

    assign data_out = out + 8'd128;

/* TODO: End of implementation */
endmodule