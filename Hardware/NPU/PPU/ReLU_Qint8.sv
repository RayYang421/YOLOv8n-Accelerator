module ReLU_Qint8 (
    input en,
    input [7:0] data_in,
    output logic [7:0] data_out
);
/* TODO: Start writing your implementation here */

    always_comb begin
        if(en) begin
            data_out = (data_in[7] == 1'b1)? 8'b0000_0000  : data_in;
        end else begin
            data_out = data_in;
        end
    end

/* TODO: End of implementation */
endmodule