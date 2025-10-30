/*
======================= START OF LICENSE NOTICE =======================
  Copyright (C) 2024 Akshat Ramachandran (GT), Souvik Kundu (Intel), Tushar Krishna (GT). All Rights Reserved

  NO WARRANTY. THE PRODUCT IS PROVIDED BY DEVELOPER "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL DEVELOPER BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE PRODUCT, EVEN
  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
======================== END OF LICENSE NOTICE ========================
  Primary Author: Akshat Ramachandran (GT)

*/
`timescale 1ns / 1ps

module scope #(
    parameter DATA_WIDTH = 4,
    parameter COMMAND_WIDTH = 3,  // Total 8 commands supported in each switch
    parameter NUM_INPUT_OUTPUT_PORTS = 8
) (
    // timing signals
    clk,
    rst_n,

    row_activation_flat,  // Flattened row_activation array
    // data signals
    i_valid,  // valid input data signal
    i_data_bus,  // input data bus coming into distribute switch

    o_valid,  // output valid
    o_data_bus,  // output data This will be sent to the control unit to be sent to the correct row

    // control signals
    i_en,  // distribute switch enable
    i_cmd_flat  // Flattened i_cmd array
);

  localparam [31:0] NUM_SWITCH_IN = NUM_INPUT_OUTPUT_PORTS;
  localparam [31:0] LEVEL = $clog2(NUM_INPUT_OUTPUT_PORTS);
  localparam [31:0] TOTAL_STAGE = LEVEL;
  localparam [31:0] WIDTH_INPUT_DATA = NUM_INPUT_OUTPUT_PORTS * DATA_WIDTH;
  localparam [31:0] NUM_ACTIVATION_ROWS = LEVEL + 1;

  // Flattened input and control signals
  input clk;
  input rst_n;
  input [NUM_INPUT_OUTPUT_PORTS-1:0] i_valid;
  input [WIDTH_INPUT_DATA-1:0] i_data_bus;
  output [NUM_INPUT_OUTPUT_PORTS-1:0] o_valid;
  output [WIDTH_INPUT_DATA-1:0] o_data_bus;

  input i_en;
  input [COMMAND_WIDTH * (LEVEL+1) * NUM_SWITCH_IN - 1:0] i_cmd_flat;
  input [DATA_WIDTH * (LEVEL+1) * NUM_SWITCH_IN - 1:0] row_activation_flat;

  wire [    DATA_WIDTH-1:0] connection      [                       0:LEVEL] [0:2*NUM_SWITCH_IN-1];
  wire                      connection_valid[                       0:LEVEL] [0:2*NUM_SWITCH_IN-1];
  wire                      dummy_o_valid   [    0:NUM_INPUT_OUTPUT_PORTS-1];
  wire [DATA_WIDTH - 1 : 0] dummy_o_data    [0 : NUM_INPUT_OUTPUT_PORTS - 1];

  // Unflatten the arrays within the module
  reg  [ COMMAND_WIDTH-1:0] i_cmd           [                     0 : LEVEL] [0 : NUM_SWITCH_IN-1];
  reg  [    DATA_WIDTH-1:0] row_activation  [                     0 : LEVEL] [0 : NUM_SWITCH_IN-1];

  integer u, v;
  always @* begin
    for (u = 0; u <= LEVEL; u = u + 1) begin
      for (v = 0; v < NUM_SWITCH_IN; v = v + 1) begin
        i_cmd[u][v] = i_cmd_flat[(u*NUM_SWITCH_IN+v)*COMMAND_WIDTH+:COMMAND_WIDTH];
        row_activation[u][v] = row_activation_flat[(u*NUM_SWITCH_IN+v)*DATA_WIDTH+:DATA_WIDTH];
      end
    end
  end

  genvar i, j, k, s, p;
  generate
    // Input
    for (i = 0; i < NUM_SWITCH_IN; i = i + 1) begin : input_level

      scope_switch_2x2 #(
          .DATA_WIDTH(DATA_WIDTH),
          .COMMAND_WIDTH(COMMAND_WIDTH)
      ) first_stage (
          .clk(clk),
          .rst_n(rst_n),
          .row_activation(row_activation[0][i]),
          .i_valid({i_valid[i], 1'b0}),
          .i_data_bus({i_data_bus[i*DATA_WIDTH+:DATA_WIDTH], {DATA_WIDTH{1'b0}}}),
          .o_valid({connection_valid[0][2*i], connection_valid[0][2*i+1]}),
          .o_data_bus({connection[0][2*i], connection[0][2*i+1]}),
          .i_en(i_en),
          .i_cmd(i_cmd[0][i])
      );
    end

    // middle -> output level
    for (s = 0; s < (LEVEL - 1); s = s + 1) begin : middle_levels
      localparam [31:0] switch_group = 1 << s;
      localparam [31:0] num_jumps = (NUM_SWITCH_IN) / (2 * switch_group);
      localparam [31:0] mult_factor = switch_group * 2;

      for (k = 0; k < num_jumps; k = k + 1) begin
        for (j = 0; j < switch_group; j = j + 1) begin
          scope_switch_2x2 #(
              .DATA_WIDTH(DATA_WIDTH),
              .COMMAND_WIDTH(COMMAND_WIDTH)
          ) lower_switch (
              .clk(clk),
              .rst_n(rst_n),
              .row_activation(row_activation[s+1][mult_factor*k+j]),
              .i_valid({
                connection_valid[s][2*(mult_factor*k+j)],
                connection_valid[s][2*(mult_factor*k+j)+(mult_factor)]
              }),
              .i_data_bus({
                connection[s][2*(mult_factor*k+j)], connection[s][2*(mult_factor*k+j)+(mult_factor)]
              }),
              .o_valid({
                connection_valid[s+1][2*(mult_factor*k+j)],
                connection_valid[s+1][2*(mult_factor*k+j)+1]
              }),
              .o_data_bus({
                connection[s+1][2*(mult_factor*k+j)], connection[s+1][2*(mult_factor*k+j)+1]
              }),
              .i_en(i_en),
              .i_cmd(i_cmd[s+1][mult_factor*k+j])
          );
        end
      end

      // Second half
      for (k = 0; k < num_jumps; k = k + 1) begin
        for (j = 0; j < switch_group; j = j + 1) begin
          scope_switch_2x2 #(
              .DATA_WIDTH(DATA_WIDTH),
              .COMMAND_WIDTH(COMMAND_WIDTH)
          ) upper_switch (
              .clk(clk),
              .rst_n(rst_n),
              .row_activation(row_activation[s+1][(mult_factor*k+j)+(switch_group)]),
              .i_valid({
                connection_valid[s][2*(mult_factor*k+j)+1],
                connection_valid[s][2*(mult_factor*k+j)+(mult_factor+1)]
              }),
              .i_data_bus({
                connection[s][2*(mult_factor*k+j)+1],
                connection[s][2*(mult_factor*k+j)+(mult_factor+1)]
              }),
              .o_valid({
                connection_valid[s+1][2*(mult_factor*k+j)+(mult_factor)],
                connection_valid[s+1][2*(mult_factor*k+j)+(mult_factor+1)]
              }),
              .o_data_bus({
                connection[s+1][2*(mult_factor*k+j)+(mult_factor)],
                connection[s+1][2*(mult_factor*k+j)+(mult_factor+1)]
              }),
              .i_en(i_en),
              .i_cmd(i_cmd[s+1][(mult_factor*k+j)+(switch_group)])
          );
        end
      end
    end

    for (s = LEVEL - 1; s < (LEVEL); s = s + 1) begin : output_level
      localparam [31:0] switch_group = 1 << s;
      localparam [31:0] num_jumps = (NUM_SWITCH_IN) / (2 * switch_group);
      localparam [31:0] mult_factor = switch_group * 2;

      for (k = 0; k < num_jumps; k = k + 1) begin
        for (j = 0; j < switch_group; j = j + 1) begin
          scope_switch_2x2 #(
              .DATA_WIDTH(DATA_WIDTH),
              .COMMAND_WIDTH(COMMAND_WIDTH)
          ) lower_switch_output (
              .clk(clk),
              .rst_n(rst_n),
              .row_activation(row_activation[s+1][mult_factor*k+j]),
              .i_valid({
                connection_valid[s][2*(mult_factor*k+j)],
                connection_valid[s][2*(mult_factor*k+j)+(mult_factor)]
              }),
              .i_data_bus({
                connection[s][2*(mult_factor*k+j)], connection[s][2*(mult_factor*k+j)+(mult_factor)]
              }),
              .o_valid({o_valid[(mult_factor*k+j)+:1], dummy_o_valid[mult_factor*k+j]}),
              .o_data_bus({
                o_data_bus[(mult_factor*k+j)+:DATA_WIDTH], dummy_o_data[mult_factor*k+j]
              }),
              .i_en(i_en),
              .i_cmd(i_cmd[s+1][mult_factor*k+j])
          );
        end
      end

      // Second half
      for (k = 0; k < num_jumps; k = k + 1) begin
        for (j = 0; j < switch_group; j = j + 1) begin
          scope_switch_2x2 #(
              .DATA_WIDTH(DATA_WIDTH),
              .COMMAND_WIDTH(COMMAND_WIDTH)
          ) upper_switch_output (
              .clk(clk),
              .rst_n(rst_n),
              .row_activation(row_activation[s+1][(mult_factor*k+j)+(switch_group)]),
              .i_valid({
                connection_valid[s][2*(mult_factor*k+j)+1],
                connection_valid[s][2*(mult_factor*k+j)+(mult_factor+1)]
              }),
              .i_data_bus({
                connection[s][2*(mult_factor*k+j)+1],
                connection[s][2*(mult_factor*k+j)+(mult_factor+1)]
              }),
              .o_valid({
                o_valid[(mult_factor*k+j)+(switch_group)+:1],
                dummy_o_valid[(mult_factor*k+j)+(switch_group)]
              }),
              .o_data_bus({
                o_data_bus[(mult_factor*k+j)+(switch_group)+:DATA_WIDTH],
                dummy_o_data[(mult_factor*k+j)+(switch_group)+j]
              }),
              .i_en(i_en),
              .i_cmd(i_cmd[s+1][(mult_factor*k+j)+(switch_group)])
          );
        end
      end
    end

  endgenerate

endmodule
