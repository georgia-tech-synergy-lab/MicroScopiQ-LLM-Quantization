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

module multi_row_mux #(
    parameter PE_OUTPUT_WIDTH = 4,
    parameter NUM_INPUT_DATA = 8,
    parameter DATA_WIDTH = NUM_INPUT_DATA * PE_OUTPUT_WIDTH
) (
    // timing signals
    clk,  // clock 
    rst_n,  // Negative edge reset

    // data signals
    i_valid,  // valid input data signal, can be sent by control
    i_data_bus,  // input data bus coming into distribute switch

    o_valid,  // output valid
    o_data_bus,  // output data
    o_selected_row,  // selected row output
    // control signals
    i_en  // distribute switch enable
);
  // timing signals
  input clk;
  input rst_n;

  // interface
  input [NUM_INPUT_DATA-1:0] i_valid;
  input [NUM_INPUT_DATA*DATA_WIDTH-1:0] i_data_bus;

  output [NUM_INPUT_DATA-1 : 0] o_valid;
  output [DATA_WIDTH-1:0] o_data_bus;  //{o_data_a, o_data_b}
  output [(NUM_INPUT_DATA) - 1 : 0] o_selected_row;  // selected row output

  input i_en;
  reg [    NUM_INPUT_DATA-1 : 0] o_valid_inner;
  reg [          DATA_WIDTH-1:0] o_data_bus_inner;  //{o_data_a, o_data_b}
  reg [(NUM_INPUT_DATA) - 1 : 0] o_selected_row_inner;  // selected row register

  // Muxing Data Function
  function [DATA_WIDTH+NUM_INPUT_DATA-1:0] sel_data;
    input [NUM_INPUT_DATA*DATA_WIDTH-1:0] data_bus;
    input [NUM_INPUT_DATA-1:0] en;
    reg [(NUM_INPUT_DATA) - 1 : 0] selected_row;
    reg [NUM_INPUT_DATA - 1 : 0] i;
    reg [DATA_WIDTH-1:0] data_out;
    begin
      data_out = {DATA_WIDTH{1'b0}};
      selected_row = 0;
      for (i = 0; i < NUM_INPUT_DATA; i = i + 1) begin
        if (en[i] == 1'b1) begin
          data_out = data_bus[i*DATA_WIDTH+:DATA_WIDTH];
          selected_row = i;
        end
      end
      sel_data = {data_out, selected_row};
    end
  endfunction

  // Muxing Logic
  always @(negedge rst_n or posedge clk) begin
    if (~rst_n) begin
      o_data_bus_inner <= {DATA_WIDTH{1'b0}};
      o_selected_row_inner <= 0;
    end else begin
      {o_data_bus_inner, o_selected_row_inner} <= sel_data(i_data_bus, i_valid);
    end
  end

  // Output Valid Logic
  always @(negedge rst_n or posedge clk) begin
    if (~rst_n) begin
      o_valid_inner <= 0;
    end else begin
      o_valid_inner <= i_valid;
    end
  end

  // output logic
  assign o_data_bus = o_data_bus_inner;
  assign o_valid = o_valid_inner;
  assign o_selected_row = o_selected_row_inner;

endmodule
