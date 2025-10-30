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

module scope_switch_2x2 #(
    parameter DATA_WIDTH = 8,
    parameter COMMAND_WIDTH = 3
) (
    clk,
    rst_n,

    row_activation,  // Will be directly provided by control unit

    i_valid,
    i_data_bus,

    o_valid,
    o_data_bus,

    i_en,
    i_cmd

);
  localparam OUT_DATA_WIDTH = 2 * DATA_WIDTH;
  // interface
  input clk;
  input rst_n;
  input [DATA_WIDTH-1 : 0] row_activation;

  input [1:0] i_valid;
  input [2*DATA_WIDTH-1:0] i_data_bus;

  output [1:0] o_valid;
  output [2*DATA_WIDTH-1:0] o_data_bus;  //{o_data_a, o_data_b}

  input i_en;
  input [COMMAND_WIDTH-1:0] i_cmd;


  // 000 --> Route Left-Left and Right-Right
  // 001 --> Route Left-Right and Right-left
  // 010 --> Merge and Left
  // 011 --> Merge and right
  // 100 --> Right-Left + 0 (right)
  // 101 --> Right-Right + 0 (left)    
  // 110 --> Left-Right + 0 (left)
  // 111 --> Left-Left + 0 (right)   


  // output register
  reg [OUT_DATA_WIDTH-1:0] o_data_bus_inner;
  reg [1:0] o_valid_inner;

  // Intermediate Results
  wire [DATA_WIDTH-1:0] i_data_high = i_data_bus[DATA_WIDTH*2-1:DATA_WIDTH];
  wire [DATA_WIDTH-1:0] i_data_low = i_data_bus[DATA_WIDTH-1:0];

  wire [DATA_WIDTH-1:0] res_inner = row_activation + (i_data_high >> 1) + (i_data_low >> 2);
  wire res_valid = &i_valid;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      o_data_bus_inner <= {OUT_DATA_WIDTH{1'b0}};
    end else if (i_en) begin
      case ({
        i_cmd
      })
        3'b000: begin
          o_data_bus_inner <= {i_data_high, i_data_low};
        end
        3'b001: begin
          o_data_bus_inner <= {i_data_low, i_data_high};
        end
        3'b010: begin
          o_data_bus_inner <= {res_inner, {DATA_WIDTH{1'b0}}};
        end
        3'b011: begin
          o_data_bus_inner <= {{DATA_WIDTH{1'b0}}, res_inner};
        end
        3'b100: begin
          o_data_bus_inner <= {i_data_low, {DATA_WIDTH{1'b0}}};
        end
        3'b101: begin
          o_data_bus_inner <= {{DATA_WIDTH{1'b0}}, i_data_low};
        end
        3'b110: begin
          o_data_bus_inner <= {{DATA_WIDTH{1'b0}}, i_data_high};
        end
        3'b111: begin
          o_data_bus_inner <= {i_data_high, {DATA_WIDTH{1'b0}}};
        end
      endcase
    end else begin
      o_data_bus_inner <= {OUT_DATA_WIDTH{1'b0}};
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      o_valid_inner <= 2'b0;
    end else if (i_en) begin
      case ({
        i_cmd
      })
        3'b000: begin
          o_valid_inner <= {i_valid[0], i_valid[1]};
        end
        3'b001: begin
          o_valid_inner <= {i_valid[1], i_valid[0]};
        end
        3'b010: begin
          o_valid_inner <= {res_valid, 1'b0};
        end
        3'b011: begin
          o_valid_inner <= {1'b0, res_valid};
        end
        3'b100: begin
          o_valid_inner <= {i_valid[1], 1'b1};
        end
        3'b101: begin
          o_valid_inner <= {1'b1, i_valid[1]};
        end
        3'b110: begin
          o_valid_inner <= {1'b1, i_valid[0]};
        end
        3'b111: begin
          o_valid_inner <= {i_valid[0], 1'b1};
        end
      endcase
    end else begin
      o_valid_inner <= 2'b0;
    end
  end

  assign o_data_bus = o_data_bus_inner;
  assign o_valid = o_valid_inner;

endmodule
