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

module sram_sp_2d_array #(
    parameter SRAM_BANK_DATA_WIDTH = 8,                          //
    parameter SRAM_BANK_ADDR_WIDTH = 10,                         //
    parameter SRAM_BANK_DEPTH      = 2 ** SRAM_BANK_ADDR_WIDTH,  //
    parameter NUM_BANK             = 4                           //
) (
    clk,
    rst_n,

    // Port A
    i_sram_a_wr_data,
    i_sram_a_wr_addr,
    i_sram_a_wr_en,

    // Port B
    i_sram_b_rd_addr,
    i_sram_b_rd_en,
    o_sram_b_rd_data
);

  localparam SRAM_DATA_WIDTH = NUM_BANK * SRAM_BANK_DATA_WIDTH;
  localparam SRAM_ADDR_WIDTH = NUM_BANK * SRAM_BANK_ADDR_WIDTH;

  /*
        ports
    */
  input clk;
  input rst_n;

  input [SRAM_DATA_WIDTH    -1:0] i_sram_a_wr_data;
  input [SRAM_ADDR_WIDTH    -1:0] i_sram_a_wr_addr;
  input [NUM_BANK           -1:0] i_sram_a_wr_en;

  input [SRAM_ADDR_WIDTH    -1:0] i_sram_b_rd_addr;
  input [NUM_BANK           -1:0] i_sram_b_rd_en;
  output [SRAM_DATA_WIDTH    -1:0] o_sram_b_rd_data;


  genvar BANK_ITER;

  wire                              w_i_rd_wr_en      [0 : NUM_BANK-1];
  wire [SRAM_BANK_ADDR_WIDTH  -1:0] w_i_addr          [0 : NUM_BANK-1];
  wire [SRAM_BANK_ADDR_WIDTH  -1:0] w_i_bank_a_addr   [0 : NUM_BANK-1];
  wire                              w_i_bank_a_wr_en  [0 : NUM_BANK-1];
  wire [SRAM_BANK_DATA_WIDTH  -1:0] w_i_bank_a_wr_data[0 : NUM_BANK-1];
  wire [SRAM_BANK_ADDR_WIDTH  -1:0] w_i_bank_b_rd_addr[0 : NUM_BANK-1];
  wire                              w_i_bank_b_rd_en  [0 : NUM_BANK-1];
  wire [SRAM_BANK_DATA_WIDTH  -1:0] w_o_bank_b_rd_data[0 : NUM_BANK-1];

  generate
    for (BANK_ITER = 0; BANK_ITER < NUM_BANK; BANK_ITER = BANK_ITER + 1) begin : SP_SRAM_BANKS

      assign w_i_bank_a_wr_data   [BANK_ITER]    =   i_sram_a_wr_data [(SRAM_BANK_DATA_WIDTH*BANK_ITER)   +:  SRAM_BANK_DATA_WIDTH];
      assign w_i_bank_a_addr      [BANK_ITER]    =   i_sram_a_wr_addr [(SRAM_BANK_ADDR_WIDTH*BANK_ITER)   +:  SRAM_BANK_ADDR_WIDTH];
      assign w_i_bank_a_wr_en[BANK_ITER] = i_sram_a_wr_en[BANK_ITER];

      assign w_i_bank_b_rd_addr   [BANK_ITER]    =   i_sram_b_rd_addr [(SRAM_BANK_ADDR_WIDTH*BANK_ITER)   +:  SRAM_BANK_ADDR_WIDTH];
      assign w_i_bank_b_rd_en[BANK_ITER] = i_sram_b_rd_en[BANK_ITER];

      assign o_sram_b_rd_data     [(SRAM_BANK_DATA_WIDTH*BANK_ITER)   +:  SRAM_BANK_DATA_WIDTH]   =   w_o_bank_b_rd_data [BANK_ITER];


      sram_bank_sp #(
          .SRAM_BANK_DATA_WIDTH(SRAM_BANK_DATA_WIDTH),
          .SRAM_BANK_ADDR_WIDTH(SRAM_BANK_ADDR_WIDTH),
          .SRAM_BANK_DEPTH     (SRAM_BANK_DEPTH)
      ) sram_bank_sp_inst (
          .clk       (clk),
          .rst_n     (rst_n),
          .i_rd_wr_en(w_i_rd_wr_en[BANK_ITER]),
          .i_addr    (w_i_addr[BANK_ITER]),
          .i_wr_data (w_i_bank_a_wr_data[BANK_ITER]),
          .o_rd_data (w_o_bank_b_rd_data[BANK_ITER])
      );

      assign w_i_rd_wr_en[BANK_ITER] = (w_i_bank_a_wr_en[BANK_ITER] == 1) ? 1 : 0;
      assign  w_i_addr    [BANK_ITER] =   (w_i_bank_a_wr_en[BANK_ITER] == 1)  ?   w_i_bank_a_addr    [BANK_ITER]
                                                                                    :   w_i_bank_b_rd_addr [BANK_ITER];
    end

  endgenerate




endmodule
