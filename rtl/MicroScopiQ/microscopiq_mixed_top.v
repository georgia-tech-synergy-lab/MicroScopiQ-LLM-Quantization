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

module microscopiq_mixed_top #(
    parameter ARRAY_COL_NUM = 8,
    parameter ARRAY_ROW_NUM = 8,
    parameter LOG2_ARRAY_COL_NUM = $clog2(ARRAY_COL_NUM),
    parameter LOG2_ARRAY_ROW_NUM = $clog2(ARRAY_ROW_NUM),
    parameter NUM_SCOPES = 1,  // 0 means no NoC, 1 means global NoC, 2 means a NoC for each row

    parameter WEIGHTS_DATA_WIDTH           = 2 * ARRAY_COL_NUM,                                //
    parameter WEIGHTS_NUM_BANKS            = 1,                                                //
    parameter WEIGHTS_SRAM_DATA_WIDTH      = WEIGHTS_NUM_BANKS * WEIGHTS_DATA_WIDTH,
    parameter WEIGHTS_SRAM_BANK_ADDR_WIDTH = 2,                                                //
    parameter WEIGHTS_SRAM_ADDR_WIDTH      = WEIGHTS_NUM_BANKS * WEIGHTS_SRAM_BANK_ADDR_WIDTH, //

    parameter IACTS_DATA_WIDTH           = 4,                                            //
    parameter IACTS_NUM_BANKS            = ARRAY_COL_NUM,                                //
    parameter IACTS_SRAM_DATA_WIDTH      = IACTS_NUM_BANKS * IACTS_DATA_WIDTH,
    parameter IACTS_SRAM_BANK_ADDR_WIDTH = 2,                                            //
    parameter IACTS_SRAM_ADDR_WIDTH      = IACTS_NUM_BANKS * IACTS_SRAM_BANK_ADDR_WIDTH, //

    parameter PE_OUTPUT_WIDTH = 4,  //
    parameter COMMAND_WIDTH   = 3,

    parameter INSTR_SRAM_BANK_ADDR_WIDTH = 2  //
) (
    clk,
    rst_n,
    mode,

    i_top_en,

    //Weights from DRAM
    i_weights_write_valid,
    i_weights_write_data,
    i_weights_write_addr,
    i_weights_write_addr_end,

    //iActs from DRAM
    i_iacts_write_valid,
    i_iacts_write_data,
    i_iacts_write_addr,
    i_iacts_write_addr_end,

    //instruction data // This will be the permutation instructions for the NoC
    i_instr_write_valid,
    i_instr_write_data,
    i_instr_write_addr,

    i_all_buf_pingpong_config,
    row_activation_fifo_out,
    i_iacc_noc_pe,

    //Quantization value                    
    i_scale_val                             , // Will be a 32-bit value with all scale factors necessary for computation including micro-exponents
    o_scale_val,
    i_scale_compute_en,
    o_scale_val_en
);

  localparam INSTR_SRAM_BANK_DATA_WIDTH = COMMAND_WIDTH * ($clog2(
      ARRAY_COL_NUM
  ) + 1) * ARRAY_COL_NUM;  //  instruction is only SCOPE instruction here
  localparam WEIGHTS_DEPTH = ARRAY_ROW_NUM;  //
  localparam IACTS_PINGPONG_CONFIG_WIDTH = 4;
  localparam WEIGHTS_PINGPONG_CONFIG_WIDTH = 4;
  localparam ALL_BUF_CONFIG_WIDTH = IACTS_PINGPONG_CONFIG_WIDTH + WEIGHTS_PINGPONG_CONFIG_WIDTH;
  localparam SCALE_VALUE_WIDTH = 16;  // This value is fixed

  /*
        ports
    */
  input clk;
  input rst_n;
  input mode;
  input i_top_en;

  input i_weights_write_valid;
  input [WEIGHTS_SRAM_DATA_WIDTH        -1:0] i_weights_write_data;
  input [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1:0] i_weights_write_addr;
  input [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1:0] i_weights_write_addr_end;

  input i_iacts_write_valid;
  input [IACTS_SRAM_DATA_WIDTH          -1:0] i_iacts_write_data;
  input [IACTS_SRAM_BANK_ADDR_WIDTH     -1:0] i_iacts_write_addr;
  input [IACTS_SRAM_BANK_ADDR_WIDTH     -1:0] i_iacts_write_addr_end;

  input i_instr_write_valid;
  input [INSTR_SRAM_BANK_DATA_WIDTH     -1:0] i_instr_write_data;
  input [INSTR_SRAM_BANK_ADDR_WIDTH     -1:0] i_instr_write_addr;

  input [ALL_BUF_CONFIG_WIDTH           -1:0] i_all_buf_pingpong_config;
  input [PE_OUTPUT_WIDTH * ($clog2(
ARRAY_COL_NUM
)+1) * ARRAY_COL_NUM - 1:0] row_activation_fifo_out;
  input i_iacc_noc_pe;
  input [SCALE_VALUE_WIDTH              -1:0] i_scale_val;
  output [SCALE_VALUE_WIDTH              -1:0] o_scale_val;
  input i_scale_compute_en;
  output o_scale_val_en;

  /*
        MicroscopiQ Contoller signals
    */
  wire [IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_from_ctrl_to_dpe;
  wire [IACTS_NUM_BANKS                    -1:0] w_iacts_valid_from_ctrl_to_dpe;
  wire [ARRAY_COL_NUM                        -1:0] w_weights_valid_from_ctrl_to_dpe;
  wire [WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_from_ctrl_to_dpe;
  wire w_weights_ping_pong_sel;
  wire [INSTR_SRAM_BANK_DATA_WIDTH             -1:0] w_scope_instr;

  wire [(ARRAY_COL_NUM) - 1 : 0] o_selected_row;
  /*
        MicroScopiQ Controller + Buffers' signals
    */
  wire [IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_sram_a_wr_data_ping;
  wire [IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_a_wr_addr_ping;
  wire [IACTS_NUM_BANKS                    -1:0] w_iacts_sram_a_wr_en_ping;

  wire [IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_b_rd_addr_ping;
  wire [IACTS_NUM_BANKS                    -1:0] w_iacts_sram_b_rd_en_ping;
  wire [IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_sram_b_rd_data_ping;

  wire [IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_sram_a_wr_data_pong;
  wire [IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_a_wr_addr_pong;
  wire [IACTS_NUM_BANKS                    -1:0] w_iacts_sram_a_wr_en_pong;
  wire [IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_b_rd_addr_pong;
  wire [IACTS_NUM_BANKS                    -1:0] w_iacts_sram_b_rd_en_pong;
  wire [IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_sram_b_rd_data_pong;

  wire [WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_sram_a_wr_data_ping;
  wire [WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_a_wr_addr_ping;
  wire [WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_a_wr_en_ping;
  wire [WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_b_rd_addr_ping;
  wire [WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_b_rd_en_ping;
  wire [WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_sram_b_rd_data_ping;
  wire [WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_sram_a_wr_data_pong;
  wire [WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_a_wr_addr_pong;
  wire [WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_a_wr_en_pong;
  wire [WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_b_rd_addr_pong;
  wire [WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_b_rd_en_pong;
  wire [WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_sram_b_rd_data_pong;

  wire [INSTR_SRAM_BANK_DATA_WIDTH         -1:0] w_instr_sram_a_wr_data;
  wire [INSTR_SRAM_BANK_ADDR_WIDTH         -1:0] w_instr_sram_a_wr_addr;
  wire w_instr_sram_a_wr_en;
  wire [INSTR_SRAM_BANK_ADDR_WIDTH         -1:0] w_instr_sram_b_rd_addr;
  wire w_instr_sram_b_rd_en;
  wire [INSTR_SRAM_BANK_DATA_WIDTH         -1:0] w_instr_sram_b_rd_data;

  wire [ARRAY_COL_NUM * PE_OUTPUT_WIDTH - 1 : 0] w_o_scope_data_bus;
  wire [ARRAY_COL_NUM - 1 : 0] w_o_scope_data_bus_valid;

  wire [ARRAY_COL_NUM * PE_OUTPUT_WIDTH - 1 : 0] scope_to_pe_data_bus;
  wire [ARRAY_ROW_NUM-1 : 0] scope_to_pe_valid;

  /*
        Scope signals
    */
  wire w_dpe_weights_ping_pong_sel[0 : ARRAY_ROW_NUM][0 : ARRAY_COL_NUM-1];
  wire [IACTS_DATA_WIDTH                   -1:0]
      w_dpe_iacts[0 : ARRAY_ROW_NUM][0 : ARRAY_COL_NUM-1];
  wire w_dpe_iacts_valid[0 : ARRAY_ROW_NUM][0 : ARRAY_COL_NUM-1];
  wire [WEIGHTS_DATA_WIDTH     -1:0] w_dpe_weights[0 : ARRAY_ROW_NUM][0 : ARRAY_COL_NUM-1];
  wire w_dpe_weights_valid[0 : ARRAY_ROW_NUM][0 : ARRAY_COL_NUM-1];

  wire [(PE_OUTPUT_WIDTH*ARRAY_COL_NUM)*ARRAY_ROW_NUM      -1:0] w_pe_row_out_to_mux_data_bus;
  wire [ARRAY_ROW_NUM      -1:0] w_pe_row_out_to_mux_data_bus_valid;

  wire [PE_OUTPUT_WIDTH*ARRAY_COL_NUM                    -1:0] w_o_mux_data;
  wire [ARRAY_COL_NUM                    -1:0] w_o_mux_data_valid;



  /*
Scale Factor Computation Block
*/

  scale_compute #(
      .SCALE_VALUE_WIDTH(SCALE_VALUE_WIDTH)
  ) scale_compute_INST (
      .clk(clk),
      .rst_n(rst_n),
      .i_scale_compute_en(i_scale_compute_en),
      .i_scale_val(i_scale_val),
      .o_scale_val(o_scale_val),
      .o_scale_val_en(o_scale_val_en)
  );

  /*
____________________________________________________________________________________________________________________________
Weights PING BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
  sram_sp_2d_array #(
      .SRAM_BANK_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
      .SRAM_BANK_ADDR_WIDTH(WEIGHTS_SRAM_BANK_ADDR_WIDTH),
      .SRAM_BANK_DEPTH     (2 ** WEIGHTS_SRAM_BANK_ADDR_WIDTH),
      .NUM_BANK            (WEIGHTS_NUM_BANKS)
  ) WEIGHTS_PING_BUFFER (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_sram_a_wr_data(w_weights_sram_a_wr_data_ping),
      .i_sram_a_wr_addr(w_weights_sram_a_wr_addr_ping),
      .i_sram_a_wr_en  (w_weights_sram_a_wr_en_ping),
      .i_sram_b_rd_addr(w_weights_sram_b_rd_addr_ping),
      .i_sram_b_rd_en  (w_weights_sram_b_rd_en_ping),
      .o_sram_b_rd_data(w_weights_sram_b_rd_data_ping)
  );


  /*
____________________________________________________________________________________________________________________________
Weights PONG BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
  sram_sp_2d_array #(
      .SRAM_BANK_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
      .SRAM_BANK_ADDR_WIDTH(WEIGHTS_SRAM_BANK_ADDR_WIDTH),
      .SRAM_BANK_DEPTH     (2 ** WEIGHTS_SRAM_BANK_ADDR_WIDTH),
      .NUM_BANK            (WEIGHTS_NUM_BANKS)
  ) WEIGHTS_PONG_BUFFER (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_sram_a_wr_data(w_weights_sram_a_wr_data_pong),
      .i_sram_a_wr_addr(w_weights_sram_a_wr_addr_pong),
      .i_sram_a_wr_en  (w_weights_sram_a_wr_en_pong),
      .i_sram_b_rd_addr(w_weights_sram_b_rd_addr_pong),
      .i_sram_b_rd_en  (w_weights_sram_b_rd_en_pong),
      .o_sram_b_rd_data(w_weights_sram_b_rd_data_pong)
  );


  /*
____________________________________________________________________________________________________________________________
IACTS PING BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
  sram_sp_2d_array #(
      .SRAM_BANK_DATA_WIDTH(IACTS_DATA_WIDTH),
      .SRAM_BANK_ADDR_WIDTH(IACTS_SRAM_BANK_ADDR_WIDTH),
      .SRAM_BANK_DEPTH     (2 ** IACTS_SRAM_BANK_ADDR_WIDTH),
      .NUM_BANK            (IACTS_NUM_BANKS)
  ) IACTS_PING_SRAM (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_sram_a_wr_data(w_iacts_sram_a_wr_data_ping),
      .i_sram_a_wr_addr(w_iacts_sram_a_wr_addr_ping),
      .i_sram_a_wr_en  (w_iacts_sram_a_wr_en_ping),
      .i_sram_b_rd_addr(w_iacts_sram_b_rd_addr_ping),
      .i_sram_b_rd_en  (w_iacts_sram_b_rd_en_ping),
      .o_sram_b_rd_data(w_iacts_sram_b_rd_data_ping)
  );
  /*
____________________________________________________________________________________________________________________________
IACTS PONG BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
  sram_sp_2d_array #(
      .SRAM_BANK_DATA_WIDTH(IACTS_DATA_WIDTH),
      .SRAM_BANK_ADDR_WIDTH(IACTS_SRAM_BANK_ADDR_WIDTH),
      .SRAM_BANK_DEPTH     (2 ** IACTS_SRAM_BANK_ADDR_WIDTH),
      .NUM_BANK            (IACTS_NUM_BANKS)
  ) IACTS_PONG_SRAM (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_sram_a_wr_data(w_iacts_sram_a_wr_data_pong),
      .i_sram_a_wr_addr(w_iacts_sram_a_wr_addr_pong),
      .i_sram_a_wr_en  (w_iacts_sram_a_wr_en_pong),
      .i_sram_b_rd_addr(w_iacts_sram_b_rd_addr_pong),
      .i_sram_b_rd_en  (w_iacts_sram_b_rd_en_pong),
      .o_sram_b_rd_data(w_iacts_sram_b_rd_data_pong)
  );


  /*
____________________________________________________________________________________________________________________________
INSTRUCTION BUFFER

PORT A - Write Port
PORT B - Read  Port
____________________________________________________________________________________________________________________________
*/
  sram_sp_2d_array #(
      .SRAM_BANK_DATA_WIDTH(INSTR_SRAM_BANK_DATA_WIDTH),
      .SRAM_BANK_ADDR_WIDTH(INSTR_SRAM_BANK_ADDR_WIDTH),
      .SRAM_BANK_DEPTH     (2 ** INSTR_SRAM_BANK_ADDR_WIDTH),
      .NUM_BANK            (1)
  ) INSTR_SRAM (
      .clk             (clk),
      .rst_n           (rst_n),
      .i_sram_a_wr_data(w_instr_sram_a_wr_data),
      .i_sram_a_wr_addr(w_instr_sram_a_wr_addr),
      .i_sram_a_wr_en  (w_instr_sram_a_wr_en),
      .i_sram_b_rd_addr(w_instr_sram_b_rd_addr),
      .i_sram_b_rd_en  (w_instr_sram_b_rd_en),
      .o_sram_b_rd_data(w_instr_sram_b_rd_data)
  );
  /*
____________________________________________________________________________________________________________________________
microscopiq Controller
____________________________________________________________________________________________________________________________
*/  //
  microscopiq_controller #(
      .WEIGHTS_DATA_WIDTH          (WEIGHTS_DATA_WIDTH),
      .WEIGHTS_NUM_BANKS           (WEIGHTS_NUM_BANKS),
      .WEIGHTS_SRAM_DATA_WIDTH     (WEIGHTS_SRAM_DATA_WIDTH),
      .WEIGHTS_SRAM_BANK_ADDR_WIDTH(WEIGHTS_SRAM_BANK_ADDR_WIDTH),
      .WEIGHTS_SRAM_ADDR_WIDTH     (WEIGHTS_SRAM_ADDR_WIDTH),
      .IACTS_DATA_WIDTH            (IACTS_DATA_WIDTH),
      .IACTS_NUM_BANKS             (IACTS_NUM_BANKS),
      .IACTS_SRAM_DATA_WIDTH       (IACTS_SRAM_DATA_WIDTH),
      .IACTS_SRAM_BANK_ADDR_WIDTH  (IACTS_SRAM_BANK_ADDR_WIDTH),
      .IACTS_SRAM_ADDR_WIDTH       (IACTS_SRAM_ADDR_WIDTH),
      .SCALE_VALUE_WIDTH           (SCALE_VALUE_WIDTH),
      .PE_OUTPUT_WIDTH             (PE_OUTPUT_WIDTH),
      .ARRAY_COL_NUM               (ARRAY_COL_NUM),
      .ARRAY_ROW_NUM               (ARRAY_ROW_NUM),
      .LOG2_ARRAY_COL_NUM          (LOG2_ARRAY_COL_NUM),
      .LOG2_ARRAY_ROW_NUM          (LOG2_ARRAY_ROW_NUM),
      .INSTR_SRAM_BANK_ADDR_WIDTH  (INSTR_SRAM_BANK_ADDR_WIDTH),
      .INSTR_SRAM_BANK_DATA_WIDTH  (INSTR_SRAM_BANK_DATA_WIDTH)
  ) microscopiq_CONTROLLER_INST (
      .clk                     (clk),                      // inputs from top
      .rst_n                   (rst_n),
      .i_top_en                (i_top_en),
      .i_weights_write_valid   (i_weights_write_valid),
      .i_weights_write_data    (i_weights_write_data),
      .i_weights_write_addr    (i_weights_write_addr),
      .i_weights_write_addr_end(i_weights_write_addr_end),

      .i_iacts_write_valid   (i_iacts_write_valid),
      .i_iacts_write_data    (i_iacts_write_data),
      .i_iacts_write_addr    (i_iacts_write_addr),
      .i_iacts_write_addr_end(i_iacts_write_addr_end),

      .i_instr_write_valid(i_instr_write_valid),
      .i_instr_write_data (i_instr_write_data),
      .i_instr_write_addr (i_instr_write_addr),

      .i_all_buf_pingpong_config(i_all_buf_pingpong_config),

      .i_data_bus_from_scope      (w_o_scope_data_bus),        // inputs from scope
      .i_data_bus_from_scope_valid(w_o_scope_data_bus_valid),
      .i_selected_row_mux         (o_selected_row),
      .o_weights_ping_pong_sel    (w_weights_ping_pong_sel),
      .o_scope_instr              (w_scope_instr),

      .o_iacts_from_ctrl_to_dpe        (w_iacts_from_ctrl_to_dpe),         // Outputs to birrd
      .o_iacts_valid_from_ctrl_to_dpe  (w_iacts_valid_from_ctrl_to_dpe),
      .o_weights_from_ctrl_to_dpe      (w_weights_from_ctrl_to_dpe),
      .o_weights_valid_from_ctrl_to_dpe(w_weights_valid_from_ctrl_to_dpe),

      .o_data_bus_from_scope_to_pe_rows(scope_to_pe_data_bus),
      .o_data_bus_from_scope_valid_to_pe_row(scope_to_pe_valid),
      .o_iacts_sram_a_wr_data_ping(w_iacts_sram_a_wr_data_ping),  // I-O from-to iActs Ping
      .o_iacts_sram_a_wr_addr_ping(w_iacts_sram_a_wr_addr_ping),
      .o_iacts_sram_a_wr_en_ping(w_iacts_sram_a_wr_en_ping),
      .o_iacts_sram_b_rd_addr_ping(w_iacts_sram_b_rd_addr_ping),
      .o_iacts_sram_b_rd_en_ping(w_iacts_sram_b_rd_en_ping),
      .i_iacts_sram_b_rd_data_ping(w_iacts_sram_b_rd_data_ping),
      .o_iacts_sram_a_wr_data_pong(w_iacts_sram_a_wr_data_pong),  // I-O from-to iActs Pong
      .o_iacts_sram_a_wr_addr_pong(w_iacts_sram_a_wr_addr_pong),
      .o_iacts_sram_a_wr_en_pong(w_iacts_sram_a_wr_en_pong),
      .o_iacts_sram_b_rd_addr_pong(w_iacts_sram_b_rd_addr_pong),
      .o_iacts_sram_b_rd_en_pong(w_iacts_sram_b_rd_en_pong),
      .i_iacts_sram_b_rd_data_pong(w_iacts_sram_b_rd_data_pong),
      .o_weights_sram_a_wr_data_ping(w_weights_sram_a_wr_data_ping),  // I-O from-to Weights Ping

      .o_weights_sram_a_wr_addr_ping(w_weights_sram_a_wr_addr_ping),
      .o_weights_sram_a_wr_en_ping  (w_weights_sram_a_wr_en_ping),

      .o_weights_sram_b_rd_addr_ping(w_weights_sram_b_rd_addr_ping),
      .o_weights_sram_b_rd_en_ping  (w_weights_sram_b_rd_en_ping),

      .i_weights_sram_b_rd_data_ping(w_weights_sram_b_rd_data_ping),
      .o_weights_sram_a_wr_data_pong(w_weights_sram_a_wr_data_pong),  // I-O from-to Weights Pong

      .o_weights_sram_a_wr_addr_pong(w_weights_sram_a_wr_addr_pong),
      .o_weights_sram_a_wr_en_pong  (w_weights_sram_a_wr_en_pong),

      .o_weights_sram_b_rd_addr_pong(w_weights_sram_b_rd_addr_pong),
      .o_weights_sram_b_rd_en_pong  (w_weights_sram_b_rd_en_pong),
      .i_weights_sram_b_rd_data_pong(w_weights_sram_b_rd_data_pong),

      .o_instr_sram_a_wr_data(w_instr_sram_a_wr_data),  // I-O from-to INSTR Buffer
      .o_instr_sram_a_wr_addr(w_instr_sram_a_wr_addr),
      .o_instr_sram_a_wr_en  (w_instr_sram_a_wr_en),
      .o_instr_sram_b_rd_addr(w_instr_sram_b_rd_addr),
      .o_instr_sram_b_rd_en  (w_instr_sram_b_rd_en),
      .i_instr_sram_b_rd_data(w_instr_sram_b_rd_data)
  );


  //_________________________________________________________________________________________________________________________
  // connecting data and control to scope input
  genvar GENVAR_VEC_TO_ARR_COL_ITER;

  generate
    for (
        GENVAR_VEC_TO_ARR_COL_ITER = 0;
        GENVAR_VEC_TO_ARR_COL_ITER < ARRAY_COL_NUM;
        GENVAR_VEC_TO_ARR_COL_ITER = GENVAR_VEC_TO_ARR_COL_ITER + 1
    ) begin : feather_SRAM_BANK_TO_DPE_COL

      assign w_dpe_iacts[0][GENVAR_VEC_TO_ARR_COL_ITER]                     =   w_iacts_from_ctrl_to_dpe           [(GENVAR_VEC_TO_ARR_COL_ITER*IACTS_DATA_WIDTH)      +:  IACTS_DATA_WIDTH];
      assign w_dpe_iacts_valid[0][GENVAR_VEC_TO_ARR_COL_ITER]               =   w_iacts_valid_from_ctrl_to_dpe     [GENVAR_VEC_TO_ARR_COL_ITER];

      assign w_dpe_weights[0][GENVAR_VEC_TO_ARR_COL_ITER]                   =   w_weights_from_ctrl_to_dpe         [(GENVAR_VEC_TO_ARR_COL_ITER*(WEIGHTS_DATA_WIDTH/ARRAY_COL_NUM))    +:  WEIGHTS_DATA_WIDTH/ARRAY_COL_NUM];
      assign w_dpe_weights_valid[0][GENVAR_VEC_TO_ARR_COL_ITER]             =   w_weights_valid_from_ctrl_to_dpe   [GENVAR_VEC_TO_ARR_COL_ITER];

      assign w_dpe_weights_ping_pong_sel[0][GENVAR_VEC_TO_ARR_COL_ITER] = w_weights_ping_pong_sel;
    end

  endgenerate


  //===============================================================================================//
  // Systolic PE ARRAY
  wire                              o_acc_valid_down   [ 0:ARRAY_ROW_NUM * ARRAY_COL_NUM - 1];
  wire [   PE_OUTPUT_WIDTH - 1 : 0] o_acc_down         [ 0:ARRAY_ROW_NUM * ARRAY_COL_NUM - 1];

  wire                              o_weight_valid_down[ 0:ARRAY_ROW_NUM * ARRAY_COL_NUM - 1];
  wire [WEIGHTS_DATA_WIDTH - 1 : 0] o_weight_down      [ 0:ARRAY_ROW_NUM * ARRAY_COL_NUM - 1];

  wire [  IACTS_DATA_WIDTH - 1 : 0] o_PE_right         [ 0:ARRAY_ROW_NUM * ARRAY_COL_NUM - 1];
  wire                              o_valid_right      [00:ARRAY_ROW_NUM * ARRAY_COL_NUM - 1];

  genvar gi, gj;
  genvar i, j;
  /*
        instaniate 2D PE array
    */
  generate
    for (gi = 0; gi < ARRAY_ROW_NUM; gi = gi + 1) begin : pe_row
      for (gj = 0; gj < ARRAY_COL_NUM; gj = gj + 1) begin : pe_col
        if (gi == 0 && gj == 0) begin : top_left
          microscopiq_mixed_pe #(
              .IACTS_DATA_WIDTH(IACTS_DATA_WIDTH),
              .WEIGHTS_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
              .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH)
          ) pe_inst_0_0 (
              .clk(clk),
              .rst_n(rst_n),
              .mode(mode),
              .i_iacts(w_dpe_iacts[gi][gj]),
              .i_iacts_valid(w_dpe_iacts_valid[gi][gj]),
              .i_iacc_noc(scope_to_pe_data_bus[gj*PE_OUTPUT_WIDTH+:PE_OUTPUT_WIDTH]),
              .i_iacc_valid_noc(scope_to_pe_valid[gi]),
              .i_iacc_pe(0),
              .i_iacc_valid_pe(0),
              .i_iacc_noc_pe(i_iacc_noc_pe),
              .i_weights(w_dpe_weights[gi][gj]),
              .i_weights_valid(w_dpe_weights_valid[gi][gj]),
              .i_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi][gj]),
              .o_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi+1][gj]),
              .o_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj]),
              .o_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj]),
              .o_weights(o_weight_down[gi*ARRAY_COL_NUM+gj]),
              .o_weights_valid(o_weight_valid_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data(o_acc_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data_valid(o_acc_valid_down[gi*ARRAY_COL_NUM+gj])
          );
        end else if (gi == 0) begin : top
          microscopiq_mixed_pe #(
              .IACTS_DATA_WIDTH(IACTS_DATA_WIDTH),
              .WEIGHTS_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
              .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH)
          ) pe_inst_i_0 (
              .clk(clk),
              .rst_n(rst_n),
              .mode(mode),
              .i_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj-1]),
              .i_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj-1]),
              .i_iacc_noc(scope_to_pe_data_bus[gj*PE_OUTPUT_WIDTH+:PE_OUTPUT_WIDTH]),
              .i_iacc_valid_noc(scope_to_pe_valid[gi]),
              .i_iacc_pe({PE_OUTPUT_WIDTH{1'b0}}),
              .i_iacc_valid_pe(1'b0),
              .i_iacc_noc_pe(i_iacc_noc_pe),
              .i_weights(w_dpe_weights[gi][gj]),
              .i_weights_valid(w_dpe_weights_valid[gi][gj]),
              .i_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi][gj]),
              .o_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi+1][gj]),
              .o_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj]),
              .o_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj]),
              .o_weights(o_weight_down[gi*ARRAY_ROW_NUM+gj]),
              .o_weights_valid(o_weight_valid_down[gi*ARRAY_ROW_NUM+gj]),
              .o_out_data(o_acc_down[gi*ARRAY_ROW_NUM+gj]),
              .o_out_data_valid(o_acc_valid_down[gi*ARRAY_ROW_NUM+gj])
          );
        end else if (gj == 0) begin : left
          microscopiq_mixed_pe #(
              .IACTS_DATA_WIDTH(IACTS_DATA_WIDTH),
              .WEIGHTS_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
              .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH)
          ) pe_inst_0_j (
              .clk(clk),
              .rst_n(rst_n),
              .mode(mode),
              .i_iacts(w_dpe_iacts[gi][gj]),
              .i_iacts_valid(w_dpe_iacts_valid[gi][gj]),
              .i_iacc_noc(scope_to_pe_data_bus[gj*PE_OUTPUT_WIDTH+:PE_OUTPUT_WIDTH]),
              .i_iacc_valid_noc(scope_to_pe_valid[gi]),
              .i_iacc_pe(o_acc_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_iacc_valid_pe(o_acc_valid_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_iacc_noc_pe(i_iacc_noc_pe),
              .i_weights(o_weight_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_weights_valid(o_weight_valid_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi][gj]),
              .o_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi+1][gj]),
              .o_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj]),
              .o_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj]),
              .o_weights(o_weight_down[gi*ARRAY_COL_NUM+gj]),
              .o_weights_valid(o_weight_valid_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data(o_acc_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data_valid(o_acc_valid_down[gi*ARRAY_COL_NUM+gj])
          );
        end else if (gj == (ARRAY_COL_NUM - 1)) begin : right
          microscopiq_mixed_pe #(
              .IACTS_DATA_WIDTH(IACTS_DATA_WIDTH),
              .WEIGHTS_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
              .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH)
          ) pe_inst_0_j (
              .clk(clk),
              .rst_n(rst_n),
              .mode(mode),
              .i_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj-1]),
              .i_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj-1]),
              .i_iacc_noc(scope_to_pe_data_bus[gj*PE_OUTPUT_WIDTH+:PE_OUTPUT_WIDTH]),
              .i_iacc_valid_noc(scope_to_pe_valid[gi]),
              .i_iacc_pe(o_acc_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_iacc_valid_pe(o_acc_valid_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_iacc_noc_pe(i_iacc_noc_pe),
              .i_weights(o_weight_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_weights_valid(o_weight_valid_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi][gj]),
              .o_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi+1][gj]),
              .o_iacts(),
              .o_iacts_valid(),
              .o_weights(o_weight_down[gi*ARRAY_COL_NUM+gj]),
              .o_weights_valid(o_weight_valid_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data(o_acc_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data_valid(o_acc_valid_down[gi*ARRAY_COL_NUM+gj])
          );
        end else begin : other
          microscopiq_mixed_pe #(
              .IACTS_DATA_WIDTH(IACTS_DATA_WIDTH),
              .WEIGHTS_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
              .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH)
          ) pe_inst_i_j (
              .clk(clk),
              .rst_n(rst_n),
              .mode(mode),
              .i_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj-1]),
              .i_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj-1]),
              .i_iacc_noc(scope_to_pe_data_bus[gj*PE_OUTPUT_WIDTH+:PE_OUTPUT_WIDTH]),
              .i_iacc_valid_noc(scope_to_pe_valid[gi]),
              .i_iacc_pe(o_acc_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_iacc_valid_pe(o_acc_valid_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_iacc_noc_pe(i_iacc_noc_pe),
              .i_weights(o_weight_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_weights_valid(o_weight_valid_down[(gi-1)*ARRAY_ROW_NUM+gj]),
              .i_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi][gj]),
              .o_weights_ping_pong_sel(w_dpe_weights_ping_pong_sel[gi+1][gj]),
              .o_iacts(o_PE_right[gi*ARRAY_COL_NUM+gj]),
              .o_iacts_valid(o_valid_right[gi*ARRAY_COL_NUM+gj]),
              .o_weights(o_weight_down[gi*ARRAY_COL_NUM+gj]),
              .o_weights_valid(o_weight_valid_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data(o_acc_down[gi*ARRAY_COL_NUM+gj]),
              .o_out_data_valid(o_acc_valid_down[gi*ARRAY_COL_NUM+gj])
          );
        end
      end
    end
  endgenerate


  genvar GENVAR_COL_MUX_SCOPE;
  generate
    for (gi = 0; gi < ARRAY_ROW_NUM; gi = gi + 1) begin : pe_row_to_mux
      for (
          GENVAR_COL_MUX_SCOPE = 0;
          GENVAR_COL_MUX_SCOPE < ARRAY_ROW_NUM;
          GENVAR_COL_MUX_SCOPE = GENVAR_COL_MUX_SCOPE + 1
      ) begin : PE_ROW_TO_BUS
        assign  w_pe_row_out_to_mux_data_bus      [(gi*(GENVAR_COL_MUX_SCOPE*PE_OUTPUT_WIDTH) + GENVAR_COL_MUX_SCOPE*PE_OUTPUT_WIDTH)  +:  PE_OUTPUT_WIDTH]    =  o_acc_down[gi*ARRAY_COL_NUM + GENVAR_COL_MUX_SCOPE];
        assign  w_pe_row_out_to_mux_data_bus_valid[gi]                                           =     o_acc_valid_down[gi*ARRAY_COL_NUM + GENVAR_COL_MUX_SCOPE];
      end

    end
  endgenerate

  //// Mult-Stage muxing tree [Per Column]
  multi_row_mux #(
      .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH),
      .NUM_INPUT_DATA (ARRAY_COL_NUM),
      .DATA_WIDTH     (ARRAY_COL_NUM * PE_OUTPUT_WIDTH)
  ) MUX_INST (
      .clk           (clk),
      .rst_n         (rst_n),
      .i_valid       (w_pe_row_out_to_mux_data_bus_valid),
      .i_data_bus    (w_pe_row_out_to_mux_data_bus),
      .o_selected_row(o_selected_row),
      .o_valid       (w_o_mux_data_valid),
      .o_data_bus    (w_o_mux_data),
      .i_en          (i_top_en)
  );

  //// Scope
  scope #(
      .COMMAND_WIDTH         (COMMAND_WIDTH),
      .DATA_WIDTH            (PE_OUTPUT_WIDTH),
      .NUM_INPUT_OUTPUT_PORTS(ARRAY_COL_NUM)
  ) scope_INST (
      .clk                (clk),
      .rst_n              (rst_n),
      .row_activation_flat(row_activation_fifo_out),
      .i_valid            (w_o_mux_data_valid),
      .i_data_bus         (w_o_mux_data),
      .o_valid            (w_o_scope_data_bus_valid),
      .o_data_bus         (w_o_scope_data_bus),
      .i_en               (i_top_en),
      .i_cmd_flat         (w_scope_instr)
  );
  //###############################################################################################//

endmodule
