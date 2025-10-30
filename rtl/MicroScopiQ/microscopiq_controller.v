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

module microscopiq_controller #(
    parameter WEIGHTS_DATA_WIDTH           = 2,                                                 //
    parameter WEIGHTS_NUM_BANKS            = 1,                                                 //
    parameter WEIGHTS_SRAM_DATA_WIDTH      = WEIGHTS_NUM_BANKS * WEIGHTS_DATA_WIDTH,            //
    parameter WEIGHTS_SRAM_BANK_ADDR_WIDTH = 2,                                                 //
    parameter WEIGHTS_SRAM_ADDR_WIDTH      = WEIGHTS_NUM_BANKS * WEIGHTS_SRAM_BANK_ADDR_WIDTH,  //
    parameter scope_IN_COMMAND_WIDTH       = 3,
    parameter IACTS_DATA_WIDTH             = 8,                                                 //
    parameter IACTS_NUM_BANKS              = 8,                                                 //
    parameter IACTS_SRAM_DATA_WIDTH        = IACTS_NUM_BANKS * IACTS_DATA_WIDTH,                //
    parameter IACTS_SRAM_BANK_ADDR_WIDTH   = 2,                                                 //
    parameter IACTS_SRAM_ADDR_WIDTH        = IACTS_NUM_BANKS * IACTS_SRAM_BANK_ADDR_WIDTH,      //

    parameter SCALE_VALUE_WIDTH = 32,  //

    parameter PE_OUTPUT_WIDTH = 8,  //

    parameter ARRAY_COL_NUM      = 8,  //
    parameter ARRAY_ROW_NUM      = 8,  //
    parameter LOG2_ARRAY_COL_NUM = 3,  //
    parameter LOG2_ARRAY_ROW_NUM = 3,  //

    parameter INSTR_SRAM_BANK_ADDR_WIDTH = 2,  //
    parameter INSTR_SRAM_BANK_DATA_WIDTH = 32
) (

    // signals from MicroscopiQ TOP
    clk,
    rst_n,
    i_top_en,

    i_weights_write_valid,
    i_weights_write_data,
    i_weights_write_addr,
    i_weights_write_addr_end,

    i_iacts_write_valid,
    i_iacts_write_data,
    i_iacts_write_addr,
    i_iacts_write_addr_end,

    i_instr_write_valid,
    i_instr_write_data,
    i_instr_write_addr,

    i_all_buf_pingpong_config,

    // inputs from scope
    i_data_bus_from_scope,
    i_data_bus_from_scope_valid,
    i_selected_row_mux,

    // Output Scope data to PEs
    o_data_bus_from_scope_to_pe_rows,
    o_data_bus_from_scope_valid_to_pe_row,

    // Outputs to PE Array
    o_weights_ping_pong_sel,
    o_scope_instr,

    // I-O from-to iActs Ping
    o_iacts_sram_a_wr_data_ping,
    o_iacts_sram_a_wr_addr_ping,
    o_iacts_sram_a_wr_en_ping,
    o_iacts_sram_b_rd_addr_ping,
    o_iacts_sram_b_rd_en_ping,
    i_iacts_sram_b_rd_data_ping,

    // I-O from-to iActs Pong
    o_iacts_sram_a_wr_data_pong,
    o_iacts_sram_a_wr_addr_pong,
    o_iacts_sram_a_wr_en_pong,
    o_iacts_sram_b_rd_addr_pong,
    o_iacts_sram_b_rd_en_pong,
    i_iacts_sram_b_rd_data_pong,

    // I-O from-to Weights Ping
    o_weights_sram_a_wr_data_ping,
    o_weights_sram_a_wr_addr_ping,
    o_weights_sram_a_wr_en_ping,
    o_weights_sram_b_rd_addr_ping,
    o_weights_sram_b_rd_en_ping,
    i_weights_sram_b_rd_data_ping,

    // I-O from-to Weights Pong
    o_weights_sram_a_wr_data_pong,
    o_weights_sram_a_wr_addr_pong,
    o_weights_sram_a_wr_en_pong,
    o_weights_sram_b_rd_addr_pong,
    o_weights_sram_b_rd_en_pong,
    i_weights_sram_b_rd_data_pong,

    // Outputs to PE ARRAY
    o_iacts_from_ctrl_to_dpe,
    o_iacts_valid_from_ctrl_to_dpe,
    o_weights_from_ctrl_to_dpe,
    o_weights_valid_from_ctrl_to_dpe,
    // I-O from-to INSTR Buffer
    o_instr_sram_a_wr_data,
    o_instr_sram_a_wr_addr,
    o_instr_sram_a_wr_en,
    o_instr_sram_b_rd_addr,
    o_instr_sram_b_rd_en,
    i_instr_sram_b_rd_data

);

  localparam IACTS_PINGPONG_CONFIG_WIDTH = 4;
  localparam WEIGHTS_PINGPONG_CONFIG_WIDTH = 4;
  localparam ALL_BUF_CONFIG_WIDTH = IACTS_PINGPONG_CONFIG_WIDTH + WEIGHTS_PINGPONG_CONFIG_WIDTH;

  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_IDLE = 0;
  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_FILL_PING = 1;
  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG = 2;
  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING = 3;
  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_DRAIN_PONG = 4;
  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_DRAIN_PING = 5;
  localparam [IACTS_PINGPONG_CONFIG_WIDTH     -1:0] IACTS_PINGPONG_FILL_PONG = 6;

  localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1:0] WEIGHTS_PINGPONG_IDLE = 0;
  localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1:0] WEIGHTS_PINGPONG_FILL_PING = 1;
  localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1:0] WEIGHTS_PINGPONG_PING_FEED_DPE = 2;
  localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1:0] WEIGHTS_PINGPONG_FILL_PONG = 3;
  localparam [WEIGHTS_PINGPONG_CONFIG_WIDTH   -1:0] WEIGHTS_PINGPONG_PONG_FEED_DPE = 4;


  /*
        ports
    */
  input clk;
  input rst_n;
  input i_top_en;

  input i_iacts_write_valid;
  input [IACTS_SRAM_DATA_WIDTH          -1:0] i_iacts_write_data;
  input [IACTS_SRAM_BANK_ADDR_WIDTH     -1:0] i_iacts_write_addr;
  input [IACTS_SRAM_BANK_ADDR_WIDTH     -1:0] i_iacts_write_addr_end;

  input i_weights_write_valid;
  input [WEIGHTS_SRAM_DATA_WIDTH        -1:0] i_weights_write_data;
  input [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1:0] i_weights_write_addr;
  input [WEIGHTS_SRAM_BANK_ADDR_WIDTH   -1:0] i_weights_write_addr_end;

  input i_instr_write_valid;
  input [INSTR_SRAM_BANK_DATA_WIDTH     -1:0] i_instr_write_data;
  input [INSTR_SRAM_BANK_ADDR_WIDTH     -1:0] i_instr_write_addr;

  input [ALL_BUF_CONFIG_WIDTH           -1:0] i_all_buf_pingpong_config;

  // Outputs to scope
  output o_weights_ping_pong_sel;
  output [INSTR_SRAM_BANK_DATA_WIDTH         -1:0] o_scope_instr;

  // TO PE array
  output [ARRAY_ROW_NUM - 1 : 0] o_data_bus_from_scope_valid_to_pe_row;
  output      [ARRAY_COL_NUM * PE_OUTPUT_WIDTH -1: 0]  o_data_bus_from_scope_to_pe_rows; // A unit in the top-level should handle splitting to per-PE

  // inputs from scope
  input [ARRAY_COL_NUM * PE_OUTPUT_WIDTH -1:0] i_data_bus_from_scope;
  input [ARRAY_COL_NUM-1:0] i_data_bus_from_scope_valid;
  input [ARRAY_ROW_NUM - 1 : 0] i_selected_row_mux;

  // I-O from-to iActs Ping
  output [IACTS_SRAM_DATA_WIDTH          -1:0] o_iacts_sram_a_wr_data_ping;
  output [IACTS_SRAM_ADDR_WIDTH          -1:0] o_iacts_sram_a_wr_addr_ping;
  output [IACTS_NUM_BANKS                -1:0] o_iacts_sram_a_wr_en_ping;
  output [IACTS_SRAM_ADDR_WIDTH          -1:0] o_iacts_sram_b_rd_addr_ping;
  output [IACTS_NUM_BANKS                -1:0] o_iacts_sram_b_rd_en_ping;
  input [IACTS_SRAM_DATA_WIDTH          -1:0] i_iacts_sram_b_rd_data_ping;

  // I-O from-to iActs Pong
  output [IACTS_SRAM_DATA_WIDTH          -1:0] o_iacts_sram_a_wr_data_pong;
  output [IACTS_SRAM_ADDR_WIDTH          -1:0] o_iacts_sram_a_wr_addr_pong;
  output [IACTS_NUM_BANKS                -1:0] o_iacts_sram_a_wr_en_pong;
  output [IACTS_SRAM_ADDR_WIDTH          -1:0] o_iacts_sram_b_rd_addr_pong;
  output [IACTS_NUM_BANKS                -1:0] o_iacts_sram_b_rd_en_pong;
  input [IACTS_SRAM_DATA_WIDTH          -1:0] i_iacts_sram_b_rd_data_pong;

  // I-O from-to Weights Ping
  output [WEIGHTS_SRAM_DATA_WIDTH        -1:0] o_weights_sram_a_wr_data_ping;
  output [WEIGHTS_SRAM_ADDR_WIDTH        -1:0] o_weights_sram_a_wr_addr_ping;
  output [WEIGHTS_NUM_BANKS              -1:0] o_weights_sram_a_wr_en_ping;
  output [WEIGHTS_SRAM_ADDR_WIDTH        -1:0] o_weights_sram_b_rd_addr_ping;
  output [WEIGHTS_NUM_BANKS              -1:0] o_weights_sram_b_rd_en_ping;
  input [WEIGHTS_SRAM_DATA_WIDTH        -1:0] i_weights_sram_b_rd_data_ping;

  // I-O from-to Weights Pong
  output [WEIGHTS_SRAM_DATA_WIDTH        -1:0] o_weights_sram_a_wr_data_pong;
  output [WEIGHTS_SRAM_ADDR_WIDTH        -1:0] o_weights_sram_a_wr_addr_pong;
  output [WEIGHTS_NUM_BANKS              -1:0] o_weights_sram_a_wr_en_pong;
  output [WEIGHTS_SRAM_ADDR_WIDTH        -1:0] o_weights_sram_b_rd_addr_pong;
  output [WEIGHTS_NUM_BANKS              -1:0] o_weights_sram_b_rd_en_pong;
  input [WEIGHTS_SRAM_DATA_WIDTH        -1:0] i_weights_sram_b_rd_data_pong;

  output [IACTS_SRAM_DATA_WIDTH          -1:0] o_iacts_from_ctrl_to_dpe;
  output [IACTS_NUM_BANKS                -1:0] o_iacts_valid_from_ctrl_to_dpe;
  output [ARRAY_COL_NUM                    -1:0] o_weights_valid_from_ctrl_to_dpe;
  output [WEIGHTS_SRAM_DATA_WIDTH        -1:0] o_weights_from_ctrl_to_dpe;


  output [INSTR_SRAM_BANK_DATA_WIDTH     -1:0] o_instr_sram_a_wr_data;
  output [INSTR_SRAM_BANK_ADDR_WIDTH     -1:0] o_instr_sram_a_wr_addr;
  output o_instr_sram_a_wr_en;
  output [INSTR_SRAM_BANK_ADDR_WIDTH     -1:0] o_instr_sram_b_rd_addr;
  output o_instr_sram_b_rd_en;
  input [INSTR_SRAM_BANK_DATA_WIDTH     -1:0] i_instr_sram_b_rd_data;


  // internal signals

  wire [  IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_from_ctrl_to_dpe;
  wire [  IACTS_NUM_BANKS                    -1:0] w_iacts_valid_from_ctrl_to_dpe;
  wire [ARRAY_COL_NUM                        -1:0] w_weights_valid_from_ctrl_to_dpe;
  wire [  WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_from_ctrl_to_dpe;


  reg  [  IACTS_PINGPONG_CONFIG_WIDTH        -1:0] r_acts_buf_ping_pong_state;
  reg  [  WEIGHTS_PINGPONG_CONFIG_WIDTH      -1:0] r_weights_buf_ping_pong_state;

  reg  [  IACTS_SRAM_BANK_ADDR_WIDTH         -1:0] r_iacts_pingpong_rd_addr;
  reg  [  WEIGHTS_SRAM_BANK_ADDR_WIDTH       -1:0] r_weights_pingpong_rd_addr;

  wire [  IACTS_PINGPONG_CONFIG_WIDTH        -1:0] w_iacts_pingpong_config;
  wire [  WEIGHTS_PINGPONG_CONFIG_WIDTH      -1:0] w_weights_pingpong_config;
  reg  [  scope_IN_COMMAND_WIDTH             -1:0] r_scope_instr;

  reg                                              r_weights_ping_pong_sel;

  wire [  IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_sram_a_wr_data_ping;
  wire [  IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_a_wr_addr_ping;
  wire [  IACTS_NUM_BANKS                    -1:0] w_iacts_sram_a_wr_en_ping;
  wire [  IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_b_rd_addr_ping;
  wire [  IACTS_NUM_BANKS                    -1:0] w_iacts_sram_b_rd_en_ping;
  wire [  IACTS_SRAM_DATA_WIDTH              -1:0] w_iacts_sram_a_wr_data_pong;
  wire [  IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_a_wr_addr_pong;
  wire [  IACTS_NUM_BANKS                    -1:0] w_iacts_sram_a_wr_en_pong;
  wire [  IACTS_SRAM_ADDR_WIDTH              -1:0] w_iacts_sram_b_rd_addr_pong;
  wire [  IACTS_NUM_BANKS                    -1:0] w_iacts_sram_b_rd_en_pong;
  wire [  WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_sram_a_wr_data_ping;
  wire [  WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_a_wr_addr_ping;
  wire [  WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_a_wr_en_ping;
  wire [  WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_b_rd_addr_ping;
  wire [  WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_b_rd_en_ping;
  wire [  WEIGHTS_SRAM_DATA_WIDTH            -1:0] w_weights_sram_a_wr_data_pong;
  wire [  WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_a_wr_addr_pong;
  wire [  WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_a_wr_en_pong;
  wire [  WEIGHTS_SRAM_ADDR_WIDTH            -1:0] w_weights_sram_b_rd_addr_pong;
  wire [  WEIGHTS_NUM_BANKS                  -1:0] w_weights_sram_b_rd_en_pong;

  // extracting ping pong FSM state instruction from i_all_buf_pingpong_config - top
  assign w_iacts_pingpong_config = i_all_buf_pingpong_config[0+:IACTS_PINGPONG_CONFIG_WIDTH];
  assign w_weights_pingpong_config = i_all_buf_pingpong_config[IACTS_PINGPONG_CONFIG_WIDTH     +:  WEIGHTS_PINGPONG_CONFIG_WIDTH];

  /*
        Reading/Writing Weights, iActs and Instruction from SRAMs
    */
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_scope_instr                 <= 0;
      r_iacts_pingpong_rd_addr      <= 0;
      r_weights_pingpong_rd_addr    <= 0;
      r_weights_ping_pong_sel       <= 0;
      r_acts_buf_ping_pong_state    <= IACTS_PINGPONG_IDLE;
      r_weights_buf_ping_pong_state <= WEIGHTS_PINGPONG_IDLE;
    end else begin

      //________________________________________________________________________________________________________________________
      //  r_acts_buf_ping_pong_state  FSM
      case (r_acts_buf_ping_pong_state)
        IACTS_PINGPONG_IDLE: begin
          //(w_iacts_pingpong_config ==  IACTS_PINGPONG_FILL_PING)
          if (i_top_en == 1) begin
            r_acts_buf_ping_pong_state <= IACTS_PINGPONG_FILL_PING;
            r_iacts_pingpong_rd_addr   <= 0;
          end
        end


        IACTS_PINGPONG_FILL_PING: begin
          //(w_iacts_pingpong_config ==  IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG)
          if (i_iacts_write_addr == i_iacts_write_addr_end) begin
            r_acts_buf_ping_pong_state <= IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG;
            r_iacts_pingpong_rd_addr   <= 0;
          end
        end


        IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG: begin
          if (r_iacts_pingpong_rd_addr == i_iacts_write_addr_end) begin
            if (w_iacts_pingpong_config == IACTS_PINGPONG_DRAIN_PONG) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_DRAIN_PONG;
            end else if (w_iacts_pingpong_config == IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING;
            end

            r_iacts_pingpong_rd_addr <= 0;
          end else begin
            r_iacts_pingpong_rd_addr <= r_iacts_pingpong_rd_addr + 1;
          end
        end


        IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING: begin
          if (r_iacts_pingpong_rd_addr == i_iacts_write_addr_end) begin
            if (w_iacts_pingpong_config == IACTS_PINGPONG_DRAIN_PING) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_DRAIN_PING;
            end else if (w_iacts_pingpong_config == IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_PING_FEED_DPE_FILL_PONG;
            end
            r_iacts_pingpong_rd_addr <= 0;
          end else begin
            r_iacts_pingpong_rd_addr <= r_iacts_pingpong_rd_addr + 1;
          end

        end


        IACTS_PINGPONG_DRAIN_PONG: begin
          if (i_iacts_write_addr == i_iacts_write_addr_end) begin
            if (w_iacts_pingpong_config == IACTS_PINGPONG_FILL_PONG) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_FILL_PONG;
            end else if (w_iacts_pingpong_config == IACTS_PINGPONG_FILL_PING) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_FILL_PING;
            end
          end
        end

        IACTS_PINGPONG_DRAIN_PING: begin
          if (i_iacts_write_addr == i_iacts_write_addr_end) begin
            if (w_iacts_pingpong_config == IACTS_PINGPONG_FILL_PONG) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_FILL_PONG;
            end else if (w_iacts_pingpong_config == IACTS_PINGPONG_FILL_PING) begin
              r_acts_buf_ping_pong_state <= IACTS_PINGPONG_FILL_PING;
            end
          end
        end


        IACTS_PINGPONG_FILL_PONG: begin
          if (i_iacts_write_addr == i_iacts_write_addr_end) begin
            r_acts_buf_ping_pong_state <= IACTS_PINGPONG_PONG_FEED_DPE_FILL_PING;
          end
        end


        default: begin
          r_acts_buf_ping_pong_state <= IACTS_PINGPONG_IDLE;
          r_iacts_pingpong_rd_addr   <= 0;
        end
      endcase
      //________________________________________________________________________________________________________________________




      //________________________________________________________________________________________________________________________
      case (r_weights_buf_ping_pong_state)
        WEIGHTS_PINGPONG_IDLE: begin
          if (i_top_en == 1) begin
            r_weights_buf_ping_pong_state <= WEIGHTS_PINGPONG_FILL_PING;
            r_weights_pingpong_rd_addr    <= 0;
          end
        end


        WEIGHTS_PINGPONG_FILL_PING: begin
          if (i_weights_write_addr == i_weights_write_addr_end) begin
            if (w_weights_pingpong_config == WEIGHTS_PINGPONG_PING_FEED_DPE) begin
              r_weights_buf_ping_pong_state <= WEIGHTS_PINGPONG_PING_FEED_DPE;
              r_weights_pingpong_rd_addr    <= 0;
            end
          end
        end


        WEIGHTS_PINGPONG_PING_FEED_DPE: begin
          if (r_weights_pingpong_rd_addr == i_weights_write_addr_end) begin
            if (w_weights_pingpong_config == WEIGHTS_PINGPONG_FILL_PONG) begin
              r_weights_buf_ping_pong_state <= WEIGHTS_PINGPONG_FILL_PONG;
            end
            r_weights_pingpong_rd_addr <= 0;
          end
          begin
            r_weights_pingpong_rd_addr <= r_weights_pingpong_rd_addr + 1;
          end
        end


        WEIGHTS_PINGPONG_FILL_PONG: begin
          if (i_weights_write_addr == i_weights_write_addr_end) begin
            r_weights_buf_ping_pong_state <= WEIGHTS_PINGPONG_PONG_FEED_DPE;
            r_weights_pingpong_rd_addr    <= 0;
          end
        end


        WEIGHTS_PINGPONG_PONG_FEED_DPE: begin
          if (r_weights_pingpong_rd_addr == i_weights_write_addr_end) begin
            r_weights_buf_ping_pong_state <= WEIGHTS_PINGPONG_FILL_PING;
            r_weights_pingpong_rd_addr    <= 0;
          end
          begin
            r_weights_pingpong_rd_addr <= r_weights_pingpong_rd_addr + 1;
          end
        end


        default: begin
          r_weights_buf_ping_pong_state <= 0;
        end
      endcase
      //________________________________________________________________________________________________________________________

      //************************************************************************************************//
      r_scope_instr <= i_instr_sram_b_rd_data[scope_IN_COMMAND_WIDTH-1 : 0];
      //************************************************************************************************//

    end
  end


  // for iActs Ping, port A
  assign w_iacts_sram_a_wr_data_ping = i_iacts_write_data;
  assign w_iacts_sram_a_wr_addr_ping = {IACTS_NUM_BANKS{i_iacts_write_addr}};
  assign w_iacts_sram_a_wr_en_ping = {IACTS_NUM_BANKS{i_iacts_write_valid}};

  // for iActs Ping, port B
  assign w_iacts_sram_b_rd_addr_ping = {IACTS_NUM_BANKS{r_iacts_pingpong_rd_addr}};
  assign w_iacts_sram_b_rd_en_ping = ~0;

  // for iActs Pong, port A
  assign w_iacts_sram_a_wr_data_pong = i_iacts_write_data;
  assign w_iacts_sram_a_wr_addr_pong = {IACTS_NUM_BANKS{i_iacts_write_addr}};
  assign w_iacts_sram_a_wr_en_pong = {IACTS_NUM_BANKS{i_iacts_write_valid}};

  // for iActs Pong, port B
  assign w_iacts_sram_b_rd_addr_pong = {IACTS_NUM_BANKS{r_iacts_pingpong_rd_addr}};
  assign w_iacts_sram_b_rd_en_pong = ~0;

  // iActs from ctrl to dpe
  assign w_iacts_from_ctrl_to_dpe = i_iacts_sram_b_rd_data_ping;
  assign w_iacts_valid_from_ctrl_to_dpe = ~0;

  // Weights Ping, A
  assign w_weights_sram_a_wr_data_ping = i_weights_write_data;
  assign w_weights_sram_a_wr_addr_ping = {WEIGHTS_NUM_BANKS{i_weights_write_addr}};
  assign w_weights_sram_a_wr_en_ping = {WEIGHTS_NUM_BANKS{i_weights_write_valid}};

  // Weights Ping, B
  assign w_weights_sram_b_rd_addr_ping = {WEIGHTS_NUM_BANKS{r_weights_pingpong_rd_addr}};
  assign w_weights_sram_b_rd_en_ping = ~0;

  // Weights Pong, A
  assign w_weights_sram_a_wr_data_pong = i_weights_write_data;
  assign w_weights_sram_a_wr_addr_pong = {WEIGHTS_NUM_BANKS{i_weights_write_addr}};
  assign w_weights_sram_a_wr_en_pong = {WEIGHTS_NUM_BANKS{i_weights_write_valid}};

  // Weights Pong, B
  assign w_weights_sram_b_rd_addr_pong = {WEIGHTS_NUM_BANKS{r_weights_pingpong_rd_addr}};
  assign w_weights_sram_b_rd_en_pong = ~0;

  // Weights from ctrl to dpe
  assign w_weights_from_ctrl_to_dpe = i_weights_sram_b_rd_data_ping;
  assign w_weights_valid_from_ctrl_to_dpe = ~0;


  //__________________________________________________________________________________________________________________________
  //__________________________________________________________________________________________________________________________

  assign o_data_bus_from_scope_valid_to_pe_row = i_selected_row_mux;


  // connect to ports

  assign o_iacts_from_ctrl_to_dpe = w_iacts_from_ctrl_to_dpe;
  assign o_iacts_valid_from_ctrl_to_dpe = w_iacts_valid_from_ctrl_to_dpe;
  assign o_weights_from_ctrl_to_dpe = w_weights_from_ctrl_to_dpe;
  assign o_weights_valid_from_ctrl_to_dpe = w_weights_valid_from_ctrl_to_dpe;

  assign o_weights_ping_pong_sel = r_weights_ping_pong_sel;
  assign o_scope_instr = r_scope_instr;
  assign o_iacts_sram_a_wr_data_ping = w_iacts_sram_a_wr_data_ping;
  assign o_iacts_sram_a_wr_addr_ping = w_iacts_sram_a_wr_addr_ping;
  assign o_iacts_sram_a_wr_en_ping = w_iacts_sram_a_wr_en_ping;
  assign o_iacts_sram_b_rd_addr_ping = w_iacts_sram_b_rd_addr_ping;
  assign o_iacts_sram_b_rd_en_ping = w_iacts_sram_b_rd_en_ping;
  assign o_iacts_sram_a_wr_data_pong = w_iacts_sram_a_wr_data_pong;
  assign o_iacts_sram_a_wr_addr_pong = w_iacts_sram_a_wr_addr_pong;
  assign o_iacts_sram_a_wr_en_pong = w_iacts_sram_a_wr_en_pong;
  assign o_iacts_sram_b_rd_addr_pong = w_iacts_sram_b_rd_addr_pong;
  assign o_iacts_sram_b_rd_en_pong = w_iacts_sram_b_rd_en_pong;
  assign o_weights_sram_a_wr_data_ping = w_weights_sram_a_wr_data_ping;
  assign o_weights_sram_a_wr_addr_ping = w_weights_sram_a_wr_addr_ping;
  assign o_weights_sram_a_wr_en_ping = w_weights_sram_a_wr_en_ping;
  assign o_weights_sram_b_rd_addr_ping = w_weights_sram_b_rd_addr_ping;
  assign o_weights_sram_b_rd_en_ping = w_weights_sram_b_rd_en_ping;
  assign o_weights_sram_a_wr_data_pong = w_weights_sram_a_wr_data_pong;
  assign o_weights_sram_a_wr_addr_pong = w_weights_sram_a_wr_addr_pong;
  assign o_weights_sram_a_wr_en_pong = w_weights_sram_a_wr_en_pong;
  assign o_weights_sram_b_rd_addr_pong = w_weights_sram_b_rd_addr_pong;
  assign o_weights_sram_b_rd_en_pong = w_weights_sram_b_rd_en_pong;


  assign o_instr_sram_a_wr_data = i_instr_write_data;
  assign o_instr_sram_a_wr_addr = i_instr_write_addr;
  assign o_instr_sram_a_wr_en = i_instr_write_valid;
  assign o_instr_sram_b_rd_addr = r_iacts_pingpong_rd_addr;
  assign o_instr_sram_b_rd_en = (|(w_iacts_sram_b_rd_en_ping)) | (|(w_iacts_sram_b_rd_en_pong));

  assign o_data_bus_from_scope_to_pe_rows = i_data_bus_from_scope;

endmodule
