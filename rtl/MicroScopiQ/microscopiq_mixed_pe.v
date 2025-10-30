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

module microscopiq_mixed_pe #(
    parameter IACTS_DATA_WIDTH   = 4,  //  Iacts data width
    parameter WEIGHTS_DATA_WIDTH = 4,  //  Weights data width
    parameter PE_OUTPUT_WIDTH    = 8   //  Width of PE output
) (
    // timing signals
    clk,
    rst_n,
    mode,

    // Input Activation data signals
    i_iacts,  //  Iacts data
    i_iacts_valid,  //  Iacts data valid to register the data and control flow

    // Input Accumulation data signals
    i_iacc_noc,  //  Iacc data
    i_iacc_valid_noc,  //  Iacc data valid to register the data and control flow

    i_iacc_pe,  //  Iacc data
    i_iacc_valid_pe,  //  Iacc data valid to register the data and control flow

    i_iacc_noc_pe,  // To select if the acc data comes from NoC or PE above in the same column

    // Input Weights data signals
    i_weights,  //  Weights data
    i_weights_valid,  //  Weights data valid to register the data and control flow

    // control signals for FIFO+PE operation
    i_weights_ping_pong_sel,  //  Weights internal Ping-pong buffer selection

    o_weights_ping_pong_sel,  //  registered output of i_weights_ping_pong_sel, to next PE

    o_iacts,  //  registered output of i_iacts, to next PE
    o_iacts_valid,  //  registered output of i_iacts_valid, to next PE

    o_weights,  //  registered output of i_weights, to next PE
    o_weights_valid,  //  registered output of i_weights_valid, to next PE

    o_out_data,  //  Computed PE output
    o_out_data_valid  //  data valid for the PE output
);
  /*
        ports
    */
  input clk;
  input rst_n;
  input mode;
  input [IACTS_DATA_WIDTH   -1:0] i_iacts;
  input i_iacts_valid;

  input [PE_OUTPUT_WIDTH   -1:0] i_iacc_noc;
  input i_iacc_valid_noc;

  input [PE_OUTPUT_WIDTH   -1:0] i_iacc_pe;
  input i_iacc_valid_pe;

  input i_iacc_noc_pe;

  input [WEIGHTS_DATA_WIDTH -1:0] i_weights;
  input i_weights_valid;

  input i_weights_ping_pong_sel;
  output o_weights_ping_pong_sel;

  output [IACTS_DATA_WIDTH    -1:0] o_iacts;
  output o_iacts_valid;

  output [WEIGHTS_DATA_WIDTH  -1:0] o_weights;
  output o_weights_valid;

  output [PE_OUTPUT_WIDTH     -1:0] o_out_data;
  output o_out_data_valid;

  /*
        inner logics
    */
  reg     [   WEIGHTS_DATA_WIDTH                             -1:0] r_local_weights_buffer_ping;
  reg     [   WEIGHTS_DATA_WIDTH                             -1:0] r_local_weights_buffer_pong;

  reg                                                              r_weights_ping_pong_sel;
  reg     [   PE_OUTPUT_WIDTH                               -1:0] r_iacts;
  reg                                                              r_iacts_valid;

  reg     [   PE_OUTPUT_WIDTH                               -1:0] r_iacc;
  reg                                                              r_iacc_valid;

  reg     [   WEIGHTS_DATA_WIDTH                             -1:0] r_weights;
  reg                                                              r_weights_valid;

  reg     [                                 PE_OUTPUT_WIDTH-1 : 0] r_out_data;

  wire    [   IACTS_DATA_WIDTH                               -1:0] w_iacts;
  wire    [   PE_OUTPUT_WIDTH                               -1:0] w_iacc;
  wire    [                              WEIGHTS_DATA_WIDTH - 1 : 0] w_selected_weight;
  wire                                                             last_compute_valid;

  wire    [   PE_OUTPUT_WIDTH                               -1:0] w_mul_iacts_weights;
  reg     [   PE_OUTPUT_WIDTH                               -1:0] r_mul_iacts_weights;

  reg                                                              r_out_data_valid;
  reg     [PE_OUTPUT_WIDTH                               - 2 : 0] activation_magnitude_zero = 0;

  integer                                                          i;

    mixed_multiplier#(
        .WEIGHTS_DATA_WIDTH(WEIGHTS_DATA_WIDTH),
        .IACTS_DATA_WIDTH(IACTS_DATA_WIDTH),
        .PE_OUTPUT_WIDTH(PE_OUTPUT_WIDTH)
    )
    mixed_multiplier_INST(
        .clk(clk),
        .rst_n(rst_n),

        .weight(w_selected_weight),
        .activation(w_iacts),
        .result(w_mul_iacts_weights)
    );

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_iacts                     <= 0;
      r_weights                   <= 0;

      r_weights_ping_pong_sel     <= 0;
      r_iacts_valid               <= 0;
      r_weights_valid             <= 0;

      r_local_weights_buffer_ping <= 0;
      r_local_weights_buffer_pong <= 0;
    end else begin
      r_iacts_valid <= i_iacts_valid;
      r_iacc_valid <= i_iacc_noc_pe == 0 ? i_iacc_valid_noc : i_iacc_noc_pe;

      r_iacts <= i_iacts_valid ? i_iacts : {IACTS_DATA_WIDTH{1'b0}};
      r_iacc                      <=  (i_iacc_noc_pe == 0) ? (i_iacc_valid_noc == 1 ? i_iacc_noc : {PE_OUTPUT_WIDTH{1'b0}}) : (i_iacc_valid_pe == 1 ? i_iacc_pe : {PE_OUTPUT_WIDTH{1'b0}}) ;

      r_weights_valid <= i_weights_valid;
      r_weights <= i_weights;

      r_weights_ping_pong_sel <= i_weights_ping_pong_sel;

      // Check if this PE has been selected and then load weights (infrequent operation)
      if (i_weights_valid) begin
        if (i_weights_ping_pong_sel == 0) begin
          r_local_weights_buffer_ping <= i_weights;
        end else begin
          r_local_weights_buffer_pong <= i_weights;
        end
      end
    end
  end

  /*
        MAC for the weights and iacts logic
    */
  assign w_iacts = r_iacts;
  assign w_iacc = r_iacc;

  assign w_selected_weight = (i_weights_ping_pong_sel == 1) ? r_local_weights_buffer_ping : r_local_weights_buffer_pong;

// This will be handled by the mixed-multiplier
//   assign w_mul_iacts_weights = {
//     w_iacts[IACTS_DATA_WIDTH-1] ^ w_selected_weight[WEIGHTS_DATA_WIDTH-1],
//     (w_selected_weight[0] == 0) ? activation_magnitude_zero : w_iacts[IACTS_DATA_WIDTH-2 : 0]
//   };

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_out_data       <= 0;
      r_out_data_valid <= 0;
    end else begin
      r_mul_iacts_weights <= w_mul_iacts_weights;

      r_out_data          <= (mode == 0) ? {w_mul_iacts_weights[PE_OUTPUT_WIDTH/2 +: PE_OUTPUT_WIDTH/2] + w_iacc,w_mul_iacts_weights[0 +: PE_OUTPUT_WIDTH/2] + w_iacc} : w_mul_iacts_weights + w_iacc;
      r_out_data_valid    <= r_iacc_valid & r_iacts_valid;
    end
  end

  assign o_weights_ping_pong_sel = r_weights_ping_pong_sel;
  assign o_iacts                 = r_iacts;
  assign o_iacts_valid           = r_iacts_valid;
  assign o_weights               = r_weights;
  assign o_weights_valid         = r_weights_valid;
  assign o_out_data              = r_out_data;  // Ensure accumulation out is PE_OUTPUT_WIDTH bits
  assign o_out_data_valid        = r_out_data_valid;

endmodule
