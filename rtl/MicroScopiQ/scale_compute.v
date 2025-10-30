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

module scale_compute #(
    SCALE_VALUE_WIDTH = 16
) (
    clk,
    rst_n,

    i_scale_compute_en,
    i_scale_val,

    o_scale_val,
    o_scale_val_en
);
  input clk;
  input rst_n;
  input [SCALE_VALUE_WIDTH              -1:0] i_scale_val;
  output [SCALE_VALUE_WIDTH              -1:0] o_scale_val;
  input i_scale_compute_en;
  output o_scale_val_en;

  reg [SCALE_VALUE_WIDTH/4              -1:0] computed_scale_inlier_inner;
  reg [SCALE_VALUE_WIDTH/4              -1:0] computed_scale_outlier_inner;
  reg [  SCALE_VALUE_WIDTH              -1:0] i_scale_inner;
  reg                                         scale_valid;


  always @(posedge clk) begin
    if (~rst_n) begin
      computed_scale_inlier_inner  <= 0;
      computed_scale_outlier_inner <= 0;
      scale_valid                  <= 0;
    end else begin
      i_scale_inner <= i_scale_val;
      scale_valid <= i_scale_compute_en;
      computed_scale_inlier_inner <= i_scale_val[3 : 0] + i_scale_val[SCALE_VALUE_WIDTH-1 : 8];
      computed_scale_outlier_inner <= i_scale_val[6:4] - i_scale_val[3:0] + i_scale_val[7] + i_scale_val[SCALE_VALUE_WIDTH-1 : 8];
    end
  end

  assign o_scale_val = {
    i_scale_inner[SCALE_VALUE_WIDTH-1 : 8],
    computed_scale_outlier_inner,
    computed_scale_inlier_inner
  };
  assign o_scale_val_en = scale_valid;
endmodule
