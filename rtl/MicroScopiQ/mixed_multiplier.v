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
`timescale 1ns/1ps

module mixed_multiplier #(
    parameter IACTS_DATA_WIDTH   = 4,  //  Iacts data width
    parameter WEIGHTS_DATA_WIDTH = 4,  //  Weights data width
    parameter PE_OUTPUT_WIDTH    = 4   //  Width of PE output

) (
    clk,
    rst_n,
    mode,
    weight,
    activation,
    result
);


input clk;
input rst_n;

input mode;

input [WEIGHTS_DATA_WIDTH-1 : 0] weight;

input [IACTS_DATA_WIDTH-1 : 0] activation;

output [PE_OUTPUT_WIDTH-1 : 0] result;

// Bitshifter architecture focuses on unsigned multiplication, 
// the signed inputs must be converted to an absolute value before the multiplication, 
// and the final output will be converted back to the signed value.

wire [IACTS_DATA_WIDTH - 1 : 0] mul_p00, mul_p01, mul_p10, mul_p11, 
                                mul_p01_shift2, mul_p00_p01_shift2, mul_p11_shift2,
                                mul_p11_shift4, mul_p10_p11_shift2, mul_p00_p11_shift4,
                                mul_p10_shift2, mul_p01_shift2_p10_shift2;

wire [PE_OUTPUT_WIDTH - 1 : 0] all_partial_sums;
assign mul_p00 = weight[0 +: WEIGHTS_DATA_WIDTH/2] * activation[0 +: IACTS_DATA_WIDTH/2];
assign mul_p01 = weight[0 +: WEIGHTS_DATA_WIDTH/2] * activation[IACTS_DATA_WIDTH/2 +: IACTS_DATA_WIDTH/2];

assign mul_p10 = weight[WEIGHTS_DATA_WIDTH/2 +: WEIGHTS_DATA_WIDTH/2] * activation[0 +: IACTS_DATA_WIDTH/2];
assign mul_p11 = weight[WEIGHTS_DATA_WIDTH/2 +: WEIGHTS_DATA_WIDTH/2] * activation[IACTS_DATA_WIDTH/2 +: IACTS_DATA_WIDTH/2];

// If we are doing 2-bit precision
assign mul_p01_shift2 = mul_p01 << 2;
assign mul_p00_p01_shift2 = mul_p00_p01_shift2 + mul_p00;

assign mul_p11_shift2 = mul_p11 << 2;
assign mul_p10_p11_shift2 = mul_p11_shift2 + mul_p10;

// Now higher precision starts
assign mul_p11_shift4 = mul_p11_shift2 << 4;
assign mul_p00_p11_shift4 = mul_p00  + mul_p11_shift4;

assign mul_p10_shift2 = mul_p10 << 2;

assign mul_p01_shift2_p10_shift2 = mul_p10_shift2 + mul_p00_p01_shift2;

assign all_partial_sums = mul_p01_shift2_p10_shift2 + mul_p00_p11_shift4;

assign result = mode == 0 ? {mul_p10_p11_shift2, mul_p00_p01_shift2} : all_partial_sums;




    
endmodule