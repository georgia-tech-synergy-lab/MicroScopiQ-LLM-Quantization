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

module fifo #(
    parameter DATA_WIDTH = 8,  // Width of the data entries
    parameter FIFO_DEPTH = 4,  // Depth of the FIFO, must be a power of 2
    parameter ADDR_WIDTH = 2   // Address width, derived from FIFO_DEPTH
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  write_en,   // Write enable signal
    input  wire                  read_en,    // Read enable signal
    input  wire [DATA_WIDTH-1:0] data_in,    // Data input
    output reg  [DATA_WIDTH-1:0] data_out,   // Data output
    output wire                  fifo_full,  // FIFO is full
    output wire                  fifo_empty  // FIFO is empty
);

  // Internal memory for FIFO
  reg [DATA_WIDTH-1:0] fifo_mem[FIFO_DEPTH-1:0];

  // Read and write pointers
  reg [ADDR_WIDTH-1:0] w_ptr;
  reg [ADDR_WIDTH-1:0] r_ptr;
  reg [ADDR_WIDTH:0] count;  // Extending count to ADDR_WIDTH+1 to manage full and empty states

  // Write operation
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      w_ptr <= 0;
    end else if (write_en && !fifo_full) begin
      fifo_mem[w_ptr] <= data_in;
      w_ptr <= (w_ptr + 1) & (FIFO_DEPTH - 1);
    end
  end

  // Read operation
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_ptr <= 0;
    end else if (read_en && !fifo_empty) begin
      data_out <= fifo_mem[r_ptr];
      r_ptr <= (r_ptr + 1) & (FIFO_DEPTH - 1);
    end
  end

  // Manage full and empty states
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      count <= 0;
    end else if (write_en && !fifo_full && !(read_en && !fifo_empty)) begin
      count <= count + 1;
    end else if (read_en && !fifo_empty && !(write_en && !fifo_full)) begin
      count <= count - 1;
    end
  end

  // Output status signals
  assign fifo_full  = (count == FIFO_DEPTH);
  assign fifo_empty = (count == 0);

endmodule
