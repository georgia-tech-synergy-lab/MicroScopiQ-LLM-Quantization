# Run the FFN layer with MXFP6_e3m2
python scratch_3.py --w_elem_format "fp6_e3m2" --a_elem_format "fp6_e3m2" --scale_bits 4 --block_size 32 --bfloat 16

# Note that ffn_mx_auto.py is hardcoded with the above config.