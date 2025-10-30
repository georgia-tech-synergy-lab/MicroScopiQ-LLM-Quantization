# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization_channel --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20 > group4bit.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization_kc_vt --attention_number 40 --quantize_bit 2 --streaming --streaming_gap 20  > kivi2bit.txt

# python evaluation_mmlu_cot.py --model meta-llama/Llama-2-7b-hf --batch_size 4 --max_new_tokens 256 --compress_method quantize_with_lrap --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20  > mmlu.txt


# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outliterquantize_with_lrap --attention_number 40 --quantize_bit 2 --left 0.05 --rank 0.01 --rankv 0.03 --loop 3 --streaming --streaming_gap 20 > gear2bit.txt


# python evaluation_mmlu_cot.py --model meta-llama/Llama-2-7b-hf --batch_size 4 --max_new_tokens 256 --compress_method group_channel_with_lrap --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20  > mmlu_groupq_lrap4bit.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outliterquantize_with_lrap --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 128 --group_num 128 > KIVI_128.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > gearl_channel_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > gearl_token_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > gearl_kcvt_4b_2s_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method outliterquantize_with_lrap --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 128 --group_num 128 > KIVI_128.txt


# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > gears_channel_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > gears_token_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > gears_kcvt_4b_2s_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARL --attention_number 40 --quantize_bit 4 --group_size 32 --streaming --streaming_gap 32 > gearl_KIVI_4b_32.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARL --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > gearl_KIVI_4b_64.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --group_size 128 --streaming --streaming_gap 128 > gearl_KIVI_4b_128.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > gearl_channel_4b_44_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > gearl_token_4b_44_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > gearl_kcvt_4b_44_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.06 --loop 3 --streaming --streaming_gap 20 > gearl_channel_4b_26_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.06 --loop 3 --streaming --streaming_gap 20 > gearl_token_4b_26_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.06 --loop 3 --streaming --streaming_gap 20 > gearl_kcvt_4b_26_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > gearl_channel_4b_24_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > gearl_token_4b_24_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > gearl_kcvt_4b_24_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 > kcvt_4b_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --group_size 32 --stremaing --streaming_gap 32 

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARS --attention_number 40 --quantize_bit 3 --left 0.10 --streaming --streaming_gap 20 > gears_channel_3b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARS --attention_number 40 --quantize_bit 3 --left 0.10 --streaming --streaming_gap 20 > gears_token_3b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS --attention_number 40 --quantize_bit 3 --left 0.10 --streaming --streaming_gap 20 > gears_kcvt_3b_2s_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARS --attention_number 40 --quantize_bit 2 --left 0.10 --streaming --streaming_gap 20 > gears_channel_4b_10s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARS --attention_number 40 --quantize_bit 2 --left 0.10 --streaming --streaming_gap 20 > gears_token_4b_10s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS --attention_number 40 --quantize_bit 2 --left 0.05 --streaming --streaming_gap 20 > gears_kcvt_4b_5s_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 512 --compress_method None --attention_number 40 > llama3_gsm8k.txt


# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > llama3_gears_channel_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > llama3_gears_token_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > llama3_gears_kcvt_4b_2s_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 128 > llama3_KIVI_4b_64_128.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 128 > test.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 > llama3_kcvt_4b_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method None --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > llama3_gsm8k.txt


# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 10 > gearl_channel_4b_44_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 10 > gearl_token_4b_44_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 10 > gearl_kcvt_4b_44_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method channelQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 10 > gears_channel_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method tokenQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 10 > gears_token_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS --attention_number 40 --quantize_bit 4 --left 0.02 --streaming --streaming_gap 10 > gears_kcvt_4b_2s_gsm8k.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARSL --attention_number 40 --quantize_bit 4 --left 0.02 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > llama3_gsm8k_gearsl_0.02.txt
# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixedGEARS+L --attention_number 40 --quantize_bit 4 --left 0.02 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 40  > llama3_bbh_gearsl_0.02.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS+L --attention_number 40 --quantize_bit 3 --left 0.02 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > llama3_gsm8k_gearsl_0.02_3b.txt
# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixedGEARS+L --attention_number 40 --quantize_bit 3 --left 0.02 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 40  > llama3_bbh_gearsl_0.02_3b.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARS+L --attention_number 40 --quantize_bit 3 --left 0.02 --rank 0.02 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > llama3_gsm8k_gearsl_0.024_3b.txt
# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixedGEARS+L --attention_number 40 --quantize_bit 3 --left 0.02 --rank 0.02 --rankv 0.04 --loop 3 --streaming --streaming_gap 40  > llama3_bbh_gearsl_0.024_3b.txt

# # python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARSL --attention_number 40 --quantize_bit 2 --left 0.02 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > llama3_2bit_kvctgearsl.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 3 --group_size 64 --streaming --streaming_gap 64 > llama3_gsm8k_kivi3bit_64.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQGEARL --attention_number 40 --quantize_bit 3 --group_size 64 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 64 > llama3_gsm8k_gearlkivi_0.04_0.04_64_3bit.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 3 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 64 > llama3_gsm8k_gearlkcvt_0.04_0.04_3bit.txt


# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20 > llama3_gsm8k_gearlkcvt_4bit.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 3 --streaming --streaming_gap 20 > llama3_gsm8k_gearlkcvt_3bit.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 2 --streaming --streaming_gap 20 > llama3_gsm8k_gearlkcvt_2bit.txt


# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 4 --rankv 2 --loop 3 --streaming --streaming_gap 64 > llama3_gsm8k_gearlkcvt_4bit_4_2.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 3 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 40 > llama3_gsm8k_gearlkcvt_3bit_0.04.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 2 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 40 > llama3_gsm8k_gearlkcvt_2bit_0.04.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > llama3_gsm8k_gearlkcvt_4bit_0.02_0.02.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --loop 3 --streaming --streaming_gap 20 > llama3_gsm8k_gearlkcvt_4bit_0.04_0.04.txt
# python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20 > llama13_aqua_kcvt_4bit.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20 > llama3_gsm8k_kcvt_4bit.txt

# python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40  > bbh_kcvt_4_llama13b.txt

# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40  > bbh_kcvt_4_llama8b.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 20 > llama3_gsm8k_kcvt_4bit.txt
# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method kcvtQfixedGEARL --attention_number 40 --quantize_bit 4 --rank 0.04 --rankv 0.04 --streaming --loop 3 --streaming_gap 20 > llama3_gsm8k_kcvt_0.04_0.04_4bit.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > llama3_8b_gsm8k_flexgen_64_64_4bit.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > llama2_13b_gsm8k_flexgen_64_64_4bit.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64 > llama3_8b_gsm8k_flexgen_64_64_2bit.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64 > llama2_13b_gsm8k_flexgen_64_64_2bit.txt

# python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method Flexgen --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > llama2_13b_aqua_flexgen_64_64_4bit.txt

# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method Flexgen --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > llama3_8b_aqua_flexgen_64_64_4bit.txt

# python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64 > llama2_13b_aqua_flexgen_64_64_2bit.txt

# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64 > llama3_8b_aqua_flexgen_64_64_2bit.txt

# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64  > llama3_8b_bbh_flexgen_64_64_4bit.txt

# python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 4 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64  > llama2_13b_bbh_flexgen_64_64_4bit.txt

# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64  > llama3_8b_bbh_flexgen_64_64_2bit.txt

# python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 4 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64  > llama2_13b_bbh_flexgen_64_64_2bit.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method GEARSLKIVI --attention_number 40 --quantize_bit 2 --left 0.02 --rank 4 --rankv 4 --loop 3 --group_size 64 --streaming --streaming_gap 64 > llama3_8b_gsm8k_gearskivi_0.02_64_64_2bit.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method GEARSKIVI --attention_number 40 --quantize_bit 2 --left 0.02 --group_size 64 --streaming --streaming_gap 64 > llama2_13b_gsm8k_gearskivi_0.02_64_64_2bit.txt

# python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 4 --max_new_tokens 256 --compress_method GEARSKIVI --attention_number 40 --quantize_bit 2 --group_size 64 --left 0.02 --streaming --streaming_gap 64  > bbh_gearskivi_2_64_64_0.02_llama13b.txt

# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method GEARSKIVI --attention_number 40 --quantize_bit 2 --group_size 64 --left 0.02 --streaming --streaming_gap 64  > bbh_gearskivi_2_64_64_0.02_llama3_8b.txt

#########
# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --group_size 64 --rank 4 --rankv 4 --loop 3 --streaming --streaming_gap 64 > llama3-8B_newgearlkivi_aqua.txt # 33.86

# python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --group_size 64 --rank 4 --rankv 4 --loop 3 --streaming --streaming_gap 64 > llama2-13B_newgearlkivi_aqua.txt # 21.26

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --rank 4 --rankv 4 --loop 3 --group_size 64 --streaming --streaming_gap 64 > llama3_8b_gsm8k_newgearlkivi_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64 > llama2_13b_gsm8k_newgearlkivi_gsm8k.txt


# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKCVT --attention_number 40 --quantize_bit 4 --rank 4 --rankv 4 --loop 3 --streaming --streaming_gap 20 > llama3-8B_newgearlkcvt_aqua.txt 

# python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKCVT --attention_number 40 --quantize_bit 4 --rank 4 --rankv 4 --loop 3 --streaming --streaming_gap 20 > llama2-13B_newgearlkcvt_aqua.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKCVT --attention_number 40 --quantize_bit 4 --rank 4 --rankv 4 --loop 3 --streaming --streaming_gap 20 > llama3_8b_gsm8k_newgearlkcvt_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKCVT --attention_number 40 --rank 4 --rankv 4 --loop 3 --quantize_bit 4 --streaming --streaming_gap 20 > llama2_13b_gsm8k_newgearlkcvt_gsm8k.txt
######
# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --group_size 64 --prefillrank 4 --prefillrankv 4 --rank 4 --rankv 4 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3-8B_newgearlkivi_aqua.txt # 33.86

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 4 --rankv 4 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3_8b_gsm8k_newgearlkcvt_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 4 --rankv 4 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama2_13b_gsm8k_newgearlkcvt_gsm8k.txt

#### BBH
# python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 4 --max_new_tokens 256 --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64  > llama2_13b_bbh_newgearslkivi_2bit_64_4rank.txt

# python evaluation_bbh_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 4 --max_new_tokens 256 --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3_8b_bbh_newgearslkivi_2bit_64_4rank.txt

# python evaluation_aqua_cot.py --model meta-llama/Meta-Llama-3-8B --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --left 0.02 --group_size 64 --stream_grouping --streaming --streaming_gap 64 > llama3-8B_newgearslkivi_aqua.txt 

# python evaluation_aqua_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --left 0.02 --group_size 64 --stream_grouping --streaming --streaming_gap 64 > llama2-13B_newgearslkivi_aqua.txt

# python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --left 0.01 --group_size 64 --stream_grouping --streaming --streaming_gap 64 > llama3_8b_gsm8k_newgearslkivi_gsm8k.txt

# python evaluation_gsm8k.py --model meta-llama/Llama-2-13b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --left 0.01 --group_size 64 --stream_grouping --streaming --streaming_gap 64 > llama2_13b_gsm8k_newgearslkivi_gsm8k.txt



# python evaluation_aqua_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method None > Aqua_mistral7b.txt

# python evaluation_aqua_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --left 0.02 --group_size 64 --stream_grouping --streaming --streaming_gap 64 > mistral7b_newgearslkivi_aqua.txt

# python evaluation_aqua_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --group_size 64 --stream_grouping --streaming --streaming_gap 64 > mistral7b_newgearlkivi_aqua.txt

# python evaluation_aqua_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 6 --max_new_tokens 196 --model_max_length 4096 --root_output_dir ./aqua --compress_method NEWGEARLKCVT --attention_number 40 --quantize_bit 4 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --streaming_gap 20 > mistral7b_newgearlkcvt_aqua.txt

# python evaluation_gsm8k.py --model mistralai/Mistral-7B-v0.1 --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64 > mistral7b_gsm8k_newgearlkivi_gsm8k.txt

# python evaluation_gsm8k.py --model mistralai/Mistral-7B-v0.1 --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARLKCVT --attention_number 40 --quantize_bit 4 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64 > mistral7b_gsm8k_newgearlkcvt_gsm8k.txt

# python evaluation_gsm8k.py --model mistralai/Mistral-7B-v0.1 --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method NEWGEARSLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --left 0.02 --streaming --stream_grouping --streaming_gap 64 > mistral7b_gsm8k_newgearslkivi_gsm8k.txt


# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 40  > mistral_bbh_kcvt_4bit.txt

# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --stream_grouping --streaming_gap 64  > mistral_bbh_kivi_4bit_64.txt

# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method NEWGEARLKCVT --attention_number 40 --quantize_bit 4 --prefillrank 4 --prefillrankv 4 --rank 4 --rankv 4 --loop 3 --streaming --streaming_gap 40  > mistral_bbh_newgearlkcvt_4bit.txt

# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method NEWGEARSLKCVT --attention_number 40 --quantize_bit 4 --prefillrank 4 --prefillrankv 4 --rank 4 --rankv 4 --loop 3 --left 0.02 --streaming --streaming_gap 40  > mistral_bbh_newgearslkcvt_4bit.txt

# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method Flexgen --attention_number 40 --quantize_bit 2 --group_size 64 --streaming --streaming_gap 64  > mistral_bbh_flexgen_2bit_64.txt

# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --group_size 64 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64  > mistral_bbh_newgearlkivi_2bit_64.txt

# python evaluation_bbh_cot.py --model mistralai/Mistral-7B-v0.1 --batch_size 4 --max_new_tokens 256 --compress_method NEWGEARSLKIVI --attention_number 40 --group_size 64 --left 0.02 --quantize_bit 2 --prefillrank 4 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64  > mistral_bbh_newgearslkivi_2bit_64.txt

# python evaluation_bbh_cot.py --model meta-llama/Llama-2-13b-hf --batch_size 4 --max_new_tokens 256 --compress_method kcvtQfixed --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64  > llama2_13b_bbh_newgearslkivi_2bit_64_4rank.txt


python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 12 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 2 --group_size 64 --prefillrankv 2 --rank 2 --rankv 2 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3_gsm8k_2r.txt
python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 12 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 4 --group_size 64 --prefillrankv 4 --rank 4 --rankv 4 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3_gsm8k_4r.txt
python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 12 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 6 --group_size 64 --prefillrankv 6 --rank 6 --rankv 6 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3_gsm8k_6r.txt
python evaluation_gsm8k.py --model meta-llama/Meta-Llama-3-8B --prompt_file gsm8k_prompt_original.txt --batch_size 12 --max_new_tokens 256 --compress_method NEWGEARLKIVI --attention_number 40 --quantize_bit 2 --prefillrank 8 --group_size 64 --prefillrankv 8 --rank 8 --rankv 8 --loop 3 --streaming --stream_grouping --streaming_gap 64 > llama3_gsm8k_8r.txt














