python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 128 --group_size 64 > KIVI_64_st128.txt

python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --streaming --streaming_gap 128 --group_size 32 > KIVI_32_st128.txt

python evaluation_bbh_cot.py --model meta-llama/Llama-2-7b-hf --batch_size 4 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --group_size 128 --streaming --streaming_gap 128 > bbh_cot_KIVI_128.txt

python evaluation_bbh_cot.py --model meta-llama/Llama-2-7b-hf --batch_size 4 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --group_size 64 --streaming --streaming_gap 64 > bbh_cot_KIVI_64.txt


# python evaluation_bbh_cot.py --model meta-llama/Llama-2-7b-hf --batch_size 4 --max_new_tokens 256 --compress_method KIVI --attention_number 40 --quantize_bit 4 --group_size 32 --streaming --streaming_gap 32 > bbh_cot_KIVI_32.txt