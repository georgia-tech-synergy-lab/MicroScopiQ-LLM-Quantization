python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method new_groupquant_channel --attention_number 40 --quantize_bit 4 --group_num 128 >gsm8kcot_groupQ128.txt
python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method new_groupquant_channel --attention_number 40 --quantize_bit 4 --group_num 64 >gsm8kcot_groupQ64.txt
python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method groupquantization_channel --attention_number 40 --quantize_bit 4 >gsm8kcot_groupQ_channel.txt

python evaluation_gsm8k.py --model meta-llama/Llama-2-7b-hf --prompt_file gsm8k_prompt_original.txt --batch_size 8 --max_new_tokens 256 --compress_method new_groupquant_channel --attention_number 40 --quantize_bit 4 --group_num 32 --streaming --streaming_gap 32 >gsm8kcot_groupQ32.txt

