kvcache_avg = [945,1330,1048]
model_dict = {"llama2-13b":5120,"Llama3b":3200,"Mistral":4096}
head_dict = {"llama2-13b":128,"Llama3b":100,"Mistral":128}
model_name_list = ["Llama3b","llama2-13b", "Mistral"]
sparsity = 0.00
bit = 2
backbone = "KIVI"
residual = 64
group_size = 64
rank = 4
head_dim = 128
base_list = []
for model_name in model_name_list:
    for kv_avg_length in kvcache_avg:
        quant_base = bit / 16
        if backbone == "KIVI":
            mn_scale_base = 2 / group_size
            residual_base = residual / kv_avg_length
        elif backbone == "KCVT":
            mn_scale_base = 2 / kv_avg_length + 2 / model_dict[model_name]
            residual_base = residual / kv_avg_length
        if rank != 0:
            rank_base = rank * (head_dict[model_name] + kv_avg_length) / (head_dict[model_name] * kv_avg_length) / 2
        else:
            rank_base = 0.0
        if sparsity != 0:
            if bit == 4:
                sparsity_base = sparsity
            else:
                sparsity_base = sparsity * 2
        else:
            sparsity_base = 0.0
        base_all = quant_base + mn_scale_base + residual_base + rank_base + sparsity_base
        base_list.append(base_all)
    print(base_list)
avg_base = sum(base_list)  / len(base_list)
print(avg_base)
    
    
    
    