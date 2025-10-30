from pytorch_pretrained_vit import ViT
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from transformers import BertModel, LlamaForCausalLM, AutoModelForCausalLM, GPTNeoForCausalLM, AutoModelForSeq2SeqLM

# Function to calculate outlier statistics for each channel using the 3-sigma rule
def analyze_outliers_olive(model):
    outlier_counts = []
    outlier_positions = {}
    layer_index = 0
    adjacent_outliers_counts = []

    # Iterate through layers
    for layer_name, layer in (model.named_parameters()):
            if 'weight' in layer_name or "proj" in layer_name:
                weights = layer.data.cpu().numpy()
                # Calculate mean and standard deviation
                mean = np.mean(weights)
                std = np.std(weights)
                # print(mean, std)

                # Identify outliers using the 3-sigma rule
                outliers = np.abs(weights - mean) > 2.5 * std
                # print("SHAPES=",weights.shape, outliers.shape)
                # Count number of outliers in each channel
                # if(len(outliers.shape) == 1):
                #      channel_outliers = np.sum(outliers, axis = 0)
                # else:
                #print(np.unique(outliers))
                channel_outliers = np.count_nonzero(outliers)
                outlier_counts.append(channel_outliers)
                outlier_positions[layer_index] = np.where(outliers)
                adjacent_outliers = np.abs(np.diff(np.where(outliers > 0))) == 1
                adjacent_outliers_count = np.sum(adjacent_outliers)
                adjacent_outliers_counts.append(adjacent_outliers_count)
                
                total_weights = reduce(lambda x,y: x*y, weights.shape)

                print(f"Layer {layer_index}: Count of Adjacent Outliers: {adjacent_outliers_counts[layer_index]} out of total outliers: {outlier_counts[layer_index]}")
                print(f"Percentage stats: Adjacent Percentage: {adjacent_outliers_counts[layer_index]/outlier_counts[layer_index]}, Total Outlier Percentage: {outlier_counts[layer_index]/total_weights}")
                layer_index += 1

    return outlier_counts, outlier_positions


def check_closely_located_outliers(outlier_positions):
    for layer_index, positions in outlier_positions.items():
        if positions[0].size > 0:
            print(f"Outliers in Layer {layer_index + 1}: {positions}")




if __name__ == "__main__":
    llm_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-base", local_files_only=True)
    #GPTNeoForCausalLM.from_pretrained("./gpt-neo-125m", local_files_only=True)  #AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", torch_dtype=torch.int8, attn_implementation="flash_attention_2")
    #
    #AutoModelForCausalLM.from_pretrained("microsoft/phi-2")#LlamaForCausalLM.from_pretrained("./")
    #llm_model = BertModel.from_pretrained("bert-base-cased")
    vit_model = ViT('L_32', pretrained=True)
    # Analyze outliers and check closely located outliers
    outlier_counts, outlier_positions = analyze_outliers_olive(llm_model)
    # print(outlier_counts)
    #check_closely_located_outliers(outlier_positions)