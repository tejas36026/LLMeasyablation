!pip install transformers accelerate -q

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize model and tokenizer
model_dir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=torch.float16
)

def test_generation(prompt, max_length=30):
    """Test generation with given prompt"""
    print(f"\n🤖 GENERATING: '{prompt}'")
    print("-" * 40)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Result: {generated_text}")
    return generated_text

def find_microsoft_neurons_enhanced(prompt="Microsoft", top_k=5, threshold_ratio=0.1):
    """Find neurons with activation above a threshold"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    neuron_activations = {}
    
    def create_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation_output = output[0]
            else:
                activation_output = output
                
            if len(activation_output.shape) == 3:
                activations = activation_output[0, -1, :]  # Last token
            else:
                activations = activation_output[-1, :]
                
            # Store both positive and negative activations
            neuron_activations[layer_idx] = {
                'activations': activations.detach(),
                'abs_activations': activations.abs().detach()
            }
        return hook
    
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(create_hook(i)))
    
    with torch.no_grad():
        model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    # Find significant neurons using threshold
    significant_neurons = {}
    for layer_idx, data in neuron_activations.items():
        activations = data['activations']
        abs_activations = data['abs_activations']
        
        # Method 1: Top K neurons
        top_values, top_indices = torch.topk(abs_activations, top_k)
        
        # Method 2: Threshold-based (neurons above X% of max activation)
        threshold = threshold_ratio * abs_activations.max()
        above_threshold = abs_activations > threshold
        threshold_indices = torch.where(above_threshold)[0]
        
        significant_neurons[layer_idx] = {
            'topk_indices': top_indices.cpu().numpy(),
            'threshold_indices': threshold_indices.cpu().numpy(),
            'values': activations.cpu().numpy(),
            'abs_values': abs_activations.cpu().numpy()
        }
        
        print(f"Layer {layer_idx:2d}: {len(threshold_indices)} neurons above threshold")
    
    return significant_neurons

def ablate_neurons(neurons_data, strategy='threshold'):
    """Ablate (zero out) specific neurons in the model"""
    print("\n⚡ ABLATING MICROSOFT NEURONS")
    print("=" * 50)
    
    def ablation_hook_factory(layer_idx, neuron_indices):
        def hook(module, input, output):
            # output is a tuple, we need to modify the first element (hidden states)
            if isinstance(output, tuple):
                output_tuple = list(output)
                hidden_states = output_tuple[0]
                
                # Zero out the ablated neurons
                if len(hidden_states.shape) == 3:  # (batch, seq, hidden)
                    hidden_states[:, :, neuron_indices] = 0
                elif len(hidden_states.shape) == 2:  # (seq, hidden)
                    hidden_states[:, neuron_indices] = 0
                    
                output_tuple[0] = hidden_states
                return tuple(output_tuple)
            else:
                # If output is not a tuple, modify directly
                if len(output.shape) == 3:
                    output[:, :, neuron_indices] = 0
                elif len(output.shape) == 2:
                    output[:, neuron_indices] = 0
                return output
        return hook
    
    # Register ablation hooks
    ablation_hooks = []
    ablated_count = 0
    
    for layer_idx, neuron_data in neurons_data.items():
        if layer_idx < len(model.model.layers):
            # Get indices based on strategy
            if strategy == 'topk':
                indices = neuron_data['topk_indices']
            elif strategy == 'threshold':
                indices = neuron_data['threshold_indices']
            elif strategy == 'positive_only':
                # Only ablate positively activated neurons
                values = neuron_data['values']
                positive_mask = values > 0
                indices = np.where(positive_mask)[0]
            else:
                indices = neuron_data['threshold_indices']  # default
            
            if len(indices) > 0:
                hook = model.model.layers[layer_idx].register_forward_hook(
                    ablation_hook_factory(layer_idx, indices)
                )
                ablation_hooks.append(hook)
                print(f"Ablated {len(indices)} neurons in layer {layer_idx}")
                ablated_count += len(indices)
    
    print(f"Total neurons ablated: {ablated_count}")
    return ablation_hooks

def ablate_gradual(neurons_data, strength=0.5, strategy='threshold'):
    """Gradual ablation (reduce activation by percentage)"""
    print(f"\n⚡ GRADUAL ABLATION ({strength*100}% strength)")
    print("=" * 50)
    
    def gradual_hook_factory(layer_idx, neuron_indices, ablation_strength):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output_tuple = list(output)
                hidden_states = output_tuple[0]
                
                # Reduce activation by percentage instead of complete zeroing
                if len(hidden_states.shape) == 3:
                    hidden_states[:, :, neuron_indices] *= (1 - ablation_strength)
                elif len(hidden_states.shape) == 2:
                    hidden_states[:, neuron_indices] *= (1 - ablation_strength)
                    
                output_tuple[0] = hidden_states
                return tuple(output_tuple)
            return output
        return hook
    
    ablation_hooks = []
    ablated_count = 0
    
    for layer_idx, neuron_data in neurons_data.items():
        if layer_idx < len(model.model.layers):
            if strategy == 'topk':
                indices = neuron_data['topk_indices']
            elif strategy == 'threshold':
                indices = neuron_data['threshold_indices']
            elif strategy == 'positive_only':
                values = neuron_data['values']
                positive_mask = values > 0
                indices = np.where(positive_mask)[0]
            else:
                indices = neuron_data['threshold_indices']
            
            if len(indices) > 0:
                hook = model.model.layers[layer_idx].register_forward_hook(
                    gradual_hook_factory(layer_idx, indices, strength)
                )
                ablation_hooks.append(hook)
                print(f"Reduced {len(indices)} neurons in layer {layer_idx} by {strength*100}%")
                ablated_count += len(indices)
    
    print(f"Total neurons affected: {ablated_count}")
    return ablation_hooks

# Main experiment
print("🧪 NEURAL ABLATION EXPERIMENT: REMOVING MICROSOFT KNOWLEDGE")
print("=" * 60)

try:
    # Find Microsoft-related neurons
    microsoft_neurons = find_microsoft_neurons_enhanced()
    
    print(f"\n📍 Found Microsoft-related neurons in {len(microsoft_neurons)} layers")
    
    # Show neuron statistics
    total_threshold_neurons = 0
    for layer_idx, data in microsoft_neurons.items():
        threshold_count = len(data['threshold_indices'])
        total_threshold_neurons += threshold_count
        print(f"Layer {layer_idx:2d}: {threshold_count:3d} neurons above threshold")
    
    print(f"Total neurons above threshold: {total_threshold_neurons}")

    print("\n" + "🔷 BEFORE ABLATION - NORMAL MODEL")
    normal_output = test_generation("Microsoft ")
    normal_output2 = test_generation("The company Microsoft")
    normal_output3 = test_generation("Bill Gates founded")

    # Test different ablation strategies
    print("\n" + "🎯 TESTING DIFFERENT ABLATION STRATEGIES")
    print("=" * 50)
    
    # Strategy 1: Complete ablation (threshold neurons)
    print("\n🔶 STRATEGY 1: COMPLETE ABLATION (threshold neurons)")
    ablation_hooks = ablate_neurons(microsoft_neurons, strategy='threshold')
    ablated_output1 = test_generation("Microsoft ")
    ablated_output2 = test_generation("The company Microsoft")
    
    # Remove hooks
    for hook in ablation_hooks:
        hook.remove()

    # Strategy 2: Gradual ablation (50% strength)
    print("\n🔶 STRATEGY 2: GRADUAL ABLATION (50% strength)")
    gradual_hooks = ablate_gradual(microsoft_neurons, strength=0.5, strategy='threshold')
    gradual_output = test_generation("Microsoft ")
    
    for hook in gradual_hooks:
        hook.remove()
    
    # Strategy 3: Top-K neurons only
    print("\n🔶 STRATEGY 3: TOP-K NEURONS ONLY")
    topk_hooks = ablate_neurons(microsoft_neurons, strategy='topk')
    topk_output = test_generation("Microsoft ")
    
    for hook in topk_hooks:
        hook.remove()

    # Final test with complete ablation
    print("\n" + "🔶 FINAL COMPLETE ABLATION TEST")
    final_hooks = ablate_neurons(microsoft_neurons, strategy='threshold')
    
    # Test various Microsoft-related prompts
    test_results = {}
    test_prompts = [
        "Microsoft develops",
        "Microsoft is a",
        "Windows is",
        "Bill Gates is",
        "Azure cloud",
        "Xbox gaming"
    ]
    
    for prompt in test_prompts:
        test_results[prompt] = test_generation(prompt)

    # Test other companies (control group)
    print("\n" + "🔍 CONTROL GROUP: OTHER COMPANIES")
    print("=" * 40)
    control_prompts = ["Apple is", "Google is", "Amazon is", "Facebook is"]
    for prompt in control_prompts:
        test_generation(prompt)

    # Remove ablation hooks to restore model
    for hook in final_hooks:
        hook.remove()

    # Compare results
    print("\n" + "📊 COMPARISON SUMMARY")
    print("=" * 50)
    print("BEFORE ABLATION:")
    print(f"  'Microsoft ' → {normal_output}")
    print(f"  'The company Microsoft' → {normal_output2}")
    print("\nAFTER COMPLETE ABLATION:") 
    print(f"  'Microsoft ' → {ablated_output1}")
    print(f"  'The company Microsoft' → {ablated_output2}")
    print("\nAFTER GRADUAL ABLATION (50%):")
    print(f"  'Microsoft ' → {gradual_output}")
    
    print("\nAFTER TOP-K ABLATION:")
    print(f"  'Microsoft ' → {topk_output}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
