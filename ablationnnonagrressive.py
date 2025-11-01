import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gdown
import os
import zipfile

def download_and_extract_model(model_file_id: str, output_path: str):
    """
    Downloads a model zip file from Google Drive and extracts it.
    """
    zip_path = "/content/model.zip"

    # Check if already extracted
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f"✅ Model already found at: {output_path}")
        return output_path

    print("📂 Downloading model zip file from Google Drive...")

    try:
        # Download the zip file using the direct download link
        url = f"https://drive.google.com/uc?id={model_file_id}"
        gdown.download(url, zip_path, quiet=False)

        if not os.path.exists(zip_path):
            print(f"❌ Failed to download zip file")
            return None

        print(f"✅ Download complete: {zip_path}")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Extract the zip file
        print("📦 Extracting model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)

        print(f"✅ Extraction complete: {output_path}")

        # List extracted files
        print(f"📁 Contents of extracted model:")
        model_files = os.listdir(output_path)
        for file in model_files:
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   📄 {file} ({size:,} bytes)")
            else:
                print(f"   📁 {file}/")

        # Clean up zip file
        os.remove(zip_path)
        print("🧹 Cleaned up zip file")

        return output_path

    except Exception as e:
        print(f"🚨 Download/extraction error: {e}")
        return None
def download_small_test_model():
    """
    Download a small test model if the main model fails.
    """
    print("🔄 Falling back to small test model...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import requests

        model_path = "/content/test_model"
        os.makedirs(model_path, exist_ok=True)

        print("📥 Downloading TinyLlama model...")
        
        # ADD THESE LINES FOR BETTER VISIBILITY:
        print("⏳ This may take 2-5 minutes for 1.1B model...")
        print("📊 Downloading tokenizer and model files...")
        
        # Add timeout and progress indication
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Add this line
        )

        # Save locally
        print("💾 Saving model locally...")
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)

        print(f"✅ Test model saved to: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Failed to download test model: {e}")
        return None



def generate_response_with_comparison(original_model, ablated_model, tokenizer, prompt, device):
    """
    Generate response showing both BEFORE and AFTER ablation for every input.
    """
    print(f"\n🔍 RESPONSE COMPARISON FOR: '{prompt}'")
    print("=" * 60)
    model_dtype = next(original_model.parameters()).dtype

    # Generate BEFORE ablation response
    print("🟢 BEFORE ABLATION:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs.input_ids = inputs.input_ids.to(model_dtype)
    if inputs.attention_mask is not None:
        inputs.attention_mask = inputs.attention_mask.to(model_dtype)

    # ADD THESE 3 PRINT STATEMENTS HERE:
    print(f"🧠 LAYER TYPES ACTIVATED: {[module.__class__.__name__ for name, module in original_model.named_modules() if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'attention', 'mlp'])]}")
    print(f"🔢 TENSOR LAYERS SHAPES: {[f'{name}: {list(module.weight.shape) if hasattr(module, "weight") else "No weight"}' for name, module in original_model.named_modules() if hasattr(module, 'weight')][:10]}")  # First 10 only
    print(f"🕸️ NEURAL NETWORK PATHS: {[name for name, module in original_model.named_modules() if any(key in name.lower() for key in ['attention', 'mlp', 'feedforward'])][:15]}")  # First 15 only
    
    with torch.no_grad():
        outputs_before = original_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    response_before_clean = response_before.replace(prompt, "").strip()
    print(f"   {response_before_clean}")
    
    # Generate AFTER ablation response
    print("🔴 AFTER ABLATION:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # ADD THE SAME 3 PRINT STATEMENTS FOR ABLATED MODEL:
    print(f"🧠 LAYER TYPES ACTIVATED: {[module.__class__.__name__ for name, module in ablated_model.named_modules() if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'attention', 'mlp'])]}")
    print(f"🔢 TENSOR LAYERS SHAPES: {[f'{name}: {list(module.weight.shape) if hasattr(module, "weight") else "No weight"}' for name, module in ablated_model.named_modules() if hasattr(module, 'weight')][:10]}")
    print(f"🕸️ NEURAL NETWORK PATHS: {[name for name, module in ablated_model.named_modules() if any(key in name.lower() for key in ['attention', 'mlp', 'feedforward'])][:15]}")
    
    with torch.no_grad():
        outputs_after = ablated_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    response_after_clean = response_after.replace(prompt, "").strip()
    print(f"   {response_after_clean}")
    
    print("=" * 60)
    
    return response_after_clean




def download_text_file(file_id: str, output_path: str):
    """
    Downloads a text file from Google Drive.
    """
    if os.path.exists(output_path):
        print(f"✅ Text file found at: {output_path}")
        return output_path

    print("📂 Downloading text file from Google Drive...")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"✅ Download complete: {output_path}")
        return output_path
    except Exception as e:
        print(f"🚨 Download error: {e}")
        return None

def extract_text_from_file(file_path):
    """
    Extract text from the downloaded file. Returns text and list of words.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read all lines and strip whitespace
            lines = [line.strip() for line in f.readlines()]
            # Remove empty lines
            lines = [line for line in lines if line]

        print(f"✅ Extracted {len(lines)} words/phrases from {file_path}")

        # Show first 10 items as preview
        print(f"📝 Preview (first 10 items):")
        for i, line in enumerate(lines[:10], 1):
            print(f"   {i}. {line}")
        if len(lines) > 10:
            print(f"   ... and {len(lines) - 10} more")

        # Return as single text (for tokenization) and word list
        return ' '.join(lines), lines
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return "", []

def get_tokens_from_text(text, tokenizer, word_list=None):
    """
    Extract unique tokens from COMPLETE WORDS only.
    Only ablates tokens that represent full, specific words/phrases.
    """
    unique_token_ids = set()

    if word_list:
        print(f"\n📝 Analyzing {len(word_list)} words to find complete token matches...")

        # Filter: Only include tokens for COMPLETE single-token words
        # Or multi-token words where we want all parts ablated
        single_token_words = []
        multi_token_words = []

        for word in word_list:

            word_tokens = tokenizer.encode(" " + word.strip(), add_special_tokens=False)

            if len(word_tokens) == 1:
                # Single complete token - safe to ablate
                single_token_words.append((word, word_tokens[0]))
                unique_token_ids.add(word_tokens[0])
            elif len(word_tokens) == 2 and len(word) >= 5:
                # Two-token word that's long enough to be specific
                multi_token_words.append((word, word_tokens))
                unique_token_ids.update(word_tokens)

        print(f"📊 Found {len(single_token_words)} single-token words")
        print(f"📊 Found {len(multi_token_words)} multi-token words")
        print(f"📊 Total unique token IDs to ablate: {len(unique_token_ids)}")

        # Show examples
        print(f"\n📝 Examples of single-token words to ablate:")
        for word, token_id in single_token_words[:5]:
            print(f"   '{word}' → Token {token_id} ('{tokenizer.decode([token_id])}')")

        if multi_token_words:
            print(f"\n📝 Examples of multi-token words to ablate:")
            for word, tokens in multi_token_words[:3]:
                decoded = [tokenizer.decode([t]) for t in tokens]
                print(f"   '{word}' → Tokens {tokens} ({decoded})")

    return unique_token_ids


def test_model_before_ablation(model, tokenizer, test_words, device):
    """
    Test the model BEFORE ablation to see normal outputs with detailed layer analysis.
    """
    print("\n" + "="*60)
    print("TESTING MODEL BEFORE ABLATION")
    print("="*60)

    # ========== COMPREHENSIVE LAYER ANALYSIS ==========
    print("\n🧠 NEURAL NETWORK ARCHITECTURE ANALYSIS (BEFORE ABLATION)")
    print("-" * 50)
    
    # Analyze model layers and structure
    layer_types = {}
    total_layers = 0
    attention_layers = []
    mlp_layers = []
    transformer_blocks = []
    
    print("📊 DETECTED NEURAL NETWORK LAYERS:")
    for name, module in model.named_modules():
        layer_type = module.__class__.__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        total_layers += 1
        
        # Categorize important layers
        if 'attention' in name.lower():
            attention_layers.append((name, layer_type))
        elif 'mlp' in name.lower() or 'feedforward' in name.lower():
            mlp_layers.append((name, layer_type))
        elif any(block in name.lower() for block in ['block', 'layer']):
            if name.count('.') <= 3:  # Main blocks only
                transformer_blocks.append((name, layer_type))
    
    # Print layer statistics
    print(f"📈 TOTAL LAYERS DETECTED: {total_layers}")
    print(f"🎯 ATTENTION LAYERS: {len(attention_layers)}")
    print(f"🔧 MLP/FEEDFORWARD LAYERS: {len(mlp_layers)}")
    print(f"🏗️  TRANSFORMER BLOCKS: {len(transformer_blocks)}")
    
    # Show top layer types
    print(f"\n🏆 TOP LAYER TYPES:")
    for layer_type, count in sorted(layer_types.items(), key=lambda x: x[1], reverse=True)[:8]:
        percentage = (count / total_layers) * 100
        stars = "⭐" * min(count // max(1, total_layers // 20), 5)
        print(f"   {layer_type}: {count} layers ({percentage:.1f}%) {stars}")

    # ========== NEURAL NETWORK LINKAGE ANALYSIS ==========
    print(f"\n🔗 NEURAL NETWORK LINKAGE PATTERNS:")
    print("-" * 40)
    
    # Analyze transformer block structure
    if transformer_blocks:
        print("🏗️  TRANSFORMER BLOCK HIERARCHY:")
        for block_name, block_type in transformer_blocks[:6]:  # Show first 6
            depth = block_name.count('.')
            indent = "  " * depth
            print(f"   {indent}📦 {block_name} → {block_type}")
        if len(transformer_blocks) > 6:
            print(f"   ... and {len(transformer_blocks) - 6} more blocks")
    
    # Attention layer analysis
    if attention_layers:
        print(f"\n👁️  ATTENTION MECHANISM STRUCTURE:")
        for attn_name, attn_type in attention_layers[:4]:
            print(f"   🔍 {attn_name} → {attn_type}")
        if len(attention_layers) > 4:
            print(f"   ... and {len(attention_layers) - 4} more attention layers")
    
    # MLP layer analysis
    if mlp_layers:
        print(f"\n⚡ FEEDFORWARD NETWORK STRUCTURE:")
        for mlp_name, mlp_type in mlp_layers[:4]:
            print(f"   🧩 {mlp_name} → {mlp_type}")
        if len(mlp_layers) > 4:
            print(f"   ... and {len(mlp_layers) - 4} more MLP layers")

    # ========== PARAMETER ANALYSIS ==========
    print(f"\n📊 MODEL PARAMETER DISTRIBUTION:")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧮 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    print(f"📈 Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Parameter distribution by layer type
    embed_params = 0
    attention_params = 0
    mlp_params = 0
    norm_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        if 'embed' in name:
            embed_params += param_count
        elif 'attention' in name:
            attention_params += param_count
        elif 'mlp' in name or 'feedforward' in name:
            mlp_params += param_count
        elif 'norm' in name or 'ln' in name:
            norm_params += param_count
    
    print(f"\n📋 PARAMETER DISTRIBUTION:")
    if embed_params > 0:
        print(f"   🔤 Embedding Layers: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
    if attention_params > 0:
        print(f"   👁️  Attention Layers: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
    if mlp_params > 0:
        print(f"   ⚡ MLP Layers: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
    if norm_params > 0:
        print(f"   📐 Normalization Layers: {norm_params:,} ({norm_params/total_params*100:.1f}%)")

    # ========== MODEL INFERENCE TESTING ==========
    print(f"\n🧪 MODEL INFERENCE TESTING WITH SAMPLE WORDS:")
    print("-" * 50)

    model.eval()

    for i, word in enumerate(test_words, 1):
        print(f"\n🔍 TEST {i}: '{word}'")
        print(f"   {'─' * (len(word) + 10)}")

        # Tokenize the word
        inputs = tokenizer(word, return_tensors="pt").to(device)
        input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"   🎯 Input tokens: {input_tokens}")
        print(f"\nPRE-ABLATION NEURAL NETWORK SUMMARY:")
        with torch.no_grad():
            try:
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_tokens = tokenizer.convert_ids_to_tokens(outputs[0])
                
                print(f"   💭 Generated: {response}")
                print(f"   🔤 Output tokens: {response_tokens[:10]}{'...' if len(response_tokens) > 10 else ''}")
                print(f"   📏 Response length: {len(response_tokens)} tokens")
                
                print("\n" + "─" * 60)
                print("🎯 DETAILED LAYER ACCESS & HIERARCHY INSPECTION")
                print("─" * 60)
                print(f"   model.layers.0.mlp → {model.model.layers[0].mlp}")
                print(f"   model.layers.0.post_attention_layernorm → {model.model.layers[0].post_attention_layernorm}")
                # Print specific layer: model.layers.0.mlp
                print(f"   📦 model.layers.0.mlp → {model.model.layers[0].mlp}")
                print("\nTRANSFORMER BLOCK HIERARCHY (first 2 layers):")
                for i in range(min(2, len(model.model.layers))):
                    l = model.model.layers[i]
                    print(f"   model.layers.{i} → {l.__class__.__name__}")
                    print(f"     ├─ self_attn → {l.self_attn.__class__.__name__}")
                    print(f"     ├─ input_layernorm → {l.input_layernorm.__class__.__name__}")
                    print(f"     ├─ mlp → {l.mlp.__class__.__name__}")
                    print(f"     └─ post_attention_layernorm → {l.post_attention_layernorm.__class__.__name__}")
               
                print("\nDETECTED NEURAL NETWORK LAYERS:")
                print(f"TOTAL LAYERS DETECTED: {total_layers}")
                print(f"ATTENTION LAYERS: {len(attention_layers)}")
                print(f"MLP/FEEDFORWARD LAYERS: {len(mlp_layers)}")
                print(f"TRANSFORMER BLOCKS: {len(transformer_blocks)}")
                          

                print("\nPARAMETER DISTRIBUTION:")
                print(f"   Embedding Layers: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
                print(f"   Attention Layers: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
                print(f"   MLP Layers: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
                print(f"   Normalization Layers: {norm_params:,} ({norm_params/total_params*100:.1f}%)")
                # Print post-attention layernorm
                print(f"   📐 model.layers.0.post_attention_layernorm → {model.model.layers[0].post_attention_layernorm}")
                
                # Print full transformer block hierarchy (first 2 layers)
                print("\n🏗️  TRANSFORMER BLOCK HIERARCHY:")
                for i in range(min(2, len(model.model.layers))):
                    layer = model.model.layers[i]
                    print(f"   📦 model.layers.{i} → {layer.__class__.__name__}")
                    print(f"     ├─ self_attn → {layer.self_attn.__class__.__name__}")
                    print(f"     ├─ input_layernorm → {layer.input_layernorm.__class__.__name__}")
                    print(f"     ├─ mlp → {layer.mlp.__class__.__name__}")
                    print(f"     └─ post_attention_layernorm → {layer.post_attention_layernorm.__class__.__name__}")
                
                # Reuse existing stats from earlier in the function
                print("\n📊 DETECTED NEURAL NETWORK LAYERS:")
                print(f"📈 TOTAL LAYERS DETECTED: {total_layers}")
                print(f"🎯 ATTENTION LAYERS: {len(attention_layers)}")
                print(f"🔧 MLP/FEEDFORWARD LAYERS: {len(mlp_layers)}")
                print(f"🏗️  TRANSFORMER BLOCKS: {len(transformer_blocks)}")
                
                print("\n📋 PARAMETER DISTRIBUTION:")
                print(f"   🔤 Embedding Layers: {embed_params:,} ({embed_params/total_params*100:.1f}%)")
                print(f"   👁️  Attention Layers: {attention_params:,} ({attention_params/total_params*100:.1f}%)")
                print(f"   ⚡ MLP Layers: {mlp_params:,} ({mlp_params/total_params*100:.1f}%)")
                print(f"   📐 Normalization Layers: {norm_params:,} ({norm_params/total_params*100:.1f}%)")
                
                print("\n🧪 MODEL INFERENCE TESTING WITH SAMPLE WORDS")



                # Show token distribution
                unique_tokens = len(set(response_tokens))
                print(f"   🎲 Unique tokens: {unique_tokens} (diversity: {unique_tokens/len(response_tokens)*100:.1f}%)")
                
            except Exception as e:
                print(f"   ❌ Generation error: {str(e)[:100]}")

    # ========== FINAL SUMMARY ==========
    print(f"\n🎯 PRE-ABLATION NEURAL NETWORK SUMMARY:")
    print("-" * 45)
    print(f"   🏗️  Architecture: {model.__class__.__name__}")
    print(f"   📊 Total Layers: {total_layers}")
    print(f"   🧮 Model Parameters: {total_params:,}")
    print(f"   🔗 Layer Types: {len(layer_types)} unique types")
    print(f"   ⚡ Ready for ablation: {len(test_words)} test words configured")
    
    if hasattr(model, 'config'):
        config = model.config
        print(f"   🔧 Hidden Size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"   🎯 Vocab Size: {getattr(config, 'vocab_size', 'N/A')}")
        print(f"   🔄 Layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
    
    print("="*60 + "\n")




def ablate_specific_tokens_output_only(model, tokenizer, token_ids_to_ablate):
    """
    Ablate tokens only for output generation, not input processing.
    This way inputs work normally but outputs containing ablated tokens are degraded.
    """
    print(f"\n🔧 Starting OUTPUT-ONLY token embedding ablation...")
    print(f"📊 Ablating {len(token_ids_to_ablate)} token embeddings for output generation only")

    # Get the embedding layer
    embeddings = model.get_input_embeddings()

    # SAFETY CHECK: Don't ablate critical tokens (0-100 are usually special/common)
    safe_token_ids = set(range(0, 100))
    tokens_to_skip = token_ids_to_ablate & safe_token_ids

    if tokens_to_skip:
        print(f"⚠️  Skipping {len(tokens_to_skip)} tokens in safe range (0-100)")
        token_ids_to_ablate = token_ids_to_ablate - safe_token_ids

    # Store original embeddings for input processing
    original_embeddings = {}
    with torch.no_grad():
        for token_id in token_ids_to_ablate:
            original_embeddings[token_id] = embeddings.weight[token_id].clone()
            embeddings.weight[token_id, :] = 0

    # Verify ablation
    with torch.no_grad():
        ablated_count = sum(1 for tid in token_ids_to_ablate
                           if torch.all(embeddings.weight[tid] == 0).item())

    print(f"✅ Successfully ablated {ablated_count}/{len(token_ids_to_ablate)} token embeddings for output")

    # Store originals for restoring during input processing
    model.original_embeddings = original_embeddings
    model.ablated_token_ids = token_ids_to_ablate

    return model

# Custom forward pass that restores embeddings for input processing
def custom_forward(self, input_ids, **kwargs):
    # Restore original embeddings for input processing
    if hasattr(self, 'original_embeddings') and hasattr(self, 'ablated_token_ids'):
        embeddings = self.get_input_embeddings()
        with torch.no_grad():
            for token_id, original_embed in self.original_embeddings.items():
                embeddings.weight[token_id] = original_embed

    # Call original forward
    result = self.original_forward(input_ids, **kwargs)

    # Re-ablated for next output generation
    if hasattr(self, 'original_embeddings') and hasattr(self, 'ablated_token_ids'):
        embeddings = self.get_input_embeddings()
        with torch.no_grad():
            for token_id in self.ablated_token_ids:
                embeddings.weight[token_id, :] = 0

    return result

def patch_model_for_output_only_ablation(model):
    """
    Patch the model to restore embeddings during input processing
    but keep them ablated for output generation.
    """
    model.original_forward = model.forward
    model.forward = custom_forward.__get__(model, type(model))
    return model

def find_token_in_word_list(token_id, tokenizer, word_list_path="./word_list_to_ablate.txt"):
    """
    Find which words from the drive list contain this token.
    """
    try:
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]

        matching_words = []
        token_text = tokenizer.decode([token_id]).strip()

        for word in words:
            # Tokenize the word and check if it contains our token
            word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
            if token_id in word_tokens:
                matching_words.append(word)
                if len(matching_words) >= 5:  # Limit to 5 matches for brevity
                    break

        return matching_words
    except Exception as e:
        print(f"        ❌ Could not search word list: {e}")
        return []

def analyze_prompt_ablation(model, tokenizer, prompt, device):
    """
    Analyze which tokens in the prompt are ablated and show details.
    """
    print(f"\n🔍 ANALYZING PROMPT: '{prompt}'")
    print("-" * 50)

    # Tokenize the prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    token_texts = [tokenizer.decode([t]) for t in tokens]

    embeddings = model.get_input_embeddings()

    print(f"📝 Token breakdown:")
    ablated_count = 0

    for i, (token_id, token_text) in enumerate(zip(tokens, token_texts)):
        with torch.no_grad():
            is_ablated = torch.all(embeddings.weight[token_id] == 0).item()

        status = "🔴 ABLATED" if is_ablated else "✅ Normal"

        # Show token details
        print(f"   Token {i:2d}: ID {token_id:6d} -> '{token_text:15s}' : {status}")

        if is_ablated:
            ablated_count += 1

            # Try to find which word from the drive list caused this ablation
            found_in_words = find_token_in_word_list(token_id, tokenizer)
            if found_in_words:
                print(f"        📌 Ablated because of: {found_in_words[:3]}")  # Show first 3 matches

    print(f"📊 Summary: {ablated_count}/{len(tokens)} tokens ablated in this prompt")
    print("-" * 50)

    return ablated_count > 0

def test_ablation_after(model, tokenizer, test_words, device):
    """
    Test the model AFTER ablation to see the effect.
    """
    print("\n" + "="*60)
    print("TESTING MODEL AFTER ABLATION (OUTPUT-ONLY)")
    print("="*60)

    model.eval()

    for word in test_words:
        print(f"\n🧪 Testing AFTER ablation: '{word}'")

        # Try to generate with this word as prompt
        inputs = tokenizer(word, return_tensors="pt").to(device)

        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   Output: {response[:100]}{'...' if len(response) > 100 else ''}")

                # Check if response contains ablated words
                response_tokens = tokenizer.encode(response, add_special_tokens=False)
                embeddings = model.get_input_embeddings()
                ablated_in_response = sum(1 for tid in response_tokens
                                        if torch.all(embeddings.weight[tid] == 0).item())
                if ablated_in_response > 0:
                    print(f"   ⚠️  Response contains {ablated_in_response} ablated tokens")

            except Exception as e:
                print(f"   Error: {str(e)[:100]}")

    print("="*60 + "\n")


def compare_before_after_ablation(original_model, ablated_model, tokenizer, prompt, device):
    """
    Show both before AND after ablation responses for any input.
    """
    print(f"\n🔍 COMPARING RESPONSES FOR: '{prompt}'")
    print("=" * 60)
    
    # Test BEFORE ablation
    print("🟢 BEFORE ABLATION:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs_before = original_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    response_before_clean = response_before.replace(prompt, "").strip()
    print(f"   {response_before_clean}")
    
    # Test AFTER ablation
    print("🔴 AFTER ABLATION:")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs_after = ablated_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    response_after_clean = response_after.replace(prompt, "").strip()
    print(f"   {response_after_clean}")
    
    print("=" * 60)


def load_local_model(model_path):
    """
    Load model from local directory with proper error handling.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return None, None, None

    print(f"📂 Loading model from: {model_path}")

    # Check what files are in the model directory
    model_files = os.listdir(model_path)
    print(f"📁 Model files found: {model_files}")

    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # Load model
        print("📥 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )

        # Move to device
        print(f"📍 Moving model to {device}...")
        model.to(device)
        model.eval()

        print("✅ Model loaded successfully!")
        return model, tokenizer, device

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None




def chat_with_model(model_path: str, text_file_path: str = None):
    """
    Loads the LOCAL model, ablates tokens from the text file, and starts chat.
    """
    # Load the local model
    model, tokenizer, device = load_local_model(model_path)

    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Store original model BEFORE ablation for comparison
    import copy
    original_model = copy.deepcopy(model)
    ablated_model = None

    # If a text file is provided, extract tokens and ablate them
    if text_file_path and os.path.exists(text_file_path):
        print(f"\n📂 Reading words from: {text_file_path}")
        text, word_list = extract_text_from_file(text_file_path)

        # IMPORTANT: Filter to only meaningful company/brand names
        print(f"\n🔍 Filtering word list...")
        original_count = len(word_list)

        # Keep only words that are likely to be specific entities
        word_list = [w for w in word_list
                    if len(w) >= 4
                    and not w.isdigit()
                    and any(c.isalpha() for c in w)
                    and not w.lower() in ['the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 'been', 'are', 'not']]

        # Further filter: Only keep words that tokenize to 1-2 tokens
        filtered_list = []
        for word in word_list:
            tokens = tokenizer.encode(" " + word, add_special_tokens=False)
            if len(tokens) <= 2:
                filtered_list.append(word)

        word_list = filtered_list

        print(f"   Original: {original_count} words")
        print(f"   After filtering: {len(word_list)} words")
        print(f"   Removed: {original_count - len(word_list)} words")

        if text and word_list:
            # Create test words
            test_words_in_list = word_list[:3] if len(word_list) >= 3 else word_list
            test_words_not_in_list = ["happiness", "beautiful", "wonderful"]
            test_words = test_words_in_list + test_words_not_in_list

            print(f"\n🧪 Testing BEFORE ablation:")
            print(f"   Words FROM your list: {test_words_in_list}")
            print(f"   Words NOT in your list: {test_words_not_in_list}")

            # TEST BEFORE ABLATION
            test_model_before_ablation(original_model, tokenizer, test_words, device)

            # Get tokens from the text
            tokens_to_ablate = get_tokens_from_text(text, tokenizer, word_list)

            # Ablate these specific tokens for OUTPUT ONLY
            model = ablate_specific_tokens_output_only(model, tokenizer, tokens_to_ablate)
            model = patch_model_for_output_only_ablation(model)
            ablated_model = model
            
            print(f"\n💾 Saving ablated model for future use...")
            model.save_pretrained("/content/ablated_model_output_only")
            tokenizer.save_pretrained("/content/ablated_model_output_only")
            print("✅ Ablated model saved! You can now load it directly without re-ablating.")

            # TEST AFTER ABLATION
            print(f"\n🧪 Testing AFTER ablation (OUTPUT-ONLY):")
            print(f"   Words FROM your list (output should be degraded): {test_words_in_list}")
            print(f"   Words NOT in your list (output should work normally): {test_words_not_in_list}")

            test_ablation_after(ablated_model, tokenizer, test_words, device)
    else:
        print(f"\n⚠ No text file provided or file not found. Skipping ablation.")
        ablated_model = model  # Use original model if no ablation

    print("\n✅ Model loaded successfully. Starting chat session.")
    print("   Type 'exit' or 'quit' to end the conversation.")
    print("   Type 'analyze [text]' to see ablation details for any text.")
    print("   Type 'compare [text]' to see BEFORE and AFTER ablation responses")
    print("="*60)

    while True:
        prompt = input("\nYou: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Ending chat. Goodbye!")
            break

        if prompt.lower().startswith("analyze "):
            text_to_analyze = prompt[8:]
            analyze_prompt_ablation(ablated_model, tokenizer, text_to_analyze, device)
            continue

        if prompt.lower().startswith("compare "):
            text_to_compare = prompt[8:]
            compare_before_after_ablation(original_model, ablated_model, tokenizer, text_to_compare, device)
            continue

        # Generate response with BOTH before and after comparison for EVERY input
        response_text = generate_response_with_comparison(original_model, ablated_model, tokenizer, prompt, device)
        
        # You can still show just the after response if you want
        # print(f"Model: {response_text}")








if __name__ == "__main__":
    # ========== CONFIGURATION ==========

    # 1. Your Google Drive file with the word list
    TEXT_FILE_GDRIVE_ID = "1Y_kEtPunBz2RJmu9iBhtG7glHgHg4SaT"
    TEXT_FILE_PATH = "./word_list_to_ablate.txt"

    # 2. Your Google Drive MODEL ZIP file ID
    MODEL_ZIP_FILE_ID = "1iC1vXqg7Uf6v1qjQ9bWQ7a9Y8ZzQ6X9Y"  # Replace with your actual file ID
    MODEL_EXTRACT_PATH = "/content/ablated_model"

    print("\n" + "="*60)
    print("TOKEN ABLATION SETUP - OUTPUT ONLY")
    print("="*60)
    print(f"Model zip file ID: {MODEL_ZIP_FILE_ID}")
    print(f"Extraction path: {MODEL_EXTRACT_PATH}")
    print(f"Word list file ID: {TEXT_FILE_GDRIVE_ID}")
    print("="*60)

    # ========== STEP 1: Download and extract model ==========
    print("\n📥 STEP 1: Downloading and extracting model from Google Drive...")
    model_path = download_and_extract_model(MODEL_ZIP_FILE_ID, MODEL_EXTRACT_PATH)

    # If model download fails, use a small test model
    if not model_path:
        print("❌ Failed to download your model. Trying fallback...")
        model_path = download_small_test_model()

    if not model_path:
        print("❌ Could not load any model. Exiting.")
        exit()

    # ========== STEP 2: Download word list ==========
    print("\n📥 STEP 2: Downloading word list from Google Drive...")
    text_file = download_text_file(TEXT_FILE_GDRIVE_ID, TEXT_FILE_PATH)

    if not text_file:
        print("❌ Failed to download word list. Exiting.")
        exit()

    # Show preview of the file
    print(f"\n📄 File preview (first 500 characters):")
    print("-" * 60)
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            preview = f.read(500)
            print(preview)
            if len(preview) == 500:     
                print("...")
    except Exception as e:
        print(f"Could not read file: {e}")
    print("-" * 60)

    # ========== CREATIVE NEURAL NETWORK REPORT ==========
    print("\n" + "="*60)
    print("🧠 CREATIVE NEURAL NETWORK ANALYSIS REPORT")
    print("="*60)
    
    # Load model for analysis
    model, tokenizer, device = load_local_model(model_path)
    if model:
        print("\n📊 DETECTED MODEL ARCHITECTURE LAYERS:")
        print("-" * 40)
        layer_count = 0
        for name, module in model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['layer', 'block', 'attention', 'mlp']):
                print(f"🔍 {name}: {module.__class__.__name__}")
                layer_count += 1
                if layer_count >= 15:  # Limit output
                    print("   ... (showing first 15 layers)")
                    break
        
        # 2. Creative neural network report
        print("\n🌟 NEURAL NETWORK PERFORMANCE METRICS:")
        print("-" * 40)
        layer_counts = {}
        for name, module in model.named_modules():
            layer_type = module.__class__.__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        print("🏆 TOP PERFORMING LAYERS:")
        for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            stars = "⭐" * min(count, 5)
            print(f"   {layer_type}: {count} instances {stars}")
        
        # 3. Model configuration analysis
        print("\n⚡ MODEL CONFIGURATION ANALYSIS:")
        print("-" * 40)
        if hasattr(model, 'config'):
            config = model.config
            print(f"📐 Model Type: {getattr(config, 'model_type', 'Unknown')}")
            print(f"🧠 Hidden Size: {getattr(config, 'hidden_size', 'Unknown')}")
            print(f"🔢 Vocabulary Size: {getattr(config, 'vocab_size', 'Unknown')}")
            print(f"🔄 Number of Layers: {getattr(config, 'num_hidden_layers', 'Unknown')}")
            print(f"👁️ Attention Heads: {getattr(config, 'num_attention_heads', 'Unknown')}")
        else:
            print("❌ No configuration found")
    
    print("="*60)

    # ========== STEP 3: Load local model and ablate ==========
    print(f"\n📥 STEP 3: Loading LOCAL model and performing OUTPUT-ONLY ablation...")
    chat_with_model(model_path, text_file_path=text_file)
