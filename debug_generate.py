"""
Debug script to inspect model's next move predictions.
"""

import torch
from src.model import ChessForCausalLM
from src.tokenizer import ChessTokenizer, PIECE_TOKENS

# Load model and tokenizer
model_path = "./my_model/epoch1_v4"
print(f"Loading model from {model_path}...")

tokenizer = ChessTokenizer.from_pretrained(model_path)
model = ChessForCausalLM.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Vocab size: {tokenizer.vocab_size}")
print()

def manual_smart_generate(model, input_ids, max_new_tokens=64):
    """Greedy-generate tokens until EOS or max_new_tokens is reached."""
    generated = input_ids.clone()
    
    for i in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token_str = tokenizer.convert_ids_to_tokens([next_token_id])[0]
            
            # Stop if EOS
            if next_token_id == tokenizer.eos_token_id:
                break

            generated = torch.cat([generated, torch.tensor([[next_token_id]]).to(device)], dim=1)
            
    return generated

def generate_next_move(game_history, num_tokens=6, top_k=10):
    """Generate next move and show detailed output."""
    print("=" * 60)
    print(f"Input: '{game_history}'")
    
    # Tokenize properly - don't add EOS when we're continuing generation
    if game_history:
        # Tokenize game history without special tokens first
        tokens = tokenizer.tokenize(game_history)
        token_ids = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokens)
    else:
        # Just BOS for empty input
        token_ids = [tokenizer.bos_token_id]
    
    input_ids = torch.tensor([token_ids]).to(device)
    
    # Decode each token ID to see what's happening
    decoded_tokens = [tokenizer.convert_ids_to_tokens([t.item()])[0] for t in input_ids[0]]
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Each ID decoded: {decoded_tokens}")
    print()
    
    # Get logits for next token
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        
        # Top-k predictions
        probs = torch.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        print(f"Top {top_k} predictions for next token:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.convert_ids_to_tokens([idx.item()])[0]
            print(f"  {i+1}. '{token}' (id={idx.item()}, prob={prob.item():.4f})")
        print()
        
        # Generate multiple tokens manually
        generated = manual_smart_generate(model, input_ids, max_new_tokens=num_tokens)
        
        new_tokens = generated[0, input_ids.shape[1]:]
        new_token_list = [tokenizer.convert_ids_to_tokens([t.item()])[0] for t in new_tokens]
        print(f"Generated token IDs: {new_tokens.tolist()}")
        print(f"Generated tokens: {new_token_list}")
        
        # Try to form a move
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Generated move text: '{generated_text}'")
    
    return generated_text

# Check what the training data looks like
print("=" * 60)
print("CHECKING TRAINING DATA FORMAT")
print("=" * 60)

sample_move = "WPe2e4"
tokens = tokenizer.tokenize(sample_move)
print(f"Sample move: '{sample_move}'")
print(f"Tokenized: {tokens}")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {ids}")
print()

# Check full encoding with special tokens
full_encoding = tokenizer(sample_move, add_special_tokens=True)
print(f"Full encoding with special tokens: {full_encoding['input_ids']}")
decoded = [tokenizer.convert_ids_to_tokens([t])[0] for t in full_encoding['input_ids']]
print(f"Decoded: {decoded}")
print()

# Test game sequences
test_games = [
    "",  # Empty - should predict first move
    "WPe2e4",  # After e4
    "WPe2e4 BPe7e5",  # After e4 e5
]

print("\n" + "=" * 60)
print("TESTING MODEL PREDICTIONS")
print("=" * 60 + "\n")

# First, let's check what the training data looks like
print("=" * 60)
print("SIMULATING TRAINING DATA TOKENIZATION")
print("=" * 60)

sample_game = "WPa2a4 BPh7h5"
game_with_bos = tokenizer.bos_token + " " + sample_game
print(f"Raw input (as in data.py): '{game_with_bos}'")

# What _tokenize produces
raw_tokens = tokenizer.tokenize(game_with_bos)
print(f"After tokenize(): {raw_tokens}")

# Full encoding with special tokens
full_encoding = tokenizer(game_with_bos, add_special_tokens=True)
full_ids = full_encoding['input_ids']
print(f"Full encoding IDs: {full_ids}")
decoded_each = [tokenizer.convert_ids_to_tokens([t])[0] for t in full_ids]
print(f"Each ID decoded: {decoded_each}")
print()

# What the model sees during training:
print("During training, model learns to predict:")
for i in range(1, min(len(full_ids), 10)):
    context = decoded_each[:i]
    target = decoded_each[i]
    print(f"  Given {context} -> predict '{target}'")
print()

for game in test_games:
    generate_next_move(game)

print("\nDone!")

