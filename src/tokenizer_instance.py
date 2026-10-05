from tokenizer import ChessTokenizer

# 只要这一行，不需要传任何 vocab_file
tokenizer = ChessTokenizer()

# 测试一下
print(tokenizer.vocab_size)
print(tokenizer.convert_ids_to_tokens(tokenizer.encode("WPe2e4")))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode("WBb5c6(x)")))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode("WPe7e8=Q")))
print(tokenizer.encode("WPe2e4"))
print(tokenizer.encode("WBb5c6(x)"))
print(tokenizer.encode("WPe7e8=Q"))