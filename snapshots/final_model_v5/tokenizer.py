import json, os, re
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

SQUARE_RE = re.compile(r"(?:[WB][PNBRQK])?([a-h][1-8])([a-h][1-8])([qrbnQRBN])?")

PIECE_TOKENS: List[str] = []

FILES = "abcdefgh"
RANKS = "12345678"
ALL_SQUARES = [f + r for r in RANKS for f in FILES]

class ChessTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    def __init__(self, vocab_file: Optional[str] = None, vocab: Optional[Dict[str,int]] = None, **kwargs):
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)

        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._build_fixed_vocab()

        self._ids_to_tokens = {v:k for k,v in self._vocab.items()}

        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )

    def _build_fixed_vocab(self) -> Dict[str,int]:
        vocab: Dict[str,int] = {}
        for t in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            vocab[t] = len(vocab)

        for sq in ALL_SQUARES:
            vocab[sq] = len(vocab)

        promotion_squares = [sq for sq in ALL_SQUARES if sq.endswith("1") or sq.endswith("8")]
        for sq in promotion_squares:
            for promo in ["q", "r", "b", "n"]:
                vocab[f"{sq}{promo}"] = len(vocab)

        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str,int]:
        return dict(self._vocab)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            raise ValueError("No pair inputs.")
        # Avoid adding BOS twice if it's already at the start
        if token_ids_0 and token_ids_0[0] == self.bos_token_id:
            return token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}

        normalized = text
        for marker in special:
            normalized = normalized.replace(marker, f" {marker} ")

        for chunk in normalized.split():
            if chunk in special:
                tokens.append(chunk)
                continue

            for match in SQUARE_RE.finditer(chunk):
                from_sq = match.group(1)
                to_sq = match.group(2)
                promo = match.group(3)

                tokens.append(from_sq)
                to_token = to_sq + (promo.lower() if promo else "")
                tokens.append(to_token)

        if not tokens:
            tokens.append(self.UNK_TOKEN)

        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab[self.UNK_TOKEN])

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        special_drop = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        filtered = [t for t in tokens if t not in special_drop]

        moves: List[str] = []
        for i in range(0, len(filtered) - 1, 2):
            from_sq = filtered[i]
            to_sq = filtered[i + 1]
            moves.append(f"{from_sq}{to_sq}")

        return " ".join(moves)


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        return (vocab_file,)
