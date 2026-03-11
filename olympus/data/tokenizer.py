"""
BPE Tokenizer wrapper supporting tiktoken and sentencepiece backends,
with a built-in character-level fallback.

Special tokens are reserved for GENESIS reasoning and memory operations.
"""

from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

# Special token definitions
SPECIAL_TOKENS: Dict[str, str] = {
    "pad": "<|pad|>",
    "bos": "<|bos|>",
    "eos": "<|eos|>",
    "reason_start": "<|reason_start|>",
    "reason_end": "<|reason_end|>",
    "memory_read": "<|memory_read|>",
    "memory_write": "<|memory_write|>",
    "tier_escalate": "<|tier_escalate|>",
    "trace_start": "<|trace_start|>",
    "trace_end": "<|trace_end|>",
}


class _CharLevelBackend:
    """Minimal character-level tokenizer used when no BPE backend is available."""

    def __init__(self, vocab_size: int = 32000) -> None:
        self._vocab_size = vocab_size
        # Reserve first 256 IDs for raw bytes, then special tokens
        self._special_token_offset = 256
        self._special_to_id: Dict[str, int] = {}
        self._id_to_special: Dict[int, str] = {}
        for i, tok in enumerate(SPECIAL_TOKENS.values()):
            tid = self._special_token_offset + i
            self._special_to_id[tok] = tid
            self._id_to_special[tid] = tok

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self._special_to_id.items():
                if text[i:].startswith(tok):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[i]) % 256)
                i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        parts: List[str] = []
        for tid in ids:
            if tid in self._id_to_special:
                parts.append(self._id_to_special[tid])
            elif 0 <= tid < 256:
                parts.append(chr(tid))
            else:
                parts.append(f"<unk_{tid}>")
        return "".join(parts)

    def special_token_id(self, name: str) -> Optional[int]:
        tok = SPECIAL_TOKENS.get(name)
        if tok is None:
            return None
        return self._special_to_id.get(tok)


class TokenizerWrapper:
    """Unified tokenizer interface for GENESIS / OLYMPUS.

    Attempts to use tiktoken first (cl100k_base encoding), then sentencepiece,
    and finally falls back to a character-level tokenizer.

    Args:
        backend: Preferred backend - ``"tiktoken"``, ``"sentencepiece"``,
                 or ``"char"``.  If the requested backend is unavailable the
                 next one is tried automatically.
        vocab_size: Target vocabulary size.  For tiktoken / sentencepiece the
                    actual vocab may differ; this is used for the char-level
                    fallback and exposed via :pyattr:`vocab_size`.
        sp_model_path: Path to a sentencepiece ``.model`` file (required only
                       when ``backend="sentencepiece"``).
    """

    def __init__(
        self,
        backend: str = "tiktoken",
        vocab_size: int = 32000,
        sp_model_path: Optional[str] = None,
    ) -> None:
        self._backend_name: str = "none"
        self._tiktoken_enc = None
        self._sp_processor = None
        self._char_backend: Optional[_CharLevelBackend] = None
        self._vocab_size = vocab_size

        # --- Try tiktoken ---
        if backend in ("tiktoken", "auto"):
            try:
                import tiktoken  # type: ignore

                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                self._vocab_size = self._tiktoken_enc.n_vocab + len(SPECIAL_TOKENS)
                self._backend_name = "tiktoken"
                logger.info("Tokenizer: using tiktoken (cl100k_base), vocab=%d", self._vocab_size)
            except Exception:
                logger.info("tiktoken not available, trying sentencepiece")

        # --- Try sentencepiece ---
        if self._backend_name == "none" and backend in ("sentencepiece", "tiktoken", "auto"):
            try:
                import sentencepiece as spm  # type: ignore

                if sp_model_path is not None:
                    self._sp_processor = spm.SentencePieceProcessor()
                    self._sp_processor.Load(sp_model_path)
                    self._vocab_size = self._sp_processor.GetPieceSize() + len(SPECIAL_TOKENS)
                    self._backend_name = "sentencepiece"
                    logger.info("Tokenizer: using sentencepiece, vocab=%d", self._vocab_size)
                else:
                    logger.info("sentencepiece available but no model path given, falling back")
            except Exception:
                logger.info("sentencepiece not available, falling back to char-level")

        # --- Char-level fallback ---
        if self._backend_name == "none":
            self._char_backend = _CharLevelBackend(vocab_size=vocab_size)
            self._vocab_size = vocab_size
            self._backend_name = "char"
            logger.info("Tokenizer: using char-level fallback, vocab=%d", self._vocab_size)

        # Build special token ID map regardless of backend
        self._special_ids: Dict[str, int] = {}
        if self._backend_name == "tiktoken":
            base = self._tiktoken_enc.n_vocab  # type: ignore[union-attr]
            for i, name in enumerate(SPECIAL_TOKENS):
                self._special_ids[name] = base + i
        elif self._backend_name == "sentencepiece":
            base = self._sp_processor.GetPieceSize()  # type: ignore[union-attr]
            for i, name in enumerate(SPECIAL_TOKENS):
                self._special_ids[name] = base + i
        else:
            # char backend has its own special token mapping
            for name in SPECIAL_TOKENS:
                tid = self._char_backend.special_token_id(name)  # type: ignore[union-attr]
                if tid is not None:
                    self._special_ids[name] = tid

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode *text* into a list of token IDs."""
        if self._backend_name == "tiktoken":
            return self._tiktoken_enc.encode(text, allowed_special="all")  # type: ignore[union-attr]
        elif self._backend_name == "sentencepiece":
            return self._sp_processor.Encode(text)  # type: ignore[union-attr]
        else:
            return self._char_backend.encode(text)  # type: ignore[union-attr]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text."""
        if self._backend_name == "tiktoken":
            # Filter out special IDs that tiktoken doesn't know about
            base = self._tiktoken_enc.n_vocab  # type: ignore[union-attr]
            clean_ids = [i for i in ids if i < base]
            return self._tiktoken_enc.decode(clean_ids)  # type: ignore[union-attr]
        elif self._backend_name == "sentencepiece":
            base = self._sp_processor.GetPieceSize()  # type: ignore[union-attr]
            clean_ids = [i for i in ids if i < base]
            return self._sp_processor.Decode(clean_ids)  # type: ignore[union-attr]
        else:
            return self._char_backend.decode(ids)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._vocab_size

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend_name

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self._special_ids["pad"]

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return self._special_ids["bos"]

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self._special_ids["eos"]

    def special_token_id(self, name: str) -> Optional[int]:
        """Return the ID for a named special token, or ``None``."""
        return self._special_ids.get(name)

    def special_token_ids(self) -> Dict[str, int]:
        """Return a copy of all special token name -> ID mappings."""
        return dict(self._special_ids)

    def __repr__(self) -> str:
        return (
            f"TokenizerWrapper(backend={self._backend_name!r}, "
            f"vocab_size={self._vocab_size})"
        )
