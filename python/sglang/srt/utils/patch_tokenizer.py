import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        tokenizer = _SpecialTokensCachePatcher.patch(tokenizer)
        tokenizer = _BasetenkenizerSpeedPatcher.patch(tokenizer)

    return tokenizer


def unpatch_tokenizer(tokenizer):
    tokenizer = _BasetenkenizerSpeedPatcher.unpatch(tokenizer)
    return _SpecialTokensCachePatcher.unpatch(tokenizer)


def _is_kimi_tiktoken_tokenizer(tokenizer):
    cls = type(tokenizer)
    class_name = cls.__name__
    module_name = cls.__module__ or ""
    return class_name == "TikTokenTokenizer" and "tokenization_kimi" in module_name


def decode_without_hf_kwargs(tokenizer, token_ids, skip_special_tokens):
    if skip_special_tokens:
        special_ids = getattr(tokenizer, "all_special_ids_set", None)
        if special_ids is None:
            special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids is not None:
            special_ids_set = set(special_ids)
            token_ids = [tid for tid in token_ids if tid not in special_ids_set]
    return tokenizer.decode(token_ids)


class _SpecialTokensCachePatcher:
    _PATCHED_FLAG = "_sglang_special_tokens_patched"
    _CACHED_TOKENS_ATTR = "_sglang_cached_special_tokens"
    _CACHED_IDS_ATTR = "_sglang_cached_special_ids"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._original_all_special_tokens = (
            tokenizer_cls.all_special_tokens.fget
        )
        tokenizer_cls._original_all_special_ids = tokenizer_cls.all_special_ids.fget
        tokenizer_cls._original_add_special_tokens = tokenizer_cls.add_special_tokens
        tokenizer_cls._original_add_tokens = tokenizer_cls.add_tokens

        patched_all_special_tokens = _make_cached_property(
            cls._CACHED_TOKENS_ATTR, tokenizer_cls._original_all_special_tokens
        )
        patched_all_special_ids = _make_cached_property(
            cls._CACHED_IDS_ATTR, tokenizer_cls._original_all_special_ids
        )

        def patched_add_special_tokens(self, *args, **kwargs):
            assert (
                False
            ), "Cannot modify special tokens after patch. Call unpatch_tokenizer first."

        def patched_add_tokens(self, new_tokens, special_tokens=False):
            assert (
                not special_tokens
            ), "Cannot add special tokens after patch. Call unpatch_tokenizer first."
            return tokenizer_cls._original_add_tokens(
                self, new_tokens, special_tokens=False
            )

        tokenizer_cls.all_special_tokens = patched_all_special_tokens
        tokenizer_cls.all_special_ids = patched_all_special_ids
        tokenizer_cls.add_special_tokens = patched_add_special_tokens
        tokenizer_cls.add_tokens = patched_add_tokens
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls.all_special_tokens = property(
            tokenizer_cls._original_all_special_tokens
        )
        tokenizer_cls.all_special_ids = property(
            tokenizer_cls._original_all_special_ids
        )
        tokenizer_cls.add_special_tokens = tokenizer_cls._original_add_special_tokens
        tokenizer_cls.add_tokens = tokenizer_cls._original_add_tokens

        del tokenizer_cls._original_all_special_tokens
        del tokenizer_cls._original_all_special_ids
        del tokenizer_cls._original_add_special_tokens
        del tokenizer_cls._original_add_tokens
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        for attr in [cls._CACHED_TOKENS_ATTR, cls._CACHED_IDS_ATTR]:
            if hasattr(tokenizer, attr):
                delattr(tokenizer, attr)

        logger.info(f"Unpatched special tokens cache for {tokenizer_cls.__name__}")
        return tokenizer


def _make_cached_property(cache_attr, original_fn):
    @property
    def cached_prop(self):
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop


# Sentinel so the cache can remember "already tried and failed" separately
# from "never attempted", instead of retrying a broken conversion on every call.
_UNSET = object()
_basetenkenizer_cache = {}


def _get_basetenkenizer_for_kimi(tokenizer):
    """Lazily build (and cache) a basetenkenizer.Tokenizer for this Kimi
    tokenizer's tiktoken.model, keyed by vocab_file path rather than tokenizer
    instance -- this is model-derived, not instance-derived, so it must be
    shared across every TikTokenTokenizer instance for the same model
    regardless of which instance happened to trigger the class-level patch.

    Returns None if basetenkenizer isn't installed or conversion fails;
    callers must fall back to the original tiktoken-based path in that case.
    """
    vocab_file = getattr(tokenizer, "vocab_file", None)
    if not vocab_file:
        return None

    cached = _basetenkenizer_cache.get(vocab_file, _UNSET)
    if cached is not _UNSET:
        return cached

    fast_tokenizer = None
    try:
        import json
        import os

        import basetenkenizer

        model_dir = os.path.dirname(vocab_file)
        tokenizer_json_str = basetenkenizer.tiktoken_model_to_tokenizer_json(model_dir)
        if tokenizer_json_str is not None:
            spec = json.loads(tokenizer_json_str)
            decoder = spec.get("decoder")
            if isinstance(decoder, dict) and decoder.get("type") == "ByteLevel":
                # basetenkenizer 0.2.8 emits a bare {"type": "ByteLevel"} decoder,
                # which the currently-released `tokenizers` crate (0.22.x) refuses
                # to deserialize ("PyDecoderWrapper" error) without these fields
                # explicitly present -- they're the same fields already set on
                # the sibling post_processor block in the same spec.
                decoder.setdefault("add_prefix_space", False)
                decoder.setdefault("trim_offsets", False)
                decoder.setdefault("use_regex", False)
            fast_tokenizer = basetenkenizer.Tokenizer.from_json_str(json.dumps(spec))
    except ImportError:
        pass
    except Exception:
        logger.exception(
            "Failed to build basetenkenizer backend for Kimi tokenizer at %s; "
            "falling back to the native tiktoken-based encode/decode.",
            vocab_file,
        )

    # This, not `_BasetenkenizerSpeedPatcher.patch()` running, is the actual
    # signal that the fast path is live: the patcher wraps the tokenizer's
    # methods unconditionally, but each wrapped call still falls back to the
    # slow native path here if this resolves to None (basetenkenizer missing
    # or conversion failed). Logged once per vocab_file (cached above).
    if fast_tokenizer is not None:
        logger.info(
            "basetenkenizer engaged for Kimi tokenizer at %s: fast encode/decode is live.",
            vocab_file,
        )
    else:
        logger.warning(
            "basetenkenizer NOT engaged for Kimi tokenizer at %s: falling back to the "
            "native tiktoken-based encode/decode for every call.",
            vocab_file,
        )

    _basetenkenizer_cache[vocab_file] = fast_tokenizer
    return fast_tokenizer


class _BasetenkenizerSpeedPatcher:
    """Swap Kimi's TikTokenTokenizer encode/decode/call/batch-decode leaf calls
    for basetenkenizer's Rust-backed equivalents, without touching chat
    rendering at all.

    The real win requires batching across *requests*, not just across the
    internal chunks of one string -- an earlier version of this patch only
    touched `_encode_text_piece`/`decode` and was measured to be a net
    *regression* for the common single-prompt case (0.81x encode, 0.40x
    decode versus unpatched), because a single `encode_segments`/`decode`
    call has more Python/FFI overhead than tiktoken's own single call. Only
    `__call__`/`batch_decode` -- which SGLang's TokenizerManager and
    DetokenizerManager call with the *whole batch of prompts/outputs already
    in hand* -- can turn that into a real win via `encode_batch`/`decode_batch`.

      - `_encode_text_piece` / plain `decode()`: the safe choke points for
        single-string encode/decode, routed through `encode_segments()` (never
        flat `encode()`, which has no equivalent of tiktoken's
        `disallowed_special=()` and will happily turn literal `<|open|>`-shaped
        *untrusted* text into real control tokens). Passes the whole string as
        one segment with `tiktoken_safe=True` rather than re-running Kimi's own
        `_split_whitespaces_or_nonwhitespaces` chunking first -- profiled that
        loop alone costs more than the actual Rust BPE encode on a ~1M-token
        input (~138ms vs ~101ms), capping this patch at ~1.4x there instead of
        the 8x+ it gets once that redundant Python pass is removed.
        `tiktoken_safe=True` reproduces the same chunk boundaries natively
        (verified byte-identical against Kimi's own chunked output, including
        at that same ~1M-token scale).
      - `__call__`: only takes the fast path for the exact shape
        TokenizerManager/AsyncDynamicbatchTokenizer actually use (plain str or
        list[str], no text_pair/text_target, no `return_token_type_ids`).
        Verified empirically that Kimi's *unpatched* `__call__` already
        ignores `add_special_tokens` entirely (`_tokenize` always calls
        `self.encode(text)` with its own default) -- so this always encodes
        permissively too, matching existing behavior rather than diverging
        from it. Anything outside the verified shape falls back to the
        original `__call__` untouched.
      - `batch_decode`: SGLang's DetokenizerManager passes
        `skip_special_tokens`/`spaces_between_special_tokens` directly to this
        method when `is_fast=True`. `spaces_between_special_tokens` has no
        basetenkenizer equivalent and is dropped -- verified that today's
        slow-path decode (`decode_without_hf_kwargs`) never uses it either, so
        this isn't a new gap. `skip_special_tokens` is NOT delegated to
        basetenkenizer's own notion of "special" -- empirically that disagrees
        with Kimi's actual runtime special-token set (e.g. `<|end_of_msg|>` is
        declared `"special": true` in tokenizer_config.json but Kimi's own
        `additional_special_tokens` default doesn't include it, so native
        `decode(skip_special_tokens=True)` keeps it while basetenkenizer would
        strip it). Instead this pre-filters ids using `all_special_ids_set`
        (the same cache `_SpecialTokensCachePatcher` already maintains),
        exactly mirroring `decode_without_hf_kwargs`, then calls
        `decode_batch(..., skip_special_tokens=False)` since filtering already
        happened -- so behavior matches Kimi's real special-token set, not
        basetenkenizer's tokenizer.json-declared one.
      - `is_fast`: read-only `property` hardcoded to `False` on the slow base
        class, so it needs the same class-level property-override technique
        `_SpecialTokensCachePatcher` already uses. Only safe to flip now that
        `__call__`/`batch_decode` are actually fast -- flipping it alone,
        without those, would silently switch TokenizerManager onto the slow
        generic `_tokenize`/id-string-id round trip for the regular batched
        path, which is strictly worse than today's direct `.encode()` loop.
    """

    _PATCHED_FLAG = "_sglang_basetenkenizer_speed_patched"

    # The only kwargs __call__ can take while still using the fast path --
    # anything else falls back to the original implementation untouched.
    _CALL_SUPPORTED_KWARGS = frozenset({"add_special_tokens"})

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._original_encode_text_piece = tokenizer_cls._encode_text_piece
        tokenizer_cls._original_decode = tokenizer_cls.decode
        tokenizer_cls._original_call = tokenizer_cls.__call__
        tokenizer_cls._original_batch_decode = tokenizer_cls.batch_decode
        tokenizer_cls._original_is_fast = tokenizer_cls.is_fast.fget

        def patched_encode_text_piece(self, text, allow_special_tokens=True):
            fast_tokenizer = _get_basetenkenizer_for_kimi(self)
            if fast_tokenizer is None:
                return tokenizer_cls._original_encode_text_piece(
                    self, text, allow_special_tokens=allow_special_tokens
                )

            if not text:
                return []

            # Do NOT re-run Kimi's own `_split_whitespaces_or_nonwhitespaces`
            # chunking here. Profiled empirically: for a ~1M-token input, that
            # pure-Python character-by-character scan alone costs ~138ms --
            # *more* than the actual Rust BPE encode (~101ms). Re-running it in
            # front of encode_segments() was capping this patch at ~1.4x on
            # large inputs, when the real win is 8x+. `tiktoken_safe=True`
            # reproduces the same chunk boundaries natively (verified
            # byte-identical against Kimi's own chunked output on both a 40k
            # highly-mergeable stress case and a real ~1M-token/~6M-char input),
            # so the whole text can go through in one call.
            return fast_tokenizer.encode_segments(
                [(text, allow_special_tokens)], tiktoken_safe=True
            ).ids

        def patched_decode(self, token_ids, **kwargs):
            if kwargs:
                return tokenizer_cls._original_decode(self, token_ids, **kwargs)

            fast_tokenizer = _get_basetenkenizer_for_kimi(self)
            if fast_tokenizer is None:
                return tokenizer_cls._original_decode(self, token_ids)

            if isinstance(token_ids, int):
                token_ids = [token_ids]
            return fast_tokenizer.decode(token_ids)

        def patched_call(
            self,
            text=None,
            text_pair=None,
            text_target=None,
            text_pair_target=None,
            **kwargs,
        ):
            fast_tokenizer = _get_basetenkenizer_for_kimi(self)
            is_plain_text = isinstance(text, str) or (
                isinstance(text, (list, tuple))
                and all(isinstance(t, str) for t in text)
            )
            if (
                fast_tokenizer is not None
                and text is not None
                and is_plain_text
                and text_pair is None
                and text_target is None
                and text_pair_target is None
                and set(kwargs) <= cls._CALL_SUPPORTED_KWARGS
            ):
                if isinstance(text, str):
                    encodings = [fast_tokenizer.encode_segments([(text, True)])]
                else:
                    encodings = fast_tokenizer.encode_batch(list(text))
                return {
                    "input_ids": [e.ids for e in encodings]
                    if not isinstance(text, str)
                    else encodings[0].ids,
                    "attention_mask": [e.attention_mask for e in encodings]
                    if not isinstance(text, str)
                    else encodings[0].attention_mask,
                }

            return tokenizer_cls._original_call(
                self,
                text=text,
                text_pair=text_pair,
                text_target=text_target,
                text_pair_target=text_pair_target,
                **kwargs,
            )

        def patched_batch_decode(
            self,
            sequences,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
            **kwargs,
        ):
            fast_tokenizer = _get_basetenkenizer_for_kimi(self)
            if fast_tokenizer is None or kwargs:
                return tokenizer_cls._original_batch_decode(
                    self,
                    sequences,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    **kwargs,
                )

            # Pre-filter using Kimi's own actual special-id set -- the same
            # cache `_SpecialTokensCachePatcher` maintains, and the same
            # approach `decode_without_hf_kwargs` already uses for the slow
            # path -- rather than trusting basetenkenizer's own
            # tokenizer.json-declared "special" flags, which were found to
            # disagree with Kimi's real runtime set for at least one token.
            if skip_special_tokens:
                special_ids = getattr(self, "all_special_ids_set", None)
                if special_ids is None:
                    special_ids = getattr(self, "all_special_ids", None)
                if special_ids is not None:
                    special_ids_set = set(special_ids)
                    sequences = [
                        [tid for tid in ids if tid not in special_ids_set]
                        for ids in sequences
                    ]

            return fast_tokenizer.decode_batch(sequences, skip_special_tokens=False)

        tokenizer_cls._encode_text_piece = patched_encode_text_piece
        tokenizer_cls.decode = patched_decode
        tokenizer_cls.__call__ = patched_call
        tokenizer_cls.batch_decode = patched_batch_decode
        tokenizer_cls.is_fast = property(lambda self: True)
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._encode_text_piece = tokenizer_cls._original_encode_text_piece
        tokenizer_cls.decode = tokenizer_cls._original_decode
        tokenizer_cls.__call__ = tokenizer_cls._original_call
        tokenizer_cls.batch_decode = tokenizer_cls._original_batch_decode
        tokenizer_cls.is_fast = property(tokenizer_cls._original_is_fast)

        del tokenizer_cls._original_encode_text_piece
        del tokenizer_cls._original_decode
        del tokenizer_cls._original_call
        del tokenizer_cls._original_batch_decode
        del tokenizer_cls._original_is_fast
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        logger.info(
            f"Unpatched basetenkenizer speed patch for {tokenizer_cls.__name__}"
        )
        return tokenizer
