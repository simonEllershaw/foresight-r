import pytest
from foresight_r.inference.prompt_template import truncate_ehr, create_prompt


class MockTokenizer:
    """A simple mock tokenizer for testing."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        if text == "\n":
            return [198]  # Mock newline ID
        if text == "##":
            return [35, 35]  # Mock ## IDs

        # Determine tokens based on words
        if not text:
            return []

        # SIMULATE TOKEN MERGING:
        # If text contains "TemplateStart" and "TemplateEnd" directly adjacent (no space),
        # return 1 merged token.
        # Otherwise return them as separate tokens.
        if "TemplateStartTemplateEnd" in text:
            # This simulates the "Empty" state merging
            # We'll treat this whole string as 1 token
            # But we need to handle the rest of the string slightly carefully if simplistic
            return [9999]

        # Determine tokens
        tokens = text.split()
        ids = []
        for t in tokens:
            if t == "[BOS]":
                ids.append(1)
            elif t == "[EOS]":
                ids.append(2)
            elif t == "TemplateStart":
                ids.append(100)
            elif t == "TemplateEnd":
                ids.append(101)
            else:
                ids.append(999)  # Generic word ID

        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        # Return a string with that many words
        return " ".join(["word"] * len(token_ids))

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        # Simulate adding special tokens via template
        content = messages[0]["content"]
        # We construct a template that allows simulating the merge when content is empty
        # If content is empty: "TemplateStart" + "" + "TemplateEnd" -> "TemplateStartTemplateEnd"
        # If content is not empty: "TemplateStart" + " " + content + " " + "TemplateEnd" -> separate
        if not content:
            return "TemplateStartTemplateEnd"
        return f"TemplateStart {content} TemplateEnd"


@pytest.fixture
def tokenizer():
    return MockTokenizer()


def test_truncate_ehr_no_truncation_needed(tokenizer):
    task_description = "Task"
    ehr_text = "word " * 10
    max_prompt_length = 1000

    truncated = truncate_ehr(ehr_text, tokenizer, max_prompt_length, task_description)

    assert truncated == ehr_text


def test_truncate_ehr_truncation_needed(tokenizer):
    task_description = "Task"
    max_prompt_length = 50
    ehr_text = "word " * 100

    truncated = truncate_ehr(ehr_text, tokenizer, max_prompt_length, task_description)

    # Verify resulting prompt fits
    final_prompt = create_prompt(truncated, task_description, tokenizer)
    final_ids = tokenizer.encode(final_prompt)

    assert len(final_ids) <= max_prompt_length


def test_special_tokens_accounted_for(tokenizer):
    """Test that special tokens introduced by chat template are counted in overhead."""
    task_description = "Task"

    # NOTE: With our new MockTokenizer logic for this test file,
    # apply_chat_template now returns "TemplateStart ... TemplateEnd".
    # so we rely on that behavior for this test too.

    _ = create_prompt("", task_description, tokenizer)
    # prompt is "TemplateStartTemplateEnd" -> [9999] (1 token)

    # But for this test we want to measure "overhead" as the parts surrounding content.
    # The new logic in truncate_ehr calculates:
    # num_drop = len(full) - max.

    # Let's test the logic robustly.
    # Full prompt with content: TemplateStart (1) + Content (5) + TemplateEnd (1) = 7 tokens.
    # Empty prompt: TemplateStartTemplateEnd (1) = 1 token.

    # If we used OLD logic: Overhead = 1.
    # If Max = 6.
    # Old logic: Available = 6 - 1 = 5.
    # Content is 5 tokens ("a b c d e"). 5 <= 5. No truncation.
    # Resulting prompt: 1+5+1 = 7 tokens. 7 > 6. OVERFLOW!

    # NEW Logic:
    # Full = 7. Max = 6.
    # Drop = 7 - 6 = 1.
    # Content = 5.
    # Available = 5 - 1 = 4.
    # Truncate content to 4 tokens.
    # New prompt: 1 + 4 + 1 = 6 tokens. Fits!

    # Patch PROMPT_TEMPLATE to ensure overhead is small and predictable
    import foresight_r.inference.prompt_template

    original_template = foresight_r.inference.prompt_template.PROMPT_TEMPLATE
    foresight_r.inference.prompt_template.PROMPT_TEMPLATE = "{ehr}"

    try:
        max_length = 6
        ehr_text = "a b c d e"  # 5 tokens

        truncated = truncate_ehr(
            ehr_text,
            tokenizer,
            max_prompt_length=max_length,
            task_description=task_description,
        )

        # With new logic, it should truncate to 4 tokens
        truncated_ids = tokenizer.encode(truncated)
        assert len(truncated_ids) == 4

        final_prompt = create_prompt(truncated, task_description, tokenizer)
        final_ids = tokenizer.encode(final_prompt)
        assert len(final_ids) <= max_length
    finally:
        foresight_r.inference.prompt_template.PROMPT_TEMPLATE = original_template


def test_token_merging_edge_case(tokenizer):
    """
    Test specifically for the off-by-one error caused by token merging.
    Simulated by MockTokenizer merging "TemplateStart" and "TemplateEnd"
    only when directly adjacent (empty content).
    """
    task_description = "Task"

    # Scenario:
    # Empty Template: "TemplateStartTemplateEnd" -> 1 token [9999]
    # Filled Template: "TemplateStart" (1) + "word" (1) + "TemplateEnd" (1) -> 3 tokens

    # Case: Max Length = 2.
    # Content = "word" (1 token).

    # OLD Logic (Buggy):
    # Overhead = len(encode("TemplateStartTemplateEnd")) = 1.
    # Available = Max(2) - Overhead(1) = 1.
    # Content len = 1.
    # 1 <= 1. Return content as is.
    # Final Prompt: "TemplateStart word TemplateEnd" -> 3 tokens.
    # 3 > 2. FAILURE.

    # NEW Logic (Fixed):
    # Full Prompt = "TemplateStart word TemplateEnd" -> 3 tokens.
    # Drop = 3 - 2 = 1.
    # Content len = 1.
    # Available = 1 - 1 = 0.
    # Returns "" (empty string).
    # Final Prompt: "TemplateStartTemplateEnd" -> 1 token.
    # 1 <= 2. SUCCESS.

    # Monkeypatch PROMPT_TEMPLATE to be empty for this test so we control exact token counts
    import foresight_r.inference.prompt_template

    original_template = foresight_r.inference.prompt_template.PROMPT_TEMPLATE
    foresight_r.inference.prompt_template.PROMPT_TEMPLATE = "{ehr}"

    try:
        max_length = 2
        ehr_text = "word"

        # With max_length 2 and 3 tokens required (TemplateStart, word, TemplateEnd),
        # drop is 1. Available is 1 - 1 = 0.
        # This should now raise ValueError as 0 available tokens is considered invalid.
        with pytest.raises(ValueError, match="Available tokens"):
            truncate_ehr(
                ehr_text,
                tokenizer,
                max_prompt_length=max_length,
                task_description=task_description,
            )

    finally:
        # Restore
        foresight_r.inference.prompt_template.PROMPT_TEMPLATE = original_template


def test_overhead_calculation_is_token_based(tokenizer):
    """Regression test for the bug where overhead was char-based."""
    task_description = "Task"
    # Calculate overhead
    # In the bug: overhead was len(string).
    # In the fix: overhead is len(tokens).

    # Let's verify we can fit an EHR that WOULD be truncated by char count but FITS by token count.

    # 1. Get overhead string length
    dummy_prompt = create_prompt("", task_description, tokenizer)
    overhead_char_len = len(dummy_prompt)
    overhead_token_len = len(tokenizer.encode(dummy_prompt))

    # Assumption for this test to be meaningful:
    assert (
        overhead_char_len > overhead_token_len * 2
    )  # Standard text has more chars than words/tokens

    # Set max_length such that:
    # overhead_token_len + ehr_len <= max_length
    # overhead_char_len + ehr_len > max_length (if bug existed)

    # Example:
    # Overhead tokens = 50
    # Overhead chars = 300
    # EHR tokens = 60
    # Total tokens = 110
    # Set max_length = 120

    # If correct: 120 - 50 = 70 available. 60 fits.
    # If bug: 120 - 300 = negative! (Or just very small if positive).

    max_length = overhead_token_len + 100
    ehr_tokens_count = 50
    ehr_text = " ".join(["word"] * ehr_tokens_count)

    truncated = truncate_ehr(
        ehr_text,
        tokenizer,
        max_prompt_length=max_length,
        task_description=task_description,
    )

    # Should not be truncated
    assert truncated == ehr_text
