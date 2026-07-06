"""Tests for text utilities module."""

import pytest


class TestExtractText:
    """Test extract_text function."""

    def test_extract_none(self):
        """Test extracting from None returns empty string."""
        from runLLMAgentForAgenticDevs import extract_text
        
        result = extract_text(None)
        assert result == ""

    def test_extract_string(self):
        """Test extracting from string returns same string."""
        from runLLMAgentForAgenticDevs import extract_text
        
        text = "Hello, world!"
        result = extract_text(text)
        assert result == text

    def test_extract_list_of_strings(self):
        """Test extracting from list of strings joins them."""
        from runLLMAgentForAgenticDevs import extract_text
        
        content = ["Hello", "World"]
        result = extract_text(content)
        assert result == "Hello\nWorld"

    def test_extract_list_with_dicts_text_type(self):
        """Test extracting from list with dict containing type=text."""
        from runLLMAgentForAgenticDevs import extract_text
        
        content = [
            "Prefix",
            {"type": "text", "text": "Middle"},
            "Suffix",
        ]
        result = extract_text(content)
        assert "Prefix" in result
        assert "Middle" in result
        assert "Suffix" in result

    def test_extract_list_with_dicts_text_key(self):
        """Test extracting from list with dict containing text key."""
        from runLLMAgentForAgenticDevs import extract_text
        
        content = [{"text": "Hello"}]
        result = extract_text(content)
        assert "Hello" in result

    def test_extract_list_mixed(self):
        """Test extracting from mixed list types."""
        from runLLMAgentForAgenticDevs import extract_text
        
        content = [
            "String",
            {"type": "text", "text": "Dict"},
            {"text": "Another"},
        ]
        result = extract_text(content)
        assert "String" in result
        assert "Dict" in result
        assert "Another" in result

    def test_extract_other_types_converted(self):
        """Test extracting from other types converts to string."""
        from runLLMAgentForAgenticDevs import extract_text
        
        result = extract_text(123)
        assert result == "123"


class TestHardClip:
    """Test hard_clip function."""

    def test_clip_empty_string(self):
        """Test clipping empty string."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        result = hard_clip("")
        assert result == ""

    def test_clip_within_limit(self):
        """Test clipping text within limit."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        text = "Hello"
        result = hard_clip(text, max_chars=100)
        assert result == text

    def test_clip_exceeds_limit(self):
        """Test clipping text exceeding limit."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        text = "Hello World This Is A Long String"
        result = hard_clip(text, max_chars=10)
        assert len(result) == 10
        assert result == "ong String"  # Last 10 chars

    def test_clip_exact_limit(self):
        """Test clipping text exactly at limit."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        text = "Hello"
        result = hard_clip(text, max_chars=5)
        assert result == text

    def test_clip_keeps_end(self):
        """Test that clipping keeps the end of the string."""
        from runLLMAgentForAgenticDevs import hard_clip
        
        text = "ABCDEFGHIJ"
        result = hard_clip(text, max_chars=3)
        assert result == "HIJ"  # Last 3 chars
