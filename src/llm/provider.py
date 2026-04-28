"""
LLM provider for QueryMind - Anthropic Claude integration.

Wraps the Anthropic Messages API into a single function with a clean
interface - callable by the pipeline orchestrator. Designed to be swappable -
replacing this module with an OpenAI or Ollama equivalent only requires
matching the call_llm() function format.

Usage:
    from src.llm.provider import call_llm

    system_prompt = "You are a SQL expert..."
    messages = [{"role": "user", "content": "..."}]
    response = call_llm(system_prompt, messages)
"""

import os
import logging

import anthropic
from dotenv import load_dotenv

from src.config import get_settings

# Load .env file so ANTHROPIC_API_KEY is available
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model, max_tokens, and temperature are loaded from config/settings.yaml
# via src.config.get_settings(). See call_llm below.


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when the LLM call fails for any reason.
    
    Wraps all provider-specific exceptions (network errors, rate limits,
    content filtering) into a single exception type that the pipeline
    can catch without importing anthropic-specific errors.
    """
    pass

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
        system_prompt: str,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
) -> str:
    """Send a prompt to the Anthropic API and return the raw text response.
    
    This function is the single integration point between QueryMind and
    the LLM provider. The pipeline calls it with the system prompt and
    messages assembled by prompts.build_messages(), and receives back
    the raw LLM output (expected to be SQL or a CANNOT_ANSWER response).

    Args:
        system_prompt: The system-level instructions for the LLM.
        messages: List of message dicts with 'role' and 'content' keys,
            as produced by prompts.build_messages().
        model: Anthropic model identifier. If None, falls back to
            settings.llm.model.
        max_tokens: Maximum tokens in the response. If None, falls back
            to settings.llm.max_tokens.
        temperature: Sampling temperature (0.0 = deterministic, the
            default for SQL generation). If None, falls back to
            settings.llm.temperature.

    Returns:
        The raw text response from the LLM. Format depends on the caller's
        prompt (SQL, CANNOT_ANSWER, classification label, narration
        text, etc.).

    Raises:
        LLMError: If the API call fails for any reason (auth, network,
            rate limit, content filter, unexpected response format, etc.)
    """
    # Resolve defaults from config at call time. Callers can still override
    # any field explicitly; None here means "use the configured value".
    settings = get_settings().llm
    if model is None:
        model = settings.model
    if max_tokens is None:
        max_tokens = settings.max_tokens
    if temperature is None:
        temperature = settings.temperature

    # Validate API key is available
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError(
            "ANTHROPIC_API_KEY not found in environment variables. "
            "Ensure your .env file contains the key."
        )
    
    try:
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )

        # Extract text from the response.
        # Anthropic returns a list of content blocks - for text-to-SQL
        # we expect exactly one TextBlock.
        if not response.content:
            raise LLMError("LLM returned an empty response.")
        
        # Concatenate all text blocks (should realistically be one)
        text_parts = [
            block.text
            for block in response.content
            if block.type == "text"
        ]

        if not text_parts:
            raise LLMError(
                "LLM response contained no text blocks. "
                f"Content types: {[b.type for b in response.content]}"
            )

        raw_output = "\n".join(text_parts).strip()

        logger.info(
            f"LLM response received: {len(raw_output)} chars, "
            f"model={response.model}, "
            f"usage={response.usage.input_tokens}in/"
            f"{response.usage.output_tokens}out"
        )

        return raw_output
    
    except anthropic.AuthenticationError:
        raise LLMError(
            "Anthropic API authentication failed. Check your API key."
        )
    except anthropic.RateLimitError:
        raise LLMError(
            "Anthropic API rate limit exceeded. Wait a moment and retry."
        )
    except anthropic.APIConnectionError:
        raise LLMError(
            "Could not connect to the Anthropic API. Check your network."
        )
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error (status {e.status_code}): {e}")
    except LLMError:
        # Re-raise our own exception (instead of wrapping them again)
        raise
    except Exception as e:
        raise LLMError(f"Unexpected error during LLM call: {e}")
