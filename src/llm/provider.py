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
from time import perf_counter

import anthropic
from dotenv import load_dotenv

from src.config import get_settings

from dataclasses import dataclass

# Load .env file so ANTHROPIC_API_KEY is available
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached Anthropic client
# ---------------------------------------------------------------------------
# Lazy singleton: client is created on first call_llm() instance and is
# reused across the rest of the process. Anthropic SDK is designed to be
# a long-lived object - reusing one client lets it pool HTTP connections
# instead of opening a fresh one per request.
# Trade off:
#   We read ANTHROPIC_API_KEY once, on first use. If env var changes
#   mid-process, call _reset_client() before the next call_llm().
_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    # Return the cached Anthropic client, creating it on first call
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Ensure your .env file contains the key."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _reset_client() -> None:
    # Clear the cached client. Primarily used for internal tests
    global _client
    _client = None
    
     
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
# Response container
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """The full result of an LLM call.

    Wraps the model's text output along with the token usage from
    Anthropic's response, so downstream code can surface cost and
    token counts to the user without making a second API call.
    
    Attributes:
        text: Raw text response. Same content that earlier versions
            of call_llm() returned directly.
        input_tokens: Number of tokens in the prompt (system + message())
        output_tokens: Number of tokens in the model's response.
        model: The actual model snapshot Anthropic resolved (e.g.
            "claude-sonnet-4-5-20250929" even when we asked for the
            "claude-sonnet-4-5" alias). Useful for reproducibility -
            eval results can be tagged with exact snapshot used.
            Defaults to "" so test instances don't need to populate it.
        latency_s: Wall-clock seconds for API round-trip. Lets the
        pipeline distinguis "the LLM was slow" from "our orchestration
        was slow". Defaults to 0.0 so test fixtures don't need it.
    """
    text: str
    input_tokens: int
    output_tokens: int
    model: str = ""
    latency_s: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
        system_prompt: str,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
) -> LLMResponse:
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
        An LLMResponse with the raw text plus token usage. The text
        format depends on the caller's prompt (SQL, CANNOT_ANSWER,
        classification label, narration text, etc.).

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
  
    try:
        client = _get_client()

        # Wall-clock measurement around just the API call. perf_counter
        # is monotonic and high-resolution - right tool for timing
        # short network operations. Excludes _get_client() (cheap
        # after first call) and the response-parsing below (microseconds).
        start = perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=messages,
        )
        latency_s = perf_counter() - start

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
            f"latency={latency_s:.2f}s, "
            f"usage={response.usage.input_tokens}in/"
            f"{response.usage.output_tokens}out"
        )

        return LLMResponse(
            text=raw_output,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            latency_s=latency_s,
        )

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
