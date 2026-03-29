"""Unified LLM client factory using LangChain for model-agnostic access.

Supports OpenAI, Anthropic, Ollama, and any LangChain-compatible provider.
See: https://python.langchain.com/docs/integrations/chat/

Usage:
    from shared.llm_clients import get_chat_model, chat_completion, get_embeddings

    # Default (OpenAI gpt-4o-mini)
    answer = chat_completion("What is a knowledge graph?")

    # Switch provider
    answer = chat_completion("What is a knowledge graph?", provider="anthropic", model="claude-sonnet-4-20250514")

    # Get the raw LangChain model for advanced usage
    llm = get_chat_model(provider="openai", model="gpt-4o")
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from shared.config import get_openai_api_key, get_anthropic_api_key

# Provider type alias
Provider = Literal["openai", "anthropic", "ollama"]

# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.1",
}

DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}


def get_chat_model(
    provider: Provider = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """Get a LangChain chat model for any supported provider.

    Docs: https://python.langchain.com/docs/integrations/chat/

    Args:
        provider: One of "openai", "anthropic", "ollama"
        model: Model name. If None, uses the default for the provider.
        temperature: Sampling temperature (0.0 = deterministic)
        **kwargs: Additional provider-specific arguments

    Returns:
        A LangChain BaseChatModel instance
    """
    model = model or DEFAULT_MODELS[provider]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=get_openai_api_key(),
            **kwargs,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=get_anthropic_api_key(),
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            temperature=temperature,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'.")


def get_embeddings(
    provider: Provider = "openai",
    model: str | None = None,
    **kwargs,
) -> Embeddings:
    """Get a LangChain embeddings model.

    Docs: https://python.langchain.com/docs/integrations/text_embedding/

    Args:
        provider: One of "openai", "ollama"
        model: Model name. If None, uses the default for the provider.

    Returns:
        A LangChain Embeddings instance
    """
    model = model or DEFAULT_EMBEDDING_MODELS.get(provider, "text-embedding-3-small")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=model,
            api_key=get_openai_api_key(),
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, **kwargs)

    else:
        raise ValueError(f"Embeddings not supported for provider: {provider}")


def chat_completion(
    prompt: str,
    system: str = "You are a helpful assistant.",
    provider: Provider = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    response_format: dict | None = None,
) -> str:
    """Simple chat completion using LangChain — works with any provider.

    Args:
        prompt: The user message
        system: System message for the LLM
        provider: LLM provider ("openai", "anthropic", "ollama")
        model: Model name (defaults per provider)
        temperature: Sampling temperature
        response_format: JSON format spec (OpenAI-specific, ignored for other providers)

    Returns:
        The LLM's response as a string
    """
    kwargs = {}
    if response_format and provider == "openai":
        kwargs["model_kwargs"] = {"response_format": response_format}

    llm = get_chat_model(provider=provider, model=model, temperature=temperature, **kwargs)

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content


def get_embedding(
    text: str,
    provider: Provider = "openai",
    model: str | None = None,
) -> list[float]:
    """Get an embedding vector for a single text string.

    Args:
        text: Text to embed
        provider: Embedding provider ("openai", "ollama")
        model: Model name (defaults per provider)

    Returns:
        List of floats representing the embedding vector
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return embeddings.embed_query(text)


def get_embedding_batch(
    texts: list[str],
    provider: Provider = "openai",
    model: str | None = None,
) -> list[list[float]]:
    """Get embedding vectors for multiple texts efficiently.

    Args:
        texts: List of texts to embed
        provider: Embedding provider
        model: Model name

    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings(provider=provider, model=model)
    return embeddings.embed_documents(texts)


# ── Convenience: structured output with LangChain ──────────────────────

def chat_completion_structured(
    prompt: str,
    output_schema,
    system: str = "You are a helpful assistant.",
    provider: Provider = "openai",
    model: str | None = None,
    temperature: float = 0.0,
):
    """Chat completion that returns a structured Pydantic object.

    Uses LangChain's .with_structured_output() for type-safe extraction.
    Docs: https://python.langchain.com/docs/how_to/structured_output/

    Args:
        prompt: The user message
        output_schema: A Pydantic BaseModel class defining the output structure
        system: System message
        provider: LLM provider
        model: Model name
        temperature: Sampling temperature

    Returns:
        An instance of output_schema populated with the LLM's response
    """
    llm = get_chat_model(provider=provider, model=model, temperature=temperature)
    structured_llm = llm.with_structured_output(output_schema)

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]
    return structured_llm.invoke(messages)
