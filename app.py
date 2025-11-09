import chainlit as cl
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import os

# ======================================================
# ✅ Gemini Client
# ======================================================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool]
)


# ======================================================
# ✅ Format answer in clean markdown
# ======================================================
def format_markdown_answer(answer: str) -> str:
    return f"""
{answer}

---
"""


# ======================================================
# ✅ Add inline citations safely
# ======================================================
def add_citations(response):
    if not hasattr(response, "text"):
        return "No response text."

    text = response.text.strip()

    # Extract metadata safely
    try:
        meta = response.candidates[0].grounding_metadata
    except Exception:
        return text

    if meta is None:
        return text

    supports = getattr(meta, "grounding_supports", None)
    chunks = getattr(meta, "grounding_chunks", None)

    if not supports or not chunks:
        return text

    # Sort supports to avoid index shifting
    try:
        sorted_supports = sorted(
            supports, key=lambda s: s.segment.end_index, reverse=True
        )
    except Exception:
        return text

    for support in sorted_supports:
        seg = support.segment
        end_idx = seg.end_index

        if not support.grounding_chunk_indices:
            continue

        citation_links = []
        for idx in support.grounding_chunk_indices:
            if idx < len(chunks):
                try:
                    uri = chunks[idx].web.uri
                    citation_links.append(f"[{idx+1}]({uri})")
                except:
                    pass

        if citation_links:
            text = text[:end_idx] + " " + ", ".join(citation_links) + text[end_idx:]

    return text


# ======================================================
# ✅ Chainlit UI
# ======================================================

@cl.on_chat_start
async def start():
    await cl.Message("""
### 🤖 Gemini 2.5 Flash Chatbot  

""").send()


@cl.on_message
async def main(message: cl.Message):
    user_msg = message.content

    # Gemini API Call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_msg,
        config=config,
    )

    # Inline citations applied
    grounded_answer = add_citations(response)

    # Clean markdown version
    final_answer_md = format_markdown_answer(grounded_answer)

    # Display final message
    await cl.Message(content=final_answer_md).send()

    # ======================================================
    # ✅ Display Sources at the end
    # ======================================================
    try:
        meta = response.candidates[0].grounding_metadata

        if meta and meta.grounding_chunks:
            chunks = meta.grounding_chunks

            src_list = "\n".join(
                [
                    f"**[{i+1}]** [{chunk.web.title}]({chunk.web.uri})"
                    for i, chunk in enumerate(chunks)
                    if hasattr(chunk, "web")
                ]
            )

            await cl.Message(f"""
### 📚 Sources  
{src_list}
""").send()

    except Exception:
        pass
