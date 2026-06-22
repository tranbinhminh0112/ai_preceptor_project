"""
Question generation from the knowledge base.

Reuses the retriever to ground generation, then asks Qwen 2.5 to produce a
batch of QnA / MCQ / short-answer items with metadata (difficulty, domain,
knowledge source). Mirrors the logic from ai_model_interaction_v2.ipynb.
"""

from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama

from config import LLM_MODEL, LLM_TEMPERATURE_GENERATE


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    return cleaned


def _extract_json_payload(text: str) -> str:
    cleaned = _strip_code_fences(text)
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        json.loads(candidate)
        return candidate

    raise ValueError("The model response did not contain valid JSON.")


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
def _build_generation_context(retrieved_docs, max_docs=6) -> str:
    chunks = []
    for idx, doc in enumerate((retrieved_docs or [])[:max_docs], start=1):
        source = doc.metadata.get("source", "Unknown Source")
        hierarchy = doc.metadata.get("hierarchy", "Unknown Section")
        content = doc.metadata.get("original_content", doc.page_content)
        chunks.append(f"[Doc {idx}] Source: {source}\nHierarchy: {hierarchy}\nContent: {content}")
    return "\n\n---\n\n".join(chunks)


def _summarize_sources(retrieved_docs, max_items=8):
    seen = set()
    candidates = []
    for doc in retrieved_docs or []:
        source = doc.metadata.get("source", "Unknown Source")
        hierarchy = doc.metadata.get("hierarchy", "Unknown Section")
        asset = doc.metadata.get("asset", "None")
        key = (source, hierarchy, asset)
        if key in seen:
            continue
        seen.add(key)
        if asset and str(asset).lower() != "none":
            label = f"{source} | {hierarchy} | {asset}"
        else:
            label = f"{source} | {hierarchy}"
        candidates.append(label)
        if len(candidates) >= max_items:
            break
    return candidates


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_question_batch(
    retriever,
    topic="Mental health nursing",
    qna_count=2,
    mcq_count=2,
    short_answer_count=2,
    difficulty_levels=("easy", "medium", "hard"),
    domain_knowledge="mental health and nursing",
    max_context_docs=6,
    sources=None,
    vector_db=None,
):
    """
    Generate one grounded batch of questions.

    `retriever` is the hybrid retriever from RAGEngine.
    `sources` optionally restricts grounding to specific source documents; when
    given (with `vector_db`), retrieval is done straight from the vector store
    filtered to those sources instead of the full hybrid retriever.
    Returns a dict with keys: topic, qna, mcq, short_answer, all_questions,
    source_candidates.
    """
    retrieval_query = f"{topic} clinical questions, definitions, and learning objectives"

    sources = [s for s in (sources or []) if s]
    if sources and vector_db is not None:
        # Pull more candidates than we need so the chosen sources are well covered.
        flt = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": sources}}
        retrieved_docs = vector_db.similarity_search(
            retrieval_query, k=max(max_context_docs * 2, 8), filter=flt
        )
        if not retrieved_docs:
            raise RuntimeError(
                "No content found in the selected source(s). Try different sources or leave the filter empty."
            )
    else:
        retrieved_docs = retriever.invoke(retrieval_query)
        if not retrieved_docs:
            raise RuntimeError("No documents were retrieved for question generation.")

    source_candidates = _summarize_sources(retrieved_docs, max_items=max_context_docs)
    context_text = _build_generation_context(retrieved_docs, max_docs=max_context_docs)
    # format="json" makes Ollama emit grammar-constrained valid JSON, which is far
    # more reliable than asking for JSON in the prompt alone.
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE_GENERATE, format="json")

    system_prompt = "\n".join([
        "You are a senior nursing educator generating a question bank from the supplied clinical context.",
        "",
        "Create exactly:",
        f"- {qna_count} QnA items",
        f"- {mcq_count} MCQ items",
        f"- {short_answer_count} short-answer items",
        "",
        "Rules:",
        "1. Use ONLY the supplied context. Do not invent facts.",
        "2. Keep questions clear, clinically relevant, and non-duplicated.",
        "3. Each item must include these fields: question_type, question, answer, difficulty_level, domain_knowledge, knowledge_source.",
        "4. `question_type` must be exactly `qna`, `mcq`, or `short_answer`.",
        f"5. `difficulty_level` must be one of these values: {list(difficulty_levels)}.",
        f"6. `domain_knowledge` should stay within: {domain_knowledge}.",
        "7. `knowledge_source` must be one or more exact strings copied from the `SOURCE_CANDIDATES` list below.",
        "8. For MCQ items, include `options` as a list of 4 answer choices and `correct_answer` as the exact correct option text.",
        "9. Return JSON only. No markdown, no commentary, no code fences.",
        "10. Top-level keys must be: topic, qna, mcq, short_answer.",
        "11. Each of qna, mcq, short_answer must be a JSON array of OBJECTS (never plain strings), following EXAMPLE_FORMAT exactly.",
        "",
        "EXAMPLE_FORMAT (copy this structure, replace the values with real content):",
        json.dumps({
            "topic": "string",
            "qna": [{
                "question_type": "qna", "question": "string", "answer": "string",
                "difficulty_level": "easy", "domain_knowledge": "string",
                "knowledge_source": ["string"],
            }],
            "mcq": [{
                "question_type": "mcq", "question": "string",
                "options": ["A", "B", "C", "D"], "correct_answer": "A", "answer": "A",
                "difficulty_level": "medium", "domain_knowledge": "string",
                "knowledge_source": ["string"],
            }],
            "short_answer": [{
                "question_type": "short_answer", "question": "string", "answer": "string",
                "difficulty_level": "hard", "domain_knowledge": "string",
                "knowledge_source": ["string"],
            }],
        }, indent=2),
        "",
        "SOURCE_CANDIDATES:",
        json.dumps(source_candidates, indent=2, ensure_ascii=False),
        "",
        "CONTEXT:",
        context_text,
    ])

    user_prompt = (
        f"Generate the batch for topic: {topic}. "
        f"Return {qna_count} QnA items, {mcq_count} MCQ items, and {short_answer_count} short-answer items. "
        "Keep answers concise but complete."
    )

    # Try a couple of times — generation is stochastic and can occasionally
    # produce an unparseable response even with format="json".
    payload = None
    last_raw = ""
    last_error = None
    for attempt in range(3):
        response = llm.invoke([
            ("system", system_prompt),
            ("human", user_prompt),
        ])
        last_raw = response.content or ""
        try:
            payload = json.loads(_extract_json_payload(last_raw))
            break
        except Exception as exc:
            last_error = exc

    if payload is None:
        snippet = last_raw.strip().replace("\n", " ")[:200] or "(empty response)"
        raise ValueError(
            f"The model did not return valid JSON after 3 attempts. "
            f"Last error: {last_error}. Response started with: {snippet}"
        )

    def _normalize_item(item, expected_type):
        # Be tolerant of imperfect model output: an item may arrive as a bare
        # string, or fields may be missing / wrongly typed.
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            return None

        question = str(item.get("question") or item.get("q") or "").strip()
        if not question:
            return None

        knowledge_source = item.get("knowledge_source", [])
        if isinstance(knowledge_source, str):
            knowledge_source = [knowledge_source]
        elif not isinstance(knowledge_source, list):
            knowledge_source = []

        options = item.get("options", [])
        if not isinstance(options, list):
            options = []

        return {
            "question_type": expected_type,
            "question": question,
            "answer": str(item.get("answer", "")),
            "difficulty_level": item.get("difficulty_level", "medium"),
            "domain_knowledge": item.get("domain_knowledge", domain_knowledge),
            "knowledge_source": [str(s) for s in knowledge_source],
            "options": [str(o) for o in options],
            "correct_answer": str(item.get("correct_answer", "")),
        }

    def _normalize_list(key, expected_type):
        raw = payload.get(key, [])
        if not isinstance(raw, list):
            raw = [raw]
        return [x for x in (_normalize_item(i, expected_type) for i in raw) if x]

    qna_items = _normalize_list("qna", "qna")
    mcq_items = _normalize_list("mcq", "mcq")
    short_items = _normalize_list("short_answer", "short_answer")

    return {
        "topic": payload.get("topic", topic),
        "qna": qna_items,
        "mcq": mcq_items,
        "short_answer": short_items,
        "all_questions": qna_items + mcq_items + short_items,
        "source_candidates": source_candidates,
    }
