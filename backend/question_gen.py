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
        candidate = cleaned[start : end + 1]
        json.loads(candidate)
        return candidate

    raise ValueError("The model response did not contain valid JSON.")


def _invoke_json(system_prompt: str, user_prompt: str, attempts: int = 3, temperature: float = LLM_TEMPERATURE_GENERATE):
    llm = ChatOllama(model=LLM_MODEL, temperature=temperature, format="json")
    last_raw = ""
    last_error = None
    for _ in range(attempts):
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        last_raw = response.content or ""
        try:
            return json.loads(_extract_json_payload(last_raw)), last_raw, None
        except Exception as exc:
            last_error = exc
    return None, last_raw, last_error


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


def _dedupe_items(items):
    seen = set()
    unique = []
    for item in items:
        key = (item.get("question_type"), item.get("question", "").strip().lower())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


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
    requested_counts = {
        "qna": int(qna_count),
        "mcq": int(mcq_count),
        "short_answer": int(short_answer_count),
    }

    sources = [s for s in (sources or []) if s]
    if sources and vector_db is not None:
        flt = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": sources}}
        retrieved_docs = vector_db.similarity_search(retrieval_query, k=max(max_context_docs * 2, 8), filter=flt)
        if not retrieved_docs:
            raise RuntimeError("No content found in the selected source(s). Try different sources or leave the filter empty.")
    else:
        retrieved_docs = retriever.invoke(retrieval_query)
        if not retrieved_docs:
            raise RuntimeError("No documents were retrieved for question generation.")

    source_candidates = _summarize_sources(retrieved_docs, max_items=max_context_docs)
    context_text = _build_generation_context(retrieved_docs, max_docs=max_context_docs)

    system_prompt = "\n".join(
        [
            "You are a senior nursing educator generating a question bank from the supplied clinical context.",
            "",
            "Create exactly:",
            f"- {requested_counts['qna']} QnA items",
            f"- {requested_counts['mcq']} MCQ items",
            f"- {requested_counts['short_answer']} short-answer items",
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
            "11. Each of qna, mcq, short_answer must be a JSON array of OBJECTS.",
            "",
            "EXAMPLE_FORMAT:",
            json.dumps(
                {
                    "topic": "string",
                    "qna": [
                        {
                            "question_type": "qna",
                            "question": "string",
                            "answer": "string",
                            "difficulty_level": "easy",
                            "domain_knowledge": "string",
                            "knowledge_source": ["string"],
                        }
                    ],
                    "mcq": [
                        {
                            "question_type": "mcq",
                            "question": "string",
                            "options": ["A", "B", "C", "D"],
                            "correct_answer": "A",
                            "answer": "A",
                            "difficulty_level": "medium",
                            "domain_knowledge": "string",
                            "knowledge_source": ["string"],
                        }
                    ],
                    "short_answer": [
                        {
                            "question_type": "short_answer",
                            "question": "string",
                            "answer": "string",
                            "difficulty_level": "hard",
                            "domain_knowledge": "string",
                            "knowledge_source": ["string"],
                        }
                    ],
                },
                indent=2,
            ),
            "",
            "SOURCE_CANDIDATES:",
            json.dumps(source_candidates, indent=2, ensure_ascii=False),
            "",
            "CONTEXT:",
            context_text,
        ]
    )

    user_prompt = (
        f"Generate the batch for topic: {topic}. "
        f"Return {requested_counts['qna']} QnA items, {requested_counts['mcq']} MCQ items, "
        f"and {requested_counts['short_answer']} short-answer items. Keep answers concise but complete."
    )

    payload, last_raw, last_error = _invoke_json(system_prompt, user_prompt, attempts=3)
    payload = payload or {"topic": topic}

    def _normalize_item(item, expected_type):
        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            return None

        question = str(item.get("question") or item.get("q") or "").strip()
        if not question:
            return None

        knowledge_source = item.get("knowledge_source", item.get("knowledge_sources", item.get("source", [])))
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
            "answer": str(item.get("answer", item.get("sample_answer", item.get("correct_answer", "")))),
            "difficulty_level": str(item.get("difficulty_level", item.get("difficulty", "medium"))).lower() or "medium",
            "domain_knowledge": item.get("domain_knowledge", domain_knowledge),
            "knowledge_source": [str(s) for s in knowledge_source],
            "options": [str(o) for o in options],
            "correct_answer": str(item.get("correct_answer", item.get("correct_option", ""))),
        }

    def _normalize_list(raw_items, expected_type):
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            raw_items = []
        return [item for item in (_normalize_item(raw, expected_type) for raw in raw_items) if item]

    alias_map = {
        "qna": "qna",
        "qa": "qna",
        "mcq": "mcq",
        "multiple_choice": "mcq",
        "multiple-choice": "mcq",
        "short": "short_answer",
        "short_answer": "short_answer",
        "short-answer": "short_answer",
    }

    def _fallback_pool():
        pooled = payload.get("all_questions", payload.get("questions", []))
        if isinstance(pooled, dict):
            pooled = [pooled]
        if not isinstance(pooled, list):
            return {"qna": [], "mcq": [], "short_answer": []}

        buckets = {"qna": [], "mcq": [], "short_answer": []}
        for raw in pooled:
            if not isinstance(raw, dict):
                continue
            raw_type = str(raw.get("question_type", raw.get("type", ""))).strip().lower()
            expected_type = alias_map.get(raw_type)
            if not expected_type:
                continue
            normalized = _normalize_item(raw, expected_type)
            if normalized:
                buckets[expected_type].append(normalized)
        return buckets

    fallback = _fallback_pool()
    qna_items = _dedupe_items(_normalize_list(payload.get("qna", []), "qna") + fallback["qna"])
    mcq_items = _dedupe_items(_normalize_list(payload.get("mcq", []), "mcq") + fallback["mcq"])
    short_items = _dedupe_items(_normalize_list(payload.get("short_answer", []), "short_answer") + fallback["short_answer"])

    def _recover_missing_items(expected_type: str, remaining_count: int):
        if remaining_count <= 0:
            return []

        type_labels = {"qna": "QnA", "mcq": "MCQ", "short_answer": "short-answer"}
        example = {
            "qna": {
                "question_type": "qna",
                "question": "string",
                "answer": "string",
                "difficulty_level": "easy",
                "domain_knowledge": domain_knowledge,
                "knowledge_source": ["string"],
            },
            "mcq": {
                "question_type": "mcq",
                "question": "string",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "answer": "A",
                "difficulty_level": "medium",
                "domain_knowledge": domain_knowledge,
                "knowledge_source": ["string"],
            },
            "short_answer": {
                "question_type": "short_answer",
                "question": "string",
                "answer": "string",
                "difficulty_level": "hard",
                "domain_knowledge": domain_knowledge,
                "knowledge_source": ["string"],
            },
        }[expected_type]

        repair_system_prompt = "\n".join(
            [
                "You are repairing a grounded nursing question-generation batch.",
                f"Generate exactly {remaining_count} {type_labels[expected_type]} item(s).",
                "Use ONLY the supplied context.",
                "Do not repeat earlier questions.",
                "Return JSON only with top-level keys: topic, items.",
                f"Each item must match this structure exactly: {json.dumps(example, ensure_ascii=False)}",
                f"`difficulty_level` must be one of: {list(difficulty_levels)}.",
                "`knowledge_source` must use exact strings from SOURCE_CANDIDATES.",
                "",
                "SOURCE_CANDIDATES:",
                json.dumps(source_candidates, indent=2, ensure_ascii=False),
                "",
                "CONTEXT:",
                context_text,
            ]
        )
        repair_user_prompt = f"Generate {remaining_count} additional {type_labels[expected_type]} item(s) for topic: {topic}."
        repair_payload, _, _ = _invoke_json(repair_system_prompt, repair_user_prompt, attempts=2, temperature=0.0)
        if not repair_payload:
            return []
        repaired = repair_payload.get("items", repair_payload.get(expected_type, repair_payload.get("questions", [])))
        return _normalize_list(repaired, expected_type)

    qna_items.extend(_recover_missing_items("qna", max(0, requested_counts["qna"] - len(qna_items))))
    mcq_items.extend(_recover_missing_items("mcq", max(0, requested_counts["mcq"] - len(mcq_items))))
    short_items.extend(_recover_missing_items("short_answer", max(0, requested_counts["short_answer"] - len(short_items))))

    qna_items = _dedupe_items(qna_items)[: requested_counts["qna"]]
    mcq_items = _dedupe_items(mcq_items)[: requested_counts["mcq"]]
    short_items = _dedupe_items(short_items)[: requested_counts["short_answer"]]

    generated_counts = {
        "qna": len(qna_items),
        "mcq": len(mcq_items),
        "short_answer": len(short_items),
    }
    all_questions = qna_items + mcq_items + short_items

    if not all_questions and sum(requested_counts.values()) > 0:
        snippet = last_raw.strip().replace("\n", " ")[:200] or "(empty response)"
        raise RuntimeError(
            "The model returned no usable questions after retries and recovery. "
            f"Last parser error: {last_error}. Response started with: {snippet}"
        )

    generation_note = None
    if any(generated_counts[key] < requested_counts[key] for key in requested_counts):
        generation_note = (
            "The generator recovered a partial batch. "
            f"Returned counts: QnA {generated_counts['qna']}/{requested_counts['qna']}, "
            f"MCQ {generated_counts['mcq']}/{requested_counts['mcq']}, "
            f"Short answer {generated_counts['short_answer']}/{requested_counts['short_answer']}."
        )

    return {
        "topic": payload.get("topic", topic),
        "qna": qna_items,
        "mcq": mcq_items,
        "short_answer": short_items,
        "all_questions": all_questions,
        "source_candidates": source_candidates,
        "requested_counts": requested_counts,
        "generated_counts": generated_counts,
        "generation_note": generation_note,
    }
