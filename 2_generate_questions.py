#!/usr/bin/env python3
"""
Genera domande atomiche verificabili + risposte (derivate solo dal claim) per
ogni claim di un file JSONL, e per ciascuna coppia domanda/risposta genera anche
una query di ricerca ottimizzata (Variante B: query derivate dalle risposte),
usando un modello LLM locale servito da Ollama.

Prerequisiti:
    1) Ollama installato e in esecuzione (https://ollama.com)
    2) Un modello scaricato, es.:
         ollama pull qwen2.5:7b-instruct
    3) pip install requests

Uso:
    python generate_questions.py --input claims.jsonl --output claims_with_questions.jsonl

Formato output (per ogni riga):
    {
      "id": 89296,
      "claim": "Henry Spencer is played by a Greek actor.",
      "questions": [
        {
          "question": "What nationality is the actor who plays Henry Spencer?",
          "answer": "Greek",
          "centrality": 5,
          "provenance": "claim_text",
          "query": "Henry Spencer actor nationality Greek"
        },
        ...
      ]
    }
"""

import argparse
import json
import re
import sys
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

SYSTEM_PROMPT = """You are decomposing a claim into atomic question-answer pairs.

CRITICAL RULE: both the questions and the answers must be based STRICTLY AND ONLY
on the information explicitly stated in the claim text itself.
- Do NOT use any outside/world knowledge.
- Do NOT verify, fact-check, correct, or add information to the claim.
- Do NOT invent names, dates, locations, causes, or any detail that is not
  explicitly written in the claim.
- If the claim does not mention something (e.g. no location is given), do NOT ask
  about it.

Steps:
1. Identify the distinct atomic ASSERTIONS the claim is making (each claim usually
   makes between 1 and 5 assertions — a short claim may make only one; do not
   force extra questions if there is nothing more to ask).
   - Pay special attention to quantifiers, exclusivity, and superlative words
     (e.g. "only", "first", "last", "most", "best", "single", "exclusively",
     "unique", "never", "always"). These words usually carry the MAIN assertion
     of the claim, not a peripheral detail — they must always get their own
     dedicated question, and that question must have the highest centrality (5).
     Do not let a superlative/exclusivity claim collapse into a question only
     about its surrounding context (e.g. for "X is the ONLY drama series of
     2012", the central question is about the exclusivity itself, not just
     about the genre or the year in isolation).
2. For each assertion, write a QUESTION that asks specifically about that piece of
   information, phrased so that it could later be asked about a different,
   independent source (i.e. do not phrase it as a yes/no question about the claim
   itself, e.g. avoid "Is Henry Spencer played by a Greek actor?" — prefer
   "What nationality is the actor who plays Henry Spencer?"). For
   exclusivity/superlative assertions, phrase the question so it can be checked
   against an independent source, e.g. "Is [X] the only [category] released in
   [year], or are there others?"
3. Write the ANSWER to that question using ONLY the wording/information already
   present in the claim (verbatim or minimally rephrased). Never add facts that
   are not in the claim text.
4. Assign a CENTRALITY score from 1 to 5:
   - 5 = this question targets the core assertion of the claim (this always
     includes any exclusivity/superlative assertion, if present)
   - 1 = this question targets a marginal/peripheral detail

Respond with ONLY a valid JSON object, no markdown, no commentary, in exactly this
schema:

{
  "questions": [
    {"question": "...", "answer": "...", "centrality": 1-5}
  ]
}

Examples:
Claim: "John Ritter died in October."
Output: {"questions": [{"question": "In what month did John Ritter die?", "answer": "October", "centrality": 5}]}

Claim: "13 Reasons Why is the only television series of 2012 in the drama-mystery genre."
Output: {"questions": [
  {"question": "Is 13 Reasons Why the only drama-mystery television series released in 2012, or are there others?", "answer": "It is claimed to be the only one", "centrality": 5},
  {"question": "What genre is the television series 13 Reasons Why associated with in this claim?", "answer": "Drama-mystery", "centrality": 3},
  {"question": "In what year was 13 Reasons Why released, according to this claim?", "answer": "2012", "centrality": 3}
]}
"""

USER_PROMPT_TEMPLATE = "Claim: {claim}"


SYSTEM_PROMPT_QUERY = """You are formulating search-engine queries to help verify a
claim, given a list of question-answer pairs already derived from that claim.

For each question-answer pair, produce ONE optimized search-engine query string
that could be typed into Google/Bing to find independent sources confirming or
contradicting that specific piece of information.

Rules:
- The query must be KEYWORD-BASED (like something typed into a search engine),
  NOT a full grammatical question. E.g. prefer "Henry Spencer actor nationality"
  over "What nationality is the actor who plays Henry Spencer?".
- Always include the necessary named entities from the claim (proper names,
  titles, works) so each query is unambiguous even taken in isolation — do not
  rely on context from other queries.
- Include the salient keyword(s) from the answer itself (the fact being checked).
- For questions about exclusivity/superlatives (only, first, most...), formulate
  a query aimed at finding OTHER instances that could contradict the exclusivity
  (e.g. for "is X the only drama-mystery series of 2012", produce a query like
  "drama-mystery television series 2012 list", not just "X drama-mystery 2012").
- Keep each query concise, typically under 12 words.
- Use quotation marks only around an exact multi-word proper name/title you want
  matched exactly (e.g. "13 Reasons Why").
- Preserve the exact order of the input list: output exactly one query per input
  pair, in the same order, no more, no fewer.

Respond with ONLY a valid JSON object, no markdown, no commentary, in exactly this
schema:

{
  "queries": ["...", "...", ...]
}

Example:
Claim: "13 Reasons Why is the only television series of 2012 in the drama-mystery genre."
Questions: [
  {"question": "Is 13 Reasons Why the only drama-mystery television series released in 2012, or are there others?", "answer": "It is claimed to be the only one"},
  {"question": "What genre is the television series 13 Reasons Why associated with in this claim?", "answer": "Drama-mystery"},
  {"question": "In what year was 13 Reasons Why released, according to this claim?", "answer": "2012"}
]
Output: {"queries": [
  "drama-mystery television series 2012 list",
  "\\"13 Reasons Why\\" genre drama mystery",
  "\\"13 Reasons Why\\" release year 2012"
]}
"""

USER_PROMPT_QUERY_TEMPLATE = "Claim: {claim}\nQuestions: {questions_json}"


def call_ollama_questions(claim: str, model: str, max_retries: int = 3, timeout: int = 120) -> list:
    """Chiama il server Ollama locale e restituisce la lista di domande/risposte."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(claim=claim)},
        ],
        "format": "json",  # forza output JSON valido (supportato da Ollama)
        "stream": False,
        "options": {"temperature": 0.2},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)
            questions = parsed.get("questions", [])

            # validazione minima dello schema
            cleaned = []
            for q in questions:
                if not isinstance(q, dict):
                    continue
                question = str(q.get("question", "")).strip()
                answer = str(q.get("answer", "")).strip()
                centrality = q.get("centrality", 3)
                try:
                    centrality = int(centrality)
                except (TypeError, ValueError):
                    centrality = 3
                centrality = max(1, min(5, centrality))
                if question and answer:
                    cleaned.append({
                        "question": question,
                        "answer": answer,
                        "centrality": centrality,
                        "provenance": "claim_text",
                    })
            if cleaned:
                return cleaned
            last_err = "empty/invalid questions list"
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)

        print(f"  [retry {attempt}/{max_retries}] failed: {last_err}", file=sys.stderr)
        time.sleep(2 * attempt)

    print(f"  [WARN] giving up on claim after {max_retries} attempts: {last_err}", file=sys.stderr)
    return []


def call_ollama_queries(claim: str, questions: list, model: str, max_retries: int = 3, timeout: int = 120) -> list:
    """Genera una query di ricerca per ciascuna coppia domanda/risposta (Variante B).

    Restituisce una lista di stringhe della stessa lunghezza di `questions`, nello
    stesso ordine. In caso di fallimento dopo i retry, ritorna una lista di fallback
    costruita concatenando naively domanda+risposta, così il pipeline non si blocca.
    """
    if not questions:
        return []

    # passiamo al modello solo question+answer, non centrality/provenance, per non
    # sprecare token e non confonderlo con campi irrilevanti al task di query gen.
    qa_payload = [{"question": q["question"], "answer": q["answer"]} for q in questions]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_QUERY},
            {"role": "user", "content": USER_PROMPT_QUERY_TEMPLATE.format(
                claim=claim,
                questions_json=json.dumps(qa_payload, ensure_ascii=False),
            )},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)
            queries = parsed.get("queries", [])

            if isinstance(queries, list) and len(queries) == len(questions):
                cleaned = [str(q).strip() for q in queries]
                if all(cleaned):
                    return cleaned
                last_err = "one or more empty queries returned"
            else:
                last_err = f"expected {len(questions)} queries, got {len(queries) if isinstance(queries, list) else type(queries)}"
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)

        print(f"  [retry {attempt}/{max_retries}] query generation failed: {last_err}", file=sys.stderr)
        time.sleep(2 * attempt)

    print(f"  [WARN] falling back to naive queries after {max_retries} attempts: {last_err}", file=sys.stderr)
    # fallback naive: concatena domanda-chiave (senza punteggiatura) + risposta
    fallback = []
    for q in questions:
        naive = re.sub(r"[?]", "", q["question"]).strip()
        fallback.append(f"{naive} {q['answer']}".strip())
    return fallback


def process_file(input_path: str, output_path: str, model: str, limit: int = None):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            claim_id = record.get("id")
            claim = record.get("claim", "")

            print(f"[{i}] id={claim_id} -> {claim}")
            questions = call_ollama_questions(claim, model=model)

            if questions:
                queries = call_ollama_queries(claim, questions, model=model)
                for q, query in zip(questions, queries):
                    q["query"] = query

            # ricostruisco l'oggetto per inserire "questions" subito dopo "claim"
            new_record = {}
            for key, value in record.items():
                new_record[key] = value
                if key == "claim":
                    new_record["questions"] = questions

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            fout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path al file JSONL di input")
    parser.add_argument("--output", required=True, help="Path al file JSONL di output")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Nome modello Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe (utile per test)")
    args = parser.parse_args()

    process_file(args.input, args.output, model=args.model, limit=args.limit)
    print(f"\nFatto. Output scritto in: {args.output}")


if __name__ == "__main__":
    main()