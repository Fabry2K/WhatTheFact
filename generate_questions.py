#!/usr/bin/env python3
"""
Genera domande atomiche verificabili + risposte ipotizzate per ogni claim
di un file JSONL, usando un modello LLM locale servito da Ollama.

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
          "provenance": "claim_text"
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


def call_ollama(claim: str, model: str, max_retries: int = 3, timeout: int = 120) -> list:
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
            questions = call_ollama(claim, model=model)

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