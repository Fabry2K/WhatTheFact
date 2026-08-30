#!/usr/bin/env python3
"""
Fase 2: genera domande atomiche verificabili + risposte (derivate SOLO dal testo
sorgente, mai da conoscenza esterna del modello) e, per ciascuna coppia
domanda/risposta, una query di ricerca ottimizzata (Variante B: query derivate
dalle risposte). Usa un modello LLM locale servito da Ollama.

Supporta due modalità:

  --mode claim (default se l'input NON è .csv)
      Input: JSONL con una riga per claim breve, es. {"id": 89296, "claim": "..."}
      Genera 1-5 domande per claim (un claim breve fa poche asserzioni).

  --mode document (default se l'input è .csv)
      Input: CSV con colonne (id opzionale);title;text;label(opzionale), es. un
      dataset di articoli di news. Genera 5-15 domande per documento (un articolo
      intero contiene molte più asserzioni di un claim breve), con chunking
      automatico per i testi troppo lunghi per il context window del modello.

Prerequisiti:
    1) Ollama installato e in esecuzione (https://ollama.com)
    2) Un modello scaricato, es.:
         ollama pull qwen2.5:7b-instruct
    3) pip install requests

Uso:
    # modalità claim (JSONL di claim brevi)
    python generate_questions.py --input claims.jsonl --output claims_with_questions.jsonl

    # modalità document (CSV di articoli interi)
    python generate_questions.py --input evaluation.csv --output evaluation_with_questions.jsonl --mode document

Formato output — modalità claim (per ogni riga):
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

Formato output — modalità document (per ogni riga):
    {
      "id": 0,
      "title": "Sanders back in U.S. Senate, blasts 'colonialism' in Puerto Rico",
      "label": "1",
      "questions": [
        {
          "assertion": "Bernie Sanders condemned the Puerto Rico bill as 'colonialism at its worst'",
          "centrality": 5,
          "query": "Bernie Sanders Puerto Rico bill colonialism",
          "provenance": "document_text"
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


SYSTEM_PROMPT_DOCUMENT = """You are given the title and text of a news article.

            Decompose the article into its key factual assertions. For each one, produce
            a short, atomic, independently verifiable search query to find sources that
            confirm or contradict it.

            Respond with ONLY a valid JSON object, no markdown, no commentary, in
            exactly this schema:

            {
              "assertions": [
                {"id": 1, "assertion": "short paraphrase of the fact", "centrality": 1-5, "query": "search-engine-style query"}
              ]
            }

            - "centrality" (1-5): 5 = the article's main claim/event, 3-4 = supporting
            facts, 1-2 = minor/background details.
            - "query": short and natural, not a full sentence or question.

            GROUNDING (critical): every entity, date, number, or name in a query must
            appear explicitly in the article. Never infer, round, or "correct" a date
            or number that seems plausible — if it's not stated in the text, leave it
            out. Before finalizing a query, check you could point to the exact sentence
            supporting each fact in it.

            ENTITIES: resolve nicknames, pejorative epithets, or informal monikers to
            the real name of the person/entity they refer to (e.g. "Mr. Teleprompter"
            → "Obama"). The query must use the real, searchable name — a query built
            around a nickname will not find real sources. The informal wording can stay
            in "assertion" for context, but never in "query".

            SCOPE: only extract assertions about the event/story being reported. Ignore
            metadata about the article itself (image credits, author bio, embedded
            media captions, formatting).

            ATOMICITY: one query = one verifiable fact. Don't combine unrelated facts
            into one query, and don't generate multiple queries for the same fact from
            different angles (pick the most specific one).

            Skip generic, trivial, or purely rhetorical details. Opinions/interpretations
            are fine to include if there's a verifiable fact underneath them (e.g. "X
            said Y"), even if the wording itself is rhetorical.

            Aim for ~4-8 queries depending on the article's density of distinct facts.
"""

USER_PROMPT_DOCUMENT_TEMPLATE = "Document title: {title}\nDocument text: {text}"


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


def split_into_chunks(text: str, max_words: int = 1800, overlap_words: int = 150) -> list:
    """Divide un testo lungo in chunk di circa `max_words` parole, con overlap,
    per stare dentro al context window del modello. Se il testo è già corto,
    restituisce una lista con un solo elemento (il testo intero)."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


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


def call_ollama_questions_document_chunk(title: str, text_chunk: str, model: str,
                                          max_retries: int = 3, timeout: int = 180) -> list:
    """Come call_ollama_questions, ma per un chunk di documento. Il prompt per
    documenti (SYSTEM_PROMPT_DOCUMENT) chiede al modello una LISTA JSON piatta
    [{"id":..., "assertion":..., "centrality":..., "query":...}], non l'oggetto
    {"questions": [...]} usato in modalità claim — il parsing qui sotto è
    specifico per questo schema."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_DOCUMENT},
            {"role": "user", "content": USER_PROMPT_DOCUMENT_TEMPLATE.format(title=title, text=text_chunk)},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)

            # il modello ora dovrebbe restituire {"assertions": [...]}; teniamo
            # comunque un fallback robusto nel caso avvolga diversamente o
            # restituisca una lista nuda
            if isinstance(parsed, dict) and isinstance(parsed.get("assertions"), list):
                items = parsed["assertions"]
            elif isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = next((v for v in parsed.values() if isinstance(v, list)), [])
            else:
                items = []

            cleaned = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                assertion = str(item.get("assertion", "")).strip()
                query = str(item.get("query", "")).strip()
                centrality = item.get("centrality", 3)
                try:
                    centrality = int(centrality)
                except (TypeError, ValueError):
                    centrality = 3
                centrality = max(1, min(5, centrality))
                if assertion and query:
                    cleaned.append({
                        "assertion": assertion,
                        "centrality": centrality,
                        "query": query,
                        "provenance": "document_text",
                    })
            if cleaned:
                return cleaned
            last_err = "empty/invalid assertions list"
            print(f"    [DEBUG] raw model content (primi 800 char): {content[:800]!r}", file=sys.stderr)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)
            try:
                print(f"    [DEBUG] raw response (primi 800 char): {resp.text[:800]!r}", file=sys.stderr)
            except Exception:
                pass

        print(f"    [retry {attempt}/{max_retries}] failed: {last_err}", file=sys.stderr)
        time.sleep(2 * attempt)

    print(f"    [WARN] giving up on chunk after {max_retries} attempts: {last_err}", file=sys.stderr)
    return []


def call_ollama_questions_document(title: str, text: str, model: str,
                                    max_words: int = 1800, overlap_words: int = 150) -> list:
    """Genera domande/risposte per un documento intero, spezzandolo in chunk se
    troppo lungo per il context window, e unendo i risultati di tutti i chunk."""
    chunks = split_into_chunks(text, max_words=max_words, overlap_words=overlap_words)

    all_questions = []
    for ci, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"  chunk {ci}/{len(chunks)} ({len(chunk.split())} parole)")
        chunk_questions = call_ollama_questions_document_chunk(title, chunk, model=model)
        for q in chunk_questions:
            if len(chunks) > 1:
                q["chunk_index"] = ci
        all_questions.extend(chunk_questions)

    return all_questions


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


def process_file_claims(input_path: str, output_path: str, model: str, limit: int = None):
    """Modalità 'claim': legge un JSONL con {"id", "claim"} per riga (comportamento originale)."""
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


def process_file_documents(input_path: str, output_path: str, model: str, limit: int = None,
                            max_words: int = 1800, overlap_words: int = 150,
                            include_text: bool = False, csv_delimiter: str = ";"):
    """Modalità 'document': legge un CSV con colonne (id opzionale);title;text;label
    (label opzionale) e genera domande/risposte sull'intero documento, con
    chunking automatico per i testi troppo lunghi."""
    import csv

    with open(input_path, "r", encoding="utf-8", newline="") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin, delimiter=csv_delimiter)
        # normalizza l'header: la prima colonna spesso non ha nome (indice riga)
        fieldnames = reader.fieldnames or []
        id_field = fieldnames[0] if fieldnames and fieldnames[0].strip() == "" else None

        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break

            doc_id = row.get(id_field) if id_field else row.get("id", i)
            if doc_id is None or doc_id == "":
                doc_id = i
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            label = row.get("label")

            if not text:
                print(f"[{i}] id={doc_id} -> [WARN] testo vuoto, salto")
                continue

            print(f"[{i}] id={doc_id} -> {title[:80]}")
            questions = call_ollama_questions_document(
                title, text, model=model, max_words=max_words, overlap_words=overlap_words,
            )
            # nota: con SYSTEM_PROMPT_DOCUMENT la query è già generata insieme
            # all'assertion in un unico step, quindi qui non serve più una
            # seconda chiamata a call_ollama_queries (a differenza della
            # modalità 'claim', dove restano due step separati)

            new_record = {
                "id": doc_id,
                "title": title,
            }
            if label is not None and label != "":
                new_record["label"] = label
            if include_text:
                new_record["text"] = text
            new_record["questions"] = questions

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"  -> {len(questions)} domande generate")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path al file di input (JSONL per --mode claim, CSV per --mode document)")
    parser.add_argument("--output", required=True, help="Path al file JSONL di output")
    parser.add_argument("--mode", choices=["claim", "document"], default=None,
                         help="'claim' per JSONL di claim brevi, 'document' per CSV di documenti interi. "
                              "Se omesso, viene dedotto dall'estensione del file di input (.csv -> document, altrimenti claim).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Nome modello Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe (utile per test)")
    parser.add_argument("--max-words", type=int, default=1800, help="[solo --mode document] dimensione massima (in parole) di un chunk di documento (default: 1800)")
    parser.add_argument("--overlap-words", type=int, default=150, help="[solo --mode document] overlap in parole tra chunk consecutivi (default: 150)")
    parser.add_argument("--include-text", action="store_true", help="[solo --mode document] include il testo completo del documento nell'output (di default omesso per tenere il file leggero)")
    parser.add_argument("--csv-delimiter", default=";", help="[solo --mode document] delimitatore del CSV (default: ';')")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        mode = "document" if args.input.lower().endswith(".csv") else "claim"
        print(f"[INFO] --mode non specificato, dedotto '{mode}' dall'estensione del file di input")

    if mode == "claim":
        process_file_claims(args.input, args.output, model=args.model, limit=args.limit)
    else:
        process_file_documents(
            args.input, args.output, model=args.model, limit=args.limit,
            max_words=args.max_words, overlap_words=args.overlap_words,
            include_text=args.include_text, csv_delimiter=args.csv_delimiter,
        )

    print(f"\nFatto. Output scritto in: {args.output}")


if __name__ == "__main__":
    main()