#!/usr/bin/env python3
"""
Fase 4 (step "Interrogazione Articoli", con classificazione diretta):
per ogni claim, legge le assertion prodotte dalla Fase 2 (JSONL) e le fonti
scaricate dalla Fase 3 (cartelle sources/<claim_id>/ con manifest.json + txt),
e per ogni coppia (assertion, articolo) chiede a un LLM locale via Ollama di
classificare l'esito leggendo SOLO il testo dell'articolo.

Il mapping assertion <-> articolo si basa sul query_index presente nel
manifest.json: query_index N corrisponde alla N-esima assertion nella lista
"questions" del claim (stesso ordine usato da download_sources.py per generare
i file q{N:02d}_r{M:02d}.txt).

Metodologia di verifica (blind matching per domanda/risposta):
per ogni articolo candidato, NON si chiede al modello di classificare
direttamente l'"assertion" originale (rischio di bias/leading, e la
formulazione dell'assertion appartiene all'articolo di partenza, non a quello
candidato). Si chiede invece al modello di rispondere alla "question" usando
ESCLUSIVAMENTE il testo dell'articolo candidato, e poi si confronta questa
risposta con la "answer" di riferimento (prodotta in Fase 2 leggendo l'articolo
originale) per determinare l'esito. Questo isola la lettura dell'articolo
candidato dalla formulazione della claim, riducendo l'ancoraggio del modello
all'assertion e rendendo il confronto più simile a un doppio cieco.

Output: per ogni claim, un file <output-dir>/<claim_id>.jsonl con una riga JSON per
ogni risultato di matching (uno per articolo effettivamente scaricato), scritta e
flushata su disco subito dopo essere stata calcolata — puoi quindi seguire l'avanzamento
in tempo reale con `tail -f matching_results/<claim_id>.jsonl`.

Prerequisiti:
    1) Ollama installato e in esecuzione (https://ollama.com)
    2) Modello scaricato: ollama pull qwen2.5:7b-instruct
    3) pip install requests

Uso:
    python blind_matching.py --questions claims_with_questions.jsonl \
        --sources-dir sources --output-dir matching_results
"""

import argparse
import json
import os
import re
import sys
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

MAX_ARTICLE_CHARS = 12000  # limite prudente per il contesto di un modello 7B (con num_ctx=8192 sotto)

VALID_ESITI = {"Concorda", "Contraddice", "Parzialmente concorda", "Non verificabile"}

# mappa forme normalizzate (minuscolo, senza spazi extra) -> etichetta canonica,
# per tollerare variazioni di maiuscole/minuscole o piccola difformita' nel testo
# che il modello potrebbe restituire nonostante le istruzioni
_ESITO_NORMALIZED = {re.sub(r"\s+", " ", e.strip().lower()): e for e in VALID_ESITI}


def normalize_esito(value):
    if not isinstance(value, str):
        return None
    key = re.sub(r"\s+", " ", value.strip().lower())
    return _ESITO_NORMALIZED.get(key)

SYSTEM_PROMPT = """Sei un sistema di fact-checking. Ricevi:
1. una DOMANDA su un fatto specifico;
2. una RISPOSTA DI RIFERIMENTO a quella domanda, ricavata da un articolo diverso \
(l'articolo originale sotto verifica) — NON e' detto che sia vera, e' solo cio' che \
quell'articolo sostiene;
3. il testo di un ARTICOLO CANDIDATO, trovato in modo indipendente, che potrebbe o \
meno parlare dello stesso fatto.

Il tuo compito e' in due passi, usando ESCLUSIVAMENTE le informazioni contenute \
nell'ARTICOLO CANDIDATO (mai la tua conoscenza pregressa o generale, mai informazioni \
implicite non scritte nel testo):

PASSO 1 — Rispondi alla DOMANDA basandoti SOLO sull'articolo candidato. Se l'articolo \
non risponde alla domanda, la risposta e' che l'articolo non ne parla.

PASSO 2 — Confronta la risposta che hai appena estratto dall'articolo candidato con la \
RISPOSTA DI RIFERIMENTO fornita, e classifica l'esito:
- "Concorda": la risposta dell'articolo candidato conferma sostanzialmente la stessa \
informazione della risposta di riferimento.
- "Contraddice": la risposta dell'articolo candidato e' in disaccordo/contrasto con la \
risposta di riferimento (es. numero, data, nome, o esito diverso).
- "Parzialmente concorda": l'articolo candidato conferma solo una parte dell'informazione, \
oppure la conferma con dettagli discordanti su un aspetto secondario.
- "Non verificabile": l'articolo candidato non risponde alla domanda, o non \
fornisce abbastanza informazione per rispondere.

Rispondi SOLO con un oggetto JSON con questa struttura esatta, senza testo aggiuntivo \
prima o dopo, senza markdown/backtick:

{"risposta_articolo": "<risposta alla domanda secondo SOLO l'articolo candidato, oppure \
null se l'articolo non risponde alla domanda>", \
"esito": "Concorda" | "Contraddice" | "Parzialmente concorda" | "Non verificabile", \
"evidenza": "<citazione ESATTA e VERBATIM copiata dal testo dell'articolo candidato che \
supporta la risposta_articolo, oppure null se esito e' 'Non verificabile'>", \
"motivazione": "<breve spiegazione in una frase del confronto tra risposta_articolo e \
risposta di riferimento>"}

Regole vincolanti:
- "risposta_articolo" deve derivare ESCLUSIVAMENTE dal testo dell'articolo candidato, mai \
dalla risposta di riferimento e mai da conoscenza esterna, anche se sai che la risposta di \
riferimento e' vera o falsa da altre fonti.
- "evidenza" deve essere una citazione letterale copiata parola per parola dal testo \
dell'articolo candidato (serve per una verifica automatica tramite string-matching). \
Non parafrasare, non tradurre, non correggere refusi.
- "evidenza" deve essere BREVE: al massimo una frase o circa 250 caratteri. Se il passaggio \
rilevante e' piu' lungo, scegli la porzione minima che basta a supportare la risposta.
- Se l'articolo candidato non risponde alla domanda, usa esito "Non verificabile", \
risposta_articolo null ed evidenza null.
- Non usare in nessun caso conoscenza esterna all'articolo candidato fornito."""


def sanitize_folder_name(value) -> str:
    name = str(value).strip()
    name = re.sub(r"[^\w\-.]", "_", name)
    return name or "unknown_id"


def load_assertions(questions_path: str) -> dict:
    """Carica il JSONL di Fase 2 e restituisce {claim_id: [assertion_dict, ...]}."""
    claims = {}
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            claim_id = str(record.get("id"))
            claims[claim_id] = {
                "title": record.get("title", ""),
                "label": record.get("label"),
                "assertions": record.get("questions", []),
            }
    return claims


def parse_source_file(filepath: str) -> dict:
    """Legge un file q{N}_r{M}.txt e separa header (URL/TITLE/QUERY) dal corpo."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    url, title, query, body = "", "", "", raw
    if "\n---\n" in raw:
        header, body = raw.split("\n---\n", 1)
        for line in header.splitlines():
            if line.startswith("URL: "):
                url = line[len("URL: "):]
            elif line.startswith("TITLE: "):
                title = line[len("TITLE: "):]
            elif line.startswith("QUERY: "):
                query = line[len("QUERY: "):]

    return {"url": url, "title": title, "query": query, "body": body.strip()}


def call_ollama(question_text: str, reference_answer: str, article_text: str, model: str, ollama_url: str,
                 max_retries: int = 3, timeout: int = 300) -> dict:
    """Chiama Ollama in modalita' chat, forzando output JSON. Chiede al modello di
    rispondere a `question_text` usando SOLO `article_text`, e di confrontare la
    risposta ottenuta con `reference_answer` per determinare l'esito. Ritorna il
    dict parsato, oppure un dict di fallback 'Non verificabile' in caso di errore
    persistente."""
    truncated = article_text[:MAX_ARTICLE_CHARS]

    user_content = (
        f"DOMANDA:\n{question_text}\n\n"
        f"RISPOSTA DI RIFERIMENTO (secondo l'articolo originale sotto verifica):\n{reference_answer}\n\n"
        f"TESTO ARTICOLO CANDIDATO:\n{truncated}\n\n"
        f"Rispondi con il JSON richiesto."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 600, "num_ctx": 8192},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(ollama_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # fallback: prova a isolare il blocco {...} piu' esterno, nel caso
                # ci siano caratteri residui prima/dopo (raro con format=json, ma capita
                # con output troncati)
                start = content.find("{")
                end = content.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise
                parsed = json.loads(content[start:end + 1])

            esito = normalize_esito(parsed.get("esito"))
            if esito is None:
                print(f"    [DEBUG] JSON ricevuto ma campo 'esito' assente/non valido. "
                      f"Contenuto grezzo restituito dal modello:\n{content}", file=sys.stderr)
                raise ValueError(f"esito non valido restituito dal modello: {parsed.get('esito')!r}")

            return {
                "risposta_articolo": parsed.get("risposta_articolo"),
                "esito": esito,
                "evidenza": parsed.get("evidenza"),
                "motivazione": parsed.get("motivazione", ""),
            }
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError,
                AttributeError, TypeError) as e:
            last_err = str(e)
            print(f"    [retry {attempt}/{max_retries}] chiamata Ollama fallita: {last_err}", file=sys.stderr)
            time.sleep(1.5 * attempt)

    print(f"    [WARN] Ollama non ha risposto correttamente dopo {max_retries} tentativi: {last_err}",
          file=sys.stderr)
    return {
        "risposta_articolo": None,
        "esito": "Non verificabile",
        "evidenza": None,
        "motivazione": f"[ERRORE] risposta LLM non valida dopo {max_retries} tentativi: {last_err}",
    }


def check_evidence(evidenza, article_text: str) -> bool:
    """Verifica (anti-hallucination) che la citazione sia effettivamente presente
    nel testo dell'articolo, in modo esatto o con normalizzazione minima degli spazi."""
    if not evidenza:
        return False
    normalized_article = re.sub(r"\s+", " ", article_text).strip()
    normalized_evidence = re.sub(r"\s+", " ", evidenza).strip()
    if not normalized_evidence:
        return False
    return normalized_evidence in normalized_article


def process_claim(claim_id: str, claim_data: dict, sources_dir: str, output_dir: str,
                   model: str, ollama_url: str, delay: float, skip_existing: bool, timeout: int):
    folder_name = sanitize_folder_name(claim_id)
    claim_folder = os.path.join(sources_dir, folder_name)
    manifest_path = os.path.join(claim_folder, "manifest.json")

    if not os.path.exists(manifest_path):
        print(f"  [WARN] nessun manifest.json trovato in {claim_folder}, salto claim {claim_id}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_path = os.path.join(output_dir, f"{folder_name}.jsonl")

    if skip_existing and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for line in f if line.strip())
        if existing_lines >= len(manifest):
            print(f"  -> risultato gia' completo ({output_path}, {existing_lines} righe), salto")
            return
        print(f"  -> trovato output incompleto ({existing_lines}/{len(manifest)} righe), riprocesso il claim da capo")

    assertions = claim_data["assertions"]
    n_saved = 0

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in manifest:
            query_idx = entry["query_index"]
            if query_idx < 1 or query_idx > len(assertions):
                print(f"  [WARN] query_index {query_idx} fuori range per claim {claim_id} "
                      f"({len(assertions)} assertion disponibili), salto file {entry['filename']}")
                continue

            assertion = assertions[query_idx - 1]
            assertion_text = assertion.get("assertion", "")
            # la domanda da porre all'articolo candidato e la risposta di riferimento
            # (secondo l'articolo originale) prodotte in Fase 2. Se per qualche voce
            # mancassero (dati vecchi generati prima di questa modifica), ripieghiamo
            # sull'assertion stessa per non bloccare il pipeline.
            question_text = assertion.get("question") or assertion_text
            reference_answer = assertion.get("answer") or assertion_text
            filepath = os.path.join(claim_folder, entry["filename"])

            if not os.path.exists(filepath):
                print(f"  [WARN] file mancante: {filepath}, salto")
                continue

            source = parse_source_file(filepath)

            if entry.get("content_length", 0) == 0 or not source["body"]:
                print(f"  {entry['filename']}: contenuto vuoto, esito automatico Non verificabile")
                verdict = {"risposta_articolo": None, "esito": "Non verificabile", "evidenza": None,
                           "motivazione": "contenuto della pagina non disponibile"}
                evidenza_verificata = False
            else:
                print(f"  {entry['filename']}: interrogo il modello con la domanda #{query_idx} "
                      f"({question_text[:60]}...)")
                verdict = call_ollama(question_text, reference_answer, source["body"], model=model,
                                       ollama_url=ollama_url, timeout=timeout)
                evidenza_verificata = check_evidence(verdict["evidenza"], source["body"])
                if verdict["evidenza"] and not evidenza_verificata:
                    print(f"    [WARN] evidenza restituita dal modello NON trovata verbatim nel testo "
                          f"(possibile hallucination)")

            result = {
                "claim_id": claim_id,
                "assertion_index": query_idx,
                "assertion": assertion_text,
                "question": question_text,
                "reference_answer": reference_answer,
                "centrality": assertion.get("centrality"),
                "provenance": assertion.get("provenance"),
                "source_url": source["url"],
                "source_title": source["title"],
                "source_query": entry.get("query", source["query"]),
                "source_filename": entry["filename"],
                "risposta_articolo": verdict["risposta_articolo"],
                "esito": verdict["esito"],
                "evidenza": verdict["evidenza"],
                "evidenza_verificata": evidenza_verificata,
                "motivazione": verdict["motivazione"],
            }

            print(f"    -> {result['esito']}"
                  + (" [evidenza NON verificata]" if verdict["evidenza"] and not evidenza_verificata else ""))

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            n_saved += 1

            if delay > 0:
                time.sleep(delay)

    print(f"  -> {n_saved} matching salvati in {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, help="Path al JSONL prodotto dalla Fase 2 (con campo 'questions'/assertion)")
    parser.add_argument("--sources-dir", default="sources", help="Cartella base con le sottocartelle per claim prodotte dalla Fase 3 (default: sources)")
    parser.add_argument("--output-dir", default="matching_results", help="Cartella dove salvare i risultati del matching (default: matching_results)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Nome del modello Ollama da usare (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"URL dell'endpoint chat di Ollama (default: {OLLAMA_URL})")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe del file di input (default: tutte)")
    parser.add_argument("--delay", type=float, default=0.0, help="Secondi di pausa tra una chiamata e l'altra (default: 0)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in secondi per ogni chiamata a Ollama (default: 300)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Non saltare i claim gia' processati (riprocessa tutto)")
    args = parser.parse_args()

    claims = load_assertions(args.questions)
    os.makedirs(args.output_dir, exist_ok=True)

    claim_ids = list(claims.keys())
    if args.limit is not None:
        claim_ids = claim_ids[:args.limit]

    for i, claim_id in enumerate(claim_ids):
        claim_data = claims[claim_id]
        print(f"[{i}] id={claim_id} -> {claim_data['title']} ({len(claim_data['assertions'])} assertion)")
        process_claim(
            claim_id, claim_data, args.sources_dir, args.output_dir,
            model=args.model, ollama_url=args.ollama_url,
            delay=args.delay, skip_existing=not args.no_skip_existing, timeout=args.timeout,
        )

    print("\nFatto.")


if __name__ == "__main__":
    main()