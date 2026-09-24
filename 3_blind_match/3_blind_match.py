#!/usr/bin/env python3
"""
Fase 3 (step "Interrogazione Articoli", con classificazione diretta):
per ogni claim, legge le assertion prodotte dalla Fase 1 (JSONL) e le fonti
scaricate dalla Fase 2 (database Turso/libSQL, tabelle sources + claim_source_links),
e per ogni coppia (assertion, articolo) chiede a un LLM locale via Ollama di
classificare l'esito leggendo SOLO il testo dell'articolo.

Selezione dei documenti (--mapping, default "pooled"): tutti i documenti scaricati
per un claim finiscono in un pool unico, deduplicato per URL, e ogni domanda viene
interrogata sui più pertinenti per sovrapposizione lessicale. Le query di uno stesso
claim cercano tutte la stessa storia, quindi l'articolo che risponde alla domanda 3 è
spesso fra i risultati della query 1: legare rigidamente la domanda N ai soli
risultati della query N (--mapping strict, il comportamento storico) faceva perdere
quelle corrispondenze e produceva "Non verificabile" su fonti già scaricate.
La provenienza resta tracciata in "retrieved_by_query_index".

Metodologia di verifica (blind matching in due passi separati):
non si chiede al modello di classificare direttamente l'"assertion" originale
(rischio di bias/leading, e la formulazione dell'assertion appartiene all'articolo
di partenza, non a quello candidato). Si procede invece con DUE chiamate distinte:

  1. LETTURA CIECA — il modello riceve solo la "question" e il testo dell'articolo
     candidato, e ne estrae la risposta con una citazione verbatim. La "answer" di
     riferimento NON è nel prompt.
  2. CONFRONTO — il modello riceve la domanda, la risposta appena estratta e la
     "answer" di riferimento (mai l'articolo), e classifica l'esito.

La separazione non è cosmetica: con i due passi fusi in un unico prompt il modello
aveva la risposta attesa davanti mentre "leggeva" e finiva per restituirla come
citazione dell'articolo candidato, rendendo l'esito "Concorda" privo di valore.
Tenendo la risposta di riferimento fuori dal prompt di lettura, la contaminazione
diventa strutturalmente impossibile.

Output: per ogni claim, un file <output-dir>/<claim_id>.jsonl con una riga JSON per
ogni risultato di matching (uno per articolo effettivamente scaricato), scritta e
flushata su disco subito dopo essere stata calcolata — puoi quindi seguire l'avanzamento
in tempo reale con `tail -f matching_results/<claim_id>.jsonl`.

Prerequisiti:
    1) Ollama installato e in esecuzione (https://ollama.com)
    2) Modello scaricato: ollama pull qwen2.5:7b-instruct
    3) pip install requests libsql
    4) Un database Turso (stesso usato dalla Fase 3, vedi download_sources.py)

Uso:
    python blind_matching.py --questions claims_with_questions.jsonl \
        --output-dir matching_results
    (URL/token Turso da --turso-url/--turso-token oppure da env var
    TURSO_DATABASE_URL/TURSO_AUTH_TOKEN)
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import zlib
import requests

try:
    import libsql
except ImportError:
    print("ERRORE: manca il pacchetto 'libsql'. Installa con: pip install libsql", file=sys.stderr)
    sys.exit(1)

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DB_MAX_RETRIES = 4  # tentativi per ogni operazione sul DB Turso: c'e' sempre una
                     # richiesta HTTP di mezzo, un blip di rete non deve far
                     # fallire l'intero claim
DEFAULT_KEEP_ALIVE = "30m"  # tiene il modello caricato fra una chiamata e l'altra
                             # (default Ollama: 5m -> ricarica il modello da zero se
                             # passano piu' di 5 minuti fra due chiamate, es. per delay
                             # o download lenti fra un claim e l'altro)
DEFAULT_WORKERS = 4  # chiamate concorrenti a Ollama; combacia col default server-side
                      # di OLLAMA_NUM_PARALLEL nelle versioni recenti. Su CPU (anziche'
                      # GPU) il guadagno da parallelizzare puo' essere marginale o nullo:
                      # se non noti differenze rispetto a --workers 1, il collo di
                      # bottiglia e' il calcolo, non la concorrenza di rete.

def normalize_keep_alive(value):
    """Ollama vuole un numero (secondi, o -1 per 'sempre') oppure una stringa
    di durata CON unita' (es. "30m"). Una stringa puramente numerica come "-1"
    o "300" (quella che arriva da --keep-alive via CLI) va convertita in int,
    altrimenti Ollama la rifiuta con 400 ("missing unit in duration")."""
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    return value

# sessione HTTP condivisa: riusa la connessione TCP/keep-alive verso Ollama invece
# di aprirne una nuova a ogni requests.post()
_SESSION = requests.Session()
# pool piu' ampio del default (10): con --workers > 10 servono piu' connessioni
# aperte in parallelo verso Ollama, altrimenti requests le mette comunque in coda
_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)

MAX_ARTICLE_CHARS = 12000  # limite prudente per il contesto di un modello 7B (con num_ctx=8192 sotto)
DEFAULT_DOCS_PER_QUESTION = 5
DEFAULT_MIN_CONTENT_CHARS = 500  # sotto questa soglia e' quasi sempre uno snippet, non un articolo

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

SYSTEM_PROMPT_READ = """Sei un lettore di articoli.

Rispondi alla DOMANDA interrogando con quella, l'ARTICOLO. Non usare conoscenze esterne.

Usa null per entrambi SOLO se l'articolo non tratta affatto l'argomento della domanda.
Non confondere una semplice menzione di una persona/argomento con una risposta.

Esempio:
DOMANDA: "Quanti posti di lavoro ha creato la legge?"
ARTICOLO: "...la legge, secondo l'ufficio bilancio, dovrebbe generare circa 40.000 posti entro il 2025..."
{"risposta_articolo": "circa 40.000 posti entro il 2025", "evidenza": "dovrebbe generare circa 40.000 posti entro il 2025"}


Restituisci esclusivamente questo JSON:
{"risposta_articolo": "...", "evidenza": "..."}

"risposta_articolo" deve contenere la risposta alla domanda, anche se parziale.
"evidenza" deve essere una citazione breve, esatta e verbatim della porzione dell'articolo che risponde alla domanda. Se la risposta è null, anche "evidenza" deve essere null.
"""


SYSTEM_PROMPT_COMPARE = """Confronta RISPOSTA A e RISPOSTA B rispetto alla DOMANDA.

A proviene da un articolo trovato indipendentemente. B è la risposta di riferimento e rappresenta ciò che sostiene l'articolo originale, non necessariamente la verità.

Valuta il significato, non le parole o il livello di dettaglio.

Classifica il rapporto come:

* "Concorda": A e B esprimono la stessa informazione. A può essere più dettagliata o precisa.
* "Contraddice": A e B esprimono valori incompatibili sullo stesso fatto.
* "Parzialmente concorda": A riguarda lo stesso fatto, ma ne riporta solo una parte o omette dettagli rilevanti.
* "Non verificabile": A non contiene informazioni pertinenti alla domanda.

Una differenza di dettaglio non è una contraddizione. Se A e B possono essere vere entrambe, non scegliere "Contraddice".

Rispondi esclusivamente con:
{"esito": "...", "motivazione": "..."}
"""


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
                # in modalità claim il record non ha "title": si mostra il claim stesso
                "title": record.get("title") or record.get("claim", ""),
                "label": record.get("label"),
                "assertions": record.get("questions", []),
            }
    return claims


def _db_call(fn, *args, **kwargs):
    """Esegue una chiamata al DB sotto retry, per errori di rete transitori
    (a differenza di un file locale, qui c'e' sempre una richiesta HTTP di mezzo)."""
    last_err = None
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"    [DB retry {attempt}/{DB_MAX_RETRIES}] {e}", file=sys.stderr)
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"operazione sul DB Turso fallita dopo {DB_MAX_RETRIES} tentativi: {last_err}")


def open_turso(turso_url: str, turso_token: str):
    return libsql.connect(database=turso_url, auth_token=turso_token)


def fetch_claim_rows(conn, claim_id) -> list:
    """Righe (url, query, query_index, title, body_compresso, insufficient_content)
    per un claim, via JOIN fra claim_source_links e sources. La deduplica per URL
    e' gia' garantita dallo schema (PRIMARY KEY (claim_id, url) in
    claim_source_links, scritta cosi' dalla Fase 3): non serve rifarla qui."""
    def _run():
        cur = conn.execute(
            "SELECT csl.url, csl.query, csl.query_index, s.title, s.body, "
            "s.insufficient_content "
            "FROM claim_source_links csl JOIN sources s ON csl.url = s.url "
            "WHERE csl.claim_id = ?",
            (str(claim_id),),
        )
        return cur.fetchall()

    return _db_call(_run)


def _ollama_json(system_prompt: str, user_content: str, model: str, ollama_url: str,
                  required_field: str, max_retries: int = 3, timeout: int = 300,
                  num_predict: int = 600, keep_alive: str = DEFAULT_KEEP_ALIVE) -> dict:
    """Una chiamata a Ollama in modalita' chat con output JSON forzato, con retry.
    Solleva RuntimeError se dopo tutti i tentativi non arriva un JSON valido che
    contenga `required_field`."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        # con temperature=0 un errore di JSON malformato (es. virgolette non
        # escapate copiate verbatim dall'articolo) e' completamente deterministico:
        # riprovare con gli stessi parametri rigenera byte-per-byte lo stesso output
        # rotto. Dal secondo tentativo in poi alziamo leggermente la temperature
        # per dare al modello una reale possibilita' di uscire da quello stato.
        temperature = 0 if attempt == 1 else 0.4
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "format": "json",
            "stream": False,
            "keep_alive": normalize_keep_alive(keep_alive),
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 8192},
        }
        try:
            resp = _SESSION.post(ollama_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # fallback: prova a isolare il blocco {...} piu' esterno, nel caso
                # ci siano caratteri residui prima/dopo (raro con format=json, ma capita
                # con output troncati)
                start, end = content.find("{"), content.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise
                parsed = json.loads(content[start:end + 1])

            if required_field not in parsed:
                raise ValueError(f"campo '{required_field}' assente nella risposta del modello")
            return parsed
        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError,
                AttributeError, TypeError) as e:
            last_err = str(e)
            print(f"    [retry {attempt}/{max_retries}] chiamata Ollama fallita: {last_err}", file=sys.stderr)
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"risposta LLM non valida dopo {max_retries} tentativi: {last_err}")


def call_ollama(question_text: str, reference_answer: str, article_text: str, model: str, ollama_url: str,
                 max_retries: int = 3, timeout: int = 300, keep_alive: str = DEFAULT_KEEP_ALIVE) -> dict:
    """Blind matching in DUE chiamate separate, come previsto dall'assignment.

    PASSO 1 (lettura cieca): il modello riceve la domanda e l'articolo candidato, e
    NON vede la risposta di riferimento. Questo e' il punto: quando i due passi
    stavano in un unico prompt, il modello aveva la risposta attesa sotto gli occhi
    mentre "leggeva" e finiva per ricopiarla come se l'avesse trovata nell'articolo
    — nei test la citazione restituita era letteralmente la risposta di riferimento,
    e l'esito "Concorda" era quindi privo di valore. Separando le chiamate la
    contaminazione diventa strutturalmente impossibile: quel testo non e' nel prompt.

    PASSO 2 (confronto): il modello riceve solo la domanda, la risposta estratta al
    passo 1 e la risposta di riferimento — non l'articolo. E' un compito piccolo e
    ben delimitato, molto piu' alla portata di un modello 7B rispetto al prompt unico
    che chiedeva lettura, estrazione, citazione verbatim e classificazione insieme.
    """
    try:
        read = _ollama_json(
            SYSTEM_PROMPT_READ,
            f"DOMANDA:\n{question_text}\n\n"
            f"TESTO ARTICOLO CANDIDATO:\n{article_text[:MAX_ARTICLE_CHARS]}\n\n"
            f"Rispondi con il JSON richiesto.",
            model=model, ollama_url=ollama_url, required_field="risposta_articolo",
            max_retries=max_retries, timeout=timeout, keep_alive=keep_alive,
        )
    except RuntimeError as e:
        print(f"    [WARN] passo di lettura fallito: {e}", file=sys.stderr)
        return {"risposta_articolo": None, "esito": "Non verificabile", "evidenza": None,
                "motivazione": f"[ERRORE] {e}"}

    risposta_articolo = read.get("risposta_articolo")
    evidenza = read.get("evidenza")
    if isinstance(risposta_articolo, str) and not risposta_articolo.strip():
        risposta_articolo = None

    # se l'articolo non parla dell'argomento non c'e' nulla da confrontare:
    # risparmiamo la seconda chiamata
    if not risposta_articolo:
        return {"risposta_articolo": None, "esito": "Non verificabile", "evidenza": None,
                "motivazione": "l'articolo candidato non tratta l'argomento della domanda"}

    try:
        compare = _ollama_json(
            SYSTEM_PROMPT_COMPARE,
            f"DOMANDA:\n{question_text}\n\n"
            f"RISPOSTA A (estratta dall'articolo candidato):\n{risposta_articolo}\n\n"
            f"RISPOSTA B (di riferimento, secondo l'articolo originale sotto verifica):\n{reference_answer}\n\n"
            f"Rispondi con il JSON richiesto.",
            model=model, ollama_url=ollama_url, required_field="esito",
            max_retries=max_retries, timeout=timeout, num_predict=300, keep_alive=keep_alive,
        )
    except RuntimeError as e:
        print(f"    [WARN] passo di confronto fallito: {e}", file=sys.stderr)
        return {"risposta_articolo": risposta_articolo, "esito": "Non verificabile",
                "evidenza": evidenza, "motivazione": f"[ERRORE] {e}"}

    esito = normalize_esito(compare.get("esito"))
    if esito is None:
        print(f"    [WARN] esito non riconosciuto: {compare.get('esito')!r}", file=sys.stderr)
        esito = "Non verificabile"

    return {
        "risposta_articolo": risposta_articolo,
        "esito": esito,
        "evidenza": evidenza,
        "motivazione": compare.get("motivazione", ""),
    }


_QUOTE_CHARS = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "'": "'", '"': '"',
})


def _strip_citation_noise(evidenza: str) -> str:
    """Ripulisce la citazione dai residui di sintassi JSON che il modello a volte
    lascia dentro il valore: virgolette di incapsulamento e la virgola finale del
    campo. Senza questo, una citazione corretta risulta "non verificata" solo per
    un carattere di troppo ai bordi."""
    return (evidenza or "").strip().strip(",").strip().strip("\"“”'‘’").strip()


def _normalize_for_evidence_check(text: str) -> str:
    """Normalizza spazi e varianti tipografiche di virgolette (dritte/curve,
    singole/doppie) a una forma comune, cosi' il confronto verbatim non fallisce
    solo perche' il modello ha scelto uno stile di virgolette diverso da quello
    dell'articolo originale (es. per evitare di rompere il JSON, vedi SYSTEM_PROMPT)."""
    text = text.translate(_QUOTE_CHARS)
    text = re.sub(r"[\"']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def check_evidence(evidenza, article_text: str) -> bool:
    """Verifica (anti-hallucination) che la citazione sia effettivamente presente
    nel testo dell'articolo, in modo esatto o con normalizzazione minima degli spazi
    e delle virgolette."""
    if not evidenza:
        return False
    normalized_article = _normalize_for_evidence_check(article_text)
    normalized_evidence = _normalize_for_evidence_check(_strip_citation_noise(evidenza))
    if not normalized_evidence:
        return False
    return normalized_evidence in normalized_article


# ---------------------------------------------------------------------------
# Selezione dei documenti da interrogare
#
# La Fase 3 scarica N risultati per OGNI query, e la versione precedente di questo
# script interrogava la domanda i-esima solo sui documenti recuperati dalla query
# i-esima. E' una perdita di recall notevole: le query di uno stesso claim cercano
# tutte la stessa storia, e l'articolo che risponde alla domanda 3 e' spesso fra i
# risultati della query 1 (tipicamente la cronaca completa dell'evento, che copre
# tutti i fatti). Con quel vincolo rigido quella domanda non lo vedeva mai e
# l'esito era "Non verificabile" pur avendo la fonte gia' scaricata.
#
# Qui i documenti del claim finiscono in un pool unico, deduplicato per URL, e ogni
# domanda viene confrontata con i piu' pertinenti per sovrapposizione lessicale.
# ---------------------------------------------------------------------------

_DOC_TOKEN_RE = re.compile(r"\$?\d[\d.,/]*%?|[A-Za-z][A-Za-z'’-]*")

_DOC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in", "into", "is",
    "it", "its", "of", "on", "or", "said", "say", "says", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "about", "according", "over", "under", "than", "there",
    "these", "those", "much", "many", "some", "any", "not", "no", "yes", "source", "report",
}


def _doc_tokens(text: str) -> list:
    return [t.strip(".,;:!?'’-").lower() for t in _DOC_TOKEN_RE.findall(text or "")]


def _is_specific(token: str) -> bool:
    """Numeri, importi e parole lunghe pesano di piu': sono i termini che davvero
    distinguono un articolo sullo stesso fatto da uno sullo stesso tema."""
    stripped = token.lstrip("$")
    return (bool(stripped) and stripped[0].isdigit()) or len(token) > 7


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL_RE = re.compile(r"https?://\S+")


def clean_article_text(text: str) -> str:
    """Toglie il contorno di navigazione dal testo estratto dalla pagina.

    Il contenuto che arriva da Tavily/trafilatura conserva spesso, prima
    dell'articolo vero, un blocco di menu, loghi e link social in sintassi
    markdown. Su un modello 7B quel rumore compete con il testo utile: nei test
    l'articolo giusto veniva dichiarato "non pertinente" pur contenendo la frase
    cercata poche righe piu' sotto. Qui si rimuovono immagini, URL e voci di menu,
    conservando il testo dei link (che spesso e' contenuto legittimo)."""
    if not text:
        return ""

    text = _MD_IMAGE_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _BARE_URL_RE.sub(" ", text)

    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        # voci di menu: riga di elenco molto corta, tipicamente una sola etichetta
        bullet = stripped.lstrip("*-•").strip()
        if stripped.startswith(("*", "-", "•")) and len(bullet.split()) <= 4:
            continue
        if len(stripped) <= 2:
            continue
        kept.append(stripped)

    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def load_claim_documents(turso_url, turso_token, claim_id, min_content_chars):
    conn = open_turso(turso_url, turso_token)
    rows = fetch_claim_rows(conn, claim_id)
    documents = []
    skipped_short = 0

    for url, query, query_index, title, body_blob, insufficient_content in rows:
        if insufficient_content:
            skipped_short += 1
            continue
        try:
            raw_body = zlib.decompress(body_blob).decode("utf-8") if body_blob else ""
        except (zlib.error, UnicodeDecodeError) as e:
            print(f"  [WARN] body illeggibile per {url}: {e}, salto")
            skipped_short += 1
            continue

        body = clean_article_text(raw_body)
        if len(body) < min_content_chars:
            skipped_short += 1
            continue

        documents.append({
            "url": url,
            "title": title or "",
            "query": query or "",
            "query_index": query_index,
            "body": body,
            "tokens": set(_doc_tokens(body[:MAX_ARTICLE_CHARS])),
        })

    if skipped_short:
        print(f"  [INFO] {skipped_short} documenti scartati perche' senza contenuto "
              f"sufficiente (< {min_content_chars} caratteri)")
    return documents


def score_document(document: dict, assertion: dict) -> float:
    """Quanto un documento e' promettente per QUESTA domanda: quota dei termini di
    domanda+risposta che compaiono nel testo, con peso maggiore a numeri e parole
    lunghe (i termini realmente discriminanti)."""
    needle = " ".join([
        assertion.get("question", ""), assertion.get("answer", ""), assertion.get("assertion", ""),
    ])
    terms = {t for t in _doc_tokens(needle) if t and t not in _DOC_STOPWORDS and len(t) > 2}
    if not terms:
        return 0.0

    total = matched = 0.0
    for term in terms:
        weight = 3.0 if _is_specific(term) else 1.0
        total += weight
        if term in document["tokens"]:
            matched += weight
    return matched / total if total else 0.0


def select_documents_for_assertion(documents: list, assertion: dict, assertion_index: int,
                                    max_docs: int, mapping: str) -> list:
    """Restituisce [(score, documento)] per la domanda data.

    - mapping="strict": comportamento storico, solo i documenti recuperati dalla
      query con lo stesso indice della domanda.
    - mapping="pooled": tutti i documenti del claim, ordinati per rilevanza. I
      documenti cercati proprio per questa domanda restano comunque prioritari a
      parita' di punteggio, cosi' la provenienza della ricerca continua a contare.
    """
    if mapping == "strict":
        own = [d for d in documents if d.get("query_index") == assertion_index]
        return [(score_document(d, assertion), d) for d in own][:max_docs]

    scored = [
        (score_document(d, assertion), 0 if d.get("query_index") == assertion_index else 1, d)
        for d in documents
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(score, document) for score, _, document in scored[:max_docs]]


def process_claim(turso_url, turso_token, claim_id: str, claim_data: dict, output_dir: str,
                   model: str, ollama_urls: list, delay: float, skip_existing: bool, timeout: int,
                   docs_per_question: int = DEFAULT_DOCS_PER_QUESTION, mapping: str = "pooled",
                   min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
                   keep_alive: str = DEFAULT_KEEP_ALIVE, workers: int = DEFAULT_WORKERS):
    folder_name = sanitize_folder_name(claim_id)
    output_path = os.path.join(output_dir, f"{folder_name}.jsonl")

    expected_rows = len(claim_data["assertions"]) * docs_per_question
    if skip_existing and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for line in f if line.strip())
        if existing_lines >= expected_rows:
            print(f"  -> risultato gia' completo ({output_path}, {existing_lines} righe), salto")
            return
        print(f"  -> trovato output incompleto ({existing_lines}/{expected_rows} righe), riprocesso il claim da capo")

    assertions = claim_data["assertions"]
    n_saved = 0

    documents = load_claim_documents(turso_url, turso_token, claim_id, min_content_chars=min_content_chars)
    if not documents:
        print(f"  [WARN] nessun documento utilizzabile per il claim {claim_id} nel DB, salto")
        return
    print(f"  {len(documents)} documenti utilizzabili nel pool del claim")

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for assertion_index, assertion in enumerate(assertions, start=1):
            assertion_text = assertion.get("assertion", "")
            # la domanda da porre all'articolo candidato e la risposta di riferimento
            # (secondo l'articolo originale) prodotte in Fase 2. Se per qualche voce
            # mancassero (dati vecchi generati prima di questa modifica), ripieghiamo
            # sull'assertion stessa per non bloccare il pipeline.
            question_text = assertion.get("question") or assertion_text
            reference_answer = assertion.get("answer") or assertion_text

            selected = select_documents_for_assertion(
                documents, assertion, assertion_index,
                max_docs=docs_per_question, mapping=mapping,
            )
            print(f"  domanda #{assertion_index} ({question_text[:60]}...): "
                  f"{len(selected)} documenti selezionati")

            # le chiamate ai documenti di questa domanda sono indipendenti fra loro
            # (stesso model/prompt, articoli diversi), quindi si sottomettono tutte
            # insieme al pool e si raccolgono i risultati NELL'ORDINE ORIGINALE
            # (non nell'ordine di completamento): la scrittura su disco resta
            # deterministica e identica a prima, solo le chiamate diventano concorrenti.
            # Con --workers 1 il comportamento e' identico alla versione sequenziale.
            #
            # Con piu' di un endpoint Ollama (una GPU ciascuno) i documenti vengono
            # smistati a rotazione fra tutti gli endpoint: il modello sta comodamente
            # su una sola T4 (15GB), quindi Ollama non lo spargerebbe mai da solo su
            # entrambe le GPU — l'unico modo per usarle davvero entrambe e' avere due
            # processi server separati e distribuire le chiamate fra i due.
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = [
                    executor.submit(
                        call_ollama, question_text, reference_answer, document["body"],
                        model=model, ollama_url=ollama_urls[idx % len(ollama_urls)],
                        timeout=timeout, keep_alive=keep_alive,
                    )
                    for idx, (_score, document) in enumerate(selected)
                ]

                for rank, ((score, document), future) in enumerate(zip(selected, futures), start=1):
                    verdict = future.result()
                    print(f"    [{rank}/{len(selected)}] {document['url'][:60]} "
                          f"(rilevanza {score:.2f}) {document['title'][:50]}")
                    evidenza_verificata = check_evidence(verdict["evidenza"], document["body"])
                    if verdict["evidenza"] and not evidenza_verificata:
                        print(f"      [WARN] evidenza restituita dal modello NON trovata verbatim nel testo "
                              f"(possibile hallucination)")

                    # "Non verificabile" significa che l'articolo non tratta l'argomento:
                    # se pero' il modello e' riuscito a estrarne una risposta, le due cose
                    # si contraddicono. Non riscriviamo il suo giudizio, ma lo segnaliamo:
                    # in Fase 5 una riga cosi' non va contata come "fonte che tace".
                    esito_incoerente = (
                        verdict["esito"] == "Non verificabile" and bool(verdict["risposta_articolo"])
                    )
                    if esito_incoerente:
                        print(f"      [WARN] esito 'Non verificabile' ma il modello ha comunque "
                              f"estratto una risposta dall'articolo")

                    result = {
                        "claim_id": claim_id,
                        "assertion_index": assertion_index,
                        "assertion": assertion_text,
                        "question": question_text,
                        "reference_answer": reference_answer,
                        "centrality": assertion.get("centrality"),
                        "provenance": assertion.get("provenance"),
                        "source_url": document["url"],
                        "source_title": document["title"],
                        # query che ha effettivamente recuperato questo documento: con il
                        # pooling puo' essere diversa da quella della domanda in esame,
                        # quindi la tracciabilita' richiesta dall'assignment ("sai perche'
                        # una fonte e' stata cercata") viene tenuta esplicita qui
                        "source_query": document["query"],
                        "retrieved_by_query_index": document["query_index"],
                        "selected_by": mapping,
                        "selection_rank": rank,
                        "relevance_score": round(score, 4),
                        "risposta_articolo": verdict["risposta_articolo"],
                        "esito": verdict["esito"],
                        "evidenza": verdict["evidenza"],
                        "evidenza_verificata": evidenza_verificata,
                        "esito_incoerente": esito_incoerente,
                        "motivazione": verdict["motivazione"],
                    }

                    print(f"      -> {result['esito']}"
                          + (" [evidenza NON verificata]" if verdict["evidenza"] and not evidenza_verificata else ""))

                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    os.fsync(out_f.fileno())
                    n_saved += 1

            # con chiamate concorrenti un delay per-documento non avrebbe senso (i
            # documenti partono gia' tutti insieme): il delay si applica una volta
            # per domanda, fra un gruppo di chiamate concorrenti e il successivo
            if delay > 0:
                time.sleep(delay)

    print(f"  -> {n_saved} matching salvati in {output_path}")


def build_rollup(output_dir: str, claims: dict, claim_ids: list, rollup_path: str) -> int:
    """Raccoglie i risultati per-fonte in una riga per claim.

    E' una pura riorganizzazione: conta gli esiti, non li combina. Il verdetto unico
    per claim richiede di decidere come pesare centralita', affidabilita' della fonte
    e fonti in conflitto — sono le scelte che spettano alla Fase 5, e anticiparle qui
    significherebbe deciderle di fatto senza averle discusse. Qui si prepara solo il
    materiale su cui quella formula lavorera', con `claim_id` pronto per il join con
    le label di riferimento quando saranno disponibili.

    I risultati vengono riletti da disco, non accumulati durante l'esecuzione, cosi'
    il riepilogo comprende anche i claim gia' processati in run precedenti e saltati
    da --skip-existing."""
    rows = 0
    with open(rollup_path, "w", encoding="utf-8") as out_f:
        for claim_id in claim_ids:
            claim_data = claims[claim_id]
            results_path = os.path.join(output_dir, f"{sanitize_folder_name(claim_id)}.jsonl")
            if not os.path.exists(results_path):
                continue

            with open(results_path, "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f if line.strip()]
            if not results:
                continue

            per_question = {}
            for result in results:
                index = result.get("assertion_index")
                entry = per_question.setdefault(index, {
                    "assertion_index": index,
                    "question": result.get("question", ""),
                    "reference_answer": result.get("reference_answer", ""),
                    "centrality": result.get("centrality"),
                    "n_fonti": 0,
                    "esiti": {},
                    "evidenze_verificate": 0,
                    "esiti_incoerenti": 0,
                })
                esito = result.get("esito")
                entry["n_fonti"] += 1
                entry["esiti"][esito] = entry["esiti"].get(esito, 0) + 1
                entry["evidenze_verificate"] += bool(result.get("evidenza_verificata"))
                entry["esiti_incoerenti"] += bool(result.get("esito_incoerente"))

            totali = {}
            for entry in per_question.values():
                for esito, n in entry["esiti"].items():
                    totali[esito] = totali.get(esito, 0) + n

            record = {
                "claim_id": claim_id,
                "claim": claim_data.get("title", ""),
                # presente solo se il dataset di partenza porta gia' una label
                "label": claim_data.get("label"),
                "n_domande": len(per_question),
                "n_fonti_totali": len(results),
                "esiti_totali": totali,
                "domande": [per_question[k] for k in sorted(per_question, key=lambda x: (x is None, x))],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True, help="Path al JSONL prodotto dalla Fase 2 (con campo 'questions'/assertion)")
    parser.add_argument("--turso-url", default=os.environ.get("TURSO_DATABASE_URL"),
                         help="URL del database Turso con le fonti scaricate dalla Fase 3 "
                              "(es. libsql://il-tuo-db.turso.io). Default: legge da env var "
                              "TURSO_DATABASE_URL.")
    parser.add_argument("--turso-token", default=os.environ.get("TURSO_AUTH_TOKEN"),
                         help="Auth token del database Turso. Default: legge da env var TURSO_AUTH_TOKEN.")
    parser.add_argument("--output-dir", default="matching_results", help="Cartella dove salvare i risultati del matching (default: matching_results)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Nome del modello Ollama da usare (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"URL dell'endpoint chat di Ollama (default: {OLLAMA_URL})")
    parser.add_argument("--ollama-urls", default=None,
                         help="Lista di endpoint Ollama separati da virgola, uno per GPU (es. "
                              "'http://localhost:11434/api/chat,http://localhost:11435/api/chat'). "
                              "I documenti di ogni domanda vengono smistati a rotazione fra tutti "
                              "gli endpoint indicati. Se impostato, ha la precedenza su --ollama-url.")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe del file di input (default: tutte)")
    parser.add_argument("--delay", type=float, default=0.0, help="Secondi di pausa tra una chiamata e l'altra (default: 0)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in secondi per ogni chiamata a Ollama (default: 300)")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE,
                         help="Per quanto tempo Ollama tiene il modello caricato in memoria fra una "
                              "chiamata e l'altra (default Ollama: 5m -> con pause piu' lunghe il "
                              "modello si ricarica da zero). Usa '-1' per tenerlo sempre caricato "
                              f"(default qui: {DEFAULT_KEEP_ALIVE})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Chiamate concorrenti a Ollama per domanda (default: {DEFAULT_WORKERS}). "
                              "Usa 1 per il comportamento sequenziale originale. Su GPU con VRAM "
                              "sufficiente il guadagno e' quasi lineare; su CPU puo' essere marginale "
                              "o nullo, dipende da quanti thread il calcolo puo' davvero usare")
    parser.add_argument("--no-skip-existing", action="store_true", help="Non saltare i claim gia' processati (riprocessa tutto)")
    parser.add_argument("--docs-per-question", type=int, default=DEFAULT_DOCS_PER_QUESTION,
                         help=f"Quanti documenti interrogare per ciascuna domanda (default: {DEFAULT_DOCS_PER_QUESTION})")
    parser.add_argument("--mapping", choices=["pooled", "strict"], default="pooled",
                         help="'pooled': ogni domanda viene confrontata con i documenti piu' pertinenti "
                              "fra TUTTI quelli scaricati per il claim (default). 'strict': solo i documenti "
                              "recuperati dalla query con lo stesso indice della domanda (comportamento storico)")
    parser.add_argument("--min-content-chars", type=int, default=DEFAULT_MIN_CONTENT_CHARS,
                         help=f"Scarta i documenti piu' corti di questa soglia, quasi sempre snippet di "
                              f"ranking anziche' articoli (default: {DEFAULT_MIN_CONTENT_CHARS})")
    parser.add_argument("--rollup-output", default=None,
                         help="Path del file riepilogativo con una riga per claim (default: "
                              "<output-dir>/rollup.jsonl). Conta gli esiti per domanda e per claim "
                              "senza combinarli in un verdetto unico: quella e' la Fase 5")
    parser.add_argument("--no-rollup", action="store_true",
                         help="Non generare il file riepilogativo")
    parser.add_argument("--time-budget-minutes", type=float, default=None,
                         help="Se impostato, lo script si ferma da solo (in modo pulito) dopo "
                              "questi minuti invece di farsi uccidere a meta' dal limite di "
                              "sessione della piattaforma. Il rollup viene comunque generato "
                              "su cio' che e' gia' su disco. Rilancia lo stesso comando per "
                              "riprendere: il resume salta i claim gia' processati.")
    args = parser.parse_args()

    if not args.turso_url or not args.turso_token:
        print("ERRORE: URL/token Turso mancanti. Passa --turso-url/--turso-token oppure imposta "
              "TURSO_DATABASE_URL/TURSO_AUTH_TOKEN.", file=sys.stderr)
        sys.exit(1)

    ollama_urls = [u.strip() for u in args.ollama_urls.split(",") if u.strip()] \
        if args.ollama_urls else [args.ollama_url]
    if len(ollama_urls) > 1:
        print(f"[INFO] distribuzione dei documenti su {len(ollama_urls)} endpoint Ollama: {ollama_urls}")

    claims = load_assertions(args.questions)
    os.makedirs(args.output_dir, exist_ok=True)

    claim_ids = list(claims.keys())
    if args.limit is not None:
        claim_ids = claim_ids[:args.limit]

    start_time = time.monotonic()
    stopped_early = False

    for i, claim_id in enumerate(claim_ids):
        if args.time_budget_minutes is not None and \
                (time.monotonic() - start_time) > args.time_budget_minutes * 60:
            stopped_early = True
            break
        claim_data = claims[claim_id]
        print(f"[{i}] id={claim_id} -> {claim_data['title']} ({len(claim_data['assertions'])} assertion)")
        process_claim(
            args.turso_url, args.turso_token, claim_id, claim_data, args.output_dir,
            model=args.model, ollama_urls=ollama_urls,
            delay=args.delay, skip_existing=not args.no_skip_existing, timeout=args.timeout,
            docs_per_question=args.docs_per_question, mapping=args.mapping,
            min_content_chars=args.min_content_chars, keep_alive=args.keep_alive,
            workers=args.workers,
        )

    if stopped_early:
        remaining = len(claim_ids) - i
        print(f"\n[TIME BUDGET] Limite di tempo raggiunto: {remaining} claim non ancora "
              f"processati. Rilancia lo stesso comando (Save & Run All) per riprendere da "
              f"dove si e' fermato.")

    if not args.no_rollup:
        rollup_path = args.rollup_output or os.path.join(args.output_dir, "rollup.jsonl")
        n = build_rollup(args.output_dir, claims, claim_ids, rollup_path)
        print(f"\nRiepilogo: {n} claim scritti in {rollup_path}")

    print("\nFatto.")


if __name__ == "__main__":
    main()
