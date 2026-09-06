#!/usr/bin/env python3
"""
Fase 4 (step "Interrogazione Articoli", con classificazione diretta):
per ogni claim, legge le assertion prodotte dalla Fase 2 (JSONL) e le fonti
scaricate dalla Fase 3 (cartelle sources/<claim_id>/ con manifest.json + txt),
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

SYSTEM_PROMPT_READ = """Sei un lettore di articoli. Ricevi una DOMANDA e il testo di \
un ARTICOLO CANDIDATO. Il tuo unico compito e' rispondere alla domanda usando \
ESCLUSIVAMENTE cio' che e' scritto nell'articolo — mai la tua conoscenza pregressa, mai \
informazioni implicite non presenti nel testo.

Non stai giudicando se l'articolo sia vero o falso, e non hai (giustamente) nessuna \
risposta attesa con cui confrontarti: devi solo riferire cosa dice questo testo \
sull'argomento della domanda.

Come leggere:
- Cerca nel testo tutto cio' che riguarda l'argomento della domanda e riportalo, anche se \
copre solo in parte quello che la domanda chiede, anche se e' espresso con parole diverse.
- Non pretendere di ritrovare una formulazione precisa: se l'articolo tratta lo stesso \
fatto in modo piu' generico o piu' sintetico, quella e' comunque la risposta dell'articolo.
- Solo se nel testo non c'e' proprio nulla sull'argomento, rispondi null.

Rispondi SOLO con un oggetto JSON con questa struttura esatta, senza testo aggiuntivo \
prima o dopo, senza markdown/backtick:

{"risposta_articolo": "<cio' che l'articolo dice sull'argomento della domanda, anche se \
parziale; null SOLO se l'articolo non ne parla affatto>", \
"evidenza": "<citazione ESATTA e VERBATIM copiata dal testo dell'articolo che supporta la \
risposta, oppure null se risposta_articolo e' null>"}

Regole vincolanti:
- "evidenza" deve essere una citazione letterale copiata parola per parola dall'articolo \
(serve per una verifica automatica tramite string-matching). Non parafrasare, non tradurre, \
non correggere refusi, non aggiungere virgolette di incapsulamento attorno alla citazione \
e non lasciarci dentro virgole o altri residui di sintassi JSON.
- "evidenza" deve essere BREVE: al massimo una frase o circa 250 caratteri. Se il passaggio \
rilevante e' piu' lungo, scegli la porzione minima che basta a supportare la risposta.
- IMPORTANTE per la validita' del JSON: se la citazione copiata contiene un carattere \
virgolette doppie (") al suo interno, DEVI escaparlo scrivendo \\" (backslash seguito da \
virgolette), esattamente come richiesto dallo standard JSON per le stringhe. Non scrivere \
mai un carattere " grezzo, non escapato, dentro il valore di un campo stringa: questo \
invaliderebbe l'intero JSON. Se preferisci, puoi anche sostituire le virgolette doppie \
interne con virgolette singole (') per evitare il problema, purche' il resto della \
citazione resti verbatim.
- Non usare in nessun caso conoscenza esterna all'articolo fornito."""


SYSTEM_PROMPT_COMPARE = """Confronti due risposte alla stessa DOMANDA e classifichi il \
loro rapporto.

- RISPOSTA A e' stata estratta da un articolo trovato in modo indipendente.
- RISPOSTA B e' la risposta di riferimento, cioe' cio' che sostiene l'articolo originale \
sotto verifica. NON e' detto che sia vera: e' solo la tesi da controllare.

Il confronto riguarda la SOSTANZA, non le parole usate ne' il livello di dettaglio. \
Due risposte formulate diversamente che indicano lo stesso fatto concordano.

Classifica l'esito confrontando A con B:
- "Concorda": A indica la stessa informazione di B. Rientra qui anche il caso in cui A \
sia PIU' RICCA o PIU' PRECISA di B: aggiungere dettagli che B non riporta non e' un \
disaccordo, e' la stessa risposta detta meglio. Se B dice "una rivista" e A dice "una \
rivista mensile statunitense fondata nel 1953", le due CONCORDANO.
- "Contraddice": A e B assegnano allo stesso attributo valori INCOMPATIBILI, tali che non \
possano essere veri entrambi (mese, anno, nome, luogo o esito diverso: "ottobre" contro \
"settembre", "greco" contro "americano"). Serve un'incompatibilita' reale: una differenza \
di formulazione o di grado di dettaglio non e' una contraddizione.
- "Parzialmente concorda": A riguarda lo stesso fatto e non lo smentisce, ma ne copre solo \
una parte, oppure diverge su un aspetto secondario mentre concorda su quello principale.
- "Non verificabile": A non riguarda l'argomento della domanda, quindi non permette di \
dire nulla su B.

Prima di scegliere, chiediti: "A e B possono essere vere entrambe?". Se si', l'esito NON \
puo' essere "Contraddice".

I due errori piu' frequenti, entrambi da evitare:
- classificare "Contraddice" quando A e' semplicemente piu' dettagliata o formulata in \
modo diverso da B, pur essendo compatibile con essa;
- classificare "Non verificabile" solo perche' A non contiene ESATTAMENTE il dettaglio di \
B: se A parla dello stesso fatto ma dice meno, l'esito e' "Parzialmente concorda".

Esempi (illustrativi: mostrano il criterio, non riusarne il contenuto).

1) A: "una rivista mensile statunitense fondata nel 1953" — B: "una rivista".
   Esito: "Concorda". A e' piu' specifica ma dice la stessa cosa.
2) A: "e' morto l'11 settembre 2003" — B: "e' morto in ottobre".
   Esito: "Contraddice". Stesso attributo, valori che non possono coesistere.
3) A: "l'organismo di controllo avra' sette membri" — B: "il territorio dovra' versare 370 \
milioni in cinque anni per il funzionamento di quell'organismo".
   Esito: "Parzialmente concorda". Stesso organismo, ma A non riporta la cifra.
4) A: "il documento tratta tutt'altro argomento" — B: qualunque.
   Esito: "Non verificabile".

Rispondi SOLO con un oggetto JSON con questa struttura esatta, senza testo aggiuntivo \
prima o dopo, senza markdown/backtick:

{"esito": "Concorda" | "Contraddice" | "Parzialmente concorda" | "Non verificabile", \
"motivazione": "<breve spiegazione in una frase del confronto fra A e B>"}"""


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


def _ollama_json(system_prompt: str, user_content: str, model: str, ollama_url: str,
                  required_field: str, max_retries: int = 3, timeout: int = 300,
                  num_predict: int = 600) -> dict:
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
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 8192},
        }
        try:
            resp = requests.post(ollama_url, json=payload, timeout=timeout)
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
                 max_retries: int = 3, timeout: int = 300) -> dict:
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
            max_retries=max_retries, timeout=timeout,
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
            max_retries=max_retries, timeout=timeout, num_predict=300,
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


def load_claim_documents(claim_folder: str, manifest: list, min_content_chars: int) -> list:
    """Carica una sola volta i documenti del claim, deduplicati per URL e scartando
    quelli senza contenuto utile (snippet di ranking, pagine bloccate da paywall)."""
    documents, seen_urls = [], set()
    skipped_short = 0

    for entry in manifest:
        filepath = os.path.join(claim_folder, entry["filename"])
        if not os.path.exists(filepath):
            print(f"  [WARN] file mancante: {filepath}, salto")
            continue

        url = entry.get("url", "")
        if url and url in seen_urls:
            continue

        source = parse_source_file(filepath)
        body = clean_article_text(source["body"])
        if len(body) < min_content_chars or entry.get("insufficient_content"):
            skipped_short += 1
            continue

        if url:
            seen_urls.add(url)
        documents.append({
            "filename": entry["filename"],
            "url": url or source["url"],
            "title": entry.get("title") or source["title"],
            "query": entry.get("query", source["query"]),
            "query_index": entry.get("query_index"),
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


def process_claim(claim_id: str, claim_data: dict, sources_dir: str, output_dir: str,
                   model: str, ollama_url: str, delay: float, skip_existing: bool, timeout: int,
                   docs_per_question: int = DEFAULT_DOCS_PER_QUESTION, mapping: str = "pooled",
                   min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS):
    folder_name = sanitize_folder_name(claim_id)
    claim_folder = os.path.join(sources_dir, folder_name)
    manifest_path = os.path.join(claim_folder, "manifest.json")

    if not os.path.exists(manifest_path):
        print(f"  [WARN] nessun manifest.json trovato in {claim_folder}, salto claim {claim_id}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

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

    documents = load_claim_documents(claim_folder, manifest, min_content_chars=min_content_chars)
    if not documents:
        print(f"  [WARN] nessun documento utilizzabile per il claim {claim_id}, salto")
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

            for rank, (score, document) in enumerate(selected, start=1):
                print(f"    [{rank}/{len(selected)}] {document['filename']} "
                      f"(rilevanza {score:.2f}) {document['title'][:50]}")
                verdict = call_ollama(question_text, reference_answer, document["body"], model=model,
                                       ollama_url=ollama_url, timeout=timeout)
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
                    "source_filename": document["filename"],
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
    parser.add_argument("--sources-dir", default="sources", help="Cartella base con le sottocartelle per claim prodotte dalla Fase 3 (default: sources)")
    parser.add_argument("--output-dir", default="matching_results", help="Cartella dove salvare i risultati del matching (default: matching_results)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Nome del modello Ollama da usare (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help=f"URL dell'endpoint chat di Ollama (default: {OLLAMA_URL})")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe del file di input (default: tutte)")
    parser.add_argument("--delay", type=float, default=0.0, help="Secondi di pausa tra una chiamata e l'altra (default: 0)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in secondi per ogni chiamata a Ollama (default: 300)")
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
            docs_per_question=args.docs_per_question, mapping=args.mapping,
            min_content_chars=args.min_content_chars,
        )

    if not args.no_rollup:
        rollup_path = args.rollup_output or os.path.join(args.output_dir, "rollup.jsonl")
        n = build_rollup(args.output_dir, claims, claim_ids, rollup_path)
        print(f"\nRiepilogo: {n} claim scritti in {rollup_path}")

    print("\nFatto.")


if __name__ == "__main__":
    main()