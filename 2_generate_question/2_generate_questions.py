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
          "question": "What did Bernie Sanders say about the Puerto Rico bill?",
          "answer": "He called it 'colonialism at its worst'",
          "provenance": "document_text"
        },
        ...
      ]
    }
"""

import argparse
import concurrent.futures
import difflib
import json
import os
import re
import sys
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_KEEP_ALIVE = "30m"  # tiene il modello caricato fra una chiamata e l'altra
                             # (default Ollama: 5m). Con piu' claim/documenti in
                             # parallelo (--workers) le chiamate diventano piu'
                             # fitte, ma senza questo un ritardo isolato basta a far
                             # scaricare il modello e ricaricarlo da zero
DEFAULT_WORKERS = 4  # claim/documenti processati in parallelo; ognuno fa 2-3
                      # chiamate Ollama in sequenza (decomposizione [+ retry] +
                      # query), ma claim diversi sono indipendenti fra loro

# sessione HTTP condivisa: riusa la connessione TCP/keep-alive verso Ollama invece
# di aprirne una nuova a ogni requests.post()
_SESSION = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
_SESSION.mount("http://", _adapter)
_SESSION.mount("https://", _adapter)


def normalize_keep_alive(value):
    """Ollama vuole un numero (secondi, o -1 per 'sempre') oppure una stringa di
    durata CON unita' (es. "30m"). Una stringa puramente numerica come "-1" o "300"
    va convertita in int, altrimenti Ollama la rifiuta con 400 ("missing unit in
    duration")."""
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


SYSTEM_PROMPT = """Starting with a claim, your task is to generate questions to verify the accuracy of the claim's main assertions.

You decompose a claim into atomic question-answer pairs.

Use STRICTLY AND ONLY information explicitly stated in the claim text.
Do not use outside knowledge, verify, correct, infer, or add details.
If a fact is not mentioned, do not ask about it.

For each distinct factual assertion, produce one QUESTION, its ANSWER, and a
CENTRALITY score. Extract approximately 1-5 assertions, depending on the claim;
do not force extra questions.

RULES:

1. ATOMICITY
Each pair must represent one distinct, verifiable fact. Do not combine unrelated
facts or create duplicate questions.

2. QUESTIONS
Questions must:
- be self-contained and name the relevant entities;
- target the property of a specific entity, so the answer is as unique as possible.
  For example, for "A is a painter", ask "What is A's profession?", not
  "Who is a painter?";

3. ANSWERS
Answer the question using only information stated in the claim, verbatim or
minimally rephrased. Never invent, infer, or add details.

4. CENTRALITY
Assign a score from 1 to 5:
- 5 = core assertion; exclusivity/superlative assertions always receive 5;
- 1 = marginal or peripheral detail.

Return ONLY valid JSON, with exactly this schema:
{
  "questions": [
    {"question": "...", "answer": "...", "centrality": 1}
  ]
}
"""

USER_PROMPT_TEMPLATE = "Claim: {claim}"


SYSTEM_PROMPT_DOCUMENT = """You are given the title and text of a news article.

Extract its key factual assertions. For each assertion, produce:
- "assertion": a short paraphrase of the fact;
- "centrality": 1-5, where 5 is the main claim/event;
- "question": a self-contained question that can be asked to an independent source article to check the same fact;
- "answer": the answer to that question according ONLY to this article.

Return ONLY valid JSON in this exact schema:
{
  "assertions": [
    {
      "id": 1,
      "assertion": "...",
      "centrality": 1,
      "question": "...",
      "answer": "..."
    }
  ]
}

RULES:

1. GROUNDING
Use only information explicitly stated in the article. Every name, entity, date,
number, and other detail in the question or answer must be supported by the text.
Never infer, correct, round, or invent information.

2. QUESTIONS
The question must:
- be self-contained and name the relevant entities;
- be neutral, open-ended, and start with What, Who, When, Where, How much,
  How many, Which, or Why;
- never be a yes/no question or start with Is, Are, Was, Were, Did, Does, Do,
  Has, Have, Can, or Will;
- ask about exactly the same single fact as the assertion.

The answer must directly answer the question according to this article.
It must never start with Yes or No. If the article does not provide the
requested detail, say "Not specified" rather than guessing.

3. SPECIFICITY
Prefer distinctive, verifiable details such as names, dates, amounts, counts,
quoted statements, and concrete actions. Preserve these details in the answer
when explicitly stated.

4. ENTITIES
Use real names as they appear in the article, not nicknames or informal
epithets, in questions and answers.

5. ATOMICITY AND SCOPE
Each entry must contain one distinct, verifiable fact. Avoid duplicates,
generic details, and article metadata. Extract only facts about the event or
story reported.

Aim for approximately 4-8 assertions, depending on the article's density.
"""

USER_PROMPT_DOCUMENT_TEMPLATE = "Document title: {title}\nDocument text: {text}"


SYSTEM_PROMPT_DOCUMENT_QUERY = """You generate search-engine queries to find independent
news sources reporting the SAME EVENT as the given article.

You receive the article TITLE and a numbered list of items, each containing an
ASSERTION, QUESTION, and ANSWER. Produce exactly ONE query per item, in the same order.

Return ONLY valid JSON:
{"queries": ["...", "..."]}

RULES:

1. EVENT ANCHOR
Include the main real entities, places, or organisations from the title or assertion.
The query must identify THIS specific event, not merely the general topic.

2. DISTINCTIVE DETAIL
Include a concrete detail from the answer whenever available: a name, amount, count,
date, distinctive phrase, or specific action. Do not rely on the anchor alone.

3. KEYWORD STYLE
Write concise keyword queries, not sentences or questions. Maximum 12 words.
Use double quotes only for exact multi-word names or distinctive quoted phrases.

4. AVOID GENERIC TERMS
Do not use vague words such as details, information, status, costs, funding,
performance, reaction, impact, situation, issue, statement, speech, story, news,
update, or overview as the main distinguishing detail. Replace them with concrete
facts from the answer.

5. NAMES AND EXCLUSIVITY
Use real names, not nicknames or epithets. For claims involving "only", "first",
or "most", formulate the query to help find counterexamples rather than merely
confirming the claim.

6. GROUNDING
Use only entities and details explicitly present in the title, assertion, or answer.
Do not invent missing information.

If the answer contains no useful concrete detail, use the strongest available
combination of event entities and the specific fact described by the assertion.
"""

USER_PROMPT_DOCUMENT_QUERY_TEMPLATE = "Article title: {title}\nItems:\n{items_json}"


SYSTEM_PROMPT_QUERY = """You generate search-engine queries for verifying factual claims.

You receive a list of QUESTION-ANSWER pairs. For each pair, produce exactly ONE
keyword-based query that can find independent sources confirming or contradicting
the specific fact. Preserve the input order.

Return ONLY valid JSON:
{"queries": ["...", "..."]}

RULES:

1. Include the relevant named entities (people, places, organisations, works, or
titles) so each query is unambiguous on its own.

2. Include the specific fact being checked, preferably using concrete details from
the answer: names, dates, amounts, counts, or distinctive quoted phrases.
Never rely only on generic terms such as details, information, story, status,
impact, or update.

3. Use concise search keywords, not grammatical questions. Keep queries under
12 words when possible.

4. For claims involving "only", "first", or "most", search for the broader category
or possible counterexamples rather than simply repeating the claim.

5. Use double quotation marks only around exact multi-word names or phrases that
should be matched verbatim. Never use single quotation marks.

6. Use only information present in the question or answer. Do not invent details.

Output exactly one query per input pair, no more and no fewer.
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


# ---------------------------------------------------------------------------
# Statistiche ricavate dal corpus
#
# Per decidere se una parola è un nome proprio o se è troppo comune per
# discriminare una ricerca servono due informazioni. Codificarle in liste scritte
# a mano funziona sul campione che si ha davanti e si rompe sul dataset dopo, oltre
# a spostare di fatto la conoscenza del dominio dentro il codice. Qui invece le due
# informazioni si misurano sul corpus che si sta processando:
#
#   - un NOME PROPRIO tende a comparire maiuscolo anche in mezzo alla frase
#     ("Playboy is a magazine" / "...the Playboy mansion"), mentre una parola
#     comune compare quasi sempre minuscola. Questo risolve anche l'ambiguità
#     della prima parola della frase, dove la maiuscola non dice nulla:
#     "Henry Spencer is..." vs "Alternative metal is...".
#   - un termine DISCRIMINANTE è raro nel corpus (IDF alto). "magazine" o "series"
#     compaiono ovunque e non restringono la ricerca; "Kaepernick" sì.
#
# Nessuna delle due dipende dalla lingua o dal dominio del dataset.
# ---------------------------------------------------------------------------

import math


class CorpusStats:
    """Frequenze e uso delle maiuscole misurati sul corpus in ingresso."""

    def __init__(self):
        self.n_docs = 0
        self.doc_freq = {}   # in quante frasi compare la parola
        self.cap_freq = {}   # quante volte maiuscola NON a inizio frase
        self.low_freq = {}   # quante volte minuscola

    def add(self, text: str) -> None:
        self.n_docs += 1
        seen = set()
        for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
            for position, token in enumerate(_TOKEN_RE.findall(sentence)):
                base = token.strip(".,;:!?'’\"-")
                if not base or not base[0].isalpha():
                    continue
                key = base.lower()
                seen.add(key)
                if base[0].isupper():
                    if position > 0:
                        self.cap_freq[key] = self.cap_freq.get(key, 0) + 1
                else:
                    self.low_freq[key] = self.low_freq.get(key, 0) + 1
        for key in seen:
            self.doc_freq[key] = self.doc_freq.get(key, 0) + 1

    def is_name(self, word: str) -> bool:
        """True se nel corpus la parola si comporta da nome proprio."""
        key = word.lower()
        cap, low = self.cap_freq.get(key, 0), self.low_freq.get(key, 0)
        if cap == 0 and low == 0:
            return True  # mai vista altrove: si accetta il segnale della maiuscola
        return cap > low

    def idf(self, word: str) -> float:
        """Quanto il termine è discriminante: alto = raro nel corpus."""
        if not self.n_docs:
            return 1.0
        return math.log(self.n_docs / (1 + self.doc_freq.get(word.lower(), 0)))

    @property
    def median_idf(self) -> float:
        if not self.doc_freq:
            return 0.0
        values = sorted(self.idf(w) for w in self.doc_freq)
        return values[len(values) // 2]


def build_corpus_stats(path: str, field: str = "claim", max_records: int = 50000) -> CorpusStats:
    """Costruisce le statistiche leggendo il file di input una volta sola."""
    stats = CorpusStats()
    with open(path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get(field) or ""
            if text:
                stats.add(text)
    print(f"[INFO] statistiche corpus: {stats.n_docs} testi, "
          f"{len(stats.doc_freq)} parole distinte")
    return stats


# particelle che stanno DENTRO un nome proprio senza spezzarlo ("Alice in Chains",
# "University of Texas", "Ludwig van Beethoven"): valgono solo se hanno un pezzo di
# nome sia prima sia dopo
_NAME_CONNECTORS = {"in", "of", "the", "and", "de", "del", "della", "di", "da", "van",
                    "von", "der", "den", "la", "le", "el", "al", "bin", "ibn", "y"}


def entity_phrases(text: str, stats: CorpusStats) -> list:
    """Sequenze di parole che il corpus riconosce come nomi propri."""
    phrases = []

    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        tokens = [_strip_possessive(t.strip(".,'’-")) for t in _TOKEN_RE.findall(sentence)]
        flags = []
        for position, base in enumerate(tokens):
            if not base:
                flags.append(False)
                continue
            # un numero entra nel nome solo se attaccato a un pezzo di nome
            # ("13 Reasons Why", "Apollo 11"), mai da solo
            if _is_numeric(base):
                flags.append(None)
                continue
            flags.append(base[0].isupper() and stats.is_name(base))

        current = []
        for position, base in enumerate(tokens):
            flag = flags[position]
            neighbours_are_names = (
                (position > 0 and flags[position - 1] is True)
                or (position + 1 < len(flags) and flags[position + 1] is True)
            )
            flanked_by_names = (
                position > 0 and flags[position - 1] is True
                and position + 1 < len(flags) and flags[position + 1] is True
            )
            if flag is True:
                current.append(base)
            elif flag is None and neighbours_are_names:
                current.append(base)
            elif flag is False and flanked_by_names and base and base[0].isupper():
                # parola comune incastonata fra due pezzi di nome: "Football" in
                # "National Football League" fa parte del nome, anche se da sola nel
                # corpus compare quasi sempre minuscola
                current.append(base)
            elif (base.lower() in _NAME_CONNECTORS and current
                  and position + 1 < len(flags) and flags[position + 1] is True):
                current.append(base.lower())
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []
        if current:
            phrases.append(" ".join(current))

    # un connettore non può restare in coda o in testa a un nome
    cleaned = []
    for phrase in phrases:
        words = phrase.split()
        while words and words[0].lower() in _NAME_CONNECTORS:
            words.pop(0)
        while words and words[-1].lower() in _NAME_CONNECTORS:
            words.pop()
        if words:
            cleaned.append(" ".join(words))
    return _dedupe_phrases(cleaned)


def salient_terms(text: str, stats: CorpusStats, top_k: int = 4) -> list:
    """I termini che restringono davvero una ricerca: cifre, frasi citate e le
    parole più rare del corpus, in quest'ordine di priorità."""
    terms = _quoted_phrases(text) + _numeric_phrases(text)

    scored = []
    for token in _TOKEN_RE.findall(text or ""):
        base = _norm_token(token)
        if len(base) < 3 or base in _STOPWORDS or _is_numeric(base):
            continue
        scored.append((stats.idf(base), base))
    scored.sort(reverse=True)

    # si prendono i più rari in assoluto, senza soglia: una risposta di riferimento
    # è breve, e i suoi termini di testa SONO il fatto da verificare — anche quando
    # non sono rarissimi nel corpus ("Greek", "October"). Una soglia fissa qui
    # scartava proprio l'informazione da cercare.
    for score, term in scored:
        if len(terms) >= top_k:
            break
        if term not in terms:
            terms.append(term)

    seen, unique = set(), []
    for term in terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            unique.append(term)
    return unique


def validate_claim_query(query: str, entities: list, salients: list,
                          question: str = "", exclusivity: bool = False) -> tuple:
    """Una query serve a ritrovare fonti indipendenti sullo stesso fatto: deve dire
    DI CHI si parla e SU COSA si sta verificando.

    Il secondo requisito è volutamente elastico. Può essere soddisfatto dal dettaglio
    della risposta ("October") oppure dall'attributo su cui verte la domanda
    ("death month"): pretendere sempre il primo spingerebbe verso query che
    contengono già la risposta attesa, e quindi tendono a trovare solo pagine che la
    confermano — il confirmation bias che è il punto debole noto della Variante B.

    Per i claim di esclusività ("è l'unico...", "è il primo...") il requisito
    dell'entità decade: lì la verifica si fa cercando la categoria per far emergere
    controesempi, e nominare il soggetto restringerebbe la ricerca proprio a ciò che
    si vorrebbe smentire."""
    if not query or not query.strip():
        return False, "query vuota"

    tokens = _query_tokens(query)
    stems = _stems(query)
    if len([w for w in query.split() if w]) > 12:
        return False, "troppo lunga"

    if entities and not exclusivity:
        def entity_present(entity: str) -> bool:
            entity_tokens = _query_tokens(entity)
            if entity_tokens <= tokens:
                return True
            # di un nome composto basta la parte identificante
            return len(entity_tokens) > 1 and any(len(t) > 3 for t in entity_tokens & tokens)

        if not any(entity_present(e) for e in entities):
            return False, "non nomina l'entità del claim"

    # Un nome storpiato dal modello ("Kapernick" per "Kaepernick") rende la ricerca
    # inutile pur superando gli altri controlli, perché la parte corretta del nome
    # basta a riconoscere l'entità. Si confronta quindi ogni token della query con i
    # token dei nomi del claim: somigliarsi molto senza coincidere è un refuso.
    entity_tokens = {t for e in entities for t in _query_tokens(e)}
    for token in tokens:
        if len(token) < 5 or token in entity_tokens:
            continue
        for name_token in entity_tokens:
            if len(name_token) < 5:
                continue
            if difflib.SequenceMatcher(None, token, name_token).ratio() >= 0.8:
                return False, f"nome storpiato ({token} invece di {name_token})"

    topic_stems = {s for s in _stems(question) if s not in {_stem(t) for t in _STOPWORDS}}
    carries_answer = any(_stems(t) & stems for t in salients)
    carries_topic = bool(topic_stems & stems)
    if salients and not (carries_answer or carries_topic):
        return False, "non dice su cosa si verifica il claim"

    return True, ""


def repair_claim_query(entities: list, salients: list, max_words: int = 10) -> str:
    """Ricostruisce la query da entità + fatto quando quella del modello non regge."""
    parts, covered = [], set()

    def add(fragment: str) -> None:
        tokens = _query_tokens(fragment)
        if tokens and not tokens <= covered:
            parts.append(fragment)
            covered.update(tokens)

    for entity in entities[:2]:
        if len(covered) < 5:
            add(entity)

    for term in salients[:3]:
        if len(covered) >= max_words:
            break
        # la ricerca esatta ha senso solo per una frase citata di più parole
        add(f'"{term}"' if len(term.split()) > 1 and not _is_numeric(term.split()[0]) else term)

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Qualità delle query: validazione deterministica + riparazione
#
# Il modello 7B, anche con regole esplicite nel prompt, tende a produrre query
# generiche ("Puerto Rico bill costs", "Bernie Sanders Senate speech") che
# recuperano la stessa area tematica ma un evento diverso, spesso di un altro
# anno. Il risultato a valle e' che la Fase 4 risponde correttamente "Non
# verificabile" su documenti che semplicemente non parlano di quel fatto.
# Qui sotto la query prodotta dall'LLM viene validata su due requisiti minimi
# (ANCORA all'evento + DETTAGLIO distintivo) e, se non li soddisfa, ricostruita
# deterministicamente dai termini salienti della risposta.
# ---------------------------------------------------------------------------

# la parte numerica assorbe anche le lettere che seguono, altrimenti nomi come
# "49ers" o "3M" si spezzano in un numero più un frammento senza senso ("ers")
_TOKEN_RE = re.compile(r"\$?\d[\d.,/]*%?[A-Za-z]*|[A-Za-z][A-Za-z.'’-]*")

_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "been", "but", "by", "for", "from", "had", "has",
    "have", "he", "her", "his", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "they", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "according", "about", "over", "under", "after",
    "before", "during", "than", "them", "these", "those", "there", "also", "not", "no",
    "how", "much", "many", "said", "says", "say",
}

# nomi generici che non possono MAI essere l'elemento distintivo di una query:
# matchano qualunque cosa e portano a documenti fuori tema
_FILLER_NOUNS = {
    "detail", "details", "information", "info", "status", "cost", "costs", "funding",
    "performance", "reaction", "reactions", "impact", "situation", "issue", "issues",
    "statement", "statements", "speech", "story", "news", "update", "updates", "overview",
    "summary", "analysis", "background", "response", "responses", "comment", "comments",
    "event", "events", "case", "matter", "topic", "subject", "report", "reports",
    "bill", "law", "act", "plan", "policy", "policies", "measure", "legislation",
}

# parole comuni frequentemente maiuscole nei titoli di giornale: non sono entita'
_COMMON_WORDS = {
    "a", "about", "after", "against", "all", "amid", "an", "and", "any", "are", "as", "at",
    "babbling", "back", "be", "because", "been", "before", "being", "best", "big", "both",
    "but", "by", "call", "called", "calls", "can", "could", "cut", "cuts", "day", "days",
    "did", "do", "does", "down", "during", "each", "even", "every", "first", "for", "from",
    "get", "gets", "go", "goes", "going", "got", "had", "has", "have", "he", "her", "here",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "keep", "know",
    "last", "left", "less", "let", "like", "look", "made", "make", "makes", "man", "many",
    "may", "me", "mess", "might", "money", "more", "most", "much", "must", "my", "need",
    "new", "news", "next", "no", "not", "now", "of", "off", "on", "once", "one", "only",
    "or", "other", "our", "out", "over", "own", "part", "people", "put", "puts", "right",
    "said", "same", "say", "says", "see", "she", "should", "show", "shows", "so", "some",
    "still", "such", "take", "takes", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "thing", "things", "this", "those", "three", "through",
    "time", "times", "to", "too", "top", "trash", "tries", "try", "turn", "turns", "two",
    "under", "up", "us", "use", "used", "very", "video", "want", "was", "watch", "way",
    "we", "well", "went", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "without", "would", "year", "years", "you", "your",
    # incisi e onorifici che il modello capitalizza a inizio frase ma non sono entita'
    "according", "yes", "no", "mr", "mrs", "ms", "dr", "sen", "rep", "gov", "however",
    "meanwhile", "instead", "despite", "although", "because", "since", "while",
    # riferimenti temporali: maiuscoli ma non ancorano nessuna storia in particolare
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday",
}

# entita' reali ma troppo generiche per ancorare da sole una query: "Senate" o
# "Police" ricorrono in migliaia di storie diverse. Valgono come ancora solo dentro
# un nome composto ("Oregon State Police", "U.S. District Court").
_GENERIC_ENTITIES = {
    "senate", "congress", "house", "government", "police", "court", "department",
    "administration", "state", "federal", "committee", "council", "ministry", "party",
    "republican", "republicans", "democrat", "democrats", "democratic", "twitter",
}

_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "dozen", "hundred", "thousand", "million", "billion", "trillion",
}

# parole di scala/unita' che vale la pena tenere attaccate al numero che le precede
_UNIT_WORDS = _NUMBER_WORDS | {
    "percent", "years", "year", "months", "month", "days", "day", "weeks", "week",
    "members", "member", "people", "dollars", "votes", "seats", "times",
}

# le virgolette doppie delimitano sempre una citazione; quelle singole solo se
# isolate da spazi/punteggiatura, altrimenti l'apostrofo del genitivo sassone
# ("Syria's ... Puerto Rico's") verrebbe scambiato per un delimitatore e
# catturerebbe un intero periodo come se fosse una frase citata.
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2}\b", re.IGNORECASE)

_DOUBLE_QUOTED_RE = re.compile(r"[\"“]([^\"“”\n]{4,80})[\"”]")
_SINGLE_QUOTED_RE = re.compile(r"(?:(?<=\s)|^)[‘']([^‘’'\n]{4,80})[’'](?=[\s,.;:!?)]|$)")


def _norm_token(token: str) -> str:
    return token.strip(".,;:!?'’\"-").lower()


def _query_tokens(text: str) -> set:
    return {
        _strip_possessive(_norm_token(t))
        for t in _TOKEN_RE.findall(text or "") if _norm_token(t)
    }


# suffissi inglesi regolari: bastano a far combaciare "director" con "directed" o
# "series" con "serie" senza tirare dentro un lemmatizzatore
_SUFFIXES = ("iness", "ation", "ings", "edly", "ing", "ers", "est", "ed", "es", "er",
             "ly", "s", "or", "al")


def _stem(word: str) -> str:
    """Radice approssimata, per confrontare forme flesse della stessa parola."""
    word = (word or "").lower()
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _stems(text: str) -> set:
    return {_stem(t) for t in _query_tokens(text) if t}


# quantificatori di esclusività: un claim che ne contiene uno si verifica cercando
# CONTROESEMPI, quindi la sua query legittimamente non nomina il soggetto del claim
_EXCLUSIVITY_MARKERS = {"only", "first", "last", "sole", "single", "unique", "most",
                        "best", "largest", "smallest", "oldest", "newest", "never",
                        "always", "exclusively", "any"}


def _is_numeric(token: str) -> bool:
    stripped = token.lstrip("$")
    return bool(stripped) and stripped[0].isdigit()


def _quoted_phrases(text: str) -> list:
    phrases = []
    for pattern in (_DOUBLE_QUOTED_RE, _SINGLE_QUOTED_RE):
        for match in pattern.finditer(text or ""):
            phrase = match.group(1).strip()
            # una citazione utile e' un frammento, non un periodo intero
            if phrase and ". " not in phrase and len(phrase.split()) <= 10:
                phrases.append(phrase)
    return phrases


def _strip_possessive(text: str) -> str:
    return re.sub(r"[’']s\b", "", text).strip()


def _proper_noun_phrases(text: str) -> list:
    """Sequenze di token maiuscoli che non sono parole comuni — approssimazione
    leggera di un NER, sufficiente per ancorare la query alle entita' della storia."""
    def flush(run: list, out: list) -> None:
        if not run:
            return
        # una sequenza lunghissima di token maiuscoli non e' un nome proprio: e' una
        # dateline ("WASHINGTON (Reuters) - Democratic..."), un titolo in Title Case o
        # una sfilza di hashtag. In quel caso ogni token vale per se'.
        if len(run) > 4:
            out.extend(run)
        else:
            # niente troncamento: in "Democratic Senator Robert Menendez" la parte
            # identificante e' proprio l'ultima, tagliarla svuoterebbe il nome
            out.append(" ".join(run))

    phrases = []
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        current = []
        for token in _TOKEN_RE.findall(sentence):
            base = token.strip(".,'’-")
            root = _strip_possessive(base)
            if root and root[0].isupper() and root.lower() not in _COMMON_WORDS and not _is_numeric(root):
                current.append(root)
            else:
                flush(current, phrases)
                current = []
        flush(current, phrases)
    return [p for p in phrases if p]


def _numeric_phrases(text: str) -> list:
    """Importi e cifre con la loro unita' di scala ('$370 million', '$180,000'), piu'
    i composti con trattino ('seven-member', 'trickle-down'): sono gli elementi piu'
    discriminanti per ritrovare lo stesso fatto in una fonte indipendente.

    I numeri scritti in lettere senza trattino ('five years') restano invece fuori:
    da soli non distinguono nulla e sporcano la query."""
    phrases = []
    # date esplicite ("July 1", "June 28"): molto discriminanti per ritrovare la
    # cronaca dello stesso giorno, ma il solo giorno ("1") non direbbe nulla
    phrases.extend(m.group(0) for m in _DATE_RE.finditer(text or ""))

    tokens = _TOKEN_RE.findall(text or "")
    for i, token in enumerate(tokens):
        base = token.strip(".,'’")
        if not _is_numeric(base):
            continue
        phrase = base
        if i + 1 < len(tokens):
            nxt = tokens[i + 1].strip(".,'’")
            if nxt.lower() in _UNIT_WORDS:
                phrase = f"{base} {nxt}"
        if phrase == base and not base.startswith("$") and len(base) <= 2:
            continue  # numero nudo e piccolo ("1", "5"): non distingue nulla
        phrases.append(phrase)
    for token in _TOKEN_RE.findall(text or ""):
        base = token.strip(".,'’")
        if "-" in base and len(base) > 4 and base.lower() not in _COMMON_WORDS:
            phrases.append(base)
    return phrases


def _dedupe_phrases(phrases: list, counts: dict = None) -> list:
    """Elimina le varianti contenute l'una nell'altra ('Cruz' vs 'Ted Cruz',
    'Puerto Rico' vs 'Puerto Rico’s'), tenendo di norma la prima — cioe', quando la
    lista arriva ordinata per punteggio, la piu' rilevante.

    Eccezione: se la forma piu' lunga e' a sua volta un'entita' ricorrente
    (occorre almeno due volte), vince lei — il nome completo "Ted Cruz" e' un
    ancoraggio migliore del solo cognome. Un accostamento accidentale, che compare
    una volta sola, non supera invece questa soglia e resta scartato."""
    counts = counts or {}
    result = []
    for phrase in phrases:
        tokens = _query_tokens(phrase)
        if not tokens:
            continue
        replaced = superseded = False
        for i, kept in enumerate(result):
            kept_tokens = _query_tokens(kept)
            if kept_tokens < tokens and counts.get(phrase, 0) >= 2:
                result[i] = phrase
                replaced = True
                break
            if tokens <= kept_tokens or kept_tokens <= tokens:
                superseded = True
                break
        if not replaced and not superseded:
            result.append(phrase)
    return result


def document_anchor_terms(title: str, items: list, text: str = "") -> list:
    """Entita' principali della storia, ordinate per rilevanza: ogni query deve
    contenerne almeno una, altrimenti finisce su un evento diverso dello stesso tema.

    Le entita' vengono contate sulla PROSA (il testo dell'articolo e le
    assertion/risposte), dove la maiuscola e' un segnale affidabile. I titoli di
    giornale, spesso in Title Case o tutto maiuscolo, produrrebbero invece falsi
    nomi propri ("Shattering Biker Collarbone"): da li' prendiamo solo i singoli
    token, con un bonus, perche' e' nel titolo che di solito compare il nome reale
    del protagonista quando il corpo usa un soprannome ("Mr. Teleprompter")."""
    scores = {}
    raw_counts = {}
    prose = [text or ""] + [
        f"{it.get('assertion', '')} {it.get('question', '')} {it.get('answer', '')}" for it in items
    ]
    for chunk in prose:
        for phrase in _proper_noun_phrases(chunk):
            key = phrase.strip()
            if len(key) <= 1:
                continue
            scores[key] = scores.get(key, 0) + 1
            raw_counts[key] = raw_counts.get(key, 0) + 1
            # conta anche i costituenti: un nome vero ("Ted Cruz") ricorre in tutto
            # l'articolo, mentre un accostamento accidentale ("CruzSexScandal
            # RockPrincess Rockprincess", tipico dei dump di tweet) compare una volta
            # sola e cosi' non riesce a scalzare l'entita' reale nel ranking.
            tokens = key.split()
            if len(tokens) > 1:
                for token in tokens:
                    if len(token) > 1:
                        scores[token] = scores.get(token, 0) + 1

    title_tokens = set()
    for phrase in _proper_noun_phrases(title or ""):
        for token in phrase.split():
            if len(token) > 1:
                title_tokens.add(token)
                scores.setdefault(token, 0)

    # il bonus "compare nel titolo" e' proporzionale alla quota di token coperti:
    # cosi' un'entita' vera ("CruzSexScandal") lo incassa per intero, mentre un
    # accostamento accidentale che contiene per caso quel token ("CruzSexScandal
    # RockPrincess Rockprincess") ne prende solo una frazione e non scavalca i nomi reali.
    lowered_title = {t.lower() for t in title_tokens}
    for phrase in list(scores):
        phrase_tokens = _query_tokens(phrase)
        if phrase_tokens & lowered_title:
            scores[phrase] += 3 * len(phrase_tokens & lowered_title) / len(phrase_tokens)

    ranked = [
        p for p, _ in sorted(scores.items(), key=lambda kv: (-kv[1], -len(kv[0].split()), -len(kv[0])))
    ]
    # alla deduplica servono le occorrenze reali, non i punteggi: il bonus del
    # titolo basterebbe da solo a far passare per "entita' ricorrente" un
    # accostamento visto una volta sola
    return _dedupe_phrases(ranked, counts=raw_counts)


def build_title_query(title: str, max_words: int = 8) -> str:
    """Una query ricavata direttamente dal titolo dell'articolo.

    Le query per-domanda dipendono da come il modello ha riformulato le assertion, e
    quella riformulazione a volte perde la dicitura esatta della vicenda ("Syria
    peoples' congress" diventa "Syria ethnic groups", che riporta pagine di
    demografia). Il titolo invece e' la sintesi che la testata stessa da' della
    storia, e di solito contiene la formulazione con cui la vicenda e' conosciuta.
    Cercarlo garantisce che nel pool finisca almeno un tentativo mirato all'evento;
    grazie al pooling di Fase 4 quei documenti restano poi disponibili a ogni domanda."""
    words = []
    for token in _TOKEN_RE.findall(title or ""):
        base = token.strip(".,;:!?'’\"-")
        if not base or base.lower() in _COMMON_WORDS or base.lower() in _STOPWORDS:
            continue
        if len(base) < 2 and not _is_numeric(base):
            continue
        words.append(base)
        if len(words) >= max_words:
            break
    return " ".join(words)


def document_event_term(items: list, anchors: list) -> str:
    """La parola che nomina l'EVENTO di cui parla il documento ("congress", "bill",
    "kick"), cioe' il sostantivo di contenuto che ricorre in quasi tutte le risposte.

    L'entita' da sola non basta a identificare una storia: "Russia Syria ethnic
    groups" resta agganciato alla Siria ma riporta pagine di demografia, perche' ha
    perso per strada il "congress" che e' il fatto in questione. Si richiede quindi
    che la query porti anche questo termine, quando ce n'e' uno dominante."""
    anchor_tokens = set()
    for anchor in anchors[:8]:
        anchor_tokens |= _query_tokens(anchor)

    counts = {}
    for item in items:
        seen = set()
        for token in _TOKEN_RE.findall(f"{item.get('answer', '')} {item.get('assertion', '')}"):
            base = _norm_token(token)
            if (len(base) > 3 and base not in _STOPWORDS and base not in _COMMON_WORDS
                    and base not in anchor_tokens and not _is_numeric(base)):
                seen.add(base)
        for base in seen:
            counts[base] = counts.get(base, 0) + 1

    if not counts:
        return ""
    term, freq = max(counts.items(), key=lambda kv: kv[1])
    # dominante solo se compare in almeno meta' delle voci del documento
    return term if freq >= max(2, len(items) / 2) else ""


def item_salient_terms(item: dict) -> tuple:
    """Restituisce (strong, weak): 'strong' sono i dettagli davvero discriminanti
    (numeri, frasi citate, composti con trattino, nomi propri), 'weak' i restanti
    sostantivi di contenuto non generici."""
    answer = item.get("answer", "") or ""
    assertion = item.get("assertion", "") or ""
    text = f"{answer} {assertion}"

    # contano i dettagli della RISPOSTA: sono quelli che il confronto a valle deve
    # poter ritrovare nella fonte indipendente. L'assertion serve solo come riserva
    # quando la risposta non ne contiene nessuno — altrimenti importi citati
    # altrove nell'assertion finirebbero nella query al posto del fatto in esame.
    strong = _quoted_phrases(answer) + _numeric_phrases(answer) + _proper_noun_phrases(answer)
    if not strong:
        strong = _quoted_phrases(assertion) + _numeric_phrases(assertion)

    seen, strong_unique = set(), []
    for term in strong:
        key = term.lower()
        if key and key not in seen and key not in _COMMON_WORDS:
            seen.add(key)
            strong_unique.append(term)
    strong_unique = _dedupe_phrases(strong_unique)

    weak = []
    for token in _TOKEN_RE.findall(answer):
        base = _norm_token(token)
        if (base and base not in _STOPWORDS and base not in _FILLER_NOUNS
                and base not in _COMMON_WORDS and len(base) > 3 and not _is_numeric(base)):
            if base not in weak:
                weak.append(base)
    return strong_unique, weak


def normalize_query_quotes(query: str) -> str:
    """Converte le virgolette singole in doppie attorno alle frasi di piu' parole.

    I motori di ricerca (Tavily inclusa) trattano come ricerca esatta solo le
    virgolette DOPPIE: 'colonialism at its worst' viene sparpagliato in parole
    singole, mentre "colonialism at its worst" riporta proprio gli articoli su
    quella dichiarazione. Il modello scrive spesso le singole, quindi si normalizza."""
    if not query:
        return query
    return re.sub(r"'([^']{4,80}\s[^']*)'", r'"\1"', query)


def validate_query(query: str, anchors: list, item: dict, title_only_entities: list = None,
                    event_term: str = "") -> tuple:
    """(ok, motivo). Una query e' valida se ancora la storia (almeno un'entita'
    principale) e porta un dettaglio distintivo della risposta."""
    if not query or not query.strip():
        return False, "query vuota"

    tokens = _query_tokens(query)
    words = [w for w in re.split(r"\s+", query.strip()) if w]
    if len(words) > 12:
        return False, f"troppo lunga ({len(words)} parole)"

    # l'ancora deve essere presente per intero e non puo' essere un'entita' generica
    # da sola: "bill Senate vote Wednesday" e' agganciata al Senato, non a questa storia
    matched_anchors = []
    for anchor in anchors[:8]:
        anchor_tokens = _query_tokens(anchor)
        if not anchor_tokens or not anchor_tokens <= tokens:
            continue
        if len(anchor_tokens) == 1 and next(iter(anchor_tokens)) in _GENERIC_ENTITIES:
            continue
        matched_anchors.append(anchor)
    if not matched_anchors:
        return False, "nessuna entita' principale della storia"

    # Disambiguazione: molte parole chiave sono omonime fuori contesto — "congress"
    # trova il Congresso americano, "Kremlin" trova il palazzo di Mosca. Per restare
    # sulla storia giusta la query deve portare l'entita' dominante del documento
    # ("Syria"), oppure almeno due entita' distinte che insieme la identifichino.
    primary = next(
        (a for a in anchors[:8]
         if not (len(_query_tokens(a)) == 1 and next(iter(_query_tokens(a))) in _GENERIC_ENTITIES)),
        None,
    )
    # L'entita' dominante deve esserci sempre. Le prove raccolte sui risultati reali
    # sono nette: ogni query che ne era priva ha riportato la storia sbagliata —
    # "Kremlin congress ..." il palazzo di Mosca e il Congresso USA, "National
    # Enquirer started rumors" la storia del tabloid invece dello scandalo Cruz,
    # "Oregon State Police ..." altri incidenti della stessa polizia. Un'entita'
    # secondaria, per quanto specifica, identifica un contesto, non questo evento.
    if primary:
        primary_tokens = _query_tokens(primary)
        # di un nome composto basta la parte identificante ("Edwards" per
        # "Captain Rob Edwards")
        covered = primary_tokens <= tokens or (
            len(primary_tokens) > 1
            and any(len(t) > 3 and t not in _GENERIC_ENTITIES for t in primary_tokens & tokens)
        )
        if not covered:
            return False, f"manca l'entita' dominante della storia ({primary})"

    if event_term and event_term not in tokens:
        return False, f"manca la parola che nomina l'evento ({event_term})"

    if title_only_entities:
        wanted = {t.lower() for t in title_only_entities}
        if not (wanted & tokens):
            return False, f"manca il nome reale del protagonista ({'/'.join(title_only_entities)})"

    strong, weak = item_salient_terms(item)
    if strong:
        def carries(term: str) -> bool:
            term_tokens = _query_tokens(term)
            if not term_tokens:
                return False
            if term_tokens <= tokens:
                return True
            # per un nome proprio composto basta la parte identificante: "Menendez"
            # rappresenta "Democratic Senator Robert Menendez" quanto il nome intero
            if len(term_tokens) > 1:
                shared = term_tokens & tokens
                return any(len(t) > 3 and t not in _GENERIC_ENTITIES for t in shared)
            return False

        if not any(carries(term) for term in strong):
            return False, "manca il dettaglio distintivo della risposta"
    elif weak:
        if not any(w in tokens for w in weak):
            return False, "nessun termine di contenuto della risposta"

    informative = {t for t in tokens if t not in _STOPWORDS and t not in _FILLER_NOUNS}
    anchor_tokens = set()
    for anchor in anchors[:12]:
        anchor_tokens |= _query_tokens(anchor)
    if not (informative - anchor_tokens):
        return False, "solo entita' generiche, nessun elemento discriminante"

    return True, ""


def repair_query(anchors: list, item: dict, max_words: int = 10, title: str = "",
                  event_term: str = "") -> str:
    """Ricostruisce deterministicamente una query da entita' + dettagli salienti,
    quando quella prodotta dall'LLM non supera la validazione."""
    strong, weak = item_salient_terms(item)
    quoted_in_article = {q.lower() for q in _quoted_phrases(
        f"{item.get('answer', '')} {item.get('assertion', '')}")}
    item_tokens = _query_tokens(f"{item.get('assertion', '')} {item.get('answer', '')}")

    def _is_generic(anchor: str) -> bool:
        tokens = _query_tokens(anchor)
        return len(tokens) == 1 and next(iter(tokens)) in _GENERIC_ENTITIES

    # preferisci le entita' che compaiono in QUESTO item e quelle non generiche
    # ("Puerto Rico" prima di "Senate"), poi quelle piu' rilevanti per il documento
    # (il titolo pesa gia' nel ranking di document_anchor_terms)
    ranked_anchors = sorted(
        anchors[:12],
        key=lambda a: (1 if _is_generic(a) else 0,
                       0 if _query_tokens(a) & item_tokens else 1,
                       anchors.index(a)),
    )

    parts, covered = [], set()

    def add(fragment: str, tokens: set) -> None:
        parts.append(fragment)
        covered.update(tokens)

    # l'ancora principale viene preferibilmente dal titolo: e' li' che compare il
    # nome reale del protagonista quando il corpo dell'articolo usa un soprannome
    title_tokens = {t.lower() for p in _proper_noun_phrases(title or "") for t in p.split()}
    if title_tokens:
        for anchor in ranked_anchors:
            if _query_tokens(anchor) & title_tokens and not _is_generic(anchor):
                add(anchor, _query_tokens(anchor))
                break

    for anchor in ranked_anchors:
        if len(covered) >= 4:
            break
        tokens = _query_tokens(anchor)
        if not tokens or tokens <= covered:
            continue
        add(anchor, tokens)

    if event_term and event_term not in covered:
        add(event_term, {event_term})

    for term in strong[:3]:
        if len(covered) >= max_words:
            break
        tokens = _query_tokens(term)
        if not tokens or tokens <= covered:
            continue
        # verbatim solo per le frasi citate nell'articolo e per gli importi:
        # sono esatte e discriminanti. Non per gruppi di parole comuni.
        needs_quotes = len(term.split()) > 1 and (
            term.lower() in quoted_in_article or any(_is_numeric(t) for t in term.split())
        )
        add(f'"{term}"' if needs_quotes else term, tokens)

    # se i dettagli forti scarseggiano, differenzia comunque la query con i
    # termini di contenuto della risposta (altrimenti item diversi dello stesso
    # documento collassano tutti sulla stessa query di sole entita')
    if len(strong) < 2:
        for term in weak[:2]:
            if len(covered) >= max_words:
                break
            if term in covered:
                continue
            add(term, {term})

    return " ".join(parts).strip()


_YESNO_AUX = ("is", "are", "was", "were", "did", "does", "do", "has", "have", "had",
              "can", "could", "will", "would", "should", "may", "might")


def normalize_open_question(question: str, answer: str, fallback: str = "") -> tuple:
    """Riporta a forma aperta le domande sì/no e toglie il "Yes,"/"No," iniziale
    dalla risposta di riferimento.

    Una domanda sì/no e' inutilizzabile nel blind matching: anticipa la risposta
    attesa e nessuna fonte indipendente si esprime in quella forma, quindi il
    confronto degenera. Il prompt le vieta esplicitamente, ma un modello 7B ogni
    tanto le produce lo stesso: qui c'e' la rete di sicurezza deterministica."""
    question = (question or "").strip()
    answer = (answer or "").strip()

    stripped = re.sub(r"^(yes|no)\b[\s,:.—-]*", "", answer, flags=re.IGNORECASE).strip()
    # se dopo aver tolto il "Yes"/"No" non resta nulla, la risposta era solo quello:
    # tenerla vuota lascerebbe la Fase 4 senza niente da confrontare, quindi in quel
    # caso si conserva `fallback` (l'enunciato di partenza), che è il fatto asserito
    if stripped:
        answer = stripped[0].upper() + stripped[1:]
    elif answer:
        answer = fallback.strip() or answer

    if not question:
        return question, answer

    core = question.rstrip("?").strip()
    # rimuove un eventuale inciso iniziale tipo "According to the Kremlin, ..."
    prefix_match = re.match(r"^(according to [^,]{1,60},\s*)(.*)$", core, flags=re.IGNORECASE)
    prefix, body = (prefix_match.group(1), prefix_match.group(2)) if prefix_match else ("", core)

    first_word = body.split()[0].lower() if body.split() else ""
    if first_word in _YESNO_AUX:
        rest = " ".join(body.split()[1:]).strip()
        if rest:
            question = f"{prefix}what does the source report about {rest}?".strip()
            question = question[0].upper() + question[1:]

    return question, answer


_ANCHORABLE_NOUNS = (
    "bill", "law", "act", "legislation", "board", "committee", "report", "incident",
    "officer", "case", "vote", "deal", "plan", "video", "article", "statement", "rumors",
    "scandal", "congress", "proposal", "measure",
)


_HONORIFIC_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+([A-Z][A-Za-z]+)")


def detect_epithet(title: str, text: str) -> tuple:
    """Individua il caso "soprannome al posto del nome": il corpo chiama il
    protagonista con un epiteto ("Mr. Teleprompter") e il nome reale compare solo
    nel titolo ("Obama"). Ritorna (epiteto, nome_reale) oppure (None, None).

    E' il caso che manda completamente fuori strada la ricerca: cercare
    "Mr. Teleprompter performance in Indiana" restituisce noleggi di gobbi
    elettronici, non l'intervento del presidente. Il riconoscimento e' volutamente
    stretto — servono sia l'onorifico davanti a un nome assente dal titolo, sia un
    nome nel titolo che non compaia mai nel corpo — cosi' un normale "Mr. Trump"
    in un articolo che nel titolo dice "Trump" non viene toccato."""
    body_lower = (text or "").lower()
    title_lower = (title or "").lower()

    title_only = [
        token
        for phrase in _proper_noun_phrases(title or "")
        for token in phrase.split()
        if len(token) > 2 and token.lower() not in body_lower
    ]
    if not title_only:
        return None, None

    for match in _HONORIFIC_RE.finditer(text or ""):
        candidate = match.group(1)
        if candidate.lower() not in title_lower:
            return candidate, title_only[0]
    return None, None


def resolve_epithet_in_items(items: list, epithet: str, real_name: str) -> int:
    """Sostituisce l'epiteto col nome reale in domande e risposte. Senza questo, la
    domanda posta alle fonti indipendenti nomina qualcuno che in quelle fonti non
    esiste, e nessuna potra' mai confermarla."""
    if not epithet or not real_name:
        return 0
    patterns = [
        re.compile(rf"\b(?:Mr|Mrs|Ms|Dr)\.?\s+{re.escape(epithet)}\b"),
        re.compile(rf"\b{re.escape(epithet)}\b"),
    ]
    changed = 0
    for item in items:
        for field in ("question", "answer"):
            value = item.get(field, "")
            new_value = value
            for pattern in patterns:
                new_value = pattern.sub(real_name, new_value)
            if new_value != value:
                item[field] = new_value
                changed += 1
    return changed


_SPEECH_VERBS = ("said", "says", "told", "warned", "added", "argued", "claimed", "asked",
                 "noted", "declared", "wrote", "called", "denied", "admitted")


_PERSON_TITLES = {
    "mr", "mrs", "ms", "dr", "prof", "professor", "sir", "captain", "capt", "sergeant",
    "sgt", "lieutenant", "lt", "officer", "trooper", "detective", "chief", "judge",
    "justice", "senator", "sen", "rep", "representative", "president", "governor", "gov",
    "mayor", "minister", "secretary", "chancellor", "general", "colonel", "major",
}


def _looks_like_person(anchor: str, text: str) -> bool:
    """Euristica in due segnali: il nome porta un titolo personale ("Captain Rob
    Edwards"), oppure e' soggetto di un verbo di dichiarazione ("Sanders said").
    Serve a non infilare il nome di una persona come qualificatore di un oggetto:
    "the Captain Rob Edwards officer" e' inglese rotto, e "the Bernie Sanders bill"
    attribuirebbe la legge a chi la contestava."""
    tokens = anchor.split()
    if not tokens:
        return False
    if any(t.strip(".").lower() in _PERSON_TITLES for t in tokens):
        return True

    pattern = re.compile(rf"\b{re.escape(tokens[-1])}\s+(?:{'|'.join(_SPEECH_VERBS)})\b",
                          re.IGNORECASE)
    if pattern.search(text or ""):
        return True

    # terzo segnale: una persona con nome e cognome viene poi richiamata col solo
    # cognome ("Justin Wilkens" ... "Wilkens"), cosa che non succede ai toponimi
    # ("Puerto Rico" non diventa mai "Rico"). In caso di dubbio si sbaglia per
    # eccesso: classificare come persona significa solo rinunciare a iniettare
    # quel nome nella domanda, che e' l'esito prudente.
    if len(tokens) == 2:
        surname = re.compile(rf"(?<![\w'’]){re.escape(tokens[-1])}\b")
        first = re.compile(rf"{re.escape(tokens[0])}\s+{re.escape(tokens[-1])}")
        standalone = len(surname.findall(text or "")) - len(first.findall(text or ""))
        if standalone > 0:
            return True

    return False


def anchor_question(question: str, anchors: list, text: str = "") -> tuple:
    """Assicura che la domanda nomini l'entita' della storia.

    Una domanda come "Who opposes the bill?" non e' autosufficiente: nel blind
    matching viene posta a un articolo qualsiasi, e qualunque legge di qualunque
    anno sembrera' pertinente. Se manca l'ancora, la si inserisce davanti al nome
    generico ("the bill" -> "the Puerto Rico bill"). Ritorna (domanda, ancorata).

    Come qualificatore si usano solo entita' non-persona: "the Bernie Sanders bill"
    attribuirebbe la legge a chi invece la contestava."""
    if not question or not anchors:
        return question, False

    question_tokens = _query_tokens(question)
    for anchor in anchors[:8]:
        anchor_tokens = _query_tokens(anchor)
        if anchor_tokens and anchor_tokens <= question_tokens:
            return question, True  # gia' ancorata

    # i token che compongono un nome di persona vanno esclusi tutti, non solo il nome
    # intero: il ranking delle ancore contiene anche i frammenti ("Rob", "Ted"), e
    # iniettarne uno produrrebbe "the Rob officer"
    person_tokens = set()
    for anchor in anchors[:12]:
        if _looks_like_person(anchor, text):
            person_tokens |= _query_tokens(anchor)

    def usable(anchor: str) -> bool:
        tokens = _query_tokens(anchor)
        if len(tokens) == 1 and next(iter(tokens)) in _GENERIC_ENTITIES:
            return False
        return not (tokens & person_tokens)

    primary = next((a for a in anchors[:8] if usable(a)), None)
    if not primary:
        return question, False

    for noun in _ANCHORABLE_NOUNS:
        pattern = re.compile(rf"\bthe\s+{noun}\b", re.IGNORECASE)
        if pattern.search(question):
            return pattern.sub(f"the {primary} {noun}", question, count=1), True

    return question, False


def call_ollama_document_queries(title: str, items: list, model: str,
                                  max_retries: int = 3, timeout: int = 180,
                                  keep_alive: str = DEFAULT_KEEP_ALIVE) -> list:
    """Genera una query di ricerca per ciascun item (assertion/question/answer) di un
    documento, in una chiamata dedicata e separata dalla lettura del documento.
    Ritorna una lista lunga quanto `items` (con None dove il modello ha fallito)."""
    if not items:
        return []

    payload_items = [
        {
            "id": i + 1,
            "assertion": it.get("assertion", ""),
            "question": it.get("question", ""),
            "answer": it.get("answer", ""),
        }
        for i, it in enumerate(items)
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_DOCUMENT_QUERY},
            {"role": "user", "content": USER_PROMPT_DOCUMENT_QUERY_TEMPLATE.format(
                title=title,
                items_json=json.dumps(payload_items, ensure_ascii=False, indent=1),
            )},
        ],
        "format": "json",
        "stream": False,
        "keep_alive": normalize_keep_alive(keep_alive),
        "options": {"temperature": 0.2, "num_predict": 1024},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            parsed = json.loads(resp.json()["message"]["content"])
            queries = parsed.get("queries", [])
            if isinstance(queries, list) and len(queries) == len(items):
                return [str(q).strip() if q else None for q in queries]
            last_err = f"attese {len(items)} query, ricevute {len(queries) if isinstance(queries, list) else type(queries)}"
        except (requests.RequestException, json.JSONDecodeError, KeyError, TypeError) as e:
            last_err = str(e)

        print(f"  [retry {attempt}/{max_retries}] query generation fallita: {last_err}", file=sys.stderr)
        time.sleep(1.5 * attempt)

    print(f"  [WARN] query generation fallita: uso solo la ricostruzione deterministica ({last_err})",
          file=sys.stderr)
    return [None] * len(items)


def attach_queries_to_items(title: str, items: list, model: str, text: str = "",
                             keep_alive: str = DEFAULT_KEEP_ALIVE) -> dict:
    """Genera, valida ed eventualmente ripara la query di ogni item. Modifica gli
    item in-place aggiungendo 'query' e 'query_source'. Ritorna le statistiche."""
    if not items:
        return {"llm": 0, "repaired": 0}

    # se il corpo dell'articolo usa un soprannome, il nome reale (dal titolo) deve
    # entrare in domande, risposte e query prima di qualsiasi altra cosa
    epithet, real_name = detect_epithet(title, text)
    title_only_entities = []
    if epithet:
        n = resolve_epithet_in_items(items, epithet, real_name)
        title_only_entities = [real_name]
        print(f"    [soprannome risolto] {epithet!r} -> {real_name!r} ({n} campi aggiornati)")

    llm_queries = call_ollama_document_queries(title, items, model=model, keep_alive=keep_alive)
    anchors = document_anchor_terms(title, items, text=text)
    if real_name:
        # l'epiteto non deve restare fra le ancore: cercarlo riporterebbe comunque
        # risultati sul senso letterale della parola ("Teleprompter" -> noleggi)
        epithet_tokens = _query_tokens(epithet)
        anchors = [real_name] + [
            a for a in anchors if a != real_name and not (_query_tokens(a) & epithet_tokens)
        ]

    stats = {"llm": 0, "repaired": 0, "questions_anchored": 0}
    for item in items:
        original = item.get("question", "")
        anchored_question, was_anchored = anchor_question(original, anchors, text=text)
        if anchored_question != original:
            print(f"    [domanda ancorata] {original!r} -> {anchored_question!r}")
            stats["questions_anchored"] += 1
        item["question"] = anchored_question
        item["question_self_contained"] = was_anchored

    event_term = document_event_term(items, anchors)
    if event_term:
        print(f"    [termine evento] {event_term!r} — richiesto in ogni query")

    for item, raw_query in zip(items, llm_queries):
        raw_query = normalize_query_quotes(raw_query) if raw_query else raw_query
        ok, reason = (
            validate_query(raw_query, anchors, item, title_only_entities=title_only_entities,
                           event_term=event_term)
            if raw_query else (False, "assente")
        )
        if ok:
            item["query"] = raw_query
            item["query_source"] = "llm"
            stats["llm"] += 1
        else:
            repaired = repair_query(anchors, item, title=title, event_term=event_term)
            if not repaired:
                repaired = raw_query or item.get("assertion", "")[:80]
            item["query"] = normalize_query_quotes(repaired)
            item["query_source"] = "repaired"
            stats["repaired"] += 1
            print(f"    [query scartata: {reason}] {raw_query!r} -> {item['query']!r}")
    return stats


def call_ollama_questions(claim: str, model: str, max_retries: int = 3, timeout: int = 120,
                           extra_instructions: str = "", keep_alive: str = DEFAULT_KEEP_ALIVE) -> list:
    """Chiama il server Ollama locale e restituisce la lista di domande/risposte."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + extra_instructions},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(claim=claim)},
        ],
        "format": "json",  # forza output JSON valido (supportato da Ollama)
        "stream": False,
        "keep_alive": normalize_keep_alive(keep_alive),
        "options": {"temperature": 0.2},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.post(OLLAMA_URL, json=payload, timeout=timeout)
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
                                          max_retries: int = 3, timeout: int = 180,
                                          keep_alive: str = DEFAULT_KEEP_ALIVE) -> list:
    """Come call_ollama_questions, ma per un chunk di documento. Il prompt per
    documenti (SYSTEM_PROMPT_DOCUMENT) chiede al modello una LISTA JSON piatta
    [{"id":..., "assertion":..., "centrality":..., "query":..., "question":..., "answer":...}],
    non l'oggetto {"questions": [...]} usato in modalità claim — il parsing qui
    sotto è specifico per questo schema."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_DOCUMENT},
            {"role": "user", "content": USER_PROMPT_DOCUMENT_TEMPLATE.format(title=title, text=text_chunk)},
        ],
        "format": "json",
        "stream": False,
        "keep_alive": normalize_keep_alive(keep_alive),
        "options": {"temperature": 0.2, "num_predict": 2048},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.post(OLLAMA_URL, json=payload, timeout=timeout)
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
                question = str(item.get("question", "")).strip()
                answer = str(item.get("answer", "")).strip()
                centrality = item.get("centrality", 3)
                try:
                    centrality = int(centrality)
                except (TypeError, ValueError):
                    centrality = 3
                centrality = max(1, min(5, centrality))
                # "question" e "answer" sono opzionali a livello di validazione (non
                # scartiamo l'intera voce se mancano, per non perdere assertion buone
                # quando il modello dimentica un campo), ma vengono richiesti
                # sempre nel prompt; se mancano, ripieghiamo su fallback naive
                # derivati dall'assertion, così l'entry resta completa e usabile
                # nel confronto a valle. La "query" NON viene piu' chiesta qui: e'
                # generata da una chiamata dedicata (attach_queries_to_items).
                if not question and assertion:
                    question = f"What does the article say about: {assertion}?"
                if not answer and assertion:
                    answer = assertion
                question, answer = normalize_open_question(question, answer, fallback=assertion)
                if assertion:
                    cleaned.append({
                        "assertion": assertion,
                        "centrality": centrality,
                        "question": question,
                        "answer": answer,
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
                                    max_words: int = 1800, overlap_words: int = 150,
                                    keep_alive: str = DEFAULT_KEEP_ALIVE) -> list:
    """Genera domande/risposte per un documento intero, spezzandolo in chunk se
    troppo lungo per il context window, e unendo i risultati di tutti i chunk."""
    chunks = split_into_chunks(text, max_words=max_words, overlap_words=overlap_words)

    all_questions = []
    for ci, chunk in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"  chunk {ci}/{len(chunks)} ({len(chunk.split())} parole)")
        chunk_questions = call_ollama_questions_document_chunk(title, chunk, model=model, keep_alive=keep_alive)
        for q in chunk_questions:
            if len(chunks) > 1:
                q["chunk_index"] = ci
        all_questions.extend(chunk_questions)

    return all_questions


def call_ollama_queries(claim: str, questions: list, model: str, max_retries: int = 3, timeout: int = 120,
                         keep_alive: str = DEFAULT_KEEP_ALIVE) -> list:
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
        "keep_alive": normalize_keep_alive(keep_alive),
        "options": {"temperature": 0.2},
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.post(OLLAMA_URL, json=payload, timeout=timeout)
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


RETRY_HINT_INVERTED = """

The previous attempt produced a question that does not name the entity the claim is
about, so its answer was that entity itself (e.g. asking "Who is a vegan?" and
answering with the person's name). Such a question can only be answered by a source
that enumerates every member of the category, so it is useless for verification.
Ask about the entity instead of asking for it: keep the entity in the question and
make its property the answer."""


def question_is_inverted(question: str, answer: str, entities: list) -> bool:
    """True quando la domanda non nomina nessuna entità del claim e la risposta è
    invece proprio una di esse.

    È la forma capovolta: "Who is a vegan?" / "Tilda Swinton". Per rispondere
    servirebbe una fonte che elenchi tutti i vegani, mentre le fonti che la ricerca
    riporterà parlano della persona. La forma utile chiede la proprietà dell'entità."""
    if not entities or not question:
        return False

    question_tokens = _query_tokens(question)
    if any(_query_tokens(e) & question_tokens for e in entities):
        return False

    answer_tokens = _query_tokens(answer)
    return any(_query_tokens(e) & answer_tokens for e in entities)


def build_claim_context_query(claim: str, entities: list, stats: CorpusStats,
                               max_words: int = 8) -> str:
    """Query mirata al claim nel suo complesso, cercata in Fase 3 accanto a quelle
    delle singole domande.

    Le query per-domanda dipendono da come il modello ha riformulato il claim, e
    quella riformulazione può perdere per strada il modo in cui la vicenda è
    conosciuta. Questa parte invece dal testo originale: tiene le entità e i termini
    più rari del corpus, scarta il resto. Con il pooling di Fase 4 i documenti così
    recuperati restano disponibili a tutte le domande del claim."""
    parts, covered = [], set()
    for fragment in entities[:2] + salient_terms(claim, stats, top_k=4):
        tokens = _query_tokens(fragment)
        if tokens and not tokens <= covered:
            parts.append(fragment)
            covered.update(tokens)
        if len(covered) >= max_words:
            break
    return " ".join(parts).strip() or claim[:80]


def process_one_claim(record: dict, model: str, stats: CorpusStats,
                       keep_alive: str = DEFAULT_KEEP_ALIVE) -> dict:
    """Pipeline completa (decomposizione [+ retry se capovolta] + query) per UN
    claim. Estratta dal loop di process_file_claims per poterla sottomettere a un
    pool di thread: claim diversi sono indipendenti fra loro, quindi possono girare
    in parallelo, ma i passi DENTRO un singolo claim restano sequenziali (la
    generazione delle query ha bisogno delle domande gia' pronte)."""
    claim_id = record.get("id")
    claim = record.get("claim", "")

    print(f"id={claim_id} -> {claim}")
    questions = call_ollama_questions(claim, model=model, keep_alive=keep_alive)
    entities = entity_phrases(claim, stats)

    # una domanda capovolta è inservibile a valle: si concede un solo tentativo di
    # rigenerazione con l'indicazione di cosa non andava, invece di riscriverne il
    # testo a mano (le riscritture meccaniche di frasi producono inglese
    # sgrammaticato o cambiano il significato)
    if questions and any(
        question_is_inverted(q.get("question", ""), q.get("answer", ""), entities)
        for q in questions
    ):
        print(f"    [id={claim_id}] [domanda capovolta] rigenero la decomposizione")
        retry = call_ollama_questions(claim, model=model, extra_instructions=RETRY_HINT_INVERTED,
                                       keep_alive=keep_alive)
        if retry and not any(
            question_is_inverted(q.get("question", ""), q.get("answer", ""), entities)
            for q in retry
        ):
            questions = retry
        else:
            print(f"    [id={claim_id}] [WARN] la domanda resta capovolta anche dopo il retry")
            for q in questions:
                q["question_inverted"] = question_is_inverted(
                    q.get("question", ""), q.get("answer", ""), entities)

    if questions:
        queries = call_ollama_queries(claim, questions, model=model, keep_alive=keep_alive)
        n_repaired = 0
        for item, raw_query in zip(questions, queries):
            item["question"], item["answer"] = normalize_open_question(
                item.get("question", ""), item.get("answer", ""), fallback=claim)

            raw_query = normalize_query_quotes(raw_query or "")
            salients = salient_terms(item.get("answer", ""), stats)
            exclusivity = bool(
                _query_tokens(f"{claim} {item.get('question', '')}") & _EXCLUSIVITY_MARKERS
            )
            ok, reason = validate_claim_query(
                raw_query, entities, salients,
                question=item.get("question", ""), exclusivity=exclusivity,
            )
            if ok:
                item["query"] = raw_query
                item["query_source"] = "llm"
            else:
                repaired = normalize_query_quotes(
                    repair_claim_query(entities, salients)) or raw_query
                item["query"] = repaired
                item["query_source"] = "repaired"
                n_repaired += 1
                print(f"    [id={claim_id}] [query scartata: {reason}] {raw_query!r} -> {repaired!r}")
        print(f"  [id={claim_id}] -> {len(questions)} domande, {n_repaired} query ricostruite "
              f"(entità: {entities or 'nessuna'})")

    # ricostruisco l'oggetto per inserire "questions" subito dopo "claim"
    new_record = {}
    for key, value in record.items():
        new_record[key] = value
        if key == "claim":
            new_record["questions"] = questions
            new_record["context_query"] = build_claim_context_query(claim, entities, stats)

    return new_record


def _load_done_ids(output_path, id_key="id"):
    """Legge l'output gia' scritto (se c'e') e ritorna il set degli id gia'
    processati, cosi' una run interrotta (fine sessione Kaggle) puo' ripartire
    senza rifare il lavoro gia' fatto invece di ripartire da zero."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)[id_key])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def process_file_claims(input_path: str, output_path: str, model: str, limit: int = None,
                         stats: CorpusStats = None, workers: int = DEFAULT_WORKERS,
                         keep_alive: str = DEFAULT_KEEP_ALIVE,
                         time_budget_seconds: float = None):
    """Modalità 'claim': legge un JSONL con {"id", "claim"} per riga.

    Rispetto alla modalità 'document' qui non serve nessuna delle euristiche pensate
    per i titoli di giornale e per i corpi di pagina sporchi: un claim è una frase
    pulita e autosufficiente. Restano i due passi (decomposizione, poi query) e la
    validazione delle query, che qui si appoggia interamente alle statistiche del
    corpus invece che a liste di parole scritte a mano.

    I claim vengono sottomessi tutti insieme a un pool di thread (indipendenti fra
    loro), ma i risultati sono raccolti e scritti su disco NELL'ORDINE ORIGINALE del
    file di input, non nell'ordine di completamento: l'output resta deterministico e
    identico a una versione sequenziale, solo le chiamate diventano concorrenti. Con
    --workers 1 il comportamento e' identico alla versione sequenziale originale."""
    if stats is None:
        stats = build_corpus_stats(input_path, field="claim")

    done_ids = _load_done_ids(output_path)
    if done_ids:
        print(f"[RESUME] {len(done_ids)} claim gia' presenti in output, verranno saltate")

    records = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("id") in done_ids:
                continue
            records.append(record)

    if not records:
        print("[RESUME] Niente da fare, tutte le claim sono gia' state processate.")
        return

    start_time = time.monotonic()
    stopped_early = False
    chunk_size = max(1, workers * 4)  # controlliamo il tempo residuo tra un chunk e l'altro

    with open(output_path, "a", encoding="utf-8") as fout, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:

        for chunk_start in range(0, len(records), chunk_size):
            if time_budget_seconds is not None and (time.monotonic() - start_time) > time_budget_seconds:
                stopped_early = True
                break
            chunk = records[chunk_start:chunk_start + chunk_size]
            futures = [
                executor.submit(process_one_claim, record, model, stats, keep_alive)
                for record in chunk
            ]
            for future in futures:
                new_record = future.result()
                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                fout.flush()

    if stopped_early:
        remaining = len(records) - chunk_start
        print(f"[TIME BUDGET] Limite di tempo raggiunto: {remaining} claim non ancora "
              f"processate. Rilancia lo stesso comando (Save & Run All) per riprendere "
              f"da dove si e' fermato.")


def process_one_document(doc_id, title: str, text: str, label, model: str,
                          max_words: int, overlap_words: int, include_text: bool,
                          keep_alive: str = DEFAULT_KEEP_ALIVE) -> dict:
    """Pipeline completa (lettura/chunking + query) per UN documento. Estratta dal
    loop di process_file_documents per poterla sottomettere a un pool di thread:
    documenti diversi sono indipendenti fra loro."""
    print(f"id={doc_id} -> {title[:80]}")
    questions = call_ollama_questions_document(
        title, text, model=model, max_words=max_words, overlap_words=overlap_words,
        keep_alive=keep_alive,
    )
    # la lettura del documento (assertion/question/answer) e la scrittura della
    # query di ricerca sono due chiamate separate: chiedere entrambe nello stesso
    # prompt sovraccarica un modello 7B, che rispetta lo schema JSON ma ignora le
    # regole di qualita' e produce query generiche ("Puerto Rico bill costs") che
    # recuperano l'argomento giusto ma l'evento sbagliato. Ogni query viene poi
    # validata e, se necessario, ricostruita deterministicamente dai termini
    # salienti della risposta.
    if questions:
        stats = attach_queries_to_items(title, questions, model=model, text=text, keep_alive=keep_alive)
        print(f"  [id={doc_id}] -> query: {stats['llm']} dall'LLM, {stats['repaired']} ricostruite")

    new_record = {
        "id": doc_id,
        "title": title,
        # query aggiuntiva mirata alla storia nel suo complesso, cercata in Fase 3
        # accanto a quelle delle singole domande
        "title_query": build_title_query(title),
    }
    if label is not None and label != "":
        new_record["label"] = label
    if include_text:
        new_record["text"] = text
    new_record["questions"] = questions
    print(f"  [id={doc_id}] -> {len(questions)} domande generate")
    return new_record


def process_file_documents(input_path: str, output_path: str, model: str, limit: int = None,
                            max_words: int = 1800, overlap_words: int = 150,
                            include_text: bool = False, csv_delimiter: str = ";",
                            workers: int = DEFAULT_WORKERS, keep_alive: str = DEFAULT_KEEP_ALIVE,
                            time_budget_seconds: float = None):
    """Modalità 'document': legge un CSV con colonne (id opzionale);title;text;label
    (label opzionale) e genera domande/risposte sull'intero documento, con
    chunking automatico per i testi troppo lunghi.

    Come in process_file_claims, i documenti sono sottomessi in parallelo a un pool
    di thread ma scritti su disco nell'ordine originale del file di input."""
    import csv

    done_ids = _load_done_ids(output_path)
    if done_ids:
        print(f"[RESUME] {len(done_ids)} documenti gia' presenti in output, verranno saltati")

    rows = []
    with open(input_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter=csv_delimiter)
        fieldnames = reader.fieldnames or []
        id_field = fieldnames[0] if fieldnames and fieldnames[0].strip() == "" else None

        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break

            doc_id = row.get(id_field) if id_field else row.get("id", i)
            if doc_id is None or doc_id == "":
                doc_id = i
            if doc_id in done_ids:
                continue
            title = (row.get("title") or "").strip()
            text = (row.get("text") or "").strip()
            label = row.get("label")

            if not text:
                print(f"[{i}] id={doc_id} -> [WARN] testo vuoto, salto")
                continue

            rows.append((doc_id, title, text, label))

    if not rows:
        print("[RESUME] Niente da fare, tutti i documenti sono gia' stati processati.")
        return

    start_time = time.monotonic()
    stopped_early = False
    chunk_size = max(1, workers * 4)

    with open(output_path, "a", encoding="utf-8") as fout, \
         concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:

        for chunk_start in range(0, len(rows), chunk_size):
            if time_budget_seconds is not None and (time.monotonic() - start_time) > time_budget_seconds:
                stopped_early = True
                break
            chunk = rows[chunk_start:chunk_start + chunk_size]
            futures = [
                executor.submit(
                    process_one_document, doc_id, title, text, label, model,
                    max_words, overlap_words, include_text, keep_alive,
                )
                for doc_id, title, text, label in chunk
            ]
            for future in futures:
                new_record = future.result()
                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                fout.flush()

    if stopped_early:
        remaining = len(rows) - chunk_start
        print(f"[TIME BUDGET] Limite di tempo raggiunto: {remaining} documenti non "
              f"ancora processati. Rilancia lo stesso comando (Save & Run All) per "
              f"riprendere da dove si e' fermato.")


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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Claim/documenti processati in parallelo (default: {DEFAULT_WORKERS}). "
                              "Usa 1 per il comportamento sequenziale originale.")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE,
                         help="Per quanto tempo Ollama tiene il modello caricato in memoria fra una "
                              "chiamata e l'altra (default Ollama: 5m). Usa '-1' per tenerlo sempre "
                              f"caricato (default qui: {DEFAULT_KEEP_ALIVE})")
    parser.add_argument("--time-budget-minutes", type=float, default=None,
                         help="Se impostato, lo script si ferma da solo (in modo pulito, "
                              "scrivendo tutto cio' che ha gia' completato) dopo questi minuti, "
                              "invece di farsi uccidere a meta' dal limite di sessione Kaggle. "
                              "Rilancia lo stesso comando per riprendere da dove si e' fermato "
                              "(vedi resume automatico basato sull'output gia' scritto).")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        mode = "document" if args.input.lower().endswith(".csv") else "claim"
        print(f"[INFO] --mode non specificato, dedotto '{mode}' dall'estensione del file di input")

    time_budget_seconds = args.time_budget_minutes * 60 if args.time_budget_minutes else None

    if mode == "claim":
        process_file_claims(args.input, args.output, model=args.model, limit=args.limit,
                             workers=args.workers, keep_alive=args.keep_alive,
                             time_budget_seconds=time_budget_seconds)
    else:
        process_file_documents(
            args.input, args.output, model=args.model, limit=args.limit,
            max_words=args.max_words, overlap_words=args.overlap_words,
            include_text=args.include_text, csv_delimiter=args.csv_delimiter,
            workers=args.workers, keep_alive=args.keep_alive,
            time_budget_seconds=time_budget_seconds,
        )

    print(f"\nFatto. Output scritto in: {args.output}")


if __name__ == "__main__":
    main()