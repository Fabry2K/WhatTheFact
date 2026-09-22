#!/usr/bin/env python3
"""
Fase 3 (parte "Search API"): per ogni claim del file JSONL prodotto dallo script
di Fase 2 (generate_questions.py), esegue una ricerca web per ciascuna query
associata alle domande, e salva i risultati in un database **Turso** (libSQL,
compatibile SQLite, ospitato nel cloud) invece che in cartelle+file di testo o
in un file SQLite locale.

Perché Turso invece di SQLite locale
=====================================

Le tabelle sono identiche a una versione SQLite locale (vedi sotto), ma il DB
vive nel cloud: non serve più zippare/scaricare un file `sources.db` a ogni
fine sessione Kaggle — ogni notebook (anche di sessioni diverse) si connette
allo stesso database persistente con solo URL + token. Utile anche per la
parallelizzazione multi-sessione di cui parlavamo (Colab + Kaggle in
contemporanea): entrambe le sessioni scrivono sullo STESSO database condiviso,
quindi la deduplica per URL funziona anche fra sessioni diverse, non solo
dentro una singola run.

Storage: due tabelle (stesso schema della versione SQLite locale)
====================================================================

    sources
        url (PRIMARY KEY), title, body (testo compresso con zlib), content_length,
        content_source, insufficient_content, fetched_at.
        Una riga per URL univoco, indipendentemente da quanti claim/sessioni lo
        trovano. Se l'URL è già in tabella, si riusa il body già salvato invece
        di rifare la richiesta HTTP (anche il fallback fetch diretto viene
        saltato).

    claim_source_links
        claim_id, url (FK verso sources.url), query, query_index, result_index.
        Il collegamento fra un claim e le fonti trovate per lui.

Setup Turso (una tantum, fuori da questo script)
===================================================
    1) Registrati su https://turso.tech (nessuna carta richiesta)
    2) Crea un database: `turso db create <nome>` (via CLI) o dalla dashboard
    3) Crea un token: `turso db tokens create <nome>`
    4) Ti servono due valori: TURSO_DATABASE_URL (tipo "libsql://xxx.turso.io")
       e TURSO_AUTH_TOKEN — passali con --turso-url/--turso-token oppure
       impostali come variabili d'ambiente (es. Kaggle Secrets)

Usa la Tavily Search API (https://tavily.com) per la ricerca — free tier: 1000
crediti/mese, nessuna carta di credito richiesta. Una ricerca "basic" costa 1
credito.

Prerequisiti:
    1) pip install requests libsql
    2) Una API key Tavily gratuita da https://tavily.com (formato "tvly-xxxxx")
    3) Un database Turso (vedi sopra)

Uso:
    export TAVILY_API_KEY="tvly-xxxxx"
    export TURSO_DATABASE_URL="libsql://il-tuo-db.turso.io"
    export TURSO_AUTH_TOKEN="eyJ..."
    python download_sources.py --input claims_with_questions.jsonl --num-results 5
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import re
import sys
import threading
import time
import zlib
import requests

try:
    import libsql
except ImportError:
    print("ERRORE: manca il pacchetto 'libsql'. Installa con: pip install libsql", file=sys.stderr)
    sys.exit(1)

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MIN_CONTENT_CHARS = 500

DEFAULT_WORKERS = 6  # claim processati in parallelo (ognuno fa piu' query in sequenza);
                      # il collo di bottiglia e' la rete, non la CPU, quindi qui i thread
                      # bastano (non serve multiprocessing)
DEFAULT_MAX_RPM = 90  # richieste/minuto verso Tavily, condivise fra tutti i worker.
                       # Le chiavi "Development" (quella del free tier) sono limitate a
                       # 100 RPM: 90 lascia un margine di sicurezza. Se hai una chiave
                       # "Production" puoi alzarlo (fino a ~900 con margine).

# tentativi di retry per ogni operazione sul DB Turso: a differenza di un file
# SQLite locale, qui c'e' di mezzo la rete, quindi un timeout/blip transitorio
# va gestito con un retry invece di far fallire l'intero claim
DB_MAX_RETRIES = 4

FALLBACK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# si alza quando una chiamata torna 401 (chiave non valida): e' un errore che non ha
# senso ritentare migliaia di volte in parallelo, quindi tutti i worker lo controllano
# e si fermano appena possono invece di continuare a sprecare tempo/credito
_fatal_error = threading.Event()

# protegge i print() da interleaving illeggibile quando piu' claim vengono
# processati in contemporanea da thread diversi
_print_lock = threading.Lock()

# protegge ogni accesso al DB: una singola connessione Turso condivisa fra tutti
# i thread, serializzata qui invece di aprire una connessione remota per thread
# (piu' connessioni HTTP concorrenti verso lo stesso DB non aiuterebbero, dato
# che il vero collo di bottiglia sono le chiamate a Tavily, non il DB)
_db_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


class RateLimiter:
    """Limita il numero di richieste/minuto in modo condiviso fra tutti i thread.

    Il rate limit di Tavily si applica alla API key nel suo complesso, non per
    connessione: un semplice `time.sleep(delay)` dentro ogni thread non basta a
    rispettarlo quando piu' worker girano in parallelo, perche' ogni thread conta
    la pausa per conto proprio. Qui invece la pausa e' calcolata su un orologio
    condiviso, cosi' il rate reale resta sotto il tetto anche con N worker.
    """

    def __init__(self, max_per_minute: float):
        self.min_interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self.min_interval


# ---------------------------------------------------------------------------
# Storage (Turso / libSQL)
# ---------------------------------------------------------------------------

def _db_call(fn, *args, **kwargs):
    """Esegue una chiamata al DB sotto lock, con retry per errori di rete
    transitori (a differenza di un file SQLite locale, qui c'e' sempre una
    richiesta HTTP di mezzo)."""
    last_err = None
    for attempt in range(1, DB_MAX_RETRIES + 1):
        try:
            with _db_lock:
                return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            safe_print(f"    [DB retry {attempt}/{DB_MAX_RETRIES}] {e}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"operazione sul DB Turso fallita dopo {DB_MAX_RETRIES} tentativi: {last_err}")


def open_db(turso_url: str, turso_token: str):
    conn = libsql.connect(database=turso_url, auth_token=turso_token)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            url TEXT PRIMARY KEY,
            title TEXT,
            body BLOB,                    -- testo compresso con zlib
            content_length INTEGER,       -- lunghezza del testo ORIGINALE (non compresso)
            content_source TEXT,          -- tavily_raw / tavily_snippet / direct_fetch
            insufficient_content INTEGER, -- 0/1
            fetched_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS claim_source_links (
            claim_id TEXT NOT NULL,
            url TEXT NOT NULL REFERENCES sources(url),
            query TEXT,
            query_index INTEGER,
            result_index INTEGER,
            PRIMARY KEY (claim_id, url)
        )
    """)
    conn.commit()
    return conn


def get_source(conn, url: str):
    """None se l'URL non e' mai stato salvato, altrimenti un dict con i metadati
    (senza decomprimere il body: qui serve solo per decidere se riusarlo)."""
    def _run():
        cur = conn.execute(
            "SELECT title, content_length, content_source, insufficient_content "
            "FROM sources WHERE url = ?",
            (url,),
        )
        return cur.fetchone()

    row = _db_call(_run)
    if row is None:
        return None
    title, content_length, content_source, insufficient_content = row
    return {
        "title": title,
        "content_length": content_length,
        "content_source": content_source,
        "insufficient_content": bool(insufficient_content),
    }


def insert_source(conn, url: str, title: str, body: str,
                   content_length: int, content_source: str, insufficient_content: bool):
    compressed = zlib.compress(body.encode("utf-8"))

    def _run():
        conn.execute(
            "INSERT OR IGNORE INTO sources "
            "(url, title, body, content_length, content_source, insufficient_content, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (url, title, compressed, content_length, content_source, int(insufficient_content),
             datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()

    _db_call(_run)


def link_claim_source(conn, claim_id, url: str, query: str, query_index: int, result_index: int):
    def _run():
        conn.execute(
            "INSERT OR REPLACE INTO claim_source_links "
            "(claim_id, url, query, query_index, result_index) VALUES (?, ?, ?, ?, ?)",
            (str(claim_id), url, query, query_index, result_index),
        )
        conn.commit()

    _db_call(_run)


def get_existing_claim_state(conn, claim_id):
    """Ritorna (query_ordinate, n_righe) gia' collegate a questo claim, per il
    controllo di resume. L'ordine per query_index ricostruisce la stessa sequenza
    [title_query, query domanda 1, query domanda 2, ...] usata per costruire
    `expected` in process_claim."""
    def _run():
        rows = conn.execute(
            "SELECT DISTINCT query_index, query FROM claim_source_links "
            "WHERE claim_id = ? ORDER BY query_index",
            (str(claim_id),),
        ).fetchall()
        n_rows = conn.execute(
            "SELECT COUNT(*) FROM claim_source_links WHERE claim_id = ?",
            (str(claim_id),),
        ).fetchone()[0]
        return rows, n_rows

    rows, n_rows = _db_call(_run)
    return [q for _idx, q in rows], n_rows


# ---------------------------------------------------------------------------

def sanitize_folder_name(value) -> str:
    """Non serve piu' per creare cartelle, ma resta usata per normalizzare
    claim_id a stringa in modo consistente col resto della pipeline (Fase 4/5
    usano lo stesso claim_id come chiave)."""
    name = str(value).strip()
    name = re.sub(r"[^\w\-.]", "_", name)
    return name or "unknown_id"


def tavily_search(query: str, api_key: str, num_results: int, search_depth: str,
                   rate_limiter: RateLimiter = None, max_retries: int = 3, timeout: int = 60) -> list:
    """Esegue una ricerca su Tavily e restituisce una lista di risultati
    [{"url":..., "title":..., "content":...}, ...], al massimo `num_results`.
    In caso di fallimento persistente, restituisce lista vuota (non blocca la pipeline).
    """
    if _fatal_error.is_set():
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "search_depth": search_depth,        # "basic" (1 credito) o "advanced" (2 crediti)
        "max_results": num_results,
        "include_raw_content": True,          # ci serve il contenuto della pagina, non solo lo snippet
        "include_answer": False,
        "include_images": False,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        if _fatal_error.is_set():
            return []
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            resp = requests.post(TAVILY_SEARCH_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 401:
                # chiave non valida: non ha senso ritentare, e con piu' worker in
                # parallelo un sys.exit() qui fermerebbe solo QUESTO thread (in
                # Python un SystemExit in un thread secondario non termina il
                # processo). Segnaliamo l'errore fatale a tutti gli altri worker,
                # che smettono di fare nuove richieste appena possono.
                safe_print("  [FATAL] API key non valida o mancante. Controlla --api-key / TAVILY_API_KEY.")
                _fatal_error.set()
                return []
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after else 2 * attempt
                safe_print(f"  [429] rate limit Tavily raggiunto per '{query}', "
                           f"aspetto {wait_s:.0f}s (tentativo {attempt}/{max_retries})")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return results[:num_results]
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)
            safe_print(f"  [retry {attempt}/{max_retries}] search fallita per query '{query}': {last_err}")
            time.sleep(2 * attempt)

    safe_print(f"  [WARN] nessun risultato per query '{query}' dopo {max_retries} tentativi: {last_err}")
    return []


def fallback_fetch_page(url: str, timeout: int = 20) -> str:
    """Tenta un fetch diretto della pagina quando Tavily restituisce solo uno
    snippet troppo corto (capita spesso con siti con paywall/anti-scraping,
    es. Reuters). Richiede `trafilatura` per un'estrazione di qualità; se non
    installata, ritorna stringa vuota senza bloccare la pipeline (in tal caso
    resta comunque lo snippet di Tavily come contenuto)."""
    if not HAS_TRAFILATURA:
        return ""
    try:
        resp = requests.get(url, headers=FALLBACK_HEADERS, timeout=timeout)
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        return extracted or ""
    except Exception:
        return ""


def save_result_to_db(conn, claim_id, query_idx: int, result_idx: int,
                       query: str, result: dict, min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS) -> dict:
    """Salva (o riusa, se l'URL e' gia' noto) un singolo risultato nel DB e
    collega claim_id a quella fonte. Restituisce un piccolo riepilogo per il log.

    Se il contenuto restituito da Tavily è più corto di `min_content_chars`
    (segno che probabilmente è solo lo snippet di ranking, non la pagina intera
    — capita su siti con paywall/bot-detection), tenta un fetch diretto della
    pagina come fallback — MA solo se l'URL non era gia' in tabella: se un
    claim precedente lo ha gia' scaricato e valutato, non serve rifare la
    richiesta di rete.
    """
    url = result.get("url", "")

    if url:
        existing = get_source(conn, url)
        if existing is not None:
            link_claim_source(conn, claim_id, url, query, query_idx, result_idx)
            return {
                "url": url,
                "title": existing["title"],
                "content_length": existing["content_length"],
                "content_source": existing["content_source"],
                "insufficient_content": existing["insufficient_content"],
                "reused": True,
            }

    title = result.get("title", "")
    # preferiamo il contenuto grezzo/esteso della pagina; fallback allo snippet se assente
    raw_content = result.get("raw_content")
    content = raw_content or result.get("content") or ""
    content_source = "tavily_raw" if raw_content else "tavily_snippet"

    if len(content) < min_content_chars and url:
        fallback_content = fallback_fetch_page(url)
        if len(fallback_content) > len(content):
            content = fallback_content
            content_source = "direct_fetch"

    insufficient_content = len(content) < min_content_chars

    if url:
        insert_source(conn, url, title, content, len(content), content_source, insufficient_content)
        link_claim_source(conn, claim_id, url, query, query_idx, result_idx)

    if insufficient_content:
        safe_print(f"    [WARN] contenuto molto corto ({len(content)} char, fonte: {content_source}) per {url}")

    return {
        "url": url,
        "title": title,
        "content_length": len(content),
        "content_source": content_source,
        "insufficient_content": insufficient_content,
        "reused": False,
    }


def process_claim(conn, claim_id, questions: list, num_results: int,
                   api_key: str, search_depth: str, delay: float, skip_existing: bool,
                   min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
                   title_query: str = "", rate_limiter: RateLimiter = None):
    tag = f"[id={claim_id}]"  # prefisso nei log: indispensabile per leggerli quando
                               # piu' claim vengono processati in parallelo e le righe
                               # di thread diversi si intrecciano nell'output

    current_queries = [q.get("query") or q.get("question") for q in questions]

    # supporto resume: si salta solo se il claim è già stato scaricato CON LE STESSE
    # QUERY (stesso motivo della versione a file: dopo aver migliorato la
    # generazione delle query in Fase 2, confrontare solo il conteggio faceva
    # saltare claim che in realta' andavano ricercati con le query nuove).
    if skip_existing:
        existing_queries, n_existing_rows = get_existing_claim_state(conn, claim_id)
        expected = ([title_query] if title_query else []) + [q for q in current_queries if q]
        if existing_queries:
            if existing_queries != expected:
                safe_print(f"  {tag} -> le query sono cambiate rispetto al download precedente, riscarico")
            elif n_existing_rows >= len(questions) * num_results:
                safe_print(f"  {tag} -> già scaricato completamente con le stesse query, salto "
                           f"({n_existing_rows} fonti collegate)")
                return

    if _fatal_error.is_set():
        return

    seen_urls = set()
    n_saved = 0

    # query_index 0 = query ricavata dal titolo, mirata alla storia nel suo complesso
    # e non a una singola domanda. Serve da rete di sicurezza quando le query
    # per-domanda perdono la dicitura con cui la vicenda e' conosciuta. In Fase 4 con
    # --mapping pooled questi documenti sono disponibili a tutte le domande.
    if title_query:
        safe_print(f"  {tag} query 0 (contesto complessivo): {title_query}")
        for ri, result in enumerate(
                tavily_search(title_query, api_key=api_key, num_results=num_results,
                               search_depth=search_depth, rate_limiter=rate_limiter), start=1):
            url = result.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            save_result_to_db(conn, claim_id, 0, ri, title_query, result,
                               min_content_chars=min_content_chars)
            n_saved += 1
        if delay > 0:
            time.sleep(delay)

    for qi, q in enumerate(questions, start=1):
        if _fatal_error.is_set():
            break

        query = q.get("query") or q.get("question")
        if not query:
            safe_print(f"  {tag} [WARN] domanda {qi} senza query valida, salto")
            continue

        safe_print(f"  {tag} query {qi}/{len(questions)}: {query}")
        results = tavily_search(query, api_key=api_key, num_results=num_results,
                                 search_depth=search_depth, rate_limiter=rate_limiter)

        if len(results) < num_results:
            safe_print(f"  {tag} [WARN] richiesti {num_results} risultati, ottenuti solo {len(results)}")

        for ri, result in enumerate(results, start=1):
            # query diverse dello stesso claim cercano la stessa storia e ricadono
            # spesso sulla stessa pagina: collegarla una volta sola per claim evita
            # di gonfiare il pool di Fase 4 con lo stesso documento ripetuto
            url = result.get("url", "")
            if url and url in seen_urls:
                safe_print(f"    {tag} [dup] {url} già collegato a questo claim, salto")
                continue
            if url:
                seen_urls.add(url)
            info = save_result_to_db(conn, claim_id, qi, ri, query, result,
                                      min_content_chars=min_content_chars)
            if info.get("reused"):
                safe_print(f"    {tag} [riuso] {url} gia' nel DB (trovato per un altro claim)")
            n_saved += 1

        if delay > 0:
            time.sleep(delay)

    safe_print(f"  {tag} -> {n_saved} fonti collegate nel DB")


def process_file(input_path: str, turso_url: str, turso_token: str, num_results: int, api_key: str,
                  search_depth: str, delay: float, limit: int, skip_existing: bool,
                  min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
                  workers: int = DEFAULT_WORKERS, max_rpm: float = DEFAULT_MAX_RPM,
                  time_budget_seconds: float = None):
    if not HAS_TRAFILATURA:
        print("[INFO] libreria 'trafilatura' non installata: il fallback di fetch diretto "
              "per pagine con contenuto troppo corto (es. paywall/anti-scraping) sarà "
              "disattivo. Per attivarlo: pip install trafilatura", file=sys.stderr)

    conn = open_db(turso_url, turso_token)

    records = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            records.append((i, json.loads(line)))

    # rate limiter condiviso fra tutti i thread: e' lui, non --delay, a tenere il
    # rate reale verso Tavily sotto il tetto quando piu' claim vengono processati
    # in parallelo (vedi RateLimiter)
    rate_limiter = RateLimiter(max_rpm)

    def _handle(i, record):
        claim_id = record.get("id")
        claim = record.get("claim") or record.get("title", "")
        questions = record.get("questions", [])
        # query sull'insieme del claim/documento: "context_query" in modalità
        # claim, "title_query" in modalità document (nome storico)
        title_query = record.get("context_query") or record.get("title_query", "")

        safe_print(f"[{i}] id={claim_id} -> {claim} ({len(questions)} query)")

        if not questions:
            safe_print(f"  [id={claim_id}] [WARN] nessuna domanda/query per questo claim, salto")
            return

        process_claim(
            conn, claim_id, questions,
            num_results=num_results, api_key=api_key,
            search_depth=search_depth, delay=delay,
            skip_existing=skip_existing,
            min_content_chars=min_content_chars,
            title_query=title_query,
            rate_limiter=rate_limiter,
        )

    # sottomissione a chunk (non tutto insieme): cosi', se time_budget_seconds e'
    # impostato, possiamo controllare il tempo residuo fra un chunk e l'altro e
    # smettere di avviare nuovi claim, lasciando finire quelli gia' in corso,
    # invece di farci uccidere a meta' dal limite di sessione Kaggle.
    start_time = time.monotonic()
    stopped_early = False
    chunk_size = max(1, workers * 3)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for chunk_start in range(0, len(records), chunk_size):
            if time_budget_seconds is not None and (time.monotonic() - start_time) > time_budget_seconds:
                stopped_early = True
                break
            chunk = records[chunk_start:chunk_start + chunk_size]
            futures = [executor.submit(_handle, i, record) for i, record in chunk]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # rilancia qui eventuali eccezioni inattese
            if _fatal_error.is_set():
                break

    if stopped_early:
        remaining = len(records) - chunk_start
        print(f"\n[TIME BUDGET] Limite di tempo raggiunto: circa {remaining} claim non ancora "
              f"processati. Rilancia lo stesso comando (Save & Run All) per riprendere: il "
              f"resume salta automaticamente i claim gia' scaricati.")

    n_sources = _db_call(lambda: conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
    n_links = _db_call(lambda: conn.execute("SELECT COUNT(*) FROM claim_source_links").fetchone()[0])
    print(f"\n[INFO] Turso DB: {n_sources} fonti uniche, {n_links} collegamenti claim->fonte")
    try:
        conn.close()
    except Exception:
        pass  # alcune connessioni libsql remote non richiedono/supportano close() esplicito

    if _fatal_error.is_set():
        print("\nERRORE: API key non valida, interrotto.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path al file JSONL prodotto da generate_questions.py")
    parser.add_argument("--turso-url", default=os.environ.get("TURSO_DATABASE_URL"),
                         help="URL del database Turso (es. libsql://il-tuo-db.turso.io). "
                              "Default: legge da env var TURSO_DATABASE_URL.")
    parser.add_argument("--turso-token", default=os.environ.get("TURSO_AUTH_TOKEN"),
                         help="Auth token del database Turso. Default: legge da env var TURSO_AUTH_TOKEN.")
    parser.add_argument("--num-results", "-x", type=int, default=5, help="Numero di risultati da scaricare per ogni query (X, default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe del file di input (default: tutte)")
    parser.add_argument("--api-key", default=os.environ.get("TAVILY_API_KEY"), help="Tavily API key (default: legge da env var TAVILY_API_KEY)")
    parser.add_argument("--search-depth", choices=["basic", "advanced"], default="advanced",
                         help="Profondità di ricerca Tavily: basic=1 credito, advanced=2 crediti "
                              "(default: advanced — su questo compito la rilevanza dei primi "
                              "risultati conta più del risparmio di crediti, perché una fonte "
                              "fuori tema si traduce direttamente in un 'Non verificabile')")
    parser.add_argument("--delay", type=float, default=0.0,
                         help="Pausa extra (secondi) dopo ogni query, oltre al rate limiting "
                              "condiviso gia' applicato da --max-rpm (default: 0, di solito "
                              "non serve — usa --max-rpm per il throttling vero e proprio)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Non saltare i claim già scaricati completamente (riscarica tutto)")
    parser.add_argument("--min-content-chars", type=int, default=DEFAULT_MIN_CONTENT_CHARS,
                         help=f"Soglia minima di caratteri sotto la quale si tenta il fetch diretto di fallback (default: {DEFAULT_MIN_CONTENT_CHARS})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Claim processati in parallelo (default: {DEFAULT_WORKERS}). "
                              "Task I/O-bound (rete): alzarlo aiuta finche' non si satura "
                              "--max-rpm, oltre quel punto i worker in piu' restano solo in coda.")
    parser.add_argument("--max-rpm", type=float, default=DEFAULT_MAX_RPM,
                         help=f"Richieste/minuto massime verso Tavily, condivise fra tutti i "
                              f"worker (default: {DEFAULT_MAX_RPM}). Chiave 'Development' "
                              "(free tier): limite reale 100 RPM. Chiave 'Production': fino a "
                              "1000 RPM, puoi alzarlo parecchio.")
    parser.add_argument("--time-budget-minutes", type=float, default=None,
                         help="Se impostato, lo script si ferma da solo (in modo pulito) dopo "
                              "questi minuti invece di farsi uccidere a meta' dal limite di "
                              "sessione della piattaforma. Rilancia lo stesso comando per "
                              "riprendere da dove si e' fermato (resume automatico).")
    args = parser.parse_args()

    if not args.api_key:
        print("ERRORE: nessuna API key fornita. Passa --api-key oppure imposta TAVILY_API_KEY.", file=sys.stderr)
        sys.exit(1)
    if not args.turso_url or not args.turso_token:
        print("ERRORE: URL/token Turso mancanti. Passa --turso-url/--turso-token oppure imposta "
              "TURSO_DATABASE_URL/TURSO_AUTH_TOKEN.", file=sys.stderr)
        sys.exit(1)

    process_file(
        args.input, args.turso_url, args.turso_token,
        num_results=args.num_results,
        api_key=args.api_key,
        search_depth=args.search_depth,
        delay=args.delay,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
        min_content_chars=args.min_content_chars,
        workers=args.workers,
        max_rpm=args.max_rpm,
        time_budget_seconds=args.time_budget_minutes * 60 if args.time_budget_minutes else None,
    )
    print("\nFatto.")


if __name__ == "__main__":
    main()