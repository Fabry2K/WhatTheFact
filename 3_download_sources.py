#!/usr/bin/env python3
"""
Fase 3 (parte "Search API"): per ogni claim del file JSONL prodotto dallo script
di Fase 2 (generate_questions.py), esegue una ricerca web per ciascuna query
associata alle domande, e scarica le prime X pagine di risultato in una cartella
dedicata all'id del claim.

Esempio: claim con id 36782, 3 query associate, X=5
    -> viene creata la cartella "36782" con 15 file (5 per ogni query)

Usa la Tavily Search API (https://tavily.com) — free tier: 1000 crediti/mese,
nessuna carta di credito richiesta. Una ricerca "basic" costa 1 credito.

Prerequisiti:
    1) Una API key gratuita da https://tavily.com (formato "tvly-xxxxx")
    2) pip install requests

Uso:
    export TAVILY_API_KEY="tvly-xxxxx"
    python download_sources.py --input claims_with_questions.jsonl --output-dir sources --num-results 5

    # oppure passando la chiave direttamente:
    python download_sources.py --input claims_with_questions.jsonl --output-dir sources \
        --num-results 5 --limit 10 --api-key tvly-xxxxx

Struttura di output:
    sources/
      36782/
        manifest.json              <- indice di tutti i risultati scaricati per questo claim
        q01_r01.txt                <- risultato 1 della query 1
        q01_r02.txt
        ...
        q03_r05.txt                <- risultato 5 della query 3
"""

import argparse
import json
import os
import re
import sys
import time
import requests

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MIN_CONTENT_CHARS = 500

FALLBACK_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def sanitize_folder_name(value) -> str:
    """Rende sicuro come nome di cartella un id qualsiasi (int o stringa)."""
    name = str(value).strip()
    name = re.sub(r"[^\w\-.]", "_", name)
    return name or "unknown_id"


def tavily_search(query: str, api_key: str, num_results: int, search_depth: str,
                   max_retries: int = 3, timeout: int = 60) -> list:
    """Esegue una ricerca su Tavily e restituisce una lista di risultati
    [{"url":..., "title":..., "content":...}, ...], al massimo `num_results`.
    In caso di fallimento persistente, restituisce lista vuota (non blocca la pipeline).
    """
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
        try:
            resp = requests.post(TAVILY_SEARCH_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 401:
                print("  [FATAL] API key non valida o mancante. Controlla --api-key / TAVILY_API_KEY.", file=sys.stderr)
                sys.exit(1)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return results[:num_results]
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)
            print(f"  [retry {attempt}/{max_retries}] search fallita per query '{query}': {last_err}", file=sys.stderr)
            time.sleep(2 * attempt)

    print(f"  [WARN] nessun risultato per query '{query}' dopo {max_retries} tentativi: {last_err}", file=sys.stderr)
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


def save_result_to_file(folder: str, query_idx: int, result_idx: int, query: str, result: dict,
                         min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS) -> dict:
    """Salva un singolo risultato su disco e restituisce la entry di manifest corrispondente.

    Se il contenuto restituito da Tavily è più corto di `min_content_chars`
    (segno che probabilmente è solo lo snippet di ranking, non la pagina intera
    — capita su siti con paywall/bot-detection), tenta un fetch diretto della
    pagina come fallback.
    """
    url = result.get("url", "")
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

    filename = f"q{query_idx:02d}_r{result_idx:02d}.txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"CONTENT_SOURCE: {content_source}\n")
        f.write("---\n")
        f.write(content)

    if insufficient_content:
        print(f"    [WARN] contenuto molto corto ({len(content)} char, fonte: {content_source}) per {url}")

    return {
        "query_index": query_idx,
        "result_index": result_idx,
        "query": query,
        "url": url,
        "title": title,
        "filename": filename,
        "content_length": len(content),
        "content_source": content_source,
        "insufficient_content": insufficient_content,
    }


def process_claim(claim_id, questions: list, output_dir: str, num_results: int,
                   api_key: str, search_depth: str, delay: float, skip_existing: bool,
                   min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS,
                   title_query: str = ""):
    folder_name = sanitize_folder_name(claim_id)
    folder = os.path.join(output_dir, folder_name)
    os.makedirs(folder, exist_ok=True)

    manifest_path = os.path.join(folder, "manifest.json")
    manifest = []

    current_queries = [q.get("query") or q.get("question") for q in questions]

    # supporto resume: si salta solo se il claim è già stato scaricato CON LE STESSE
    # QUERY. Confrontare il solo numero di file era una trappola: dopo aver migliorato
    # la generazione delle query in Fase 2, i claim già presenti venivano saltati e la
    # pipeline continuava a girare sui documenti trovati con le query vecchie — quindi
    # le query nuove non venivano mai effettivamente cercate.
    if skip_existing and os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_queries = []
            for entry in existing:
                if entry.get("query") not in existing_queries:
                    existing_queries.append(entry.get("query"))

            expected = ([title_query] if title_query else []) + [q for q in current_queries if q]
            if existing_queries != expected:
                print(f"  -> le query sono cambiate rispetto al download precedente, riscarico")
            elif len(existing) == len(questions) * num_results:
                print(f"  -> già scaricato completamente con le stesse query, salto "
                      f"({len(existing)} entry)")
                return
        except (json.JSONDecodeError, OSError):
            pass  # manifest corrotto/incompleto, riprocessiamo

    seen_urls = set()

    # query_index 0 = query ricavata dal titolo, mirata alla storia nel suo complesso
    # e non a una singola domanda. Serve da rete di sicurezza quando le query
    # per-domanda perdono la dicitura con cui la vicenda e' conosciuta. In Fase 4 con
    # --mapping pooled questi documenti sono disponibili a tutte le domande.
    if title_query:
        print(f"  query 0 (contesto complessivo): {title_query}")
        for ri, result in enumerate(
                tavily_search(title_query, api_key=api_key, num_results=num_results,
                               search_depth=search_depth), start=1):
            url = result.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            manifest.append(save_result_to_file(folder, 0, ri, title_query, result,
                                                 min_content_chars=min_content_chars))
        if delay > 0:
            time.sleep(delay)

    for qi, q in enumerate(questions, start=1):
        query = q.get("query") or q.get("question")
        if not query:
            print(f"  [WARN] domanda {qi} senza query valida, salto")
            continue

        print(f"  query {qi}/{len(questions)}: {query}")
        results = tavily_search(query, api_key=api_key, num_results=num_results, search_depth=search_depth)

        if len(results) < num_results:
            print(f"  [WARN] richiesti {num_results} risultati, ottenuti solo {len(results)}")

        for ri, result in enumerate(results, start=1):
            # query diverse dello stesso claim cercano la stessa storia e ricadono
            # spesso sulla stessa pagina: salvarla una volta sola evita di gonfiare
            # il pool di Fase 4 con copie dello stesso documento
            url = result.get("url", "")
            if url and url in seen_urls:
                print(f"    [dup] {url} già scaricato per questo claim, salto")
                continue
            if url:
                seen_urls.add(url)
            entry = save_result_to_file(folder, qi, ri, query, result, min_content_chars=min_content_chars)
            manifest.append(entry)

        if delay > 0:
            time.sleep(delay)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  -> {len(manifest)} pagine salvate in {folder}/")


def process_file(input_path: str, output_dir: str, num_results: int, api_key: str,
                  search_depth: str, delay: float, limit: int, skip_existing: bool,
                  min_content_chars: int = DEFAULT_MIN_CONTENT_CHARS):
    os.makedirs(output_dir, exist_ok=True)

    if not HAS_TRAFILATURA:
        print("[INFO] libreria 'trafilatura' non installata: il fallback di fetch diretto "
              "per pagine con contenuto troppo corto (es. paywall/anti-scraping) sarà "
              "disattivo. Per attivarlo: pip install trafilatura", file=sys.stderr)

    with open(input_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            claim_id = record.get("id")
            claim = record.get("claim") or record.get("title", "")
            questions = record.get("questions", [])
            # query sull'insieme del claim/documento: "context_query" in modalità
            # claim, "title_query" in modalità document (nome storico)
            title_query = record.get("context_query") or record.get("title_query", "")

            print(f"[{i}] id={claim_id} -> {claim} ({len(questions)} query)")

            if not questions:
                print("  [WARN] nessuna domanda/query per questo claim, salto")
                continue

            process_claim(
                claim_id, questions, output_dir,
                num_results=num_results, api_key=api_key,
                search_depth=search_depth, delay=delay,
                skip_existing=skip_existing,
                min_content_chars=min_content_chars,
                title_query=title_query,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path al file JSONL prodotto da generate_questions.py")
    parser.add_argument("--output-dir", default="sources", help="Cartella base dove creare le sottocartelle per claim (default: sources)")
    parser.add_argument("--num-results", "-x", type=int, default=5, help="Numero di risultati da scaricare per ogni query (X, default: 5)")
    parser.add_argument("--limit", type=int, default=None, help="Processa solo le prime N righe del file di input (default: tutte)")
    parser.add_argument("--api-key", default=os.environ.get("TAVILY_API_KEY"), help="Tavily API key (default: legge da env var TAVILY_API_KEY)")
    parser.add_argument("--search-depth", choices=["basic", "advanced"], default="advanced",
                         help="Profondità di ricerca Tavily: basic=1 credito, advanced=2 crediti "
                              "(default: advanced — su questo compito la rilevanza dei primi "
                              "risultati conta più del risparmio di crediti, perché una fonte "
                              "fuori tema si traduce direttamente in un 'Non verificabile')")
    parser.add_argument("--delay", type=float, default=0.5, help="Secondi di pausa tra una query e l'altra, per rispettare i rate limit (default: 0.5)")
    parser.add_argument("--no-skip-existing", action="store_true", help="Non saltare i claim già scaricati completamente (riscarica tutto)")
    parser.add_argument("--min-content-chars", type=int, default=DEFAULT_MIN_CONTENT_CHARS,
                         help=f"Soglia minima di caratteri sotto la quale si tenta il fetch diretto di fallback (default: {DEFAULT_MIN_CONTENT_CHARS})")
    args = parser.parse_args()

    if not args.api_key:
        print("ERRORE: nessuna API key fornita. Passa --api-key oppure imposta TAVILY_API_KEY.", file=sys.stderr)
        sys.exit(1)

    process_file(
        args.input, args.output_dir,
        num_results=args.num_results,
        api_key=args.api_key,
        search_depth=args.search_depth,
        delay=args.delay,
        limit=args.limit,
        skip_existing=not args.no_skip_existing,
        min_content_chars=args.min_content_chars,
    )
    print("\nFatto.")


if __name__ == "__main__":
    main()