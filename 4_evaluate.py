#!/usr/bin/env python3
"""
Fase 5 (Delta Analysis): aggrega i risultati del blind matching (Fase 4) in un
verdetto unico per claim, lo confronta con la label reale, e calcola precision,
recall e F-measure complessivi.

Formula di aggregazione — motivazione
======================================

Il problema ha una struttura a due livelli: ogni claim si scompone in piu' domande
(le "assertion" di Fase 2, ciascuna con una centralita' 1-5), e ogni domanda viene
verificata contro piu' fonti indipendenti (Fase 4), ciascuna con un esito
("Concorda"/"Contraddice"/"Parzialmente concorda"/"Non verificabile"). Serve quindi
un'aggregazione in due passi: prima le fonti dentro una domanda, poi le domande
dentro il claim.

PASSO 1 — punteggio di ogni domanda (aggregazione fra fonti)
--------------------------------------------------------------
Ogni esito diventa un punteggio numerico:
    Concorda              -> +1.0
    Parzialmente concorda -> +0.5
    Contraddice            -> -1.0
    Non verificabile       -> escluso dalla media (vedi sotto)

Scala simmetrica (+1/-1) fra conferma e smentita: in un compito di fact-checking
una fonte che contraddice e' un segnale forte tanto quanto una che conferma, non
c'e' ragione di dare piu' peso alla fiducia che al sospetto. "Parzialmente
concorda" vale meta' di "Concorda" perche' e' esplicitamente definito in Fase 4
come accordo sul fatto principale con una riserva su un dettaglio secondario: e'
evidenza a favore, ma piu' debole di una conferma piena.

"Non verificabile" NON entra nella media come zero: un articolo che non tratta
l'argomento non e' una via di mezzo fra confermare e smentire, e' assenza di
segnale. Farlo contare come zero diluirebbe artificialmente le fonti che
*hanno* trovato qualcosa. Viene invece escluso dalla media, e tracciato a parte
come "copertura" (passo 3): se TUTTE le fonti di una domanda sono "Non
verificabile", quella domanda non contribuisce alcun segnale.

Ogni fonte pesa secondo l'affidabilita' della sua estrazione: le fonti con
evidenza verificata verbatim (`evidenza_verificata`) pesano 1.0, quelle con
citazione non trovata nel testo (possibile riformulazione o allucinazione)
pesano meno (default 0.5, --unverified-weight). Non si escludono del tutto: nei
test di questa pipeline una citazione "non verificata" e' risultata spesso un
falso positivo del controllo string-matching (es. il modello sostituisce le
virgolette per non rompere il JSON), non una vera allucinazione — scartarle
avrebbe buttato via anche giudizi corretti.

PASSO 2 — copertura e punteggio del claim (aggregazione fra domande)
----------------------------------------------------------------------
Le domande vengono pesate per centralita' (1-5, assegnata in Fase 2): un claim
si giudica soprattutto sulla sua asserzione principale, non in modo paritario
su ogni dettaglio periferico. Il punteggio del claim e' la media delle domande
"coperte" (con almeno una fonte diversa da "Non verificabile"), pesata per
centralita'.

La COPERTURA e' la quota di peso-centralita' del claim che proviene da domande
coperte. Se la ricerca non ha trovato nulla sulla maggior parte dei fatti
centrali del claim, il sistema non ha basi sufficienti per esprimersi — e deve
dirlo, invece di azzardare un verdetto su una minoranza di prove.

PASSO 3 — decisione finale
----------------------------
    se copertura < --coverage-threshold (default 1/3):
        NOT ENOUGH INFO   (non si e' verificata una quota sufficiente del claim)
    altrimenti, sul punteggio pesato del claim:
        punteggio >= +soglia  -> SUPPORTS
        punteggio <= -soglia  -> REFUTES
        altrimenti             -> NOT ENOUGH INFO   (prove deboli o contrastanti)
soglia di default = --support-threshold = 0.4.

Le tre etichette (SUPPORTS / REFUTES / NOT ENOUGH INFO) sono le stesse usate dal
dataset FEVER di riferimento, cosi' il confronto con la label reale e' diretto.

Limite noto: senza uno score di affidabilita' per dominio (whitelist rimossa in
Fase 3, si veda il piano v2 con `domain_authority`), una singola fonte autorevole
in contraddizione puo' essere annacquata da piu' fonti minori concordi. E' una
scelta consapevole vista l'assenza di quello score, non un difetto nascosto.

NEI e copertura fonti (Wikipedia vs extra-Wikipedia)
======================================================
FEVER etichetta NOT ENOUGH INFO in base a cosa un annotatore ha trovato SOLO su
Wikipedia. La nostra pipeline cerca open-domain (Fase 3 senza whitelist), quindi
puo' trovare evidenza SUPPORTS/REFUTES per claim che FEVER etichetta NEI — senza
che questo sia un errore della pipeline. Per non falsare le metriche, ogni claim
con gold_label == NOT ENOUGH INFO viene classificato in un bucket, guardando i
`source_url` delle sole righe che hanno dato un segnale (esito != "Non
verificabile") nelle domande coperte:

    agreement                     -> anche la pipeline dice NEI (nessun disaccordo)
    wikipedia_recall_miss         -> la pipeline trova SUPPORTS/REFUTES usando
                                      SOLO fonti Wikipedia (possibile miss di
                                      recall dell'annotatore originale: da
                                      controllare, e' comunque un errore
                                      "interno" a Wikipedia)
    source_coverage_disagreement  -> la pipeline trova SUPPORTS/REFUTES con
                                      almeno una fonte extra-Wikipedia (limite
                                      noto del dataset, non un errore della
                                      pipeline)
    no_evidence_url               -> predicted label != NEI ma le righe che
                                      hanno dato segnale non hanno source_url
                                      (dato mancante, da controllare a parte)

L'accuracy "adjusted" riportata in fondo non conta come errore i claim del
bucket source_coverage_disagreement (li esclude dal denominatore), a differenza
dell'accuracy "grezza" che li conta come sbagliati come farebbe uno scoring
FEVER standard.

Uso:
    python 5_evaluate.py --matching-dir claim_matching --labels claims_labelled.jsonl
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from urllib.parse import urlparse

ESITO_SCORE = {
    "Concorda": 1.0,
    "Parzialmente concorda": 0.5,
    "Contraddice": -1.0,
    # "Non verificabile" e' assente di proposito: non ha un punteggio, va escluso
}

VALID_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
NEI_LABEL = "NOT ENOUGH INFO"

DEFAULT_CENTRALITY = 3  # fallback se una domanda non ha centrality (non dovrebbe capitare)


def is_wikipedia_domain(url: str) -> bool:
    """True se l'URL appartiene a un dominio Wikipedia (qualsiasi lingua)."""
    if not url:
        return False
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return "wikipedia.org" in host


def aggregate_question(rows: list, verified_weight: float, unverified_weight: float):
    """Aggrega le fonti di UNA domanda in (punteggio, coperta, source_urls).

    source_urls sono gli URL delle sole righe che hanno dato un segnale
    (esito diverso da "Non verificabile"): sono le fonti che hanno
    effettivamente determinato il verdetto su questa domanda.

    Ritorna (None, False, []) se nessuna fonte porta segnale (tutte "Non
    verificabile" o dati mancanti): la domanda resta non coperta.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    source_urls = []
    for row in rows:
        score = ESITO_SCORE.get(row.get("esito"))
        if score is None:
            continue
        weight = verified_weight if row.get("evidenza_verificata") else unverified_weight
        weighted_sum += score * weight
        total_weight += weight
        url = row.get("source_url")
        if url:
            source_urls.append(url)

    if total_weight == 0:
        return None, False, []
    return weighted_sum / total_weight, True, source_urls


def aggregate_claim(questions: dict, support_threshold: float, coverage_threshold: float):
    """Aggrega le domande di un claim (gia' ridotte a punteggio+centralita' dal
    passo precedente) nel verdetto finale. Ritorna (label, punteggio, copertura)."""
    total_centrality = 0.0
    covered_centrality = 0.0
    weighted_sum = 0.0

    for question_score, centrality, covered, _urls in questions.values():
        c = centrality if centrality else DEFAULT_CENTRALITY
        total_centrality += c
        if covered:
            covered_centrality += c
            weighted_sum += question_score * c

    coverage = covered_centrality / total_centrality if total_centrality else 0.0

    if coverage < coverage_threshold or covered_centrality == 0:
        return "NOT ENOUGH INFO", 0.0, coverage

    claim_score = weighted_sum / covered_centrality
    if claim_score >= support_threshold:
        label = "SUPPORTS"
    elif claim_score <= -support_threshold:
        label = "REFUTES"
    else:
        label = "NOT ENOUGH INFO"
    return label, claim_score, coverage


def classify_nei_bucket(predicted_label: str, questions: dict) -> str:
    """Classifica un claim con gold_label == NEI in uno dei quattro bucket,
    guardando i source_url delle domande COPERTE (quelle che hanno contribuito
    al verdetto)."""
    if predicted_label == NEI_LABEL:
        return "agreement"

    urls = []
    for _score, _centrality, covered, source_urls in questions.values():
        if covered:
            urls.extend(source_urls)

    if not urls:
        return "no_evidence_url"

    if all(is_wikipedia_domain(u) for u in urls):
        return "wikipedia_recall_miss"

    return "source_coverage_disagreement"


def evaluate_claim(rows: list, verified_weight: float, unverified_weight: float,
                    support_threshold: float, coverage_threshold: float) -> dict:
    """Dalle righe grezze di Fase 4 (un claim) al verdetto finale, passando dai
    punteggi per-domanda."""
    by_question = {}
    for row in rows:
        by_question.setdefault(row["assertion_index"], []).append(row)

    questions = {}
    for index, question_rows in by_question.items():
        score, covered, source_urls = aggregate_question(question_rows, verified_weight, unverified_weight)
        centrality = question_rows[0].get("centrality")
        questions[index] = (score if covered else 0.0, centrality, covered, source_urls)

    label, claim_score, coverage = aggregate_claim(questions, support_threshold, coverage_threshold)
    return {
        "predicted_label": label,
        "score": round(claim_score, 4),
        "coverage": round(coverage, 4),
        "n_domande": len(questions),
        "n_domande_coperte": sum(1 for _, _, covered, _ in questions.values() if covered),
        "n_fonti_totali": len(rows),
        "_questions": questions,  # uso interno per classify_nei_bucket, rimosso prima di scrivere su file
    }


def load_gold_labels(path: str) -> dict:
    labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            claim_id = str(record.get("id"))
            label = record.get("label")
            if label in VALID_LABELS:
                labels[claim_id] = label
    return labels


def load_claim_texts(matching_dir: str) -> dict:
    """Recupera il testo del claim dalle righe di Fase 4 quando disponibile
    (campo "assertion" non lo riporta; si usa quindi il file rollup se presente,
    altrimenti si lascia vuoto)."""
    rollup_path = os.path.join(matching_dir, "rollup.jsonl")
    texts = {}
    if os.path.exists(rollup_path):
        with open(rollup_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                texts[str(record.get("claim_id"))] = record.get("claim", "")
    return texts


def compute_metrics(pairs: list) -> dict:
    """precision/recall/F1 per classe + macro-average, su una lista di
    (predetto, reale)."""
    labels = VALID_LABELS
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for predicted, gold in pairs:
        if predicted == gold:
            tp[predicted] += 1
        else:
            fp[predicted] += 1
            fn[gold] += 1

    per_class = {}
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": tp[label] + fn[label],
        }

    accuracy = sum(1 for p, g in pairs if p == g) / len(pairs) if pairs else 0.0
    macro_precision = sum(m["precision"] for m in per_class.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in per_class.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(labels)

    return {
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "n": len(pairs),
    }


def print_report(metrics: dict, confusion: dict, nei_buckets: Counter,
                  adjusted_accuracy: float, n_adjusted: int):
    labels = VALID_LABELS
    print("\n=== Matrice di confusione (righe = reale, colonne = predetto) ===")
    header = "reale\\predetto".ljust(18) + "".join(l[:14].ljust(16) for l in labels)
    print(header)
    for gold in labels:
        row = gold.ljust(18)
        for predicted in labels:
            row += str(confusion.get((gold, predicted), 0)).ljust(16)
        print(row)

    print("\n=== Metriche per classe ===")
    print(f"{'classe':<18}{'precision':<12}{'recall':<12}{'f1':<12}{'support':<10}")
    for label in labels:
        m = metrics["per_class"][label]
        print(f"{label:<18}{m['precision']:<12.3f}{m['recall']:<12.3f}{m['f1']:<12.3f}{m['support']:<10}")

    print(f"\naccuracy grezza (standard):  {metrics['accuracy']:.3f}  (n={metrics['n']})")
    print(f"accuracy adjusted (esclude source_coverage_disagreement): {adjusted_accuracy:.3f}  (n={n_adjusted})")
    print(f"macro precision:      {metrics['macro_precision']:.3f}")
    print(f"macro recall:         {metrics['macro_recall']:.3f}")
    print(f"macro F1:             {metrics['macro_f1']:.3f}")

    n_nei = sum(nei_buckets.values())
    if n_nei:
        print(f"\n=== Breakdown claim gold=NOT ENOUGH INFO (totale {n_nei}) ===")
        for bucket in ("agreement", "wikipedia_recall_miss", "source_coverage_disagreement", "no_evidence_url"):
            count = nei_buckets.get(bucket, 0)
            pct = 100 * count / n_nei
            print(f"  {bucket:<32} {count:>5}  ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matching-dir", default="matching_results",
                         help="Cartella con i file <claim_id>.jsonl prodotti dalla Fase 4")
    parser.add_argument("--labels", required=True,
                         help="Path al file JSONL con le label reali (campo 'id' e 'label': "
                              "SUPPORTS/REFUTES/NOT ENOUGH INFO)")
    parser.add_argument("--output", default=None,
                         help="Path del file di confronto per-claim (default: <matching-dir>/verdicts.jsonl)")
    parser.add_argument("--support-threshold", type=float, default=0.4,
                         help="Soglia sul punteggio del claim per SUPPORTS/REFUTES (default: 0.4)")
    parser.add_argument("--coverage-threshold", type=float, default=1 / 3,
                         help="Quota minima di peso-centralita' coperta da evidenza per esprimere "
                              "un verdetto, altrimenti NOT ENOUGH INFO (default: 1/3)")
    parser.add_argument("--verified-weight", type=float, default=1.0,
                         help="Peso di una fonte con evidenza verificata verbatim (default: 1.0)")
    parser.add_argument("--unverified-weight", type=float, default=0.5,
                         help="Peso di una fonte con evidenza NON verificata verbatim (default: 0.5)")
    args = parser.parse_args()

    gold_labels = load_gold_labels(args.labels)
    print(f"[INFO] {len(gold_labels)} label reali caricate da {args.labels}")

    claim_texts = load_claim_texts(args.matching_dir)

    result_files = sorted(
        p for p in glob.glob(os.path.join(args.matching_dir, "*.jsonl"))
        if os.path.basename(p) != "rollup.jsonl" and os.path.basename(p) != "verdicts.jsonl"
    )
    if not result_files:
        print(f"[ERRORE] nessun file di risultati trovato in {args.matching_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(args.matching_dir, "verdicts.jsonl")
    pairs = []
    confusion = Counter()
    nei_buckets = Counter()
    n_no_gold = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in result_files:
            claim_id = os.path.splitext(os.path.basename(path))[0]
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if not rows:
                continue

            verdict = evaluate_claim(
                rows, verified_weight=args.verified_weight, unverified_weight=args.unverified_weight,
                support_threshold=args.support_threshold, coverage_threshold=args.coverage_threshold,
            )
            questions = verdict.pop("_questions")
            gold = gold_labels.get(claim_id)

            nei_bucket = None
            if gold == NEI_LABEL:
                nei_bucket = classify_nei_bucket(verdict["predicted_label"], questions)
                nei_buckets[nei_bucket] += 1

            record = {
                "claim_id": claim_id,
                "claim": claim_texts.get(claim_id, ""),
                **verdict,
                "gold_label": gold,
                "correct": (verdict["predicted_label"] == gold) if gold else None,
                "nei_bucket": nei_bucket,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if gold is None:
                n_no_gold += 1
                continue
            pairs.append((verdict["predicted_label"], gold))
            confusion[(gold, verdict["predicted_label"])] += 1

    print(f"[INFO] {len(result_files)} claim processati, {len(pairs)} con label reale disponibile"
          + (f" ({n_no_gold} senza label, esclusi dalle metriche)" if n_no_gold else ""))
    print(f"[INFO] verdetti per-claim scritti in {output_path}")

    if not pairs:
        print("[ERRORE] nessun claim con label reale trovata: impossibile calcolare le metriche", file=sys.stderr)
        sys.exit(1)

    metrics = compute_metrics(pairs)

    # accuracy "adjusted": esclude dal denominatore i claim gold=NEI finiti nel
    # bucket source_coverage_disagreement, che non sono errori della pipeline
    # ma limiti noti del dataset (evidenza extra-Wikipedia che FEVER non poteva vedere)
    n_excluded = nei_buckets.get("source_coverage_disagreement", 0)
    n_adjusted = len(pairs) - n_excluded
    n_correct_adjusted = sum(1 for p, g in pairs if p == g)
    adjusted_accuracy = n_correct_adjusted / n_adjusted if n_adjusted else 0.0

    print_report(metrics, confusion, nei_buckets, adjusted_accuracy, n_adjusted)


if __name__ == "__main__":
    main()