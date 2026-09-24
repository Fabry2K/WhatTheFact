#!/usr/bin/env python3
"""
Fase 5 (Delta Analysis): aggrega i risultati del blind matching (Fase 4) in un
verdetto unico per claim, lo confronta con la label reale, e calcola precision,
recall e F1 per classe e complessive (macro-average).

Formula di aggregazione
=======================

PASSO 1 — punteggio di ogni domanda
------------------------------------

Ogni esito di Fase 4 viene trasformato in:

    Concorda              -> +1.0
    Parzialmente concorda -> +0.5
    Contraddice           -> -1.0
    Non verificabile      -> escluso

Le evidenze verificate (`evidenza_verificata=True`) pesano 1.0.
Le evidenze non verificate pesano `unverified_weight` (default 0.5).

Le risposte marcate `esito_incoerente=True` vengono escluse: non devono
contribuire al verdetto quando la risposta non è coerente con la domanda.

PASSO 2 — aggregazione del claim
---------------------------------

Le domande vengono pesate per centralità (1-5).

La copertura è:

    covered_centrality / total_centrality

Se la copertura è inferiore a `coverage_threshold`, il claim viene classificato
come NOT ENOUGH INFO.

Altrimenti:

    score >= support_threshold  -> SUPPORTS
    score <= -support_threshold -> REFUTES
    altrimenti                  -> NOT ENOUGH INFO

Metriche
========

Due blocchi di precision/recall/F1 (per classe + macro-average):

    1. su TUTTI i claim (3 classi: SUPPORTS, REFUTES, NOT ENOUGH INFO)
    2. sui claim con gold != NOT ENOUGH INFO (2 classi: SUPPORTS, REFUTES) —
       utile per isolare la capacità della pipeline di distinguere
       SUPPORTS/REFUTES dal comportamento su NEI, che dipende molto da come
       il dataset di valutazione definisce "non abbastanza informazione".

Uso:

    python 5_evaluate.py --matching-dir claim_matching --labels claims_labelled.jsonl
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter


# ============================================================
# CONFIGURAZIONE
# ============================================================

ESITO_SCORE = {
    "Concorda": 1.0,
    "Parzialmente concorda": 0.5,
    "Contraddice": -1.0,
}

VALID_LABELS = (
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO",
)

NON_NEI_LABELS = (
    "SUPPORTS",
    "REFUTES",
)

NEI_LABEL = "NOT ENOUGH INFO"

DEFAULT_CENTRALITY = 3

DEFAULT_SUPPORT_THRESHOLD = 0.4
DEFAULT_COVERAGE_THRESHOLD = 1 / 3

DEFAULT_VERIFIED_WEIGHT = 1.0
DEFAULT_UNVERIFIED_WEIGHT = 0.5


# ============================================================
# UTILITY
# ============================================================

def is_verified(row: dict) -> bool:
    """Interpreta il campo evidenza_verificata in modo robusto (bool o stringa)."""
    value = row.get("evidenza_verificata", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "vero", "si", "sì"}
    return bool(value)


def is_incoherent(row: dict) -> bool:
    """True se la Fase 4 ha marcato la risposta come incoerente. Le risposte
    incoerenti vengono escluse dall'aggregazione."""
    value = row.get("esito_incoerente", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "vero", "si", "sì"}
    return bool(value)


# ============================================================
# AGGREGAZIONE DOMANDA
# ============================================================

def aggregate_question(rows: list, verified_weight: float, unverified_weight: float):
    """Aggrega le fonti di UNA domanda in (punteggio, coperta).

    "Non verificabile" viene escluso dalla media. Le righe con
    esito_incoerente=True vengono escluse."""
    total_weight = 0.0
    weighted_sum = 0.0

    for row in rows:
        if is_incoherent(row):
            continue

        score = ESITO_SCORE.get(row.get("esito"))
        if score is None:
            continue

        weight = verified_weight if is_verified(row) else unverified_weight
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return None, False

    return weighted_sum / total_weight, True


# ============================================================
# AGGREGAZIONE CLAIM
# ============================================================

def aggregate_claim(questions: dict, support_threshold: float, coverage_threshold: float):
    """Aggrega le domande di un claim (gia' ridotte a punteggio+centralita').

    questions: assertion_index -> (question_score, centrality, covered)"""
    total_centrality = 0.0
    covered_centrality = 0.0
    weighted_sum = 0.0

    for question_score, centrality, covered in questions.values():
        c = centrality if centrality else DEFAULT_CENTRALITY
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = DEFAULT_CENTRALITY
        c = max(1.0, min(5.0, c))

        total_centrality += c
        if covered:
            covered_centrality += c
            weighted_sum += question_score * c

    coverage = covered_centrality / total_centrality if total_centrality else 0.0

    if coverage < coverage_threshold or covered_centrality == 0:
        return NEI_LABEL, 0.0, coverage

    claim_score = weighted_sum / covered_centrality
    if claim_score >= support_threshold:
        label = "SUPPORTS"
    elif claim_score <= -support_threshold:
        label = "REFUTES"
    else:
        label = NEI_LABEL

    return label, claim_score, coverage


# ============================================================
# VALUTAZIONE DI UN CLAIM
# ============================================================

def evaluate_claim(rows: list, verified_weight: float, unverified_weight: float,
                    support_threshold: float, coverage_threshold: float) -> dict:
    """Dalle righe grezze di Fase 4 al verdetto finale del claim."""
    by_question = {}
    for row in rows:
        assertion_index = row.get("assertion_index")
        if assertion_index is None:
            continue
        by_question.setdefault(assertion_index, []).append(row)

    questions = {}
    for index, question_rows in by_question.items():
        score, covered = aggregate_question(question_rows, verified_weight, unverified_weight)
        centrality = question_rows[0].get("centrality")
        questions[index] = (score if covered else 0.0, centrality, covered)

    label, claim_score, coverage = aggregate_claim(questions, support_threshold, coverage_threshold)

    return {
        "predicted_label": label,
        "score": round(claim_score, 4),
        "coverage": round(coverage, 4),
        "n_domande": len(questions),
        "n_domande_coperte": sum(1 for _s, _c, covered in questions.values() if covered),
        "n_fonti_totali": len(rows),
    }


# ============================================================
# GOLD LABELS
# ============================================================

def load_gold_labels(path: str) -> dict:
    """Carica le label reali dal JSONL."""
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
    """Recupera il testo del claim da rollup.jsonl, quando disponibile."""
    rollup_path = os.path.join(matching_dir, "rollup.jsonl")
    texts = {}
    if not os.path.exists(rollup_path):
        return texts
    with open(rollup_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            claim_id = record.get("claim_id")
            if claim_id is not None:
                texts[str(claim_id)] = record.get("claim", "")
    return texts


# ============================================================
# METRICHE
# ============================================================

def compute_metrics(pairs: list, labels: tuple) -> dict:
    """Precision/recall/F1 per classe + macro-average + accuracy, su una lista
    di (predetto, reale). `labels` decide quali classi vengono riportate nel
    breakdown per-classe e nella macro-average — non filtra i `pairs` in sé:
    un predetto fuori da `labels` (es. NEI in un calcolo a 2 classi) conta
    comunque come falso negativo per la classe reale."""
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
        tp_v, fp_v, fn_v = tp[label], fp[label], fn[label]
        precision = tp_v / (tp_v + fp_v) if (tp_v + fp_v) else 0.0
        recall = tp_v / (tp_v + fn_v) if (tp_v + fn_v) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": tp_v + fn_v,
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


# ============================================================
# REPORT
# ============================================================

def print_metrics_block(title: str, metrics: dict, confusion: dict, labels: tuple,
                         confusion_pred_labels: tuple):
    """Stampa un blocco completo (matrice di confusione + metriche per classe +
    macro-average + accuracy) per un insieme di label dato."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    print("\n--- Matrice di confusione (righe = reale, colonne = predetto) ---")
    header = "reale\\predetto".ljust(22) + "".join(l[:14].ljust(18) for l in confusion_pred_labels)
    print(header)
    for gold in labels:
        row = gold.ljust(22)
        for predicted in confusion_pred_labels:
            row += str(confusion.get((gold, predicted), 0)).ljust(18)
        print(row)

    print("\n--- Metriche per classe ---")
    print(f"{'classe':<22}{'precision':<12}{'recall':<12}{'f1':<12}{'support':<10}")
    for label in labels:
        m = metrics["per_class"][label]
        print(f"{label:<22}{m['precision']:<12.3f}{m['recall']:<12.3f}{m['f1']:<12.3f}{m['support']:<10}")

    print(f"\naccuracy:          {metrics['accuracy']:.3f}  (n={metrics['n']})")
    print(f"macro precision:   {metrics['macro_precision']:.3f}")
    print(f"macro recall:      {metrics['macro_recall']:.3f}")
    print(f"macro F1:          {metrics['macro_f1']:.3f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matching-dir", default="matching_results",
                         help="Cartella con i file <claim_id>.jsonl prodotti dalla Fase 4")
    parser.add_argument("--labels", required=True,
                         help="Path al file JSONL con le label reali (campo 'id' e 'label': "
                              "SUPPORTS/REFUTES/NOT ENOUGH INFO)")
    parser.add_argument("--output", default=None,
                         help="Path del file di confronto per-claim (default: <matching-dir>/verdicts.jsonl)")
    parser.add_argument("--support-threshold", type=float, default=DEFAULT_SUPPORT_THRESHOLD,
                         help="Soglia sul punteggio del claim per SUPPORTS/REFUTES (default: 0.4)")
    parser.add_argument("--coverage-threshold", type=float, default=DEFAULT_COVERAGE_THRESHOLD,
                         help="Quota minima di peso-centralita' coperta da evidenza per esprimere "
                              "un verdetto (default: 1/3)")
    parser.add_argument("--verified-weight", type=float, default=DEFAULT_VERIFIED_WEIGHT,
                         help="Peso di una fonte con evidenza verificata (default: 1.0)")
    parser.add_argument("--unverified-weight", type=float, default=DEFAULT_UNVERIFIED_WEIGHT,
                         help="Peso di una fonte con evidenza non verificata (default: 0.5)")
    args = parser.parse_args()

    gold_labels = load_gold_labels(args.labels)
    print(f"[INFO] {len(gold_labels)} label reali caricate da {args.labels}")

    claim_texts = load_claim_texts(args.matching_dir)

    result_files = sorted(
        p for p in glob.glob(os.path.join(args.matching_dir, "*.jsonl"))
        if os.path.basename(p) not in {"rollup.jsonl", "verdicts.jsonl"}
    )
    if not result_files:
        print(f"[ERRORE] nessun file di risultati trovato in {args.matching_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(args.matching_dir, "verdicts.jsonl")

    pairs = []
    confusion = Counter()
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
            gold = gold_labels.get(claim_id)

            record = {
                "claim_id": claim_id,
                "claim": claim_texts.get(claim_id, ""),
                **verdict,
                "gold_label": gold,
                "correct": (verdict["predicted_label"] == gold) if gold else None,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if gold is None:
                n_no_gold += 1
                continue

            predicted = verdict["predicted_label"]
            pairs.append((predicted, gold))
            confusion[(gold, predicted)] += 1

    print(f"[INFO] {len(result_files)} claim processati, {len(pairs)} con label reale disponibile"
          + (f" ({n_no_gold} senza label, esclusi dalle metriche)" if n_no_gold else ""))
    print(f"[INFO] verdetti per-claim scritti in {output_path}")

    if not pairs:
        print("[ERRORE] nessun claim con label reale trovata: impossibile calcolare le metriche", file=sys.stderr)
        sys.exit(1)

    # --- blocco 1: tutti i claim, 3 classi ---
    metrics_all = compute_metrics(pairs, labels=VALID_LABELS)
    print_metrics_block("TUTTI I CLAIM (SUPPORTS / REFUTES / NOT ENOUGH INFO)",
                         metrics_all, confusion, VALID_LABELS, VALID_LABELS)

    # --- blocco 2: solo gold != NEI, 2 classi (il predetto puo' comunque
    # essere NEI: conta come errore, e la matrice di confusione lo mostra
    # come colonna anche se non fa parte delle 2 classi "ufficiali") ---
    non_nei_pairs = [(p, g) for p, g in pairs if g != NEI_LABEL]
    if non_nei_pairs:
        metrics_non_nei = compute_metrics(non_nei_pairs, labels=NON_NEI_LABELS)
        non_nei_confusion = Counter({k: v for k, v in confusion.items() if k[0] != NEI_LABEL})
        print_metrics_block("SOLO GOLD SUPPORTS/REFUTES (esclude i claim gold=NOT ENOUGH INFO)",
                             metrics_non_nei, non_nei_confusion, NON_NEI_LABELS, VALID_LABELS)
    else:
        print("\n[INFO] Nessun claim con gold SUPPORTS/REFUTES: secondo blocco di metriche saltato.")


if __name__ == "__main__":
    main()