#!/usr/bin/env python3
"""
PASO 1 — Enriquecimiento de metadata sobre chunks crudos.

Entrada:
  logs/documents.json    (del paso 0)
  logs/chunks.json       (del paso 0)

Salida:
  logs/chunks_enriched.json

Para cada chunk computa:
  - section_type (+ confianza)     → vía patterns + señales de posición en doc
  - economic_variables             → dict {VAR: {importance, mentions, confidence}}
  - numeric_values                 → lista [{value, unit, raw}]
  - entities                       → dict {BANCO_X: mentions, PAIS_X: mentions}
  - temporal_refs                  → lista [{type, value}]  (años, trimestres)
  - tags                           → lista (DECISION_POLITICA, FORWARD_LOOKING, ...)
  - importance_score               → 0.0–1.0, calibrado
  - is_policy_decision, is_forward_looking → booleanos derivados

Diseño frente al pipeline previo:
  1. doc_type_category e institution se HEREDAN del documento, no se re-detectan.
  2. Patterns son case/accent-insensitive (normalize_text) y con word-boundaries.
  3. La sección se elige por scoring ponderado, no por "primer match".
  4. Importance_score es una combinación explícita y calibrada (ver calculate_importance).
  5. Sin modo 'improve' legacy: un único pipeline completo.

Uso:
  python3 01_enrich_metadata.py
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from models import EnrichedChunk
from taxonomy import (
    BOILERPLATE_PATTERN,
    CRITICAL_VARIABLES,
    ECONOMIC_VARIABLES,
    FORWARD_LOOKING_PATTERN,
    MONITOR_PM_SECTION_MAP,
    MONITOR_PM_SECTIONS,
    NUMERIC_PATTERNS,
    build_entity_patterns,
    build_section_patterns,
    build_variable_patterns,
    derive_tags,
    normalize_text,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

INPUT_CHUNKS = Path("logs/chunks.json")
INPUT_DOCUMENTS = Path("logs/documents.json")
OUTPUT_CHUNKS = Path("logs/chunks_enriched.json")

# Pesos del importance_score. La suma puede exceder 1.0; el resultado se
# capa a 1.0. La calibración objetivo (tras validar sobre los PDFs de prueba):
#   - chunk narrativo sin variables ni datos → ~0.0
#   - chunk con 1 variable HIGH + datos      → ~0.4
#   - chunk con variable CRÍTICA + datos     → ~0.6
#   - chunk DECISION de política con datos   → ~0.9+
IMPORTANCE_WEIGHTS = {
    "critical_var": 0.35,       # tiene al menos una CRITICAL
    "many_critical": 0.10,      # tiene 2+ CRITICAL
    "high_var_only": 0.20,      # HIGH pero sin CRITICAL
    "medium_var_only": 0.08,    # MEDIUM pero sin HIGH/CRITICAL
    "numerics": 0.20,           # saturable en 3
    "policy_section": 0.25,     # DECISION o VOTACION
    "outlook_section": 0.15,    # PROYECCION o RIESGOS
    "analysis_section": 0.08,   # ANALISIS
    "monitor_pm_section": 0.12, # sección de Monitor PM (contenido curado de mercado)
    "forward_looking": 0.08,
    "entities": 0.10,           # mención de banco central / país (sat. en 2)
    "no_variables_penalty": 0.12,   # sin variables ni datos → ruido semántico
    "boilerplate_penalty": 0.50,    # texto legal/disclaimer → penalización fuerte
}

# Compilamos patrones una vez (al import)
VARIABLE_PATTERNS = build_variable_patterns()
SECTION_PATTERNS = build_section_patterns()
ENTITY_PATTERNS = build_entity_patterns()

# Patrones temporales
YEAR_PATTERN = re.compile(r"\b(19[8-9]\d|20[0-4]\d)\b")
QUARTER_PATTERN = re.compile(r"\b(q[1-4]|[1-4][ºo]?\s*trimestre)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Detectores
# ---------------------------------------------------------------------------

def detect_variables(text_norm: str) -> dict:
    """Detecta variables económicas con conteo de menciones y confianza."""
    found: dict[str, dict] = {}
    for var_name, pattern in VARIABLE_PATTERNS.items():
        matches = pattern.findall(text_norm)
        if not matches:
            continue
        mentions = len(matches)
        _, importance = ECONOMIC_VARIABLES[var_name]
        # Confidence: 0.5 base + 0.1 por mención adicional, cap 1.0
        confidence = min(1.0, 0.5 + (mentions - 1) * 0.1)
        found[var_name] = {
            "importance": importance,
            "mentions": mentions,
            "confidence": round(confidence, 2),
        }
    return found


def detect_section(
    text_norm: str,
    position_in_doc: int,
    total_chunks_in_doc: int,
    section_hint: Optional[str],
) -> tuple[str, float]:
    """
    Determina la sección canónica del chunk.

    Estrategia:
      1. Si section_hint del paso 0 matchea una sección canónica, úsalo con
         alta confianza.
      2. Si no, scoring por menciones de patterns; gana la de mayor score.
      3. Bias posicional suave: primer chunk tiende a ENCABEZADO, último a
         PROYECCION/RIESGOS.
    """
    # 1. hint directo (títulos detectados durante chunking)
    if section_hint:
        hint_norm = normalize_text(section_hint)
        for section in SECTION_PATTERNS:
            if section.lower() in hint_norm:
                return section, 0.95

    # 2. scoring por patterns (acepta matches débiles — confianza refleja la certeza)
    scores: dict[str, int] = {}
    for section, pattern in SECTION_PATTERNS.items():
        matches = pattern.findall(text_norm)
        if matches:
            scores[section] = len(matches)

    if scores:
        best = max(scores, key=scores.get)
        total = sum(scores.values())
        # Confianza más baja para matches únicos, más alta para matches múltiples
        confidence = round(min(1.0, 0.40 + scores[best] / (total + 1)), 2)
        return best, confidence

    # 3. bias posicional — granular según posición relativa en documento
    if total_chunks_in_doc > 0:
        rel_pos = position_in_doc / max(1, total_chunks_in_doc - 1)
        if rel_pos < 0.05:
            return "ENCABEZADO", 0.40
        if rel_pos < 0.20:
            return "CONTEXTO_EXTERNO", 0.30
        if rel_pos < 0.35:
            return "ANALISIS", 0.28
        if rel_pos > 0.85:
            return "RIESGOS", 0.28
        if rel_pos > 0.70:
            return "PROYECCION", 0.30

    return "CONTENIDO", 0.25


def detect_entities(text_norm: str) -> dict:
    """Cuenta menciones de cada entidad conocida."""
    found: dict[str, int] = {}
    for name, pattern in ENTITY_PATTERNS.items():
        matches = pattern.findall(text_norm)
        if matches:
            found[name] = len(matches)
    return found


def extract_numerics(text: str) -> list[dict]:
    """Extrae valores numéricos (sobre el texto original, sin normalizar,
    para preservar formato como '5,25%')."""
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for pattern, unit in NUMERIC_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(1)
            raw = m.group(0)
            key = (value, unit)
            if key in seen:
                continue
            seen.add(key)
            results.append({"value": value, "unit": unit, "raw": raw})
            if len(results) >= 10:  # cap
                return results
    return results


def extract_temporal(text_norm: str) -> dict:
    years = sorted(set(YEAR_PATTERN.findall(text_norm)))
    quarters = sorted(set(m.group(0) for m in QUARTER_PATTERN.finditer(text_norm)))
    out: dict = {}
    if years:
        out["years"] = years
    if quarters:
        out["quarters"] = quarters
    return out


def calculate_importance(
    variables: dict,
    numerics: list,
    section_type: str,
    entities: dict,
    is_fwd: bool,
    text_norm: str = "",
) -> float:
    # Boilerplate legal: penalización máxima antes de cualquier otro cálculo
    if text_norm and BOILERPLATE_PATTERN.search(text_norm):
        return 0.0

    levels = {d["importance"] for d in variables.values()}
    critical_count = sum(1 for d in variables.values() if d["importance"] == "CRITICAL")

    score = 0.0
    w = IMPORTANCE_WEIGHTS

    # Variables — escalones mutuamente excluyentes según nivel más alto presente.
    if "CRITICAL" in levels:
        score += w["critical_var"]
        if critical_count >= 2:
            score += w["many_critical"]
    elif "HIGH" in levels:
        score += w["high_var_only"]
    elif "MEDIUM" in levels:
        score += w["medium_var_only"]

    # Datos cuantitativos: satura en 3 menciones.
    score += w["numerics"] * min(1.0, len(numerics) / 3)

    # Sección — escalones por tipo.
    if section_type in ("DECISION", "VOTACION"):
        score += w["policy_section"]
    elif section_type in ("PROYECCION", "RIESGOS"):
        score += w["outlook_section"]
    elif section_type == "ANALISIS":
        score += w["analysis_section"]
    elif section_type in MONITOR_PM_SECTIONS:
        score += w["monitor_pm_section"]

    if is_fwd:
        score += w["forward_looking"]

    if entities:
        # satura en 2 entidades distintas
        score += w["entities"] * min(1.0, len(entities) / 2)

    # Penalizar chunks sin contenido económico: sin variables Y sin datos numéricos
    if not variables and not numerics:
        score -= w["no_variables_penalty"]

    return round(max(0.0, min(1.0, score)), 3)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def load_documents() -> dict[str, dict]:
    with open(INPUT_DOCUMENTS, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["document_id"]: d for d in docs}


def load_chunks() -> list[dict]:
    with open(INPUT_CHUNKS, "r", encoding="utf-8") as f:
        return json.load(f)


def _count_chunks_per_doc(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk["document_id"]
        counts[doc_id] = counts.get(doc_id, 0) + 1
    return counts


def _enrich_chunk(chunk: dict, doc: dict, total_chunks_in_doc: int) -> EnrichedChunk:
    text_norm = normalize_text(chunk["text"])
    doc_type = doc.get("doc_type_category", "")

    variables = detect_variables(text_norm)
    numerics = extract_numerics(chunk["text"])
    entities = detect_entities(text_norm)
    temporal = extract_temporal(text_norm)

    # Monitor PM: la sección se deriva directamente del nombre de columna
    section_hint = chunk.get("section_title_raw", "")
    if doc_type == "MONITOR_PM" and section_hint and section_hint.strip() in MONITOR_PM_SECTION_MAP:
        section_type = MONITOR_PM_SECTION_MAP[section_hint.strip()]
        section_conf = 1.0
    else:
        section_type, section_conf = detect_section(
            text_norm,
            chunk["position_in_doc"],
            total_chunks_in_doc,
            section_hint,
        )

    is_fwd = bool(FORWARD_LOOKING_PATTERN.search(text_norm))
    tags = derive_tags(text_norm, variables, numerics, section_type)
    importance = calculate_importance(variables, numerics, section_type, entities, is_fwd, text_norm)

    return EnrichedChunk(
        chunk_id=chunk["chunk_id"],
        document_id=chunk["document_id"],
        text=chunk["text"],
        char_count=chunk["char_count"],
        page_start=chunk["page_start"],
        page_end=chunk["page_end"],
        position_in_doc=chunk["position_in_doc"],
        doc_type_category=doc_type or "REPORTE_RESEARCH",
        institution=doc.get("institution", "unknown"),
        document_date=doc.get("document_date"),
        section_type=section_type,
        section_confidence=section_conf,
        economic_variables=variables,
        numeric_values=numerics,
        entities=entities,
        temporal_refs=temporal,
        tags=tags,
        importance_score=importance,
        is_policy_decision="DECISION_POLITICA" in tags,
        is_forward_looking=is_fwd,
        chunk_date=chunk.get("chunk_date"),
    )


def _print_enrichment_report(
    enriched: list[EnrichedChunk],
    section_hist: dict[str, int],
    var_hist: dict[str, int],
) -> None:
    print(f"[01] ✓ {OUTPUT_CHUNKS} ({len(enriched)} chunks enriquecidos)")
    print(f"[01] Secciones:")
    for section, count in sorted(section_hist.items(), key=lambda x: -x[1]):
        print(f"       {section:<20} {count}")
    print(f"[01] Variables económicas (top 10):")
    for var_name, count in sorted(var_hist.items(), key=lambda x: -x[1])[:10]:
        importance_level = ECONOMIC_VARIABLES[var_name][1]
        print(f"       {var_name:<30} {count:>4}  ({importance_level})")
    avg_importance = sum(e.importance_score for e in enriched) / max(1, len(enriched))
    high_importance_count = sum(1 for e in enriched if e.importance_score >= 0.6)
    print(f"[01] Importance score: media={avg_importance:.3f}, "
          f"chunks con score≥0.6: {high_importance_count} "
          f"({100 * high_importance_count // max(1, len(enriched))}%)")


def enrich_all() -> int:
    if not INPUT_CHUNKS.exists() or not INPUT_DOCUMENTS.exists():
        print("❌ Faltan archivos de entrada. Corre primero: python3 00_generate_jsons.py",
              file=sys.stderr)
        return 1

    docs = load_documents()
    chunks = load_chunks()
    chunks_per_doc = _count_chunks_per_doc(chunks)

    enriched: list[EnrichedChunk] = []
    section_hist: dict[str, int] = {}
    var_hist: dict[str, int] = {}

    print(f"[01] Enriqueciendo {len(chunks)} chunks de {len(docs)} documentos...")

    for chunk in chunks:
        doc = docs.get(chunk["document_id"], {})
        enriched_chunk = _enrich_chunk(chunk, doc, chunks_per_doc[chunk["document_id"]])
        enriched.append(enriched_chunk)

        section_hist[enriched_chunk.section_type] = section_hist.get(enriched_chunk.section_type, 0) + 1
        for var_name in enriched_chunk.economic_variables:
            var_hist[var_name] = var_hist.get(var_name, 0) + 1

    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in enriched], f, ensure_ascii=False, indent=2)

    _print_enrichment_report(enriched, section_hist, var_hist)
    return 0


if __name__ == "__main__":
    sys.exit(enrich_all())
