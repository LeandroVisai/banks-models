#!/usr/bin/env python3
"""
PASO 5 — Construcción de prompts RAG para LLMs locales.

Toma los resultados de búsqueda de 04_search.py y construye un prompt
estructurado listo para pasarle a Ollama, llama.cpp o cualquier LLM local.

El módulo NO realiza llamadas de red. Produce el prompt como string.

Funciones principales:
  format_rag_context(results, query) → str
      Formatea los fragmentos recuperados como contexto delimitado.

  build_rag_prompt(query, results, system_instruction=None) → tuple[str, list[dict]]
      Retorna (prompt_completo, citas_fuente).

Uso desde línea de comandos:
  python3 05_query.py "tasa de interés 2022" 5
  python3 05_query.py "política monetaria" 3 --json

Uso como librería:
  from 05_query import build_rag_prompt  # requiere importar con importlib
  # o directamente:
  from importlib import import_module
  q = import_module("05_query")
  prompt, citations = q.build_rag_prompt(query, results)
"""
from __future__ import annotations

import argparse
import json
import sys

# ---------------------------------------------------------------------------
# Instrucción de sistema por defecto — diseñada para el dominio financiero
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_INSTRUCTION = """\
Eres un analista experto en política monetaria y macroeconomía. \
Respondes en español con precisión técnica. \
SOLO usas información del contexto proporcionado. \
Si el contexto no contiene suficiente información para responder, \
di explícitamente "No tengo información suficiente en los documentos disponibles." \
No inventes datos, fechas ni cifras."""

# ---------------------------------------------------------------------------
# Few-shot examples — enseñan el formato de cita esperado
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "question": "¿Cuál fue la decisión de tasa del BCCh en julio 2022?",
        "context_summary": "[DOC-1] comunicado1.pdf p.1 · COMUNICADO · DECISION",
        "answer": (
            "El Banco Central de Chile acordó incrementar la Tasa de Política Monetaria "
            "en 75 puntos base, llevándola al 9,75% anual [1]. La decisión fue tomada por "
            "unanimidad del Consejo [1].\n\n"
            "Fuentes:\n[1] comunicado1.pdf, p.1 (Comunicado, Sección: DECISION)"
        ),
    },
    {
        "question": "¿Cuáles son las proyecciones de inflación para 2023?",
        "context_summary": "[DOC-1] minuta_enero2023.pdf p.4 · MINUTA · PROYECCION",
        "answer": (
            "Según las proyecciones del IPoM, se espera que la inflación converja a la meta "
            "del 3% durante el horizonte de política monetaria [1]. Los consejeros evaluaron "
            "que los riesgos inflacionarios se mantenían sesgados al alza a corto plazo [1].\n\n"
            "Fuentes:\n[1] minuta_enero2023.pdf, p.4 (Minuta, Sección: PROYECCION)"
        ),
    },
]


# ---------------------------------------------------------------------------
# Formateo de contexto
# ---------------------------------------------------------------------------

def _build_citation_header(result: dict, index: int) -> str:
    """Construye el encabezado de cada fragmento de contexto."""
    filename = result.get("filename", "desconocido")
    page_start = result.get("page_start", "?")
    page_end = result.get("page_end", page_start)
    doc_type = result.get("doc_type_category", "")
    section = result.get("section_type", "")
    date = result.get("document_date", "")

    page_ref = f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"
    date_part = f" · {date}" if date else ""

    return f"[{index}] {filename} {page_ref} · {doc_type} · {section}{date_part}"


def _build_variables_annotation(result: dict) -> str:
    """Devuelve una línea con las variables económicas detectadas, si las hay."""
    variables = result.get("economic_variables") or {}
    if not variables:
        return ""
    var_names = ", ".join(variables.keys())
    return f"    [variables: {var_names}]"


def format_rag_context(results: list[dict], query: str) -> str:
    """
    Formatea los fragmentos recuperados como bloque de contexto delimitado.

    Cada fragmento incluye:
      - encabezado con número de cita, archivo, páginas, tipo y sección
      - variables económicas detectadas (si las hay)
      - texto del fragmento

    El bloque completo va entre separadores <CONTEXTO> para que el LLM
    pueda identificarlo inequívocamente y no lo confunda con su respuesta.
    """
    if not results:
        return "<CONTEXTO>\n(Sin fragmentos recuperados)\n</CONTEXTO>"

    lines = ["<CONTEXTO>", f"Fragmentos recuperados para la consulta: «{query}»", ""]

    for index, result in enumerate(results, start=1):
        header = _build_citation_header(result, index)
        variables_line = _build_variables_annotation(result)
        text = result.get("text", "").replace("\n", " ").strip()

        lines.append(header)
        if variables_line:
            lines.append(variables_line)
        lines.append(f"    {text}")
        lines.append("")

    lines.append("</CONTEXTO>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construcción del prompt completo
# ---------------------------------------------------------------------------

def _build_few_shot_block() -> str:
    """Construye el bloque de ejemplos few-shot para el prompt."""
    lines = [
        "A continuación hay ejemplos del formato de respuesta esperado:",
        "",
    ]
    for example in FEW_SHOT_EXAMPLES:
        lines.append(f"Pregunta de ejemplo: {example['question']}")
        lines.append(f"Contexto disponible: {example['context_summary']}")
        lines.append(f"Respuesta de ejemplo:\n{example['answer']}")
        lines.append("")
    return "\n".join(lines)


def _extract_citations(results: list[dict]) -> list[dict]:
    """Extrae metadata de cita estructurada para cada resultado."""
    citations = []
    for index, result in enumerate(results, start=1):
        page_start = result.get("page_start")
        page_end = result.get("page_end", page_start)
        citations.append({
            "citation_number": index,
            "filename": result.get("filename", "desconocido"),
            "doc_type": result.get("doc_type_category", ""),
            "section": result.get("section_type", ""),
            "page_start": page_start,
            "page_end": page_end,
            "document_date": result.get("document_date"),
            "importance_score": result.get("importance_score"),
            "chunk_id": result.get("chunk_id"),
        })
    return citations


def build_rag_prompt(
    query: str,
    results: list[dict],
    system_instruction: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Construye el prompt RAG completo y las citas fuente.

    Parámetros:
      query             — consulta original del usuario
      results           — lista de chunks recuperados por 04_search.py
      system_instruction — instrucción de sistema personalizada (usa DEFAULT si None)

    Retorna:
      (prompt_str, citations)
        prompt_str  — string completo listo para enviar a un LLM
        citations   — lista de dicts con metadata de cada fuente citada
    """
    system = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
    context_block = format_rag_context(results, query)
    few_shot_block = _build_few_shot_block()
    citations = _extract_citations(results)

    prompt_parts = [
        "### INSTRUCCIÓN DE SISTEMA",
        system,
        "",
        "### EJEMPLOS DE FORMATO",
        few_shot_block,
        "### CONTEXTO RECUPERADO",
        context_block,
        "",
        "### INSTRUCCIONES DE RESPUESTA",
        "1. Responde ÚNICAMENTE basándote en los fragmentos del <CONTEXTO> anterior.",
        "2. Cita cada afirmación con el número de fragmento entre corchetes, ej. [1] o [2].",
        "3. Si varios fragmentos apoyan la misma afirmación, cítalos todos: [1][3].",
        "4. Al final de tu respuesta, incluye una sección 'Fuentes:' listando cada cita usada.",
        "5. Formato de cada entrada en 'Fuentes:': [N] nombre_archivo, p.X (Tipo, Sección).",
        "6. Si el contexto es insuficiente, responde: "
        "\"No tengo información suficiente en los documentos disponibles.\"",
        "",
        "### PREGUNTA",
        query,
        "",
        "### RESPUESTA",
    ]

    prompt = "\n".join(prompt_parts)
    return prompt, citations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Construye prompt RAG a partir de resultados de búsqueda"
    )
    ap.add_argument("query", help="Consulta en lenguaje natural")
    ap.add_argument("k", nargs="?", type=int, default=5, help="Número de resultados (default 5)")
    ap.add_argument("--no-mmr", action="store_true", help="Desactivar MMR en la búsqueda")
    ap.add_argument("--json", action="store_true", help="Salida JSON (prompt + citas)")
    ap.add_argument(
        "--system",
        default=None,
        help="Instrucción de sistema personalizada (usa DEFAULT si se omite)",
    )
    args = ap.parse_args()

    try:
        from search_module import search  # type: ignore
    except ImportError:
        # Importación directa del módulo con nombre numérico
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "search_module",
            "04_search.py",
        )
        if spec is None or spec.loader is None:
            print("❌ No se pudo importar 04_search.py", file=sys.stderr)
            return 1
        search_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(search_module)  # type: ignore[attr-defined]
        search = search_module.search

    results, parsed = search(args.query, k=args.k, use_mmr=not args.no_mmr)
    prompt, citations = build_rag_prompt(args.query, results, system_instruction=args.system)

    if args.json:
        output = {
            "query": args.query,
            "parsed_filters": {k: v for k, v in parsed.items() if k != "raw_query"},
            "prompt": prompt,
            "citations": citations,
            "result_count": len(results),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(prompt)
        print("\n" + "=" * 80)
        print("CITAS FUENTE:")
        for citation in citations:
            page_ref = (
                f"p.{citation['page_start']}"
                if citation["page_start"] == citation["page_end"]
                else f"pp.{citation['page_start']}-{citation['page_end']}"
            )
            date_part = f" ({citation['document_date']})" if citation["document_date"] else ""
            print(
                f"  [{citation['citation_number']}] {citation['filename']}, {page_ref}"
                f" · {citation['doc_type']} · {citation['section']}{date_part}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
