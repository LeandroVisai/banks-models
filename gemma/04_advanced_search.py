#!/usr/bin/env python3
"""
BÚSQUEDA AVANZADA — RAG con filtros granulares (wrapper de 04_search.py).

Extiende 04_search.py con filtros más precisos:
  - Mes (enero-diciembre) — post-filtrado
  - Institución — filtro SQL
  - Entidades mencionadas (entity filtering) — filtro SQL
  - Tags (DECISION_POLITICA, FORWARD_LOOKING, DATOS_NUMERICOS, VARIABLE_CRITICA) — filtro SQL
  - Importance score (umbral mínimo/máximo) — filtro SQL
  - Section confidence (umbral mínimo) — filtro SQL
  - Exclusión de doc_types, instituciones, boilerplate — filtro SQL
  - Resultado en JSON con metadatos completos

Uso:
  python3 04_advanced_search.py "tasa de interés" \
    --year 2022 --month "enero,febrero" \
    --institution "BANCO_CENTRAL_CHILE" \
    --min-importance 0.7 \
    --exclude-boilerplate \
    --tags "DECISION_POLITICA,VARIABLE_CRITICA" \
    -k 10 --json

  python3 04_advanced_search.py --config my_filters.json "política monetaria"

Archivo de configuración JSON (my_filters.json):
  {
    "year": [2022, 2023],
    "month": [1, 2, 3],
    "institution": ["BANCO_CENTRAL_CHILE"],
    "entities": ["BANCO_CENTRAL_CHILE", "FEDERAL_RESERVE"],
    "tags": ["DECISION_POLITICA"],
    "min_importance": 0.7,
    "min_section_confidence": 0.6,
    "exclude_boilerplate": true,
    "exclude_doc_types": ["MONITOR_PM"],
    "exclude_institutions": []
  }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Imports internos (de 04_search.py)
sys.path.insert(0, str(Path(__file__).parent))
import psycopg2
from psycopg2.extras import RealDictCursor

DB_NAME = os.getenv("PGDATABASE", "rag_banco")
TABLE_PREFIX = os.getenv("RAG_TABLE_PREFIX", "")

def _table_name(base: str) -> str:
    name = f"{TABLE_PREFIX}{base}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Nombre de tabla inválido: {name!r}")
    return name

DOCUMENTS_TABLE = _table_name("documents")
CHUNKS_TABLE = _table_name("chunks")

# ---------------------------------------------------------------------------
# Estructuras de filtros avanzados
# ---------------------------------------------------------------------------

class AdvancedFilters:
    """Contenedor de filtros avanzados para búsqueda post-SQL."""
    def __init__(self):
        self.years: list[int] = []
        self.months: list[int] = []
        self.institutions: list[str] = []
        self.entities: list[str] = []
        self.tags: list[str] = []
        self.doc_types: list[str] = []
        self.min_importance: float = 0.0
        self.max_importance: float = 1.0
        self.min_section_confidence: float = 0.0
        self.exclude_boilerplate: bool = False
        self.exclude_doc_types: list[str] = []
        self.exclude_institutions: list[str] = []

    @classmethod
    def from_dict(cls, data: dict) -> AdvancedFilters:
        """Carga filtros desde dict (JSON)."""
        f = cls()
        if "year" in data:
            y = data["year"]
            f.years = [y] if isinstance(y, int) else (y or [])
        if "month" in data:
            m = data["month"]
            f.months = [m] if isinstance(m, int) else (m or [])
        if "institution" in data:
            i = data["institution"]
            f.institutions = [i] if isinstance(i, str) else (i or [])
        if "entities" in data:
            e = data["entities"]
            f.entities = [e] if isinstance(e, str) else (e or [])
        if "tags" in data:
            t = data["tags"]
            f.tags = [t] if isinstance(t, str) else (t or [])
        if "doc_types" in data:
            d = data["doc_types"]
            f.doc_types = [d] if isinstance(d, str) else (d or [])
        f.min_importance = float(data.get("min_importance", 0.0))
        f.max_importance = float(data.get("max_importance", 1.0))
        f.min_section_confidence = float(data.get("min_section_confidence", 0.0))
        f.exclude_boilerplate = bool(data.get("exclude_boilerplate", False))
        ex_dt = data.get("exclude_doc_types", [])
        f.exclude_doc_types = [ex_dt] if isinstance(ex_dt, str) else (ex_dt or [])
        ex_inst = data.get("exclude_institutions", [])
        f.exclude_institutions = [ex_inst] if isinstance(ex_inst, str) else (ex_inst or [])
        return f

    def to_dict(self) -> dict:
        """Exporta filtros a dict."""
        return {
            "year": self.years,
            "month": self.months,
            "institution": self.institutions,
            "entities": self.entities,
            "tags": self.tags,
            "doc_types": self.doc_types,
            "min_importance": self.min_importance,
            "max_importance": self.max_importance,
            "min_section_confidence": self.min_section_confidence,
            "exclude_boilerplate": self.exclude_boilerplate,
            "exclude_doc_types": self.exclude_doc_types,
            "exclude_institutions": self.exclude_institutions,
        }

    def active_filter_count(self) -> int:
        """Cuenta cuántos filtros están activos."""
        d = self.to_dict()
        return sum(1 for v in d.values() if v and v != 1.0 and v != 0.0)

# ---------------------------------------------------------------------------
# Construcción de WHERE clause con filtros avanzados
# ---------------------------------------------------------------------------

def build_advanced_where_clause(filters: AdvancedFilters) -> tuple[str, list]:
    """
    Construye cláusula WHERE SQL con filtros avanzados.
    Retorna (sql_where_fragment, params_list).
    """
    conditions: list[str] = []
    params: list = []

    # Año
    if filters.years:
        placeholders = ",".join(["%s"] * len(filters.years))
        conditions.append(
            f"COALESCE(EXTRACT(YEAR FROM {CHUNKS_TABLE}.chunk_date)::int, "
            f"{DOCUMENTS_TABLE}.document_year) IN ({placeholders})"
        )
        params.extend(filters.years)

    # Institución (incluir)
    if filters.institutions:
        placeholders = ",".join(["%s"] * len(filters.institutions))
        conditions.append(f"{DOCUMENTS_TABLE}.institution IN ({placeholders})")
        params.extend(filters.institutions)

    # Institución (excluir)
    if filters.exclude_institutions:
        placeholders = ",".join(["%s"] * len(filters.exclude_institutions))
        conditions.append(f"{DOCUMENTS_TABLE}.institution NOT IN ({placeholders})")
        params.extend(filters.exclude_institutions)

    # Tipos de documento (incluir)
    if filters.doc_types:
        placeholders = ",".join(["%s"] * len(filters.doc_types))
        conditions.append(f"{DOCUMENTS_TABLE}.doc_type_category IN ({placeholders})")
        params.extend(filters.doc_types)

    # Tipos de documento (excluir)
    if filters.exclude_doc_types:
        placeholders = ",".join(["%s"] * len(filters.exclude_doc_types))
        conditions.append(f"{DOCUMENTS_TABLE}.doc_type_category NOT IN ({placeholders})")
        params.extend(filters.exclude_doc_types)

    # Entidades (ANY overlap)
    if filters.entities:
        placeholders = ",".join(["%s"] * len(filters.entities))
        # Usamos JSON contains: entities JSONB has keys matching entities list
        for entity in filters.entities:
            conditions.append(f"{CHUNKS_TABLE}.entities ? %s")
            params.append(entity)

    # Tags (ANY overlap con TEXT[] array)
    if filters.tags:
        placeholders = ",".join(["%s"] * len(filters.tags))
        conditions.append(f"{CHUNKS_TABLE}.tags && ARRAY[{placeholders}]")
        params.extend(filters.tags)

    # Importance score
    conditions.append(f"{CHUNKS_TABLE}.importance_score BETWEEN %s AND %s")
    params.extend([filters.min_importance, filters.max_importance])

    # Section confidence
    if filters.min_section_confidence > 0:
        conditions.append(f"{CHUNKS_TABLE}.section_confidence >= %s")
        params.append(filters.min_section_confidence)

    # Excluir boilerplate (importance_score = 0.0)
    if filters.exclude_boilerplate:
        conditions.append(f"{CHUNKS_TABLE}.importance_score > %s")
        params.append(0.0)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    return where_clause, params


def apply_month_filter(results: list[dict], months: list[int]) -> list[dict]:
    """Filtra resultados post-SQL por mes."""
    if not months:
        return results

    filtered = []
    for row in results:
        chunk_date = row.get("chunk_date")
        document_date = row.get("document_date")

        month = None
        if chunk_date:
            if hasattr(chunk_date, "month"):
                month = chunk_date.month
            elif isinstance(chunk_date, str) and len(chunk_date) >= 7:
                try:
                    month = int(chunk_date[5:7])
                except (ValueError, TypeError):
                    pass
        
        if month is None and document_date:
            if isinstance(document_date, str):
                if re.match(r"^\d{4}-\d{2}-\d{2}$", document_date):
                    try:
                        month = int(document_date[5:7])
                    except (ValueError, TypeError):
                        pass

        if month and month in months:
            filtered.append(row)

    return filtered


def apply_importance_and_confidence_filters(
    results: list[dict],
    min_importance: float,
    max_importance: float,
    min_section_confidence: float,
) -> list[dict]:
    """Filtra resultados por importance_score y section_confidence post-SQL."""
    filtered = []
    for row in results:
        imp = float(row.get("importance_score", 0))
        conf = float(row.get("section_confidence", 0))

        if min_importance <= imp <= max_importance:
            if conf >= min_section_confidence:
                filtered.append(row)

    return filtered


# ---------------------------------------------------------------------------
# Búsqueda SQL directa (fallback si 04_search.py no se puede usar)
# ---------------------------------------------------------------------------

def direct_sql_search(
    query_text: str,
    filters: AdvancedFilters,
    k: int = 5,
) -> list[dict]:
    """
    Búsqueda directa por full-text (sin embeddings).
    Fallback si 04_search.py no está disponible o embeddings fallan.
    """
    where_clause, params = build_advanced_where_clause(filters)

    sql = f"""
    SELECT
        {CHUNKS_TABLE}.chunk_id,
        {CHUNKS_TABLE}.document_id,
        {DOCUMENTS_TABLE}.doc_type_category,
        {DOCUMENTS_TABLE}.filename,
        {DOCUMENTS_TABLE}.institution,
        {CHUNKS_TABLE}.text,
        {CHUNKS_TABLE}.section_type,
        {CHUNKS_TABLE}.section_confidence,
        {CHUNKS_TABLE}.importance_score,
        {CHUNKS_TABLE}.economic_variables,
        {CHUNKS_TABLE}.numeric_values,
        {CHUNKS_TABLE}.tags,
        {CHUNKS_TABLE}.entities,
        {CHUNKS_TABLE}.chunk_date,
        {DOCUMENTS_TABLE}.document_date,
        {DOCUMENTS_TABLE}.total_pages,
        {CHUNKS_TABLE}.page_start,
        {CHUNKS_TABLE}.page_end,
        ts_rank_cd({CHUNKS_TABLE}.text_tsv, query) AS text_score
    FROM {CHUNKS_TABLE}
    JOIN {DOCUMENTS_TABLE} USING (document_id),
         plainto_tsquery('spanish', %s) query
    WHERE {where_clause}
      AND {CHUNKS_TABLE}.text_tsv @@ query
    ORDER BY text_score DESC
    LIMIT %s
    """

    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", 5432)),
            user=os.getenv("PGUSER", os.getenv("USER", "postgres")),
            password=os.getenv("PGPASSWORD", "postgres"),
            database=DB_NAME,
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, [query_text] + params + [k])
            results = cur.fetchall()
        conn.close()
        return results
    except Exception as e:
        print(f"✗ Error en búsqueda SQL: {e}")
        return []


# ---------------------------------------------------------------------------
# Wrapper que delegaDa a 04_search.py pero aplica filtros post-SQL
# ---------------------------------------------------------------------------

def advanced_search(
    query: str,
    filters: AdvancedFilters,
    k: int = 5,
    use_mmr: bool = True,
    output_json: bool = False,
) -> list[dict]:
    """
    Ejecuta búsqueda intentando usar 04_search.py, fallback a SQL directo.
    """
    try:
        from search_wrapper import search_with_advanced_filters
        return search_with_advanced_filters(query, filters, k=k, use_mmr=use_mmr)
    except (ImportError, Exception):
        # Fallback a búsqueda SQL directa
        print("⚠  Usando búsqueda SQL directa (sin embeddings)")
        return direct_sql_search(query, filters, k=k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_months(month_str: str) -> list[int]:
    """Parsea 'enero,febrero' o '1,2,3' a lista de números 1-12."""
    month_map = {
        "enero": 1, "ene": 1,
        "febrero": 2, "feb": 2,
        "marzo": 3, "mar": 3,
        "abril": 4, "abr": 4,
        "mayo": 5,
        "junio": 6, "jun": 6,
        "julio": 7, "jul": 7,
        "agosto": 8, "ago": 8,
        "septiembre": 9, "sep": 9, "sept": 9,
        "octubre": 10, "oct": 10,
        "noviembre": 11, "nov": 11,
        "diciembre": 12, "dic": 12,
    }

    parts = month_str.replace(" ", ",").split(",")
    result = []
    for part in parts:
        part = part.strip().lower()
        if part.isdigit():
            m = int(part)
            if 1 <= m <= 12:
                result.append(m)
        elif part in month_map:
            result.append(month_map[part])
    return result


def format_result_table(results: list[dict]) -> str:
    """Formatea resultados en tabla amigable."""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"\n[{i}] {r.get('filename', 'unknown')}")
        lines.append(f"    Sección: {r.get('section_type', '?')} (conf: {r.get('section_confidence', 0):.2f})")
        lines.append(f"    Importancia: {r.get('importance_score', 0):.3f} | Score: {r.get('text_score', 0):.3f}")
        lines.append(f"    Institución: {r.get('institution', '?')} | Tipo: {r.get('doc_type_category', '?')}")
        text = r.get("text", "")[:150]
        lines.append(f"    Texto: {text}...")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda RAG avanzada con filtros granulares",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Búsqueda con mes y institución
  python3 04_advanced_search.py "política monetaria" \\
    --year 2022 --month enero,febrero \\
    --institution BANCO_CENTRAL_CHILE -k 5

  # Excluir boilerplate y Monitor PM
  python3 04_advanced_search.py "inflación" \\
    --exclude-boilerplate --exclude-doc-types MONITOR_PM -k 10

  # Filtros desde archivo JSON
  python3 04_advanced_search.py --config filters.json "commodities"

  # Solo decisiones de política con variables críticas
  python3 04_advanced_search.py "tasa de interés" \\
    --tags DECISION_POLITICA,VARIABLE_CRITICA \\
    --min-importance 0.8 -k 3 --json
        """
    )

    parser.add_argument("query", nargs="?", help="Consulta de búsqueda")
    parser.add_argument("-k", "--top-k", type=int, default=5, dest="k",
                        help="Número de resultados (default: 5)")

    # Filtros de tiempo
    parser.add_argument("--year", type=str, dest="years",
                        help="Año(s): 2022 o 2022,2023")
    parser.add_argument("--month", type=str, dest="months",
                        help="Mes(es): enero o 1,2,3 o enero,febrero")

    # Filtros de entidad/institución
    parser.add_argument("--institution", type=str, dest="institutions",
                        help="Institución: BANCO_CENTRAL_CHILE, FEDERAL_RESERVE, ...")
    parser.add_argument("--entity", type=str, dest="entities",
                        help="Entidades mencionadas (separadas por coma)")
    parser.add_argument("--doc-type", type=str, dest="doc_types",
                        help="Tipos de documento (COMUNICADO, MINUTA, FED_STATEMENT, ...)")

    # Filtros de contenido
    parser.add_argument("--tags", type=str,
                        help="Tags (DECISION_POLITICA, FORWARD_LOOKING, DATOS_NUMERICOS, VARIABLE_CRITICA)")
    parser.add_argument("--min-importance", type=float, default=0.0,
                        help="Umbral mínimo de importance_score (0-1)")
    parser.add_argument("--max-importance", type=float, default=1.0,
                        help="Umbral máximo de importance_score (0-1)")
    parser.add_argument("--min-section-confidence", type=float, default=0.0,
                        help="Umbral mínimo de section_confidence (0-1)")

    # Exclusiones
    parser.add_argument("--exclude-boilerplate", action="store_true",
                        help="Excluir boilerplate/legal/copyright")
    parser.add_argument("--exclude-doc-types", type=str, dest="exclude_doc_types",
                        help="Excluir tipos de documento")
    parser.add_argument("--exclude-institutions", type=str, dest="exclude_institutions",
                        help="Excluir instituciones")

    # Configuración
    parser.add_argument("--config", type=Path,
                        help="Cargar filtros desde archivo JSON")
    parser.add_argument("--save-config", type=Path,
                        help="Guardar filtros a archivo JSON")

    # Salida
    parser.add_argument("--json", action="store_true", dest="output_json",
                        help="Salida en JSON")
    parser.add_argument("--no-mmr", action="store_false", dest="use_mmr",
                        help="Desactivar MMR reranking")

    args = parser.parse_args()

    # Cargar configuración si se especifica
    filters = AdvancedFilters()
    if args.config:
        try:
            with open(args.config) as f:
                config_data = json.load(f)
            filters = AdvancedFilters.from_dict(config_data)
            print(f"✓ Configuración cargada desde {args.config}")
        except Exception as e:
            print(f"✗ Error al cargar config: {e}")
            sys.exit(1)

    # Aplicar argumentos de línea de comando (sobrescriben config)
    if args.years:
        filters.years = [int(y) for y in args.years.replace(",", " ").split()]
    if args.months:
        filters.months = parse_months(args.months)
    if args.institutions:
        filters.institutions = [i.strip() for i in args.institutions.split(",")]
    if args.entities:
        filters.entities = [e.strip() for e in args.entities.split(",")]
    if args.doc_types:
        filters.doc_types = [d.strip() for d in args.doc_types.split(",")]
    if args.tags:
        filters.tags = [t.strip() for t in args.tags.split(",")]
    if args.exclude_doc_types:
        filters.exclude_doc_types = [d.strip() for d in args.exclude_doc_types.split(",")]
    if args.exclude_institutions:
        filters.exclude_institutions = [i.strip() for i in args.exclude_institutions.split(",")]

    filters.min_importance = args.min_importance
    filters.max_importance = args.max_importance
    filters.min_section_confidence = args.min_section_confidence
    filters.exclude_boilerplate = args.exclude_boilerplate

    # Guardar config si se solicita
    if args.save_config:
        try:
            with open(args.save_config, "w") as f:
                json.dump(filters.to_dict(), f, indent=2)
            print(f"✓ Configuración guardada en {args.save_config}")
        except Exception as e:
            print(f"✗ Error al guardar config: {e}")
            sys.exit(1)

    # Necesitamos query
    if not args.query:
        parser.print_help()
        sys.exit(1)

    # Ejecutar búsqueda
    print(f"\n🔍 Búsqueda: {args.query!r}")
    num_filters = filters.active_filter_count()
    print(f"   k={args.k}, filtros activos={num_filters}")

    results = advanced_search(
        args.query,
        filters,
        k=args.k,
        use_mmr=args.use_mmr,
        output_json=args.output_json,
    )

    if args.output_json:
        print(json.dumps([dict(r) for r in results], indent=2, ensure_ascii=False, default=str))
    else:
        if results:
            print(format_result_table(results))
        else:
            print("❌ Sin resultados.")


if __name__ == "__main__":
    main()
