"""
Taxonomía compartida: variables económicas, secciones, entidades, patrones numéricos.

Usado por:
  01_enrich_metadata.py — para etiquetar chunks.
  04_search.py          — para parsear queries de usuario.

Los patterns son case/accent-insensitive vía `normalize_text`. Cada pattern
especifica palabras con word-boundaries para evitar falsos positivos
(ej. "tasa" aislada, no "tasador").
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Remueve tildes y diacríticos preservando la letra base."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def normalize_text(text: str) -> str:
    """Lowercase + sin tildes + whitespace colapsado. Para matching robusto."""
    return re.sub(r"\s+", " ", strip_accents(text.lower())).strip()


def compile_word_pattern(words: Iterable[str]) -> re.Pattern:
    """Compila un regex que matchea cualquiera de las palabras con word-boundaries.

    Las palabras de entrada se normalizan (lowercase+sin tildes) para matchear
    contra texto que pase por normalize_text() antes.
    """
    normed = [re.escape(normalize_text(w)) for w in words if w.strip()]
    # \b no funciona bien con caracteres no-ASCII; usamos lookarounds con
    # clase de caracteres de palabra extendida para español sin tildes (ya normalizado).
    return re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(normed) + r")(?![a-z0-9])",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Variables económicas (14) con niveles de importancia
# ---------------------------------------------------------------------------
# Formato: nombre -> (lista_keywords, importance)
#   importance: CRITICAL | HIGH | MEDIUM
#
# Keywords pueden ser palabras sueltas o frases; compilamos como alternación.
# Evitamos patterns que matcheen fragmentos (ej. "tasa" como substring de "tasador").

ECONOMIC_VARIABLES: dict[str, tuple[list[str], str]] = {
    "TASA_INTERES": ([
        "tasa de interes", "tasa de politica monetaria", "tpm",
        "interest rate", "policy rate", "federal funds rate",
        "fed funds", "tasa rectora", "tasa de referencia",
        "alza de tasa", "baja de tasa", "recorte de tasa",
        "incremento de tasa", "subida de tasa",
    ], "CRITICAL"),

    "INFLACION": ([
        "inflacion", "ipc", "indice de precios al consumidor",
        "inflation", "cpi", "headline inflation", "core inflation",
        "inflacion subyacente", "presiones inflacionarias",
        "meta de inflacion", "convergencia de la inflacion",
        "inflacion general", "inflacion total",
    ], "CRITICAL"),

    "PIB": ([
        "pib", "producto interno bruto", "producto interior bruto",
        "gdp", "gross domestic product", "actividad economica",
        "imacec", "crecimiento economico", "crecimiento del pib",
        "expansion economica", "contraccion economica",
        "dinamica del pib", "recesion", "desaceleracion economica",
        "recuperacion economica", "variacion del pib",
        "output gap", "brecha del producto",
    ], "CRITICAL"),

    "TIPO_CAMBIO": ([
        "tipo de cambio", "paridad cambiaria", "exchange rate",
        "fx", "usd/clp", "dolar", "depreciacion", "apreciacion",
        "fortalecimiento del dolar", "debilitamiento",
    ], "HIGH"),

    "EMPLEO": ([
        "empleo", "desempleo", "tasa de desempleo", "ocupados",
        "unemployment", "employment", "payrolls", "labor market",
        "mercado laboral", "mercado del trabajo", "ocupacion",
        "desocupacion", "fuerza de trabajo", "tasa de participacion",
        "creacion de empleos", "perdida de empleos",
    ], "HIGH"),

    "EXPECTATIVAS_INFLACIONARIAS": ([
        "expectativas de inflacion", "expectativas inflacionarias",
        "inflation expectations", "breakeven",
        "encuesta de expectativas", "eee", "eof",
        "expectativas de mediano plazo", "expectativas a dos anos",
        "seguros de inflacion", "ber", "break even inflacion",
        "breakeven de inflacion", "compensacion inflacionaria",
    ], "HIGH"),

    "PRECIO_COMMODITIES": ([
        "cobre", "petroleo", "oro", "litio", "copper", "oil", "gold",
        "brent", "wti", "commodity prices", "commodities",
        "materias primas", "precio del cobre", "precio del petroleo",
        "gas natural", "natural gas", "mineral",
    ], "HIGH"),

    "CREDITO": ([
        "credito", "prestamos", "credit", "loans", "bank lending",
        "colocaciones", "credito bancario", "acceso al credito",
        "condiciones crediticias", "financial conditions",
        "spread corporativo", "bonos corporativos", "emision corporativa",
        "spread sobre swap", "mercado de capitales", "emision de bonos",
        "cds", "credit default swap", "riesgo soberano", "riesgo pais",
        "spread cds", "bonos en dolares", "financiamiento externo",
        "costo financiamiento exterior", "bonos emitidos en el exterior",
    ], "MEDIUM"),

    "INVERSION": ([
        "inversion", "formacion bruta de capital", "investment",
        "capex", "capital expenditure", "inversion fija",
        "inversion privada", "inversion publica", "fbcf",
    ], "MEDIUM"),

    "BALANZA_COMERCIAL": ([
        "balanza comercial", "exportaciones", "importaciones",
        "trade balance", "exports", "imports", "cuenta corriente",
        "trade war", "aranceles", "tariff", "deficit comercial",
        "superavit comercial", "comercio exterior",
    ], "MEDIUM"),

    "TASAS_LARGO_PLAZO": ([
        "tasas a largo plazo", "bonos a 10 anos", "10-year yield",
        "long-term rates", "bond yield", "rendimiento soberano",
        "tasa a 10 anos", "spread soberano", "renta fija",
        "treasury yield",
        "swap", "tasa swap", "curva swap", "spc", "tasa spc",
        "btp", "bcu", "btpcu", "spread sobre swap", "spread sobre spc",
        "tir real", "curva de tasas", "bonos soberanos",
        "tesoro", "tasa tesoro", "treasury", "bono del tesoro",
        "tesoro estadounidense", "10 anos", "rendimiento a 10",
    ], "MEDIUM"),

    "CONSUMO": ([
        "consumo privado", "consumo de hogares", "gasto de los hogares",
        "consumer spending", "household consumption",
        "demanda interna", "gasto de consumo", "retail sales",
    ], "MEDIUM"),

    "VOLATILIDAD": ([
        "volatilidad", "volatility", "vix", "riesgo de mercado",
        "market risk", "incertidumbre", "uncertainty",
        "aversion al riesgo", "risk aversion", "turbulencia",
        "ipsa", "renta variable", "mercado accionario", "bolsa",
        "s&p 500", "nasdaq", "dow jones",
    ], "MEDIUM"),

    "LIQUIDEZ": ([
        "liquidez", "liquidity", "funding", "repo",
        "condiciones de liquidez", "mercado monetario",
        "costo de financiamiento", "financiamiento local",
        "spread bonos bancarios", "spread bancario",
        "montos transados", "profundidad de mercado",
        "baja profundidad", "escasos montos", "volumen transado",
    ], "MEDIUM"),
}

# Sets precomputados por nivel (para scoring rápido)
CRITICAL_VARIABLES = {k for k, (_, imp) in ECONOMIC_VARIABLES.items() if imp == "CRITICAL"}
HIGH_VARIABLES = {k for k, (_, imp) in ECONOMIC_VARIABLES.items() if imp == "HIGH"}


def build_variable_patterns() -> dict[str, re.Pattern]:
    return {name: compile_word_pattern(words) for name, (words, _) in ECONOMIC_VARIABLES.items()}


# ---------------------------------------------------------------------------
# Secciones canónicas (consolidadas, 10 tipos)
# ---------------------------------------------------------------------------
# Cada sección tiene keywords que aumentan el score al matchear.
# Asignamos la sección con mayor score (o CONTENIDO si ninguna matcha).

SECTION_KEYWORDS: dict[str, list[str]] = {
    "ENCABEZADO": [
        "para publicacion", "for release", "embargoed", "press release",
        "correspondiente a la sesion de politica monetaria",
    ],
    "RESUMEN": [
        "resumen ejecutivo", "executive summary", "overview",
        "key takeaways", "bottom line", "en sintesis", "en resumen",
        "principales conclusiones", "key findings",
    ],
    "DECISION": [
        "acordo", "decidio", "resolvio", "aprobo", "voto por",
        "the committee decided", "fomc decided", "policy decision",
        "redujo la tpm", "incremento la tpm", "mantuvo la tasa",
        "recortar la tpm", "bajar la tasa", "subir la tasa",
        "acordar mantener", "la opcion de",
    ],
    "VOTACION": [
        "unanimidad", "disidencia", "disidente", "voto en contra",
        "dissent", "voting member", "voted against", "unanimous",
        "votaron", "voto favorable", "mayoria de los consejeros",
        "todos los consejeros",
    ],
    "CONTEXTO_EXTERNO": [
        "contexto externo", "escenario externo", "economia global",
        "global economy", "international backdrop", "emerging markets",
        "escenario internacional", "entorno externo",
        "fed", "bce", "china", "estados unidos", "zona euro",
        "economia mundial", "world economy", "comercio mundial",
        "mercados financieros internacionales",
    ],
    "CONTEXTO_INTERNO": [
        "contexto interno", "escenario interno", "economia nacional",
        "actividad local", "domestic economy",
        "economia chilena", "actividad economica local",
        "demanda interna", "indicadores locales",
        "en chile", "el pais",
    ],
    "ANALISIS": [
        "analisis", "evaluacion", "assessment", "consejeros evaluaron",
        "discutieron", "comentaron", "en su opinion", "a su juicio",
        "se observa que", "cabe destacar", "es importante",
        "tendencias", "dinamica reciente", "los datos muestran",
        "se señalo", "se comento", "se indico", "se analizo",
        "los consejeros", "los miembros",
    ],
    "PROYECCION": [
        "proyeccion", "perspectivas", "forecast", "outlook",
        "expectativas de", "se espera que", "is expected to",
        "nuestra proyeccion", "we forecast", "horizonte de proyeccion",
        "escenario central", "escenario base", "proyecta que",
        "se estima que", "estimamos", "ipom", "informe de politica",
        "proximo año", "proximos meses", "en los proximos",
    ],
    "RIESGOS": [
        "riesgos", "risks", "downside", "upside",
        "escenarios de riesgo", "tail risk",
        "riesgo al alza", "riesgo a la baja", "riesgo externo",
        "principales riesgos", "factores de riesgo",
        "incertidumbre", "vulnerabilidad",
    ],
    "DATOS": [
        "tabla", "figura", "grafico", "table", "figure", "chart",
        "source:", "fuente:", "ver figura", "see figure",
        "datos muestran", "segun los datos",
    ],
}


def build_section_patterns() -> dict[str, re.Pattern]:
    return {name: compile_word_pattern(words) for name, words in SECTION_KEYWORDS.items()}


# ---------------------------------------------------------------------------
# Entidades (para filtros opcionales y contexto)
# ---------------------------------------------------------------------------

ENTITY_KEYWORDS: dict[str, list[str]] = {
    "BANCO_CENTRAL_CHILE": [
        "banco central de chile", "bcch", "consejo del banco central",
        "reunion de politica monetaria", "rpm", "consejo del bcch",
        "el consejo acordo", "politica monetaria del banco",
    ],
    "FEDERAL_RESERVE": [
        "federal reserve", "the fed", "fomc", "fed funds",
        "jerome powell",
    ],
    "ECB": ["european central bank", "ecb", "banco central europeo"],
    "IMF": ["imf", "fmi", "fondo monetario internacional"],
    "PAIS_CHILE": [
        "chile", "chilena", "chileno",
        "clp",             # código ISO del peso chileno
        "ipsa",            # índice bursátil chileno
        "mercado local",   # contexto implícito de Monitor PM
        "curva local",     # curva de tasas local (chilena)
        "paridad local",   # tipo de cambio local (peso)
        "peso local",
    ],
    "PAIS_US": ["estados unidos", "united states", "eeuu", "ee.uu.", "dxy"],
    "PAIS_EUROZONE": ["eurozona", "euro area", "zona euro"],
    "PAIS_CHINA": ["china", "chinese", "beijing"],
}


def build_entity_patterns() -> dict[str, re.Pattern]:
    return {name: compile_word_pattern(words) for name, words in ENTITY_KEYWORDS.items()}


# ---------------------------------------------------------------------------
# Patrones numéricos (para extracción de datos cuantitativos)
# ---------------------------------------------------------------------------

NUMERIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Porcentajes: "5,25%", "5.25 %", "5,0 por ciento"
    (re.compile(r"(\d{1,3}(?:[.,]\d{1,3})?)\s*(?:%|por ciento|percent)"), "percentage"),
    # Puntos base: "75 puntos base", "75 bps", "75 pb"
    (re.compile(r"(\d{1,4})\s*(?:puntos base|pb\b|bps\b|basis points)"), "basis_points"),
    # Billones/millones (ambigüedad ES/EN; reportamos tal cual aparece)
    (re.compile(r"(\d{1,4}(?:[.,]\d{1,3})*)\s*(billones?|billion)"), "billions"),
    (re.compile(r"(\d{1,4}(?:[.,]\d{1,3})*)\s*(millones?|million)"), "millions"),
    # Montos con símbolo
    (re.compile(r"(?:USD|US\$|\$)\s*(\d{1,4}(?:[.,]\d{1,3})*)"), "usd_amount"),
]

# Palabras que señalan forward-looking / expectativa
FORWARD_LOOKING_KEYWORDS = [
    "se espera", "se proyecta", "se estima", "preve",
    "is expected", "is projected", "we forecast", "we expect",
    "outlook", "perspectivas", "proyeccion",
]

FORWARD_LOOKING_PATTERN = compile_word_pattern(FORWARD_LOOKING_KEYWORDS)

# Texto boilerplate legal/disclaimer (típico de reportes JPMorgan y similares).
# Los chunks que matcheen esto son ruido y deben recibir penalización máxima.
BOILERPLATE_KEYWORDS = [
    "legal entity responsible for the production",
    "this document is being provided for the exclusive use",
    "legal entities disclosures",
    "analyst certification",
    "reg ac research analyst",
    "country-/region-specific disclosures",
    "regulatory disclosures",
    "not regulated activities in your jurisdiction",
    "j.p. morgan securities",
    "jpmorgan chase bank",
    "securities and exchange board",
]
BOILERPLATE_PATTERN = compile_word_pattern(BOILERPLATE_KEYWORDS)

# Secciones propias del Monitor PM (para bonus de importancia base)
MONITOR_PM_SECTIONS: frozenset[str] = frozenset({
    "MERCADO_CAMBIARIO", "TASAS_TPM", "RENTA_FIJA",
    "EXPECTATIVAS_INFLACION", "RENTA_VARIABLE",
    "FINANCIAMIENTO_USD", "FINANCIAMIENTO_EXTERIOR", "FINANCIAMIENTO_PESOS",
    "SPREAD_BONOS", "SPREAD_CORPORATIVO", "RESUMEN",
})


# ---------------------------------------------------------------------------
# Secciones canónicas para Monitor PM (columna → section_type)
# ---------------------------------------------------------------------------
# La clave es el section_title_raw asignado por 00_generate_jsons.py,
# que corresponde exactamente al encabezado de columna del Excel.

MONITOR_PM_SECTION_MAP: dict[str, str] = {
    "Título":                                    "RESUMEN",
    "Mercado Cambiario":                         "MERCADO_CAMBIARIO",
    "Tasas SPC y Expectativas TPM implícita":    "TASAS_TPM",
    "Renta Fija":                                "RENTA_FIJA",
    "Expectativas de Inflación":                 "EXPECTATIVAS_INFLACION",
    "Mercado Renta Variable":                    "RENTA_VARIABLE",
    "Costo Financiamiento en dólares":           "FINANCIAMIENTO_USD",
    "Costo Financiamiento en el exterior":       "FINANCIAMIENTO_EXTERIOR",
    "Costo de Financiamiento en pesos":          "FINANCIAMIENTO_PESOS",
    "Spread bonos bancarios":                    "SPREAD_BONOS",
    "Spreads Corporativos":                      "SPREAD_CORPORATIVO",
}


# ---------------------------------------------------------------------------
# Tags (derivados de contenido — conjunto pequeño, no redundante con variables)
# ---------------------------------------------------------------------------
# Estos son banderas booleanas útiles para filtrar, NO re-encoding de variables.

def derive_tags(
    text_norm: str,
    variables: dict,
    numerics: list,
    section_type: str,
) -> list[str]:
    tags: list[str] = []

    if section_type == "DECISION":
        tags.append("DECISION_POLITICA")
    if section_type == "VOTACION":
        tags.append("VOTACION")
    if FORWARD_LOOKING_PATTERN.search(text_norm):
        tags.append("FORWARD_LOOKING")
    if numerics:
        tags.append("DATOS_NUMERICOS")
    if any(v in CRITICAL_VARIABLES for v in variables):
        tags.append("VARIABLE_CRITICA")

    return tags
