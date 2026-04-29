#!/usr/bin/env python3
"""
Orquestador del pipeline RAG.

Uso:
  python3 run.py                    # menú interactivo
  python3 run.py full               # pipeline completo (0→1→2→3 setup+load)
  python3 run.py step 0             # extracción + chunking
  python3 run.py step 1             # enriquecimiento de metadata
  python3 run.py step 2             # vectorización
  python3 run.py step 3             # setup + load en PostgreSQL
  python3 run.py step 3 setup       # solo crear schema
  python3 run.py step 3 load        # solo cargar datos
  python3 run.py step 3 reset       # DROP + recrear + cargar
  python3 run.py step 3 stats       # métricas actuales
  python3 run.py step 4 "<query>"   # búsqueda
  python3 run.py search             # búsqueda interactiva

El script delega a los .py numerados. No depende de shell/batch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(args: list[str], title: str) -> int:
    print(f"\n{'=' * 78}\n▶ {title}\n{'=' * 78}\n")
    result = subprocess.run([sys.executable, *args], cwd=HERE)
    print()
    if result.returncode != 0:
        print(f"✗ {title} falló (exit={result.returncode})")
    return result.returncode


def step_0() -> int:
    return run(["00_generate_jsons.py"], "Paso 0 — Extracción y chunking")


def step_1() -> int:
    return run(["01_enrich_metadata.py"], "Paso 1 — Enriquecimiento de metadata")


def step_2() -> int:
    return run(["02_vectorize.py"], "Paso 2 — Vectorización")


def step_3(sub: str = "setup_and_load") -> int:
    if sub == "setup_and_load":
        rc = run(["03_database.py", "setup"], "Paso 3a — Setup PostgreSQL (schema + índices)")
        if rc != 0:
            return rc
        return run(["03_database.py", "load"], "Paso 3b — Load (documents + chunks)")
    return run(["03_database.py", sub], f"Paso 3 — {sub}")


def step_4_interactive() -> int:
    print("\nBúsqueda interactiva. Escribe 'salir' para terminar.")
    while True:
        try:
            query = input("\n🔍 Consulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"salir", "exit", "quit", ""}:
            return 0
        k = input("   k [5]: ").strip() or "5"
        try:
            int(k)
        except ValueError:
            print("   (k inválido, uso 5)")
            k = "5"
        subprocess.run([sys.executable, "04_search.py", query, k], cwd=HERE)


def step_4_query(query: str, k: int = 5) -> int:
    return run(["04_search.py", query, str(k)], f"Paso 4 — Búsqueda: {query!r}")


def pipeline_full() -> int:
    stages = [("extracción", step_0), ("enriquecimiento", step_1),
              ("vectorización", step_2), ("postgres", step_3)]
    for label, fn in stages:
        if fn() != 0:
            print(f"✗ Pipeline abortado en '{label}'")
            return 1
    print("\n✓ Pipeline completo. Prueba: python3 run.py search")
    return 0


def show_menu() -> int:
    menu = """
╔════════════════════════════════════════════════════════════════════════════╗
║  RAG Banco Central — menú principal                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  1. Pipeline completo (paso 0 → 3)                                         ║
║  2. Paso 0 — Extracción y chunking desde PDFs                              ║
║  3. Paso 1 — Enriquecimiento de metadata                                   ║
║  4. Paso 2 — Vectorización                                                 ║
║  5. Paso 3 — Setup + Load PostgreSQL                                       ║
║  6. Paso 3 — Solo stats de la BD                                           ║
║  7. Paso 4 — Búsqueda interactiva                                          ║
║  0. Salir                                                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
    while True:
        print(menu)
        choice = input("Opción: ").strip()
        if choice == "1":
            pipeline_full()
        elif choice == "2":
            step_0()
        elif choice == "3":
            step_1()
        elif choice == "4":
            step_2()
        elif choice == "5":
            step_3("setup_and_load")
        elif choice == "6":
            step_3("stats")
        elif choice == "7":
            step_4_interactive()
        elif choice == "0":
            return 0
        else:
            print("✗ opción inválida")


def main() -> int:
    if len(sys.argv) == 1:
        return show_menu()

    cmd = sys.argv[1].lower()

    if cmd == "full":
        return pipeline_full()

    if cmd == "search":
        return step_4_interactive()

    if cmd == "step":
        if len(sys.argv) < 3:
            print("uso: run.py step [0|1|2|3|4] ...")
            return 1
        n = sys.argv[2]
        if n == "0":
            return step_0()
        if n == "1":
            return step_1()
        if n == "2":
            return step_2()
        if n == "3":
            sub = sys.argv[3] if len(sys.argv) > 3 else "setup_and_load"
            if sub not in {"setup", "load", "reset", "stats", "setup_and_load"}:
                print(f"✗ subcomando inválido: {sub}")
                return 1
            return step_3(sub)
        if n == "4":
            if len(sys.argv) > 3:
                query = sys.argv[3]
                k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
                return step_4_query(query, k)
            return step_4_interactive()
        print(f"✗ paso inválido: {n}")
        return 1

    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
