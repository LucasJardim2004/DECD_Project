"""
Launcher para o modelo de Árvores de Decisão treinado com o dataset 1:5.

Este ficheiro existe para permitir executar o comando:

    python treinar_modelo_decision_tree.py

quando o terminal está dentro da pasta output_decision_tree_1to5.
A lógica principal vive no script pai:

    ../treinar_modelo_decision_tree.py
"""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    parent_script = Path(__file__).resolve().parent.parent / "treinar_modelo_decision_tree.py"
    runpy.run_path(str(parent_script), run_name="__main__")
