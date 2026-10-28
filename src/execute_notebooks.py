"""
Executes all notebooks in-place using nbclient so that outputs and plots are persisted.
"""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "01_eda.ipynb",
    "02_baselines_ml.ipynb",
    "03_deep_learning.ipynb",
    "04_federated_learning.ipynb",
]

for nb_name in NOTEBOOKS:
    nb_path = Path("notebooks") / nb_name
    print(f"Executing {nb_name}...")
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute(cwd=str(Path("notebooks").resolve()))
    nbformat.write(nb, nb_path)
    print(f"Finished {nb_name} with all outputs persisted.")

print("All notebooks executed successfully!")
