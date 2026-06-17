import sys
import os

# Insert site-packages of the virtual environment to import paths
venv_site_packages = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv312/lib/python3.12/site-packages"))
sys.path.insert(0, venv_site_packages)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
