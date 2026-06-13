#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# python_e is a Windows embeddable Python build, only usable under
# Git Bash / MSYS / Cygwin. Everywhere else, fall back to system python3.
PYTHON="python3"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        if [ -f "python_e/python.exe" ]; then
            PYTHON="python_e/python.exe"
        fi
        ;;
esac

echo "Checking Python environment..."
"$PYTHON" setup_torch.py

echo
echo "Starting CaseSorter AI Server..."
exec "$PYTHON" server.py
