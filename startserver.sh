#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# python_e is a Windows embeddable Python build, only usable under
# Git Bash / MSYS / Cygwin. Everywhere else, set up a local venv so
# we don't depend on (or pollute) the system Python install.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        PYTHON="python3"
        if [ -f "python_e/python.exe" ]; then
            PYTHON="python_e/python.exe"
        fi
        ;;
    *)
        SYSTEM_PYTHON="python3"
        VENV_DIR=".venv"
        VENV_PYTHON="$VENV_DIR/bin/python"

        if [ "$EUID" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
            echo "[WARN] Running as root via sudo. A virtualenv will be created at"
            echo "       $(pwd)/$VENV_DIR owned by root. Prefer running this script"
            echo "       without sudo as your normal user."
        fi

        if [ ! -x "$VENV_PYTHON" ]; then
            echo "Creating Python virtual environment in $VENV_DIR ..."
            if ! "$SYSTEM_PYTHON" -m venv "$VENV_DIR" 2>/tmp/venv_err; then
                cat /tmp/venv_err >&2
                if grep -qi "ensurepip is not available\|No module named venv\|No module named pip" /tmp/venv_err; then
                    echo "[SETUP] python3-venv (and pip) are missing. Attempting to install them via apt..."
                    if command -v apt-get >/dev/null 2>&1; then
                        PY_MINOR_PKG="python3-venv"
                        if [ "$EUID" -eq 0 ]; then
                            apt-get update -y && apt-get install -y python3-venv python3-pip
                        elif command -v sudo >/dev/null 2>&1; then
                            sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip
                        else
                            echo "[SETUP] ERROR: need root privileges to install python3-venv/python3-pip."
                            echo "        Run: sudo apt-get install -y python3-venv python3-pip"
                            exit 1
                        fi
                    else
                        echo "[SETUP] ERROR: apt-get not found. Please install python3-venv and python3-pip manually."
                        exit 1
                    fi
                    echo "Creating Python virtual environment in $VENV_DIR ..."
                    "$SYSTEM_PYTHON" -m venv "$VENV_DIR"
                else
                    echo "[SETUP] ERROR: failed to create virtual environment."
                    exit 1
                fi
            fi
            rm -f /tmp/venv_err
        fi

        PYTHON="$VENV_PYTHON"
        # Make sure pip itself is present and up to date inside the venv.
        "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
        "$PYTHON" -m pip install --upgrade pip >/dev/null 2>&1 || true
        ;;
esac

echo "Checking Python environment..."
"$PYTHON" setup.py

echo
echo "Starting CaseSorter AI Server..."
exec "$PYTHON" server.py
