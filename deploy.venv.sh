python -m venv .venv
source .venv/bin/activate
which python

pip install --upgrade pip

pip install -r requirements.txt
pip install -e ./chatterbox

# 1. Remove the current broken torch entirely
pip uninstall torch -y
# 2. Clear the pip cache to ensure you don't reinstall the same broken version
pip cache purge
# 3. Reinstall a fresh copy
# Note: Using the standard PyPI version first to see if it resolves
pip install torch