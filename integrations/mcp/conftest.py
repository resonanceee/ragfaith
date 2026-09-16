# Present so pytest prepends this directory to sys.path (ragfaith_mcp import)
# and so the sibling integrations/proxy package resolves without an install.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy"))
