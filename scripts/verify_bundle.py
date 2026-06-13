# verify_bundle.py - confirm the bundle is complete and secret-free before review.
from collections import defaultdict
from pathlib import Path

BUNDLE = "querymind_bundle.txt"
SEPARATOR = "=" * 40
SOURCE_EXT = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".cfg", ".ipynb"}
IGNORE = {
    "querymind_bundle.txt",
    "verify_bundle.py",
}  # expected not to be in the bundle
SECRET_MARKERS = ["sk-ant-", "sk-proj-", "AKIA"]  # literal key prefixes

text = Path(BUNDLE).read_text(encoding="utf-8")
lines = text.splitlines()


def norm(p):
    return p.replace("\\", "/")


# 1. Completeness: the bundler's recorded tree vs the files it actually dumped.
tree, in_tree = [], False
for ln in lines:
    s = ln.strip()
    if s == "### DIRECTORY TREE ###":
        in_tree = True
        continue
    if in_tree:
        if s == SEPARATOR:
            break
        if s:
            tree.append(s)

dumped = set()
for ln in lines:
    s = ln.strip()
    if s.startswith("### FILE:") and s.endswith("###"):
        dumped.add(norm(s[len("### FILE:") : -len("###")].strip()))

missing, src_count = defaultdict(list), 0
for p in tree:
    if Path(p).suffix.lower() not in SOURCE_EXT:
        continue
    src_count += 1
    if Path(p).name in IGNORE:
        continue
    if norm(p) not in dumped:
        top = norm(p).split("/")[0] if "/" in norm(p) else "(root)"
        missing[top].append(norm(p))

print(f"Source-like files in repo tree: {src_count}")
print(f"Files dumped into bundle:       {len(dumped)}")
if not missing:
    print("OK - every source-like file in the tree is present in the bundle.\n")
else:
    print("\nSource files in the repo but NOT in the bundle:")
    for top in sorted(missing):
        print(f"  {top}/")
        for p in sorted(missing[top]):
            print(f"     - {p}")
    print()

# 2. Spot checks: README and the config YAMLs are load-bearing for review.
print("Spot checks:")
print(f"  README.md present:   {'README.md' in dumped}")
cfgs = sorted(p for p in dumped if p.startswith("config/"))
print(f"  config/ present ({len(cfgs)}): {cfgs}\n")

# 3. Secret scan (literal key prefixes) - your Ctrl+F, done reliably.
hits = [(i, m) for i, ln in enumerate(lines, 1) for m in SECRET_MARKERS if m in ln]
if hits:
    print("!! POSSIBLE SECRETS - do NOT upload until resolved:")
    for i, m in hits:
        print(f"   line {i}: contains '{m}'")
else:
    print("No literal API-key prefixes (sk-ant-, sk-proj-, AKIA) found.")
