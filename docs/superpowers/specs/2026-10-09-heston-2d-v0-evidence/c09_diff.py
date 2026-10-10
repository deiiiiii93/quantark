"""C09 diff: python c09_diff.py <main.json> <branch.json> -- one line per row, bit-identical or MOVED (new - old)."""
import json
import sys

if __name__ == "__main__":
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    for k in a:
        x, y = a[k], b.get(k, {})
        if "hex" in x and "hex" in y and x["hex"] == y["hex"]:
            print(f"{k:<62} bit-identical  {x['repr'][0]}")
        elif "hex" in x and "hex" in y:
            d = [float(v1) - float(v0) for v0, v1 in zip(x["repr"], y["repr"])]
            print(f"{k:<62} MOVED  {' / '.join(x['repr'])} -> {' / '.join(y['repr'])}  "
                  f"(diff {' / '.join(f'{v:+.3e}' for v in d)})")
        else:
            print(f"{k:<62} {x.get('error', x.get('repr'))} -> {y.get('error', y.get('repr'))}")
