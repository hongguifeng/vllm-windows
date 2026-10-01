import sys

for i, a in enumerate(sys.argv[1:], 1):
    print(f"argv[{i}] = {a!r}")
