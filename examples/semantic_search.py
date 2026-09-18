"""Rank six functions by whether they swallow exceptions; no code is modified."""

from diffusion_jev.demo import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["search", *sys.argv[1:]]))
