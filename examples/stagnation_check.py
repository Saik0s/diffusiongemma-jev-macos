"""Contrast two deployment investigations; no agent is stopped or launched."""

from diffusion_jev.demo import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["progress", *sys.argv[1:]]))
