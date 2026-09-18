"""Compare a patch with a later snapshot that adds regression evidence."""

from diffusion_jev.demo import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["completion", *sys.argv[1:]]))
