"""Select application logs for deeper analysis while retaining every original."""

from diffusion_jev.demo import main

if __name__ == "__main__":
    import sys

    raise SystemExit(main(["logs", *sys.argv[1:]]))
