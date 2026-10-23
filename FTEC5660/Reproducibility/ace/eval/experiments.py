"""Safe A/B/C Formula entry point: --execute explicitly opts into APIs."""
from src.experiments.runner import main

if __name__ == '__main__':
    raise SystemExit(main())
