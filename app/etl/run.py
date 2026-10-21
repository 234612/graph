import logging

from . import sync_table, sync_text


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sync_table.run()
    sync_text.run()


if __name__ == "__main__":
    main()