from . import sync_table, sync_text


def main():
    sync_table.run()
    sync_text.run()


if __name__ == "__main__":
    main()