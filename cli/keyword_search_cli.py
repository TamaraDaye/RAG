#!/usr/bin/env python3
import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build database for query")  # pyright: ignore[]

    frequency_parser = subparsers.add_parser(
        "tf", help="Prints the frequency of the term supplied"
    )

    frequency_parser.add_argument("id", type=int, help="document id to search")

    frequency_parser.add_argument("term", type=str, help="term to find its frequency")

    idf_parser = subparsers.add_parser(
        "idf", help="Prints the inverse document frequency of a term"
    )

    idf_parser.add_argument("term", type=str, help="Search term to calculate its idf")

    args = parser.parse_args()

    match args.command:
        case "search":
            result = search_command(args.query, 5)
            for i, movie in enumerate(result):
                print(f"{i} {movie['title']}")
        case "build":
            build_command()

        case "tf":
            print(tf_command(args.id, args.term))

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")


if __name__ == "__main__":
    main()
