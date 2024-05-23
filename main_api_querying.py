import asyncio
import json
import os.path

from nesy.api_querying.google_kg_search import google_kg_search


async def main():
    merged_metadata_path = os.path.join("data", "processed", "merged_metadata.json")
    merged_metadata_aug = os.path.join("data", "processed", "merged_metadata_aug.json")

    # how many items to process
    limit = 500
    with open(merged_metadata_path, "r") as f:
        with open(merged_metadata_aug, "w", encoding="UTF-8") as g:
            merged_metadata = json.load(f)

            # dictionary for reverse lookup
            books = {}
            for k, v in merged_metadata.items():
                if v["type"] == "books" and v["title"] is not None:
                    books[v["title"]] = k

            books_titles = list(books.keys())
            print(f"Remaining books: {len(books_titles)}, processing: {limit}...")

            books_info = await google_kg_search(
                titles=list(books.keys())[:limit], query_type="books"
            )

            # title is the same used for querying, the one provided by the response is disregarded
            for title, author, year in books_info:
                asin = books[title]
                merged_metadata[asin]["person"] = author
                merged_metadata[asin]["year"] = year

            json.dump(merged_metadata, g, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
