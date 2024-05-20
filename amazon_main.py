import asyncio

from nesy.reviews_preprocessing.get_books_info import get_books_info
from nesy.reviews_preprocessing.get_movies_info_v2 import get_movies_info
from nesy.reviews_preprocessing.get_records_info_v4 import get_records_info
from nesy.reviews_preprocessing.get_shows_info_v2 import get_shows_info


def pretty_print_responses(responses: list):
    if responses is not None:
        for res in responses:
            print(res)
    print()


async def main():
    books_titles = ["harry potter order of the phoenix", "I, robot", "The Shining"]
    books_responses = await get_books_info(books_titles)
    pretty_print_responses(books_responses)

    records_titles = [
        "The Dark Side of the Moon",
        "Beerbongs & Bentleys",
        "The Eminem Show",
    ]
    records_responses = await get_records_info(records_titles)
    pretty_print_responses(records_responses)

    movies_titles = [
        "Night of the Living Dead",
        "It",
        "The Lord of the Rings: The Two Towers",
    ]
    movies_responses = await get_movies_info(movies_titles)
    pretty_print_responses(movies_responses)

    # todo hunter x hunter mi trova manga e basta
    shows_titles = ["Breaking Bad", "Doctor Who", "Dr. Who", "Hunter x Hunter"]
    shows_responses = await get_shows_info(shows_titles)
    pretty_print_responses(shows_responses)


if __name__ == "__main__":
    asyncio.run(main())
