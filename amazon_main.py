from nesy.reviews_preprocessing.get_books_info import get_books_info
from nesy.reviews_preprocessing.get_records_info_v2 import get_records_info
from nesy.reviews_preprocessing.get_movies_info import get_movies_info
from nesy.reviews_preprocessing.get_shows_info import get_shows_info
import asyncio


async def main():
    # book_titles = ["harry potter order of the phoenix", "I, robot", "The Shining"]
    # responses = await get_books_info(book_titles)
    # records_titles = [
    #     "The Dark Side of the Moon",
    #     "Beerbongs & Bentleys",
    #     "The Eminem Show",
    # ]
    # responses = await get_records_info(records_titles)
    # movie_titles = [
    #     "Night of the Living Dead",
    #     "It",
    #     "The Lord of the Rings: The Two Towers",
    # ]
    # responses = await get_movies_info(movie_titles)
    show_titles = ["Breaking Bad", "Doctor Who", "Dr. Who", "Hunter x Hunter"]
    responses = await get_shows_info(show_titles)
    print(responses)


if __name__ == "__main__":
    asyncio.run(main())
