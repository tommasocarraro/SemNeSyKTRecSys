from nesy.reviews_preprocessing.get_books_info import get_books_info
import asyncio


async def main():
    book_titles = ["harry potter order of the phoenix", "I, robot", "The Shining"]
    responses = await get_books_info(book_titles)
    print(responses)


if __name__ == "__main__":
    asyncio.run(main())
