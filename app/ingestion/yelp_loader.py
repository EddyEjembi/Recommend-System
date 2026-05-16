import json
from typing import Generator


class YelpLoader:

    @staticmethod
    def stream_json(path: str) -> Generator[dict, None, None]:
        """
        Streams a JSON file line by line.
        Yelp dataset stores one JSON object per line.
        """
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                yield json.loads(line)

    @staticmethod
    def load_reviews(path: str, limit: int = None):
        """
        Load review records with optional limit.
        """
        reviews = []

        for idx, review in enumerate(YelpLoader.stream_json(path)):
            reviews.append(review)

            if limit and idx + 1 >= limit:
                break

        return reviews

    @staticmethod
    def load_businesses(path: str, limit: int = None):
        """
        Load business records with optional limit.
        """
        businesses = []

        for idx, business in enumerate(YelpLoader.stream_json(path)):
            businesses.append(business)

            if limit and idx + 1 >= limit:
                break

        return businesses

    @staticmethod
    def load_users(path: str, limit: int = None):
        """
        Load user records with optional limit.
        """
        users = []

        for idx, user in enumerate(YelpLoader.stream_json(path)):
            users.append(user)

            if limit and idx + 1 >= limit:
                break

        return users