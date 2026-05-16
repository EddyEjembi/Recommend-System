from app.ingestion.yelp_loader import YelpLoader


REVIEWS_PATH = "app/data/raw/yelp/review.json"
BUSINESS_PATH = "app/data/raw/yelp/business.json"
USER_PATH = "app/data/raw/yelp/user.json"


# Load small sample
reviews = YelpLoader.load_reviews(
    REVIEWS_PATH,
    limit=5
)

businesses = YelpLoader.load_businesses(
    BUSINESS_PATH,
    limit=2
)

users = YelpLoader.load_users(
    USER_PATH,
    limit=2
)

print("\nREVIEWS:")
print(reviews[0])

print("\nBUSINESSES:")
print(businesses[0])

print("\nUSERS:")
print(users[0])