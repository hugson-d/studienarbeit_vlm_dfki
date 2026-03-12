"""Data models for the watch store."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Watch:
    id: int
    name: str
    brand: str
    price: float
    description: str
    image_url: str
    category: str
    in_stock: bool = True
    rating: float = 4.5


WATCHES: List[Watch] = [
    Watch(
        id=1,
        name="Classic Elegance",
        brand="TimeMaster",
        price=299.99,
        description=(
            "A timeless classic with a stainless steel case and leather strap. "
            "Features a Swiss quartz movement for precise timekeeping."
        ),
        image_url="https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400",
        category="Classic",
        rating=4.8,
    ),
    Watch(
        id=2,
        name="Sport Pro X",
        brand="ActiveTime",
        price=199.99,
        description=(
            "Built for athletes and adventurers. Water-resistant to 200m, "
            "with chronograph, heart rate monitor and GPS."
        ),
        image_url="https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=400",
        category="Sport",
        rating=4.6,
    ),
    Watch(
        id=3,
        name="Luxury Gold Edition",
        brand="GoldCraft",
        price=1299.99,
        description=(
            "18k gold-plated case with sapphire crystal glass. "
            "Swiss automatic movement, 42-hour power reserve."
        ),
        image_url="https://images.unsplash.com/photo-1587836374828-4dbafa94cf0e?w=400",
        category="Luxury",
        rating=4.9,
    ),
    Watch(
        id=4,
        name="Smartwatch Ultra",
        brand="TechTime",
        price=399.99,
        description=(
            "Next-gen smartwatch with AMOLED display, health monitoring, "
            "7-day battery life and seamless smartphone integration."
        ),
        image_url="https://images.unsplash.com/photo-1579586337278-3befd40fd17a?w=400",
        category="Smart",
        rating=4.7,
    ),
    Watch(
        id=5,
        name="Diver's Watch",
        brand="OceanTime",
        price=449.99,
        description=(
            "Professional diving watch, water-resistant to 500m. "
            "Unidirectional rotating bezel and luminous hands for underwater use."
        ),
        image_url="https://images.unsplash.com/photo-1612817159949-195b6eb9e31a?w=400",
        category="Sport",
        rating=4.7,
    ),
    Watch(
        id=6,
        name="Minimalist Slim",
        brand="PureTime",
        price=149.99,
        description=(
            "Ultra-thin design at just 5.8mm. "
            "Clean dial, mesh bracelet and Japanese quartz movement."
        ),
        image_url="https://images.unsplash.com/photo-1508685096489-7aacd43bd3b1?w=400",
        category="Classic",
        rating=4.5,
    ),
]


def get_all_watches() -> List[Watch]:
    return WATCHES


def get_watch_by_id(watch_id: int):
    return next((w for w in WATCHES if w.id == watch_id), None)


def get_watches_by_category(category: str) -> List[Watch]:
    if category == "All":
        return WATCHES
    return [w for w in WATCHES if w.category == category]


def get_categories() -> List[str]:
    return ["All"] + sorted({w.category for w in WATCHES})
