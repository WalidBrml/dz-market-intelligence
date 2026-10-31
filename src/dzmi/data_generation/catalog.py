"""Synthetic catalogue, store list and customers.

Product names are GENERIC (no real brands). Prices/costs are illustrative DZD ranges chosen
for plausibility of scale only; they are NOT real market prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# base_rate: median units/store/day for a median-popularity product BEFORE demand_scale.
# ramadan / pre_fitr / pre_adha: demand multipliers (ASSUMPTIONS). summer: seasonal amplitude.
CATEGORIES: list[dict] = [
    dict(name="Flour & Semolina", code="FLO", weight=7, price=(60, 260), margin=(0.08, 0.16), base_rate=3.0,
         perishable=False, packs=(6, 12), ramadan=1.35, pre_fitr=1.15, pre_adha=1.00, summer=0.00, elasticity=1.8,
         items=["Fine semolina", "Medium semolina", "Wheat flour T55", "Wheat flour T45", "Couscous semolina"],
         sizes=["1 kg", "2 kg", "5 kg"]),
    dict(name="Pasta & Couscous", code="PAS", weight=7, price=(50, 220), margin=(0.10, 0.18), base_rate=3.0,
         perishable=False, packs=(12, 24), ramadan=1.20, pre_fitr=1.05, pre_adha=1.00, summer=0.00, elasticity=2.0,
         items=["Spaghetti", "Short pasta", "Vermicelli", "Fine couscous", "Medium couscous"],
         sizes=["500 g", "1 kg"]),
    dict(name="Oil, Sugar & Salt", code="OIL", weight=6, price=(70, 900), margin=(0.05, 0.12), base_rate=3.5,
         perishable=False, packs=(6, 12), ramadan=1.40, pre_fitr=1.10, pre_adha=1.00, summer=0.00, elasticity=1.5,
         items=["Sunflower oil", "Mixed vegetable oil", "White sugar", "Cube sugar", "Table salt"],
         sizes=["1 kg", "1 L", "2 L", "5 L"]),
    dict(name="Dairy & Eggs", code="DAI", weight=9, price=(30, 650), margin=(0.10, 0.20), base_rate=4.0,
         perishable=True, packs=(6, 12), ramadan=1.45, pre_fitr=1.10, pre_adha=1.00, summer=0.05, elasticity=1.8,
         items=["UHT milk", "Fresh milk", "Plain yogurt", "Fruit yogurt", "Processed cheese", "Eggs", "Butter"],
         sizes=["1 L", "6-pack", "200 g", "500 g"]),
    dict(name="Dates & Dried Fruit", code="DAT", weight=5, price=(250, 1800), margin=(0.14, 0.26), base_rate=1.2,
         perishable=False, packs=(6, 12), ramadan=2.40, pre_fitr=1.30, pre_adha=1.00, summer=0.00, elasticity=2.2,
         items=["Dates (standard)", "Dates (premium)", "Dried apricots", "Raisins", "Almonds"],
         sizes=["500 g", "1 kg"]),
    dict(name="Coffee, Tea & Infusions", code="COF", weight=6, price=(150, 1200), margin=(0.14, 0.24), base_rate=1.8,
         perishable=False, packs=(6, 12), ramadan=1.15, pre_fitr=1.10, pre_adha=1.00, summer=-0.05, elasticity=2.0,
         items=["Ground coffee", "Instant coffee", "Black tea", "Green tea", "Herbal infusion"],
         sizes=["250 g", "500 g", "100 bags"]),
    dict(name="Water & Soft Drinks", code="BEV", weight=9, price=(35, 260), margin=(0.12, 0.22), base_rate=4.5,
         perishable=False, packs=(6, 12), ramadan=1.20, pre_fitr=1.05, pre_adha=1.00, summer=0.30, elasticity=2.4,
         items=["Still water", "Sparkling water", "Cola drink", "Lemon soda", "Orange drink", "Fruit juice"],
         sizes=["500 ml", "1.5 L", "2 L"]),
    dict(name="Canned & Preserved", code="CAN", weight=7, price=(60, 420), margin=(0.10, 0.20), base_rate=2.5,
         perishable=False, packs=(12, 24), ramadan=1.30, pre_fitr=1.05, pre_adha=1.00, summer=0.00, elasticity=2.0,
         items=["Tomato paste", "Canned chickpeas", "Canned tuna", "Olives", "Harissa", "Canned peas"],
         sizes=["200 g", "400 g", "800 g"]),
    dict(name="Legumes & Rice", code="LEG", weight=6, price=(90, 380), margin=(0.08, 0.16), base_rate=2.2,
         perishable=False, packs=(10, 20), ramadan=1.35, pre_fitr=1.05, pre_adha=1.00, summer=0.00, elasticity=1.8,
         items=["Lentils", "Chickpeas", "White beans", "Long-grain rice", "Freekeh"],
         sizes=["1 kg", "2 kg"]),
    dict(name="Frozen Meat & Poultry", code="MEA", weight=6, price=(450, 2400), margin=(0.10, 0.20), base_rate=1.4,
         perishable=True, packs=(5, 10), ramadan=1.50, pre_fitr=1.15, pre_adha=1.60, summer=0.00, elasticity=2.2,
         items=["Frozen chicken", "Chicken escalope", "Minced beef", "Beef cubes", "Lamb pieces", "Merguez"],
         sizes=["500 g", "1 kg"]),
    dict(name="Bakery & Pastry", code="BAK", weight=7, price=(25, 700), margin=(0.20, 0.38), base_rate=3.2,
         perishable=True, packs=(6, 12), ramadan=1.50, pre_fitr=1.90, pre_adha=1.10, summer=0.00, elasticity=2.4,
         items=["Bread loaf", "Brioche", "Traditional pastry", "Honey pastry", "Biscuits", "Cake"],
         sizes=["single", "box of 6", "box of 12"]),
    dict(name="Fresh Fruit & Vegetables", code="FRV", weight=9, price=(40, 500), margin=(0.15, 0.30), base_rate=5.0,
         perishable=True, packs=(5, 10), ramadan=1.35, pre_fitr=1.10, pre_adha=1.15, summer=0.10, elasticity=1.6,
         items=["Potatoes", "Onions", "Tomatoes", "Carrots", "Oranges", "Apples", "Bananas", "Peppers"],
         sizes=["1 kg", "2 kg"]),
    dict(name="Household Cleaning", code="CLE", weight=6, price=(80, 700), margin=(0.16, 0.30), base_rate=1.5,
         perishable=False, packs=(6, 12), ramadan=1.10, pre_fitr=1.30, pre_adha=1.15, summer=0.00, elasticity=2.2,
         items=["Laundry powder", "Dish liquid", "Bleach", "Floor cleaner", "Sponges"],
         sizes=["500 ml", "1 L", "3 kg"]),
    dict(name="Personal Care", code="PER", weight=6, price=(90, 900), margin=(0.20, 0.36), base_rate=1.3,
         perishable=False, packs=(6, 12), ramadan=1.00, pre_fitr=1.25, pre_adha=1.10, summer=0.05, elasticity=2.2,
         items=["Shampoo", "Soap bar", "Toothpaste", "Deodorant", "Hand cream"],
         sizes=["100 ml", "250 ml", "400 ml"]),
]

STORES: list[dict] = [
    dict(city="Algiers", wilaya="Alger", name="Alger Centre", fmt="Hypermarket", area=(4500, 7000)),
    dict(city="Blida", wilaya="Blida", name="Blida Ville", fmt="Supermarket", area=(1500, 3000)),
    dict(city="Oran", wilaya="Oran", name="Oran Es Senia", fmt="Hypermarket", area=(4500, 7000)),
    dict(city="Constantine", wilaya="Constantine", name="Constantine Centre", fmt="Supermarket", area=(1800, 3200)),
    dict(city="Sétif", wilaya="Sétif", name="Sétif El Hidhab", fmt="Supermarket", area=(1500, 3000)),
    dict(city="Bab Ezzouar", wilaya="Alger", name="Alger Bab Ezzouar", fmt="Hypermarket", area=(5000, 8000)),
    dict(city="Annaba", wilaya="Annaba", name="Annaba Centre", fmt="Supermarket", area=(1800, 3200)),
    dict(city="Tizi Ouzou", wilaya="Tizi Ouzou", name="Tizi Ouzou Ville", fmt="Supermarket", area=(1200, 2600)),
    dict(city="Béjaïa", wilaya="Béjaïa", name="Béjaïa Port", fmt="Supermarket", area=(1200, 2600)),
    dict(city="Batna", wilaya="Batna", name="Batna Centre", fmt="Supermarket", area=(1200, 2600)),
    dict(city="Tlemcen", wilaya="Tlemcen", name="Tlemcen Ville", fmt="Express", area=(400, 1000)),
    dict(city="Biskra", wilaya="Biskra", name="Biskra Centre", fmt="Express", area=(400, 1000)),
    dict(city="Ouargla", wilaya="Ouargla", name="Ouargla Ville", fmt="Express", area=(400, 1000)),
    dict(city="Mostaganem", wilaya="Mostaganem", name="Mostaganem Centre", fmt="Express", area=(400, 1000)),
    dict(city="Médéa", wilaya="Médéa", name="Médéa Ville", fmt="Express", area=(400, 1000)),
]

FORMAT_FACTOR = {"Hypermarket": 2.2, "Supermarket": 1.0, "Express": 0.45}
HOUSEHOLD_TYPES = ["Single", "Couple", "Family (3-4)", "Large family (5+)"]
PAYMENT_METHODS = ["Cash", "CIB card", "Edahabia card", "Mobile payment"]
PAYMENT_WEIGHTS = [0.62, 0.17, 0.14, 0.07]
BRANDS = [f"Brand {c}" for c in "ABCDEFGHIJ"]


def build_categories() -> pd.DataFrame:
    return pd.DataFrame({
        "category_id": np.arange(1, len(CATEGORIES) + 1),
        "category_name": [c["name"] for c in CATEGORIES],
        "is_perishable": [c["perishable"] for c in CATEGORIES],
    })


def _allocate(n_products: int) -> np.ndarray:
    """Products per category proportional to weights, >= 1 each, summing to n_products."""
    w = np.array([c["weight"] for c in CATEGORIES], dtype=float)
    raw = w / w.sum() * n_products
    n = np.maximum(1, np.floor(raw).astype(int))
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while n.sum() < n_products:
        n[order[i % len(n)]] += 1
        i += 1
    while n.sum() > n_products:
        j = int(np.argmax(n))
        n[j] -= 1
    return n


def build_products(n_products: int, rng: np.random.Generator) -> pd.DataFrame:
    counts = _allocate(n_products)
    rows = []
    pid = 1
    for ci, (cat, cnt) in enumerate(zip(CATEGORIES, counts), start=1):
        combos = [(it, sz) for it in cat["items"] for sz in cat["sizes"]]
        rng.shuffle(combos)
        for k in range(cnt):
            item, size = combos[k % len(combos)]
            brand = BRANDS[int(rng.integers(0, len(BRANDS)))]
            lo, hi = cat["price"]
            price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))) / 5) * 5)
            price = max(price, 25.0)
            margin = rng.uniform(*cat["margin"])
            cost = float(np.round(price * (1 - margin), 2))
            rows.append(dict(
                product_id=pid, sku=f"{cat['code']}-{pid:04d}",
                product_name=f"{item} {size} ({brand})", category_id=ci, brand=brand,
                pack_size=int(rng.choice(cat["packs"])), base_price=price, base_cost=cost,
            ))
            pid += 1
    return pd.DataFrame(rows)


def build_stores(n_stores: int, start_date, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(STORES[:n_stores], start=1):
        rows.append(dict(
            store_id=i, store_code=f"ST{i:02d}", store_name=s["name"], wilaya=s["wilaya"], city=s["city"],
            store_format=s["fmt"], floor_area_sqm=int(rng.integers(*s["area"])),
            opened_date=(pd.Timestamp(start_date) - pd.Timedelta(days=int(rng.integers(365, 365 * 12)))).normalize(),
        ))
    return pd.DataFrame(rows)


def build_customers(n_customers: int, stores: pd.DataFrame, start_date, rng: np.random.Generator) -> pd.DataFrame:
    w = stores["floor_area_sqm"].to_numpy(dtype=float)
    home = rng.choice(stores["store_id"].to_numpy(), size=n_customers, p=w / w.sum())
    wilaya = stores.set_index("store_id").loc[home, "wilaya"].to_numpy()
    return pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "home_store_id": home,
        "wilaya": wilaya,
        "household_type": rng.choice(HOUSEHOLD_TYPES, size=n_customers, p=[0.15, 0.2, 0.4, 0.25]),
        "signup_date": pd.Timestamp(start_date) - pd.to_timedelta(rng.integers(1, 1500, n_customers), unit="D"),
    })
