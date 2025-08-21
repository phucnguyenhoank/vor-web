from typing import TypedDict, Any
import uuid
import copy
import json
import datetime
import os
import pandas as pd

# ----------------------------
# TypedDict input / output schemas (using modern list[...] and union |)
# ----------------------------
class Category(TypedDict):
    id: int
    name: str

class MenuItem(TypedDict):
    id: int
    name: str
    price: float
    category_id: int

class Discount(TypedDict):
    discount_percentage: float  # decimal, e.g. 0.1 for 10%
    item_ids: list[int]

class MenuData(TypedDict):
    categories: list[Category]
    items: list[MenuItem]
    discounts: list[Discount]

class ItemInput(TypedDict):
    """
    Input shape for add/set/increase functions.

    Example:
    {"item_id": 1, "quantity": 2}
    """
    item_id: int
    quantity: int

class CartItem(TypedDict):
    item_id: int
    quantity: int

class TotalsLine(TypedDict):
    item_id: int
    name: str
    quantity: int
    price_per_item: float
    line_total: float
    line_discount: float
    discount_pct: float

class TotalsResult(TypedDict):
    order_details: list[TotalsLine]
    subtotal: float
    discounts_applied: list[dict[str, Any]]
    total_price: float
    order_id: str | None
    created_at: str | None
    saved_as: str | None

# ----------------------------
# Example menu data (your original)
# ----------------------------
data: MenuData = {
    "categories": [
        {"id": 1, "name": "Burgers"},
        {"id": 2, "name": "Sides"},
        {"id": 3, "name": "Drinks"}
    ],
    "items": [
        {"id": 1, "name": "Cheeseburger", "price": 5.99, "category_id": 1},
        {"id": 2, "name": "Veggie Burger", "price": 5.49, "category_id": 1},
        {"id": 3, "name": "French Fries", "price": 2.99, "category_id": 2},
        {"id": 4, "name": "Coca-Cola", "price": 1.49, "category_id": 3},
        {"id": 5, "name": "Orange Juice", "price": 1.99, "category_id": 3}
    ],
    "discounts": [
        {"discount_percentage": 0.10, "item_ids": [2, 4]},
        {"discount_percentage": 0.05, "item_ids": [3]},
        {"discount_percentage": 0.20, "item_ids": [5]}
    ]
}
MENU_DATA = data

# ----------------------------
# Helpers & global cart
# ----------------------------
customer_order: list[CartItem] = []  # global cart

def _get_item(item_id: int) -> MenuItem:
    """Raise StopIteration if item not found (same as your original helper)."""
    return next(i for i in data["items"] if i["id"] == item_id)

def _get_item_discount(item_id: int) -> float:
    """Return discount as decimal, e.g. 0.1 for 10%."""
    return next((d["discount_percentage"] for d in data["discounts"] if item_id in d["item_ids"]), 0.0)

# ----------------------------
# Public functions with modern-type hints
# ----------------------------
def get_menu(category_ids: list[int] | None = None) -> MenuData:
    """
    Return the menu. If category_ids is provided, return the filtered menu
    only containing those categories, their items, and applicable discounts.

    Input:
      category_ids: Optional[List[int]]; e.g. [1, 3]

    Output:
      MenuData (dict with categories, items, discounts)
    """
    if not category_ids:
        return data

    filtered_categories = [cat for cat in data["categories"] if cat["id"] in category_ids]
    filtered_category_ids = {cat["id"] for cat in filtered_categories}
    filtered_items = [item for item in data["items"] if item["category_id"] in filtered_category_ids]
    filtered_item_ids = {item["id"] for item in filtered_items}

    filtered_discounts: list[Discount] = []
    for disc in data["discounts"]:
        valid_item_ids = [iid for iid in disc["item_ids"] if iid in filtered_item_ids]
        if valid_item_ids:
            filtered_discounts.append({
                "discount_percentage": disc["discount_percentage"],
                "item_ids": valid_item_ids
            })

    return {
        "categories": filtered_categories,
        "items": filtered_items,
        "discounts": filtered_discounts
    }

def get_customer_order() -> list[CartItem]:
    """
    Return the current cart as a list of CartItem (or an empty list).
    """
    return customer_order.copy()

def increase_order_items(items: list[ItemInput]) -> dict[str, Any]:
    """
    Increase the quantity of given items; add items that don't exist.

    Input:
      items: List[ItemInput] e.g. [{"item_id": 1, "quantity": 2}, ...]

    Behavior notes:
      - quantity should be an integer (can be 0 or negative; validation not enforced here).
      - existing cart item quantities are incremented by the provided quantity.
      - if item not in cart, it's appended with the provided quantity.
    """
    if not items:
        return {"error": "Cannot add empty list of items."}

    global customer_order

    for new_item in items:
        # shallow validation (helpful for other devs)
        if not isinstance(new_item.get("item_id"), int) or not isinstance(new_item.get("quantity"), int):
            return {"error": f"Invalid item shape: {new_item}"}

        found = False
        for cart_item in customer_order:
            if cart_item["item_id"] == new_item["item_id"]:
                cart_item["quantity"] += new_item["quantity"]
                found = True
                break
        if not found:
            customer_order.append(new_item.copy())

    return {"message": "Items added successfully.", "current_customer_order": customer_order.copy()}

def remove_order_items(item_ids: list[int]) -> dict[str, Any]:
    """
    Remove items from the customer order completely.

    Input:
      item_ids: List[int] e.g. [1, 2]

    Output:
      dict with message and current_customer_order
    """
    if not item_ids:
        return {"error": "No item ids provided."}

    global customer_order
    customer_order = [item for item in customer_order if item["item_id"] not in item_ids]

    return {"message": "Items removed successfully.", "current_customer_order": customer_order.copy()}

def set_order_items(items: list[ItemInput]) -> dict[str, Any]:
    """
    Set the exact quantity of each given item; add if it doesn't exist.

    Input:
      items: List[ItemInput] e.g. [{"item_id": 1, "quantity": 5}, {"item_id": 2, "quantity": 0}]

    Behavior:
      - If item exists, quantity will be overwritten with the provided value.
      - If item does not exist, it will be added with given quantity.
      - If you want items with quantity 0 to be removed, call `remove_order_items` after this.
    """
    if not items:
        return {"error": "No valid items provided."}

    global customer_order

    for new_item in items:
        if not isinstance(new_item.get("item_id"), int) or not isinstance(new_item.get("quantity"), int):
            return {"error": f"Invalid item shape: {new_item}"}

        found = False
        for cart_item in customer_order:
            if cart_item["item_id"] == new_item["item_id"]:
                cart_item["quantity"] = new_item["quantity"]
                found = True
                break
        if not found:
            customer_order.append(new_item.copy())

    return {"message": "Items set successfully.", "current_customer_order": customer_order.copy()}

def calculate_total() -> TotalsResult:
    """
    Read-only preview using the global cart.
    Returns a dict with order_details, subtotal, discounts_applied, total_price.
    """
    global customer_order

    subtotal = 0.0
    total = 0.0
    order_details: list[TotalsLine] = []
    discounts_applied: list[dict[str, Any]] = []

    for ord_item in customer_order:
        item_id = ord_item["item_id"]
        qty = int(ord_item["quantity"])
        item = _get_item(item_id)
        price = float(item["price"])
        discount = float(_get_item_discount(item_id))

        line_total = round(price * qty, 2)
        line_discount_amount = round(price * qty * discount, 2)

        subtotal += line_total
        total += (line_total - line_discount_amount)

        order_details.append({
            "item_id": item_id,
            "name": item["name"],
            "quantity": qty,
            "price_per_item": price,
            "line_total": round(line_total, 2),
            "line_discount": round(line_discount_amount, 2),
            "discount_pct": discount
        })

        if line_discount_amount > 0:
            discounts_applied.append({
                "description": f"{int(discount*100)}% off on {item['name']}",
                "amount": -round(line_discount_amount, 2)
            })

    subtotal = round(subtotal, 2)
    total = round(total, 2)

    return {
        "order_details": order_details,
        "subtotal": subtotal,
        "discounts_applied": discounts_applied,
        "total_price": total,
        "order_id": None,
        "created_at": None,
        "saved_as": None
    }

def create_order() -> TotalsResult:
    """
    Finalize & save order using pandas. Uses global cart.
    Returns order metadata + same totals. Clears cart after saving.
    """
    global customer_order

    if not customer_order:
        return {"error": "Customer order is empty. Add some items first."}  # type: ignore

    totals = calculate_total()

    now = datetime.datetime.now()
    order_id = uuid.uuid4().hex
    created_at = now.isoformat()

    rows: list[dict[str, Any]] = []
    for det in totals["order_details"]:
        rows.append({
            "order_id": order_id,
            "created_at": created_at,
            "item_id": det["item_id"],
            "item_name": det["name"],
            "quantity": det["quantity"],
            "price_per_item": det["price_per_item"],
            "discount_pct": det["discount_pct"],
            "line_total": det["line_total"],
            "line_discount": det["line_discount"],
            "line_after_discount": round(det["line_total"] - det["line_discount"], 2)
        })

    df = pd.DataFrame(rows)
    saved_folder = "orders"
    os.makedirs(saved_folder, exist_ok=True)
    filename = os.path.join(saved_folder, f"order_{order_id}.csv")
    df.to_csv(filename, index=False, encoding="utf-8")

    totals["order_id"] = order_id
    totals["created_at"] = created_at
    totals["saved_as"] = filename

    # Clear the cart after successful save
    customer_order = []

    return totals

# ----------------------------
# Quick usage examples
# ----------------------------
# increase_order_items([{"item_id": 1, "quantity": 2}])
# set_order_items([{"item_id": 1, "quantity": 5}])
# remove_order_items([1, 2])
# calculate_total()
# create_order()
