TOOLS = [
    {
        "name": "get_menu",
        "description": (
            "Return the menu. Optional input: category_ids (list of ints). "
            "If category_ids is provided, only categories/items/discounts for those IDs are returned. "
            "Returns a dict with keys: categories, items, discounts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional list of category ids to filter the menu by."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_customer_order",
        "description": (
            "Return the current customer cart as a list of objects: "
            "[{\"item_id\": int, \"quantity\": int}, ...]. No parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "increase_order_items",
        "description": (
            "Increase quantities of items in the cart or add them if missing. "
            "Input: items (array of {item_id:int, quantity:int}). "
            "Returns a dict: {message, current_customer_order}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "integer"},
                            "quantity": {"type": "integer"}
                        },
                        "required": ["item_id", "quantity"]
                    },
                    "description": "List of items to increase (item_id + quantity)."
                }
            },
            "required": ["items"]
        }
    },
    {
        "name": "remove_order_items",
        "description": (
            "Remove items from the cart completely. Input: item_ids (array of ints). "
            "Returns {message, current_customer_order}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of item_id integers to remove from the cart."
                }
            },
            "required": ["item_ids"]
        }
    },
    {
        "name": "set_order_items",
        "description": (
            "Set exact quantities for given items (overwrite or add). "
            "Input: items (array of {item_id:int, quantity:int}). "
            "Returns {message, current_customer_order}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "integer"},
                            "quantity": {"type": "integer"}
                        },
                        "required": ["item_id", "quantity"]
                    },
                    "description": "List of items to set (item_id + desired quantity)."
                }
            },
            "required": ["items"]
        }
    },
    {
        "name": "calculate_total",
        "description": (
            "Return read-only computed totals based on current cart: "
            "order_details (list), subtotal, discounts_applied, total_price. No parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "create_order",
        "description": (
            "Finalize & save current order to CSV, return totals plus metadata: "
            "{order_id, created_at, saved_as, order_details, subtotal, total_price}. No parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

TOOLS_FULL = []
for tool in TOOLS:
    TOOLS_FULL.append({"type": "function"})
    TOOLS_FULL[-1]["function"] = tool
