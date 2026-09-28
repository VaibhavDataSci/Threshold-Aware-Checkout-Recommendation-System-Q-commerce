"""
Checkout Threshold Logic Module.
Calculates cart total, remaining threshold gap to ₹150, threshold status,
and free-delivery unlock condition.
"""

FREE_DELIVERY_THRESHOLD = 150.0

def calculate_cart_total(cart_product_ids, products_df):
    """
    Calculates the total monetary value of items currently in the shopping cart.
    :param cart_product_ids: List of product_id strings in the cart.
    :param products_df: DataFrame containing product metadata with 'product_id' and 'price'.
    :return: Float cart total in INR.
    """
    if not cart_product_ids or products_df.empty:
        return 0.0

    # Filter products present in the cart
    cart_items = products_df[products_df['product_id'].isin(cart_product_ids)]
    if cart_items.empty:
        return 0.0

    # Count occurrences if duplicates exist in cart
    price_map = products_df.set_index('product_id')['price'].to_dict()
    total = sum(price_map.get(pid, 0.0) for pid in cart_product_ids if pid in price_map)
    return float(total)

def calculate_threshold_gap(cart_total, threshold=FREE_DELIVERY_THRESHOLD):
    """
    Calculates the remaining amount required to reach the free-delivery threshold.
    :param cart_total: Current cart value in INR.
    :param threshold: Fixed free-delivery threshold (default: 150.0).
    :return: Non-negative float remaining gap in INR.
    """
    gap = threshold - cart_total
    return max(0.0, float(gap))

def is_threshold_reached(cart_total, threshold=FREE_DELIVERY_THRESHOLD):
    """
    Checks if the cart total has met or exceeded the free-delivery threshold.
    :param cart_total: Current cart value in INR.
    :param threshold: Fixed free-delivery threshold (default: 150.0).
    :return: Boolean True if unlocked, False otherwise.
    """
    return float(cart_total) >= float(threshold)

def get_threshold_status(cart_total, threshold=FREE_DELIVERY_THRESHOLD):
    """
    Generates user-facing status message based on current cart total.
    :param cart_total: Current cart value in INR.
    :param threshold: Fixed free-delivery threshold (default: 150.0).
    :return: String status message.
    """
    if is_threshold_reached(cart_total, threshold=threshold):
        return "Free Delivery Unlocked"
    else:
        gap = calculate_threshold_gap(cart_total, threshold=threshold)
        return f"You are ₹{gap:.2f} away from free delivery."

def get_initial_cart(user_id, products_df, interactions_df):
    """
    Generates a deterministic initial cart for a given user.
    - Users U001-U099: Deterministic items totaling < ₹150 (approx ₹90-₹120) to demonstrate recommendation flow.
    - User U100: Deterministic items totaling >= ₹150 to demonstrate unlocked state immediately.
    """
    if interactions_df.empty or products_df.empty:
        return ["P001", "P009"]

    user_purchases = interactions_df[
        (interactions_df['user_id'] == user_id) & 
        (interactions_df['interaction'] == 'purchase')
    ]['product_id'].tolist()

    if not user_purchases:
        return ["P001", "P009"]

    price_map = products_df.set_index('product_id')['price'].to_dict()

    if user_id == "U100":
        cart = []
        total = 0.0
        for pid in user_purchases:
            cart.append(pid)
            total += price_map.get(pid, 0.0)
            if total >= 150.0:
                break
        return cart
    else:
        cart = []
        total = 0.0
        for pid in user_purchases:
            item_price = price_map.get(pid, 0.0)
            if total + item_price < 150.0:
                cart.append(pid)
                total += item_price
            if total >= 90.0 or len(cart) >= 3:
                break
        if not cart:
            cart = [user_purchases[0]]
        return cart

