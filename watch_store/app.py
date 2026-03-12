"""Flask web application for the watch store."""

import os
import secrets

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

from models import get_watch_by_id, get_watches_by_category, get_categories

app = Flask(__name__)
app.secret_key = os.environ.get("WATCH_STORE_SECRET_KEY", secrets.token_hex(32))


def get_cart():
    return session.get("cart", {})


def save_cart(cart):
    session["cart"] = cart
    session.modified = True


def cart_item_count():
    cart = get_cart()
    return sum(item["quantity"] for item in cart.values())


def cart_total():
    cart = get_cart()
    return sum(item["price"] * item["quantity"] for item in cart.values())


app.jinja_env.globals["cart_item_count"] = cart_item_count


@app.route("/")
def index():
    category = request.args.get("category", "All")
    watches = get_watches_by_category(category)
    categories = get_categories()
    return render_template("index.html", watches=watches, categories=categories, active_category=category)


@app.route("/product/<int:watch_id>")
def product(watch_id):
    watch = get_watch_by_id(watch_id)
    if watch is None:
        flash("Product not found.", "error")
        return redirect(url_for("index"))
    related = [w for w in get_watches_by_category(watch.category) if w.id != watch_id][:3]
    return render_template("product.html", watch=watch, related=related)


@app.route("/cart")
def cart():
    cart_data = get_cart()
    items = []
    for watch_id, item in cart_data.items():
        items.append({
            "id": int(watch_id),
            "name": item["name"],
            "brand": item["brand"],
            "price": item["price"],
            "quantity": item["quantity"],
            "subtotal": item["price"] * item["quantity"],
            "image_url": item["image_url"],
        })
    total = cart_total()
    return render_template("cart.html", items=items, total=total)


@app.route("/cart/add", methods=["POST"])
def add_to_cart():
    watch_id = request.form.get("watch_id", type=int)
    quantity = request.form.get("quantity", 1, type=int)
    watch = get_watch_by_id(watch_id)
    if watch is None:
        flash("Product not found.", "error")
        return redirect(url_for("index"))

    cart = get_cart()
    key = str(watch_id)
    if key in cart:
        cart[key]["quantity"] += quantity
    else:
        cart[key] = {
            "name": watch.name,
            "brand": watch.brand,
            "price": watch.price,
            "quantity": quantity,
            "image_url": watch.image_url,
        }
    save_cart(cart)
    flash(f'"{watch.name}" added to your cart.', "success")
    return redirect(request.referrer or url_for("index"))


@app.route("/cart/update", methods=["POST"])
def update_cart():
    watch_id = str(request.form.get("watch_id", type=int))
    quantity = request.form.get("quantity", type=int)
    cart = get_cart()
    if watch_id in cart:
        if quantity and quantity > 0:
            cart[watch_id]["quantity"] = quantity
        else:
            del cart[watch_id]
    save_cart(cart)
    return redirect(url_for("cart"))


@app.route("/cart/remove", methods=["POST"])
def remove_from_cart():
    watch_id = str(request.form.get("watch_id", type=int))
    cart = get_cart()
    cart.pop(watch_id, None)
    save_cart(cart)
    flash("Item removed from cart.", "info")
    return redirect(url_for("cart"))


@app.route("/checkout", methods=["GET", "POST"])
def checkout():
    cart_data = get_cart()
    if not cart_data:
        flash("Your cart is empty.", "info")
        return redirect(url_for("index"))

    items = []
    for watch_id, item in cart_data.items():
        items.append({
            "id": int(watch_id),
            "name": item["name"],
            "brand": item["brand"],
            "price": item["price"],
            "quantity": item["quantity"],
            "subtotal": item["price"] * item["quantity"],
        })
    total = cart_total()

    if request.method == "POST":
        # Validate required fields
        required_fields = ["first_name", "last_name", "email", "address", "city", "zip", "card_number"]
        errors = []
        for f in required_fields:
            if not request.form.get(f, "").strip():
                errors.append(f"{f.replace('_', ' ').title()} is required.")
        if errors:
            for error in errors:
                flash(error, "error")
            return render_template("checkout.html", items=items, total=total, form=request.form)

        # Simulate successful order placement
        order_ref = f"ORD-{secrets.token_hex(4).upper()}"
        save_cart({})
        flash(f"Order placed successfully! Your order reference is {order_ref}.", "success")
        return redirect(url_for("order_confirmation", ref=order_ref))

    return render_template("checkout.html", items=items, total=total, form={})


@app.route("/order-confirmation")
def order_confirmation():
    ref = request.args.get("ref", "N/A")
    return render_template("order_confirmation.html", ref=ref)


@app.route("/api/cart/count")
def api_cart_count():
    return jsonify({"count": cart_item_count()})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
