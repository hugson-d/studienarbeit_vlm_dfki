# WatchStore Web App

A simple Flask web application for selling watches.

## Features

- **Product listing** with category filtering (Classic, Luxury, Smart, Sport)
- **Product detail page** with description, rating and related products
- **Shopping cart** with quantity management
- **Checkout form** with shipping and payment fields
- **Order confirmation** page

## Running the App

### 1. Install dependencies

```bash
pip install flask
# or with uv from the project root:
uv sync
```

### 2. Start the server

```bash
cd watch_store
python app.py
```

The app will be available at <http://localhost:5000>.

## Project Structure

```
watch_store/
├── app.py                   # Flask application & routes
├── models.py                # Watch product data
├── templates/
│   ├── base.html            # Base layout
│   ├── index.html           # Product listing
│   ├── product.html         # Product detail
│   ├── cart.html            # Shopping cart
│   ├── checkout.html        # Checkout form
│   └── order_confirmation.html
└── static/
    ├── css/style.css        # Styles
    └── js/cart.js           # Cart badge & input formatting
```
