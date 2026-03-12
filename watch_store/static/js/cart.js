// Update cart badge dynamically after add-to-cart actions
async function refreshCartBadge() {
    try {
        const response = await fetch('/api/cart/count');
        if (response.ok) {
            const data = await response.json();
            const badge = document.getElementById('cart-badge');
            if (badge) {
                badge.textContent = data.count;
                badge.style.display = data.count > 0 ? 'inline-flex' : 'none';
            }
        }
    } catch (_) {
        // Silently ignore network errors; badge will update on next page load.
    }
}

// Auto-format card number input with spaces
const cardInput = document.getElementById('card_number');
if (cardInput) {
    cardInput.addEventListener('input', function () {
        let value = this.value.replace(/\D/g, '').substring(0, 16);
        this.value = value.replace(/(.{4})/g, '$1 ').trim();
    });
}

// Auto-format expiry date
const expiryInput = document.getElementById('expiry');
if (expiryInput) {
    expiryInput.addEventListener('input', function () {
        let value = this.value.replace(/\D/g, '').substring(0, 4);
        if (value.length >= 3) {
            value = value.substring(0, 2) + '/' + value.substring(2);
        }
        this.value = value;
    });
}

document.addEventListener('DOMContentLoaded', refreshCartBadge);
