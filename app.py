import os

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, redirect, render_template, request, session, url_for

from services import get_product_image, get_product_name, get_reviews, synthesize

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-insecure-change-for-production")

# One-time payload after POST /analyze so GET / can show results once; reload clears session.
_SESSION_VIEW_KEY = "_provenance_view"

@app.route("/")
def index():
    view = session.pop(_SESSION_VIEW_KEY, None)
    if view:
        return render_template("index.html", **view)
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    url = (request.form.get("url") or "").strip()
    if not url:
        session[_SESSION_VIEW_KEY] = {"error_message": "Please enter a URL."}
        return redirect(url_for("index"), code=303)

    try:
        product = get_product_name(url)
        try:
            product_image = get_product_image(
                product["product_name"], product["brand"]
            )
        except ValueError:
            product_image = None
        review_text = get_reviews(product["product_name"], product["brand"])
        synth = synthesize(review_text, product["product_name"], product["brand"])
    except ValueError as e:
        session[_SESSION_VIEW_KEY] = {
            "error_message": str(e),
            "submitted_url": url,
        }
        return redirect(url_for("index"), code=303)
    except Exception:
        app.logger.exception("analyze pipeline failed")
        session[_SESSION_VIEW_KEY] = {
            "error_message": "Something went wrong. Please try again.",
            "submitted_url": url,
        }
        return redirect(url_for("index"), code=303)

    session[_SESSION_VIEW_KEY] = {
        "product_name": product["product_name"],
        "brand": product["brand"],
        "category": product["category"],
        "product_image": product_image,
        "star_rating": synth["star_rating"],
        "fit": synth["fit"],
        "durability": synth["durability"],
        "quality": synth["quality"],
        "keywords": synth["keywords"],
        "submitted_url": url,
    }
    return redirect(url_for("index"), code=303)


if __name__ == "__main__":
    app.run(debug=True)
