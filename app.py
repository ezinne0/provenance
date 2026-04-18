from dotenv import load_dotenv

load_dotenv()

from flask import Flask, render_template, request

from services import get_product_name, get_reviews, synthesize

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    url = (request.form.get("url") or "").strip()
    if not url:
        return render_template(
            "index.html",
            error_message="Please enter a URL.",
        )

    try:
        product = get_product_name(url)
        review_text = get_reviews(product["product_name"], product["brand"])
        synth = synthesize(review_text, product["product_name"], product["brand"])
    except ValueError as e:
        return render_template(
            "index.html",
            error_message=str(e),
            submitted_url=url,
        )
    except Exception:
        app.logger.exception("analyze pipeline failed")
        return render_template(
            "index.html",
            error_message="Something went wrong. Please try again.",
            submitted_url=url,
        )

    return render_template(
        "index.html",
        product_name=product["product_name"],
        brand=product["brand"],
        category=product["category"],
        summary=synth["summary"],
        durability=synth["durability"],
        keywords=synth["keywords"],
        trust_score=synth["trust_score"],
        submitted_url=url,
    )


if __name__ == "__main__":
    app.run(debug=True)
