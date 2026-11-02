from app import app
from flask import render_template

with app.test_request_context("/"):
    html = render_template(
        "index.html",
        result=None,
        batch_results=None
    )

    print("Textarea found:", "<textarea" in html)
    print("Unprocessed Jinja:", "{%" in html)
    print("Form found:", 'name="email"' in html)
    print("HTML length:", len(html))