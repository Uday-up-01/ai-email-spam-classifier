from app import app

client = app.test_client()

response = client.get("/")

html = response.get_data(as_text=True)

print("HTTP Status:", response.status_code)
print("Email textarea:", "<textarea" in html)
print("Check Email button:", "Check Email" in html)
print("File upload:", 'type="file"' in html)
print("Batch form:", 'action="/batch"' in html)