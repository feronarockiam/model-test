import requests

url = 'http://localhost:5000/predict'


response = requests.post(url)
print(response)
if response.status_code == 200:
    predictions = response.json()
    print(predictions)
else:
    print(f"Error: {response.status_code}")
