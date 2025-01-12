import requests

res = requests.post('http://localhost:8080/', files={'file': open('./examples/07012025160748.pdf', 'rb')}).json()
print(res)