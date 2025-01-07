import requests
resp=requests.post("http://localhost:5000/predict",files={'file': open('C:\\Users\\mtaha\\Desktop\\ml model for breast cancer detection\\models\\test\\Screenshot 2024-12-23 153020.png','rb')})
print(resp.text)