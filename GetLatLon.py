import requests

def get_user_location():
    try:
        response = requests.get('https://ipinfo.io')
        data = response.json()
        location = data.get('loc')
        if location:
            latitude, longitude = location.split(',')
            return float(latitude), float(longitude)
        else:
            return None, None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None

latitude, longitude = get_user_location()
print(f"Latitude: {latitude}, Longitude: {longitude}")
