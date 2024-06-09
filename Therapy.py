import pandas as pd
import numpy as np

# Sample data with online availability
therapists = pd.DataFrame({
    'name': ['Therapist A', 'Therapist B', 'Therapist C'],
    'latitude': [37.7749, 37.8044, 37.7817],
    'longitude': [-122.4194, -122.2711, -122.4597],
    'specializations': [['anxiety', 'depression'], ['relationship issues', 'anxiety'], ['depression', 'stress']],
    'ratings': [4.5, 4.0, 4.8],
    'online_available': [True, False, True]
})

user_preferences = {
    'latitude': 37.7749,
    'longitude': -122.4194,
    'issues': ['anxiety'],
    'max_distance': 10,  # in kilometers
    'online_only': True  # User prefers online sessions
}
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance
# Calculate distances
therapists['distance'] = therapists.apply(lambda row: haversine(user_preferences['latitude'], user_preferences['longitude'], row['latitude'], row['longitude']), axis=1)

# Filter based on distance, specializations, and online availability
recommended_therapists = therapists[
    (therapists['distance'] <= user_preferences['max_distance']) &
    (therapists['specializations'].apply(lambda x: any(issue in x for issue in user_preferences['issues']))) &
    (therapists['online_available'] if user_preferences['online_only'] else True)
]

# Sort by ratings
recommended_therapists = recommended_therapists.sort_values(by='ratings', ascending=False)

print(recommended_therapists)
