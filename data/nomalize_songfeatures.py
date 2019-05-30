import numpy as np

"""
Loads the songfeatures for a specific Spotify URI.
"""
def normalize_songfeatures(songfeatures_path, output_path):
	songfeatures = []
	uris = []

	with open(songfeatures_path, "r") as f:
		for x in f.readlines():
			a = x.strip().split(',')                
			songfeatures.append(np.array(a[2:], dtype=np.float))
			uris.append(a[:2])
		
	songfeatures_array = np.array(songfeatures)

	# Set mean to 0.
	songfeatures_array = songfeatures_array - np.mean(songfeatures, axis=0)
	# Normalize std to 1.
	songfeatures_array = 4 * songfeatures_array / np.std(songfeatures_array, axis=0)
	# Center at 0.5.
	songfeatures_array = 0.5 + songfeatures_array / 2
	# Clip for all x to be in 0 <= x <= 1
	songfeatures_array = np.minimum(1, np.maximum(0, songfeatures_array))

	with open(output_path, "w+") as f:
		for i, uri in enumerate(uris):
			f.write(",".join(uri) + "," + ",".join(map(str, songfeatures_array[i].tolist())) + "\n")

normalize_songfeatures("data\\songfeatures_sigmoid.csv", "data\\normalized_sigmoid.csv")