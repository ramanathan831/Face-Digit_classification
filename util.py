import numpy as np
from PIL import Image
from random import shuffle

def resize_image(curr_image, resize_width, resize_height):
	im = Image.fromarray((curr_image).astype(np.uint8))
	resized_img = im.resize((resize_width, resize_height), Image.ANTIALIAS)
	curr_image = np.array(resized_img)
	return curr_image

def readImages(filename, number_of_data_points, resize_width, resize_height, indices):
	data_file = open(filename, "r")
	line = data_file.readline()
	line_num = 0
	image_array = []
	single_image_array = []
	indices_length = len(indices)

	while line:
		line = line.replace(" ","0").replace("#","1").replace("+","1").strip()
		line = list(map(int, line))

		if(1 in line):
			single_image_array.append(line)

		single_image_array.append(line)

		line_num += 1
		if(line_num % 28 == 0):
			arr = np.array(single_image_array)
			resized_img = resize_image(arr, resize_width, resize_height)
			image_array.append(resized_img)
			single_image_array = []
		line = data_file.readline()

	data_file.close()

	if len(indices) > 0:
		shuffled_image_array = []
		for value in indices:
			shuffled_image_array.append(image_array[value])
		return shuffled_image_array
	else:
		return image_array

def readLabels(filename, percentage):
	label_file = open(filename, "r")
	line = label_file.readline()
	labels = []

	shuffle_labels = [[]*10 for _ in range(0,10)]
	prior_count = [0]*10

	label_index = 0
	while line:
		label = int(line.strip())
		if(percentage != 100):
			shuffle_labels[label].append(label_index)
			prior_count[label] += 1
		label_index += 1
		labels.append(label)
		line = label_file.readline()
	label_file.close()

	indices = []
	if(percentage != 100):
		trim_labels = []
		for index in range(0,10):
			shuffle(shuffle_labels[index])
			shuffle_labels[index] = shuffle_labels[index][:int(len(shuffle_labels[index])*percentage/float(100))]
		for index in range(0,10):
			for label_index in shuffle_labels[index]:
				indices.append(label_index)
		shuffle(indices)
		for index in indices:
			trim_labels.append(labels[index])
		return trim_labels, indices
	return labels, indices
