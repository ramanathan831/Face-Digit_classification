import util
import sys
import argparse
import numpy as np
import time
import copy
import math
from timeit import default_timer as timer
np.set_printoptions(threshold=sys.maxsize)

def calc_accuracy(feature_matrix, prior_prob, images, labels, num_classes):
	true_positive_count = 0
	for outer_index in range(0, len(labels)):
		curr_image = images[outer_index]
		curr_ground_truth = labels[outer_index]

		prob = [0]*num_classes
		for index in range(0,num_classes):
			training_samples_prob = 0
			prob[index] += prior_prob[index]
			for row in range(0,images[0].shape[0]):
				for col in range(0,images[0].shape[1]):
					if(curr_image[row][col] == 0):
						ind_prob = feature_matrix[index][0][row][col]
					elif(curr_image[row][col] == 1):
						ind_prob = feature_matrix[index][1][row][col]
					if(ind_prob == 0):
						ind_prob = 0.000000001
					prob[index] += math.log(ind_prob)
		if(curr_ground_truth == np.argmax(prob)):
			true_positive_count += 1

	accuracy = true_positive_count*100/float(len(images))
	return accuracy

def fill_feature_matrix(images, labels, num_classes, gt_count):
	dictionary = {}
	empty_list = [[0]*images[0].shape[0] for _ in range(images[0].shape[1])]
	inner_dict = {}
	inner_dict[0] = copy.deepcopy(empty_list)
	inner_dict[1] = copy.deepcopy(empty_list)

	for index in range(0, num_classes):
		dictionary[index] = copy.deepcopy(inner_dict)

	for index in range(0, len(labels)):
		summation_term = 1/float(gt_count[labels[index]])
		curr_image = images[index]
		for row in range(0,images[0].shape[0]):
			for col in range(0,images[0].shape[1]):
				if(curr_image[row][col] == 0):
					dictionary[labels[index]][0][row][col] += summation_term
				elif(curr_image[row][col] == 1):
					dictionary[labels[index]][1][row][col] += summation_term
	return dictionary

def core_naive_bayes(training_images, training_labels, num_classes):
	gt_count = []
	prior_prob = []
	for index in range(0, num_classes):
		gt_count.append(0)
	for index in range(0,len(training_labels)):
		gt_count[training_labels[index]] += 1
	for index in range(0, num_classes):
		prior_prob.append(gt_count[index]/len(training_labels))
	feature_matrix = fill_feature_matrix(training_images, training_labels, num_classes, gt_count)
	return feature_matrix, prior_prob

def main():
	parser = argparse.ArgumentParser(description='Digit Classification using Perceptron')
	parser.add_argument('--image_resize_width', required=True, help='Resize Width')
	parser.add_argument('--image_resize_height', required=True, help='Resize Height')
	parser.add_argument('--training_data_path', required=True, help='Path to training data')
	parser.add_argument('--training_label_path', required=True, help='Path to training data')
	parser.add_argument('--validation_data_path', required=True, help='Path to validation data')
	parser.add_argument('--validation_label_path', required=True, help='Path to validation data')
	parser.add_argument('--test_data_path', required=True, help='Path to Testing data')
	parser.add_argument('--test_label_path', required=True, help='Path to Testing data')
	parser.add_argument('--training_data_percentage', required=True, help='Percentage of Training Data')
	args = parser.parse_args()

	training_data_percentage = int(args.training_data_percentage)
	resize_width = int(args.image_resize_width)
	resize_height = int(args.image_resize_height)

	# training_data_percentage = 10
	while(training_data_percentage <= 100):
		print("Training Percentage", training_data_percentage)
		if(training_data_percentage == 100):
			end_itr = 1
		else:
			end_itr = 10
		for iteration in range(0,end_itr):
			training_labels, indices = util.readLabels(args.training_label_path, training_data_percentage)
			training_images = util.readImages(args.training_data_path, len(training_labels), resize_width, resize_height, indices)
			num_classes = len(set(training_labels))

			validation_labels, indices = util.readLabels(args.validation_label_path, 100)
			validation_images = util.readImages(args.validation_data_path, len(validation_labels), resize_width, resize_height, indices)

			testing_labels, indices = util.readLabels(args.test_label_path, 100)
			testing_images = util.readImages(args.test_data_path, len(testing_labels), resize_width, resize_height, indices)
			start = timer()
			feature_matrix, prior_prob = core_naive_bayes(training_images, training_labels, num_classes)
			end = timer()
			print(end - start)
			start = timer()
			test_accuracy = calc_accuracy(feature_matrix, prior_prob, testing_images, testing_labels, num_classes)
			end = timer()
			print(end - start)
			print(test_accuracy)
		training_data_percentage += 10
		# exit(1)
if __name__ == '__main__':
	main()
