import util
import sys
import argparse
import numpy as np
import time
from timeit import default_timer as timer
np.set_printoptions(threshold=sys.maxsize)

def calc_accuracy(weights, images, labels, num_classes, image_width, image_height):
	true_positive_count = 0
	for index in range(0,len(images)):
		curr_image 			= images[index]
		curr_ground_truth	= labels[index]
		curr_image = curr_image.flatten()
		features = np.append(curr_image,1)
		features = features.reshape(1,image_width*image_height+1)

		predictions = []

		for inner_index in range(0,num_classes):
			product = np.inner(features, weights[inner_index])
			predictions.append(product)

		prediction_with_highest_confidence = np.argmax(predictions)
		print(prediction_with_highest_confidence, curr_ground_truth)
		if(prediction_with_highest_confidence == curr_ground_truth):
			true_positive_count += 1

	accuracy = true_positive_count*100/float(len(images))
	return accuracy

def train_perceptron(training_images, training_labels, validation_images, validation_labels, num_classes, num_epochs, image_width, image_height):
	learning_rate = 0.01
	weights = np.random.rand(num_classes,image_width*image_height+1)
	# weights = np.random.uniform(-0.05,0.05,(num_classes,image_width*image_height+1)) - better for Face classification
	best_weights = weights
	best_accuracy = -1
	best_epoch = -1
	for epoch in range(0,num_epochs):
		print ("Running on Epoch %d/%d" %(epoch+1,num_epochs) ,end="\r")
		# time.sleep(1)
		true_positive_count = 0
		for index in range(0,len(training_labels)):
			curr_image 			= training_images[index]
			curr_ground_truth	= training_labels[index]
			curr_image = curr_image.flatten() #normalizing
			features = np.append(curr_image,1) #adding bias as the last element, considering each pixel as a feature
			features = features.reshape(1,image_width*image_height+1)

			predictions = []

			for inner_index in range(0,num_classes):
				product = np.inner(features, weights[inner_index])
				predictions.append(product)
				if(product >= 0) and inner_index != curr_ground_truth:
					weights[inner_index] = weights[inner_index] - (learning_rate * features)
				if(product < 0) and inner_index == curr_ground_truth:
					weights[inner_index] = weights[inner_index] + (learning_rate * features)

			prediction_with_highest_confidence = np.argmax(predictions)
			if(prediction_with_highest_confidence == curr_ground_truth):
				true_positive_count += 1
		val_accuracy = calc_accuracy(weights, validation_images, validation_labels, num_classes, image_width, image_height)
		if(val_accuracy > best_accuracy):
			best_epoch = epoch + 1
			best_accuracy = val_accuracy
			best_weights = weights
	np.save('best_weights.npy', best_weights)
	return best_weights

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
	parser.add_argument('--num_epochs', required=True, help='Number of epochs')
	parser.add_argument('--training_data_percentage', required=True, help='Percentage of Training Data')
	args = parser.parse_args()

	num_epochs = int(args.num_epochs)
	training_data_percentage = int(args.training_data_percentage)

	resize_width = int(args.image_resize_width)
	resize_height = int(args.image_resize_height)

	while(training_data_percentage <= 100):
		print("Training Percentage", training_data_percentage)
		for iteration in range(0,1):
			training_labels, indices = util.readLabels(args.training_label_path, training_data_percentage)
			training_images = util.readImages(args.training_data_path, len(training_labels), resize_width, resize_height, indices)
			num_classes = len(set(training_labels))

			validation_labels, indices = util.readLabels(args.validation_label_path, 100)
			validation_images = util.readImages(args.validation_data_path, len(validation_labels), resize_width, resize_height, indices)

			testing_labels, indices = util.readLabels(args.test_label_path, 100)
			testing_images = util.readImages(args.test_data_path, len(testing_labels), resize_width, resize_height, indices)

			# start = timer()
			# weights = train_perceptron(training_images, training_labels, validation_images, validation_labels, num_classes, num_epochs, resize_width, resize_height)
			# end = timer()
			# print(end - start)
			start = timer()
			weights = np.load('best_weights.npy')
			test_accuracy = calc_accuracy(weights, testing_images, testing_labels, num_classes, resize_width, resize_height)
			end = timer()
			print(end - start)
			print("Test-Set Acc %f" %(test_accuracy))
		# num_epochs -= 10
		training_data_percentage += 10
		# exit(1)

if __name__ == '__main__':
	main()
