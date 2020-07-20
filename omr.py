import sys
import cv2
import numpy as np


# Convolution function
# This function convolves a two-dimensional kernel H with a grayscale image I

def general_convolution(kernel, image):

	print('convolving')

	# kernel flipping for performing convolution
	np.flip(np.flip(kernel, axis = 1), axis = 0)

	# kernel height and width - usually we will deal with square kernels
	kh, kw = kernel.shape[0], kernel.shape[1]

	# padding size
	ph, pw = kh//2, kw//2

	# image height and width
	ih, iw = image.shape[0], image.shape[1]

	# create a new padded image
	temp = np.zeros(shape = (ih+2*ph, iw+2*pw))
	th, tw = temp.shape

	# print(temp.shape)

	temp[ph:th-ph, pw:tw-pw] = image
	
	# create a zeros-filled numpy array to store output of convolution
	result = np.zeros(shape = (image.shape[0], image.shape[1]))

	# loop over all pixels in image
	for i in range(ph, th-ph):
		for j in range(pw, tw-pw):
			# print((i, j))
			result[i-ph, j-pw] = np.sum(np.multiply(temp[i-ph:i+ph+1, j-pw:j+pw+1], kernel))

	return result


# Separable Convolution function
# This function convolves two column-vector kernels with a grayscale image

def separable_convolution(kernel_1, kernel_2, image):

	kh, kw = kernel_1.shape[1], kernel_2.shape[1]

	kernel_1 = np.transpose(kernel_1)

	# image height and width
	ih, iw = image.shape[0], image.shape[1]

	# padding size
	ph, pw = kh//2, kw//2

	# create a new padded image
	temp = np.zeros(shape = (ih+2*ph, iw+2*pw))
	th, tw = temp.shape
	temp[ph:th-ph, pw:tw-pw] = image

	# create a zeros-filled numpy array to store output of convolution
	result = np.zeros(shape = (temp.shape))
	result_2 = np.zeros(shape = (temp.shape))

	# convolving with kernel_1
	for i in range(ph, th-ph):
		for j in range(pw, tw-pw):
			# print((i, j))
			result[i, j] = np.sum(np.dot(temp[i-ph:i+ph+1, j], kernel_1))

	# convolving with kernel_2
	for i in range(ph, th-ph):
		for j in range(pw, tw-pw):
			# print((i, j))
			result_2[i, j] = np.sum(np.multiply(result[i, j-pw:j+pw+1], kernel_2))

	return result_2[ph:-ph, pw:-pw]


def convert_binary(img):
	img = img/255
	img[img > 0.5] = 1
	img[img < 0.5] = 0
	return img

def convert_binary2(img):
	img = img/255
	img[img > 0.5] = 1
	img[img < 0.5] = -1
	return img

# The hamming distance function
def hamming_distance(template, reference):

#	result = general_convolution(convert_binary(template), convert_binary(reference))
	result = general_convolution(convert_binary2(template), convert_binary2(reference))

	return result

## handle kernerls with even dimensions
def assert_dimensions_of_kernel(template):
	h, w = template.shape

	if h%2 == 0:
		temp = np.zeros(shape = (h+1, w))
		temp[0:h, :] = template
		template = temp

	h, w = template.shape

	if w%2 == 0:
		temp = np.zeros(shape = (h, w+1))
		temp[:, 0:w] = template
		template = temp

	return template

# gives coordinates of all the bounding box.
def plot_bounding_box(template, reference, threshold, color_palette):

	coords = []

	kh, kw = template.shape

	ih, iw = reference.shape

	# padding size
	ph, pw = kh//2, kw//2

	temp = np.zeros(shape = (ih+2*ph, iw+2*pw))
	th, tw = temp.shape
	temp[ph:th-ph, pw:tw-pw] = reference
	temp = np.array(temp, dtype = np.uint8)
	temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

	a = hamming_distance(template, reference)
	# print(reference.shape)
	print(temp.shape)

	# cv2.imshow('img_pre', temp)
	for i in range(ph, ih-ph):
		for j in range(pw, iw-pw):
			# print((i-ph, j-pw))
			if a[i-ph, j-pw] >= threshold:
				# print(a[i-ph, j-pw])
				# coords.append((( j-pw, i-ph+1), ( j+pw,i+ph+1), color_palette))
				coords.append((( j-2*pw, i-2*ph+1), ( j,i+1), color_palette))

				# cv2.rectangle(temp, ( j-pw, i-ph+1), ( j+pw,i+ph+1), color_pallete, 2)

	# cv2.imwrite(filename, temp)
	# cv2.imshow('image', temp)
	return coords


def smooth_image(image):

	# smooth filter

	im = cv2.GaussianBlur(image, (3, 3), 1)

	return im


# plot bounding box based on coordinates
def plot_from_coordinates(templates_list, reference, original, color_palettes_list,output_detected_file_name, threshold_list):

	all_coords = []

	for template, color_palette, th in zip(templates_list, color_palettes_list,threshold_list):

		threshold = th * np.max(hamming_distance(template, reference))

		all_coords.extend(plot_bounding_box(template, reference, threshold, color_palette))

	# print(all_coords)
	original = np.array(original, dtype = np.uint8)
	original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)


	for (a, b, c) in all_coords:

		cv2.rectangle(original, a, b, c, 1)

	cv2.imwrite(output_detected_file_name, original)


# to get edge map after applying sobel filter
def get_edge_map(image, threshold):

	# Defining Sobel Filter for edge detection

	sobel_filter_x_grad = np.array([[1, 2, 1]])
	sobel_filter_y_grad = np.array([[1, 0, -1]])


	#image = smooth_image(image)

	# convolving for X axis edges
	G_x = separable_convolution(sobel_filter_x_grad, sobel_filter_y_grad, image)
	G_y = separable_convolution(sobel_filter_y_grad, sobel_filter_x_grad, image) 

	G = np.sqrt(np.power(G_x, 2) + np.power(G_y, 2))

	G = G / np.max(G)

	G[G >= threshold] = 1
	G[G < threshold] = 0

	return G

# to get Hough voting space
def hough_transform(reference_em):#image_edge_map, rho_limit, theta_limit):

	# Hough Transform for voting space of coordinates defined as (row_index, speparation_between_staves)

	iem = reference_em


	ih, iw = iem.shape

	voting_space_rows = ih - 10
	voting_space_cols = int(ih / 10)

	voting_space = np.zeros(shape = (voting_space_rows, voting_space_cols))

	# perform voting
	# increment values for that set of coordinates which contribute to the lines passing through all pixels in the image

	for i in range(1, voting_space_rows-1):
		for j in range(iw):
			for sep in range(1, voting_space_cols):

				# condition: a set of all five staves is discovered

				if i+4*sep+1 < ih-1:

					# since edge detector is not supposed to be perfect we consider pixels at row indices x-1, x and x+1

					if (iem[i-1, j] == 1 or iem[i, j] == 1 or iem[i+1, j] == 1):
						
						if (iem[i+sep-1, j] == 1 or iem[i+sep, j] == 1 or iem[i+sep+1, j] == 1):

							if (iem[i+2*sep-1, j] == 1 or iem[i+2*sep, j] == 1 or iem[i+2*sep+1, j] == 1):

								if (iem[i+3*sep-1, j] == 1 or iem[i+3*sep, j] == 1 or iem[i+3*sep+1, j] == 1):

									if (iem[i+4*sep-1, j] == 1 or iem[i+4*sep, j] == 1 or iem[i+4*sep+1, j] == 1):

										voting_space[i, sep] += 1

	return voting_space

# plot lines detecting staves on image and return the distance between lines
def plot_hough_lines(reference, original, output_file_name):


	reference_em = get_edge_map(reference, 0.3)

	voting_space_2 = hough_transform(reference_em)

	voting_threshold = 0.7*np.max(voting_space_2)

	index_list, spacing_list = np.where((voting_space_2 > voting_threshold))# & (voting_space < 20))
	original = np.array(original, dtype = np.uint8)
	original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

	temp_ind = []
	space = 0
	for ind, spc in zip(list(index_list), list(spacing_list)):
		# considering only those lines where distance in line is above 3
		if spc > 3:
			# allowing next line only if it is more than 5 rows away from previous line. This reduces plottingg line closer to each other.
			if not temp_ind:
				temp_ind.append(ind)
				space = spc
				# plotting
				cv2.line(original, (0, ind), (original.shape[1]-1, ind), (0, 0, 255), 2)
				cv2.line(original, (0, ind+spc), (original.shape[1]-1, ind+spc), (0, 0, 255), 2)
				cv2.line(original, (0, ind+2*spc), (original.shape[1]-1, 2*spc+ind), (0, 0, 255), 2)
				cv2.line(original, (0, ind+3*spc), (original.shape[1]-1, 3*spc+ind), (0, 0, 255), 2)
				cv2.line(original, (0, ind+4*spc), (original.shape[1]-1, 4*spc+ind), (0, 0, 255), 2)

			else:

				if np.abs(ind - temp_ind[-1]) > 5:				
					temp_ind.append(ind)
					space = spc
					# plotting
					cv2.line(original, (0, ind), (original.shape[1]-1, ind), (0, 0, 255), 2)
					cv2.line(original, (0, ind+spc), (original.shape[1]-1, ind+spc), (0, 0, 255), 2)
					cv2.line(original, (0, ind+2*spc), (original.shape[1]-1, 2*spc+ind), (0, 0, 255), 2)
					cv2.line(original, (0, ind+3*spc), (original.shape[1]-1, 3*spc+ind), (0, 0, 255), 2)
					cv2.line(original, (0, ind+4*spc), (original.shape[1]-1, 4*spc+ind), (0, 0, 255), 2)


	cv2.imwrite(output_file_name, original)
	return space

def scale_templates(template_1, template_2, template_3,distance_in_lines):

	temp2_scale_ratio = template_2.shape[0]/template_1.shape[0]
	temp3_scale_ratio = template_3.shape[0]/template_1.shape[0] 

	template_1 = cv2.resize(template_1,(int(template_1.shape[1]),int(template_1.shape[0]* distance_in_lines/11)))

	template_2 = cv2.resize(template_2,(template_2.shape[1],int(template_1.shape[0] * temp2_scale_ratio )))

	template_3 = cv2.resize(template_3,(template_3.shape[1],int(template_1.shape[0] * temp3_scale_ratio)))

	template_1 = assert_dimensions_of_kernel(template_1)

	template_2 = assert_dimensions_of_kernel(template_2)

	template_3 = assert_dimensions_of_kernel(template_3)

	return [template_1, template_2, template_3]


#################################################################

# def couting_pixel_to_detect_lines :
# 	# A crude way to detect staves

# 	reference = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/rach.png', 0)

# 	reference_em = get_edge_map(reference)

# 	a = np.sum(reference_em == 1.0, axis = 1)

# 	indices = np.where(a > 1000)

# 	reference = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/rach.png', 0)
# 	reference = np.array(reference, dtype = np.uint8)
# 	reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)


# 	for i in indices[0]:
# 		cv2.line(reference, (0, i), (1274, i), (0, 0, 255), 2)

# 	cv2.imwrite('new.png', reference)







#output_file_name = '/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/hough_image2.png'





if __name__ == "__main__":
	if(len(sys.argv) != 2):
		raise Exception('Please provide 2 commandline arguments.')

	# img_name = '/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/music1.png'

	img_name = sys.argv[1] 
	reference = cv2.imread(img_name, 0)
	template_1 = cv2.imread(img_name.rsplit("/",1)[0] + '/template1.png', 0)
	template_2 = cv2.imread(img_name.rsplit("/",1)[0] + '/template2.png', 0)
	template_3 = cv2.imread(img_name.rsplit("/",1)[0] + '/template3.png', 0)

	filename = img_name.rsplit("/",1)[1]
	output_hough_file_name = img_name.rsplit("/",1)[0] + '/hough_output_image_smooth' + filename.split(".")[0][-1] + '.png'
	output_detected_file_name = img_name.rsplit("/",1)[0] + '/detected_output_image_smooth' + filename.split(".")[0][-1] + '.png'


	reference = smooth_image(reference)
	original = cv2.imread(img_name, 0)
	# template_1 = smooth_image(template_1)
	# template_2 = smooth_image(template_2)
	# template_3 = smooth_image(template_3)

	threshold_list = [0.8,0.8,0.9]
	distance_in_lines = plot_hough_lines(reference, original, output_hough_file_name)
	templates = scale_templates(template_1, template_2, template_3, distance_in_lines)	
	plot_from_coordinates(templates, reference, original, [(0, 0, 255), (0, 255, 0), (255, 0, 0)],output_detected_file_name,threshold_list)


	#reference = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/music2.png', 0)
	# template_1 = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/template1.png', 0)
	# #template_1 = assert_dimensions_of_kernel(template_1)
	# template_2 = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/template2.png', 0)
	# #template_2 = assert_dimensions_of_kernel(template_2)
	# template_3 = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/template3.png', 0)
	# #template_3 = assert_dimensions_of_kernel(template_3)






# reference = cv2.imread('/media/abhirag/Data/CSCI-B657-Computer_Vision/assignments/assignment_1/anagpure-shgaikwa-a1/test-images/music3.png', 0)
# reference = np.array(reference, dtype = np.uint8)
# reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)


