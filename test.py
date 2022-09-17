
from skimage import feature
import cv2
import os
from sklearn.svm import LinearSVC
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist

desc = LocalBinaryPatterns(24, 8)
data = []
labels=[]

# loop over the training images
for imagePath in os.listdir("positive-image"):
	image = cv2.imread("./positive-image/" + imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)# lbp array
	data.append(hist)#data->list
	labels.append(imagePath.split('.') [0])#to know image

# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)# machinelearing model support vector classification
model.fit(data, labels)

# # loop over the testing images
# for imagePath in paths.list_images(args["testing"]):
# 	# load the image, convert it to grayscale, describe it,
# 	# and classify it
image = cv2.imread("Testimage/mouse.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = desc.describe(gray)
prediction = model.predict(hist.reshape(1,-1))
print(prediction)

	# # display the image and the prediction
	# cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	# 	1.0, (0, 0, 255), 3)
	# cv2.imshow("Image", image)
	# cv2.waitKey(0)