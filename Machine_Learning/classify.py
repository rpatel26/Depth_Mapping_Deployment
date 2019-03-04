
def build_model(num_classes):
	print("building model")
	model = "atudd"

	return model

if __name__ == '__main__':

   	# TODO: get labels for each class and the total number 
    classes = [x[0] for x in os.walk(dataset_path)]
    num_classes = len(classes) - 1

    print("number of classes = ", num_classes)    
    for i in classes:
        print(i)
    
    model = build_model(num_classes)
    model.load_weights(model_path)

    # TODO: load data
   	image = cv2.imread(file_list[0])

	if image is None:
		# print("Image is of type None")
		continue

	print("File detected!!")
	print(file_list)

	image = cv2.resize(image, (32,32))
	image = np.expand_dims(image, axis = 0)

	# TODO: classify data
	predicted_values = model.predict(image) # sum of every element adds up to 1
	result = classes[np.argmax(predicted_values, axis = 1)[0] + 1] 
