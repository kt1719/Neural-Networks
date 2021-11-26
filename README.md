Neural-Networks                                                                                                             

Machine Learning Coursework: Neural Networks
########################################################
Part 1 Create a neural network mini-library

LinearLayer Functions:
        * constructor: __init__(self, n_in, n_out) - Constructor of linear layer
        * forward(self, x) - Computes forward pass through the linear layer (outputs
        * backward(self, grad_z) - Given the loss function, performs back pass through the layer
        * update_params(self, learning_rate) - Performs one step of gradient descent with given learning rate

MultiLayerNetwork Functions:
        * constructor: __init__(self, input_dim, neurons, activations) - Constructor of multi layer network, stores all object instances in self._layers list
        * forward(self, x) - Computes forward pass through all the linear layers and its corresponding activation functions
        * backward(self, grad_z) - Performs backward pass through all the linear layers and activation functions given the gradient loss function
        * update_params(self, learning_rate) - Performs one step of gradient descent on all the layers given the learning rate

Trainer Functions:
        * constructor: __init__(self, network, batch_size, nb_epoch, learning_rate, loss_fun, shuffle_flag) - constructor of the Trainer class
        * shuffle(input_dataset, target_dataset) - Shuffles both the input and it's corresponding target dataset and returns the shuffled version
        * train(self, input_dataset, target_dataset) - Trains the multilayer class instance using the input_dataset and target_dataset. Performs the operations using the specified epochs, batch siz$
        * eval_loss(self, input_dataset, target_dataset) - Function used to evaluate the loss function for the given data.

Preprocessor Functions:
        * constructor __init__(self, data): Constructor for Preprocessor class. Pre stores the min data and max data
        * apply(self, data) - Normalizes the dataset using min-max normalization
        * rever(self, data) - Un-normalizes the dataset to retrieve the original one


########################################################
Part 2: Create and train a neural network for regression

Layers Class: A class used to store all the linear layer instances and activations functions into one "torch.nn.Sequential" object.
Layers Functions:
        * constructor: __init__(self, n_input_vars, activation, num_of_neurons=[8,4], n_output_vars=1) - used to create all the linear layer and activation function instances
        * forward(self, x) - used to compute a forward pass through all the layers

Regressor Class: A class for the regressor
Regressor Functions:
        * __init__(self, x, learning_rate=0.1, optimiser='Adam', num_of_neurons=[8,4], nb_epoch = 1000, activation="LeakyReLU")
                - Constructor for the Regressor Class. The parameters are used for the neural net and can be changed to try out different hyperparameters if needed
        * _preprocessor(self, x, y = None, training = False) - Preprocesses the data to be used for either training or testing purposes
        * fit(self, x, y) - Used to train the regressor, used only for training the regressor
        * predict(self, x) - Computes the corresponding values to an input data "x"
        * score(self, x, y) - Function to evaluate the model accuracy on a validation dataset

Other Functions:
        * split_data(x,y,k_folds=None) - Used shuffle the data, split it into k_folds for training and validation purposes, and returns k_folds of x and. If no value is specified for k_folds, it wi$
        *cross_validation(x,y,learning_rate, optimizer, sub_arr, no_of_epochs,k_folds=3, activation=""): This function is used for cross validation. It calls split_data to split inputs x and y into$
        * RegressorHyperParameterSearch(x, y, no_of_epochs=10, optimizer_types=['Adam', 'AdaDelta'], neuron_lists=[15,8,4], activation_functions=['LeakyReLU', 'ReLU', 'eLU'])
                - Function for searching the best hyperparameter values. Automatically computes the model with the lowest validation error looping through all the different combinations of hyperpar$
        * plot_error(model, x_train, y_train, x_val, y_val, num_epochs)
                - Used to plot the RMSE of both the training and validation datasets for a given model in relation to the number of epochs

