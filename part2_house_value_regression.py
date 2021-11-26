from operator import ne
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt


class Layers(torch.nn.Module):
    def __init__(self, n_input_vars, activation, num_of_neurons=[8,4], n_output_vars=1):
        super().__init__() # call constructor of superclass
        hidden = []
        for i in range(len(num_of_neurons)):
            #prev_layer_output = self.hidden[i-1].input?
            if i==0:
                hidden.append(torch.nn.Linear(n_input_vars, num_of_neurons[i]))
                if activation == "LeakyReLU":
                    hidden.append(torch.nn.LeakyReLU(negative_slope=0.01))
                if activation == "ReLU":
                    hidden.append(torch.nn.ReLU())
                if activation == "ELU":
                    hidden.append(torch.nn.ELU())
                if activation == "Sigmoid":
                    hidden.append(torch.nn.Sigmoid())
                if activation == "Tanh":
                    hidden.append(torch.nn.Tanh())
            else:
                hidden.append(torch.nn.Linear(num_of_neurons[i-1], num_of_neurons[i]))
                if activation == "LeakyReLU":
                    hidden.append(torch.nn.LeakyReLU(negative_slope=0.01))
                if activation == "ReLU":
                    hidden.append(torch.nn.ReLU())
                if activation == "ELU":
                    hidden.append(torch.nn.ELU())
                if activation == "Sigmoid":
                    hidden.append(torch.nn.Sigmoid())
                if activation == "Tanh":
                    hidden.append(torch.nn.Tanh())
        hidden.append(torch.nn.Linear(num_of_neurons[-1], n_output_vars))

        self.model = torch.nn.Sequential(*hidden)
        
    def forward(self, x): 
        return self.model(x)

class Regressor():
    def __init__(self, x, learning_rate=0.1, optimiser='Adam', num_of_neurons=[8,4], nb_epoch = 1000, activation="LeakyReLU"):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.input_size = x.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.x_mu = None
        self.x_sigma = None
        self.y_mu = None
        self.y_sigma = None
        self.model = Layers(self.input_size, activation, num_of_neurons)
        self.loss_func = torch.nn.MSELoss()
        if optimiser=='Adam':
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimiser=='AdaDelta':
            self.optimiser = torch.optim.Adadelta(self.model.parameters(), lr=learning_rate)
        elif optimiser=='SGD':
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.optimiser.zero_grad()
        self.lb_style = LabelBinarizer()
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        # Encoding textual values

        if training:
            n = self.lb_style.fit_transform(x['ocean_proximity'].values)
            self.x_mu = x.mean(axis=0)
            # Fill missing values with mean,
            x = x.fillna(self.x_mu)
            self.x_sigma = x.std(axis=0)
            x = x-self.x_mu
            x_preprocessed = x.div(self.x_sigma)
            x_preprocessed.ocean_proximity = n

            self.y_mu = y.mean(axis=0)
            self.y_sigma = y.std(axis=0)
            y = (y-self.y_mu)
            y_preprocessed = y.div(self.y_sigma)
        else:
            n = self.lb_style.transform(x['ocean_proximity'].values)
            x = x.fillna(self.x_mu)
            x = (x-self.x_mu)
            x_preprocessed= x.div(self.x_sigma)
            x_preprocessed.ocean_proximity = n


        # Normalising numerical values 
        return torch.tensor(x_preprocessed.values), (torch.tensor(y_preprocessed.values) if training else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        X, Y = self._preprocessor(x, y, training = True) # Compute normalised values of x and y
        for i in range(self.nb_epoch):
            # (self, n_input_vars, num_of_layers, num_of_neurons, n_output_vars=1)
            y_hat = self.model.forward(X.float())
            #   gradients sum
            self.optimiser.zero_grad()
            loss = self.loss_func(y_hat, Y.float())
            loss.backward()
            self.optimiser.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
    
        X, _ = self._preprocessor(x, training = False) # Do not forget
        predict_value = self.model.forward(X.float())
        predict_value = predict_value.detach().numpy()
        predict_value *= self.y_sigma
        predict_value += self.y_mu
        return predict_value


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        prediction = self.predict(x)
            # for i, z in zip(prediction, y.values):
            #     print(f"prediction: {i}, y: {z}")
            # print(f"min is: {min(prediction)}, max is: {max(prediction)} ")
        return mean_squared_error(prediction ,y.values, squared=False) # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def split_data(x,y,k_folds=None):
    '''

    '''
    x_df = x
    y_df = y

    assert(x.shape[0]==y.shape[0])

    p = np.random.permutation(len(x_df))
    x_df=x_df.iloc[p] 
    y_df=y_df.iloc[p]

    if k_folds == None:
        split_i = int(0.8 * x_df.shape[0])
        x_train = x_df[:split_i]
        y_train = y_df[:split_i]
        x_val = x_df[split_i:]
        y_val = y_df[split_i:]
        return x_train, y_train, x_val, y_val
    else:
        x_folds = np.array_split(x_df, k_folds)
        y_folds = np.array_split(y_df, k_folds)
        return x_folds, y_folds

def cross_validation(x,y,learning_rate, optimizer, sub_arr, no_of_epochs,k_folds=3, activation=""):
    x_folds, y_folds = split_data(x,y,k_folds)
    train_score = 0
    val_score = 0
    for (i,fold) in enumerate(zip(x_folds,y_folds)):
        
        if(i == 0):
            x_training_folds_combined = pd.concat(x_folds[i+1:])
            y_training_folds_combined = pd.concat(y_folds[i+1:])
        if(i == k_folds-1):
            x_training_folds_combined = pd.concat(x_folds[:i])
            y_training_folds_combined = pd.concat(y_folds[:i])
        if((i != 0) and (i != k_folds-1)):
            x_training_folds_combined1 = pd.concat(x_folds[:i])
            x_training_folds_combined2 = pd.concat(x_folds[i+1:])
            x_training_folds_combined =  pd.concat([x_training_folds_combined1, x_training_folds_combined2])
            y_training_folds_combined1 = pd.concat(y_folds[:i])
            y_training_folds_combined2 = pd.concat(y_folds[i+1:])
            y_training_folds_combined =  pd.concat([y_training_folds_combined1, y_training_folds_combined2])

        model = Regressor(x_training_folds_combined, learning_rate, optimizer, sub_arr, no_of_epochs, activation)
        model.fit(x_training_folds_combined, y_training_folds_combined)
        train_score += model.score(x_training_folds_combined, y_training_folds_combined)
        val_score += model.score(fold[0], fold[1])

    avg_train_err = train_score/k_folds
    avg_val_err = val_score/k_folds

    return model, avg_train_err, avg_val_err


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(x, y, no_of_epochs=10, optimizer_types=['Adam', 'AdaDelta'], neuron_lists=[15,8,4], activation_functions=['LeakyReLU', 'ReLU', 'eLU'], k_folds=None) : 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.
    
    Arguments:
        Add whatever inputs you need.
        - optimizer_types {list} -- List of different optimiser names to be tested, e.g. ['Adam', 'AdaDelta', 'SGD']
        - neuron_lists {list} -- List of neurons at each hidden layer [50,20,10, 8] if this is input then hidden layer 1 = 50 neurons.. etc
          note that: len(neuron_lists) == corresponsing num_hiden_layers
        - activation_functions {list} -- List of activation functions to choose from e.g. ['LeakyReLU', 'ReLU', 'eLU', 'Sigmoid', 'Tanh']

        
    Returns:
        The function should return your optimised hyper-parameters.
         
    
    """
    x_train, y_train, x_val, y_val = split_data(x,y)
    #float('inf')
    val_error = np.Inf
    train_error = np.Inf
    best_neur = []
    best_learn_rate = 0
    best_optimiser = ""
    best_activation = ""
    learning_rates = [0.001,0.01,0.1,1]
    for activation in activation_functions:
      for learning_rate in learning_rates:
          for optimizer in optimizer_types:
              size = 1
              while(size <= len(neuron_lists)):
                  index = 0
                  while(index < len(neuron_lists)):
                      sub_arr = neuron_lists[index:index+size] #sub array
                      if k_folds == None:
                          model = Regressor(x_train, learning_rate, optimizer, sub_arr, no_of_epochs)
                          model.fit(x_train, y_train)
                          train_score = model.score(x_train, y_train)
                          val_score = model.score(x_val, y_val)
                      else:
                          model, train_score, val_score = cross_validation(x,y,learning_rate, optimizer, sub_arr, no_of_epochs,k_folds, activation)
                      if (val_error > val_score): #we have a lower validation rms value so we want to save parameters
                          save_regressor(model)
                          train_error = train_score
                          val_error = val_score
                          best_neur = sub_arr
                          best_optimiser = optimizer
                          best_learn_rate = learning_rate
                          best_activation = activation
                      index += 1
                  size += 1
    
    return best_neur, best_learn_rate, best_optimiser, val_error, train_error, best_activation, model

    # def __init__(self, n_input_vars, num_of_neurons=[8,4], n_output_vars=1):


    # Optimiser: Adam, AdaDelta, SGD
    # Activation Functions: LeakyReLU, ReLU, eLU, Sigmoid, Tanh. List of activation functions for each hidden layer
    # Neurons = list of neurons at each hidden layer [50,20,10,8] if this is input then hidden layer 1 = 50 neurons.. etc
    # Hidden Layers = Try values from nt) number of hidden layers (each hidden layer will typically have an activation function with it specified by next param)

    # We return a tuple of ((string) optimizer_type, (list of ints) neuron list, (string list) list of activation functions for each hidden layer, rmse
    # matrix: optimizer_type, neuron_list, list_activation, number_hidden, rmse 
    # df['list_activation' == ]

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################


    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def plot_error(model, x_train, y_train, x_val, y_val, num_epochs):
        epoch_v_error = np.empty((num_epochs, 3))
        for epoch in range(num_epochs):
            model.fit(x_train, y_train)
            train_score = model.score(x_train, y_train)
            val_score = model.score(x_val, y_val)
            epoch_v_error[epoch,:] = [epoch, train_score, val_score]
            print(f'Epoch: {epoch}, TrainRMSE:{train_score}, ValidationRMSE:{val_score}')

        plt.plot(epoch_v_error[:,0], epoch_v_error[:,1], c='b', label='Training RMSE')
        plt.plot(epoch_v_error[:,0], epoch_v_error[:,2], c='g', label='Validation RMSE')
        
        plt.title('Training and Validation RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('Root Mean Square Error ($)')
        plt.legend()
        plt.rcParams['font.family'] = 'serif'

        plt.show()
        plt.savefig('rmse_epoch_plot.png', orientation='landscape', dpi=600)

        print('Plotting complete! Check file rmse_epoch_plot.png for the viz :D')

def plot_optimizers(x_train, y_train, x_val, y_val, num_epochs = 1500, optimizer_list=['SGD','Adam', 'AdaDelta']):
        optimizer_v_error = np.empty((num_epochs, 4))
        optimizer_v_error[:,0] = range(num_epochs)
        for idx, optimizer in enumerate(optimizer_list):
            print(f'Optimizer currently training: {optimizer}')
            learning_rate, sub_arr, no_of_epochs = 0.01, [50,25,20,10,5], 1
            model = Regressor(x_train, learning_rate, optimizer, sub_arr, 1)

            for epoch in range(num_epochs):
                model.fit(x_train, y_train)
                #train_score = model.score(x_train, y_train)
                val_score = model.score(x_val, y_val)
                optimizer_v_error[epoch,idx+1] = val_score
                #print(f'Epoch: {epoch}, TrainRMSE:{train_score}, ValidationRMSE:{val_score}')

        plt.plot(optimizer_v_error[:,0], optimizer_v_error[:,1], c='r', label='SGD')
        plt.plot(optimizer_v_error[:,0], optimizer_v_error[:,2], c='g', label='Adam')
        plt.plot(optimizer_v_error[:,0], optimizer_v_error[:,3], c='b', label='AdaDelta')


        plt.title('Validation RMSE of different Optimizers')
        plt.xlabel('Epoch iteration')
        plt.ylabel('Root Mean Square Error ($)')
        plt.legend()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams.update({'font.size': 28})
        plt.show()

        file_name = 'rmse_optimizer_plot.png'
        plt.savefig(file_name, orientation='landscape', dpi=600)
        print(f'Plotting complete! Check file {file_name} for the viz :D')

def plot_af(x_train, y_train, x_val, y_val, num_epochs = 1000, af_list=['ReLU','LeakyReLU', 'ELU']):
        af_v_error = np.empty((num_epochs, 4))
        af_v_error[:,0] = range(num_epochs)
        for idx, af in enumerate(af_list):
            print(f'Activation function currently training: {af}')
            learning_rate, sub_arr, no_of_epochs = 0.1, [50,25,20,10,5], 1
            model = Regressor(x_train, learning_rate=learning_rate, num_of_neurons=sub_arr, nb_epoch=1, activation = af)

            for epoch in range(num_epochs):
                model.fit(x_train, y_train)
                val_score = model.score(x_val, y_val)
                af_v_error[epoch,idx+1] = val_score

        plt.plot(af_v_error[:,0], af_v_error[:,1], c='r', label='ReLU')
        plt.plot(af_v_error[:,0], af_v_error[:,2], c='g', label='LeakyReLU')
        plt.plot(af_v_error[:,0], af_v_error[:,3], c='b', label='ELU')


        plt.title('Validation RMSE of different Activation Functions')
        plt.xlabel('Epoch iteration')
        plt.ylabel('Root Mean Square Error ($)')
        plt.legend()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams.update({'font.size': 28})
        plt.show()

        file_name = 'rmse_af_plot.png'
        plt.savefig(file_name, orientation='landscape', dpi=600)
        print(f'Plotting complete! Check file {file_name} for the viz :D')

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv") 

    # Spliting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting

    ### Very simple Regressor instantiation and error score (without validation) ###

    # regressor = Regressor(x_train, 0.01, "SGD", [1], 10, "")
    # regressor.fit(x_train, y_train)
    # save_regressor(regressor)
    # Error
    # error = regressor.score(x_train, y_train)
    # print("\nRegressor error: {}\n".format(error))

    ################################################################################

    ### Performing cross validation on a given model (specified in the parameters) ###

    model, train_score, val_score = cross_validation(x_train, y_train, 0.01, "Adam", [50, 30, 25], 278 ,k_folds=4, activation="ELU")
    print(f"train score: {train_score}")
    print(f"val score: {val_score}")
    save_regressor(model)

    ##################################################################################

    ### Uncomment these to do a full exhautive search on all the parameters listed in "RegressorHyperParameterSearch" ###
    # best_neur, best_learn_rate, best_optimiser, val_error, train_error, activ, model = RegressorHyperParameterSearch(x_train, y_train, no_of_epochs=500, optimizer_types=['Adam', 'AdaDelta'], neuron_lists=[100,50,30,25,25,10,5], activation_functions=['LeakyReLU', 'ReLU', 'ELU'], k_folds=4)
    # print(f'Best combination of neurons: {best_neur}')
    # print(f'Best learning rate: {best_learn_rate}')
    # print(f"best optimiser {best_optimiser}")
    # print(f"val_error {val_error}")
    # print(f"train_error {train_error}")
    # print(f"activation function {activ}")
    # print(model, "model")

    ######################################################################################################################


if __name__ == "__main__":
    example_main()

