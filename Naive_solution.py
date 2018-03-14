import numpy as np
import scipy
#import matlibplot


def lossFunction(X_in, Y_in, w):	
    a = 0;

def GDSolution(X_in, Y_in):
    
    lossFunction(X_in, Y_in, w);

    return w

def ZGeneration(X_in):
    Z = X_in;
    for i in range(X_in.shape[1]):
        for j in range(X_in.shape[1]):
            if (j>=i):
                temp = np.multiply(X_in[:,i], X_in[:,j]);
                Z = np.insert(Z, Z.shape[1], temp.transpose(), axis=1);
    return Z

def naiveSolution(X_in, Y_in):
    '''
    solve linear regression
    '''
    X_in_inverse = np.linalg.pinv(X_in);
    w = np.matmul(X_in_inverse,Y_in);
    return w

def main():
    """
    main function
    """
    # set-up parameters
    X_dimension = 11;
    Y_dimension = 1;
    number_of_data_points_white = 4898;
    number_of_data_points_red = 1599;
    raw_data_file = 'spambase.data';
    #raw_data_file = './winequality-overall.csv';
    using_overall = False;

    ### read file
    if using_overall:
        raw_data = np.genfromtxt(raw_data_file, delimiter = ",");
    else:
        raw_data = np.genfromtxt(raw_data_file, delimiter = ",");
    print np.shape(raw_data)
    # raw_data = np.delete(raw_data, 0, 0);
    ### randomise inputs
    raw_data_rand = raw_data[np.random.choice(raw_data.shape[0], raw_data.shape[0], replace=False), :]
    ### get data inputs
    if using_overall:
        wine_classfication_data = np.delete(raw_data, 12, 1);
        X_in = np.delete(wine_classfication_data, 11, 1);
    ### get data outputs
        Y_in = wine_classfication_data[:,11];
    else:
        X_in = np.delete(raw_data, raw_data.shape[1]-1, 1);
        Y_in = raw_data[:,raw_data.shape[1]-1];

    # Z_in = ZGeneration(X_in);
    ### using naive solution to solve linear regression
    w = naiveSolution(X_in, Y_in);
    Y_out = np.matmul(X_in, w);
    if using_overall:
        Y_final_out = np.round(Y_out);
    else:
        # Y_final_out = np.round(Y_out);
        Y_final_out = (np.sign(Y_out-0.5)+1)/2;
    error = np.equal(Y_final_out, Y_in);
    correct_no_output = np.sum(error);
    error_rate = 1 - float(correct_no_output)/np.shape(error)[0]
    print error_rate
    
def pseudoInverse(X):
    X_size = np.shape(X_in_rand);
    

if __name__ == "__main__":
    main();
