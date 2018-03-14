import numpy as np
import scipy
#import matplotlib.pyplot as plt
import cvxopt
#import matlibplot


def lossFunction(X_in, Y_in, w):	
    a = 0;

def GDSolution(X_in, Y_in):
    
    lossFunction(X_in, Y_in, w);

    return w

def kBuilder(X_in, Y_in):
    '''
    Compute quadractic coefficient for quadractic programming to find linear constant
    '''
    '''
    K = np.zeros(shape=(X_in.shape[0],0));
    X_in_trans = np.transpose(X_in);
    for i in range(X_in.shape[0]):
        temp = np.matmul(X_in, X_in[i,:].transpose());
        K = np.insert(K, K.shape[1], temp.transpose(), axis=1);
        if (i% X_in.shape[0] / 10 == 0):
            print i,np.shape(K),np.shape(temp)
    '''

    X_mult = np.matmul(X_in, X_in.transpose());
    Y_mult = np.matmul(Y_in, Y_in.transpose());
    print np.shape(Y_mult), np.shape(X_mult);
    K = np.multiply(X_mult, Y_mult);
    return K

def quadracticProgramming(X_in, Y_in):
    no_of_samples = X_in.shape[0];
    K = cvxopt.matrix(kBuilder(X_in, Y_in));
    vector_neg_ones = cvxopt.matrix(-np.ones(shape=(no_of_samples,1)));
    G = cvxopt.matrix(-np.eye(no_of_samples));
    vector_zeros = cvxopt.matrix(np.zeros(no_of_samples));
    Y_in_trans = cvxopt.matrix(Y_in.transpose());
    b = cvxopt.matrix(np.zeros(shape=(1)));
    cvxopt.solvers.options['show_progress'] = False;
    sol = cvxopt.solvers.qp(K, vector_neg_ones, G, vector_zeros, Y_in_trans, b);
    alphas = np.array(sol['x']);
    return alphas


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
    Y_in = np.reshape(Y_in, (4601,1));
    ### using naive solution to solve linear regression
    alpha = quadracticProgramming(X_in,Y_in);
    np.save('alpha_store.npy', alpha);
    print alpha
    alpha_avg = np.average(alpha);
    alpha_effective = alpha[(alpha>alpha_avg).nonzero()];
    print alpha_effective, np.shape(alpha_effective);

    '''
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
    '''

if __name__ == "__main__":
    main();
