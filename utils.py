import numpy as np
from numpy import sqrt
import random
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

def coeff_determination(y_true, y_pred):
    SS_res =  sum((y_true-y_pred)**2)
    SS_tot = sum((y_true-sum(y_true)/len(y_true))**2)
    return ( 1 - SS_res/(SS_tot) )

def percent_error(y_true, y_pred):
    return sum(abs(y_pred-y_true)/y_true)/len(y_true)

def MAE(y_true,y_pred):
    return sum(abs(y_true-y_pred))/len(y_pred)

def MSE(y_true,y_pred):
    return sum((y_true-y_pred)**2)/len(y_pred)

def RMSE(y_true,y_pred):
    return sqrt(sum((y_true-y_pred)**2)/len(y_pred))

def get_AA_ratio(seq):
    Count = []
    Count.append(seq.count("A"))
    Count.append(seq.count("C"))
    Count.append(seq.count("D"))
    Count.append(seq.count("E"))
    Count.append(seq.count("F"))
    Count.append(seq.count("G"))
    Count.append(seq.count("H"))
    Count.append(seq.count("I"))
    Count.append(seq.count("K"))
    Count.append(seq.count("L"))
    Count.append(seq.count("M"))
    Count.append(seq.count("N"))
    Count.append(seq.count("P"))
    Count.append(seq.count("Q"))
    Count.append(seq.count("R"))
    Count.append(seq.count("S"))
    Count.append(seq.count("T"))
    Count.append(seq.count("V"))
    Count.append(seq.count("W"))
    Count.append(seq.count("Y"))
    s = sum(Count)
    Count = [x/s for x in Count]
    AA_list = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    return AA_list, Count

def get_CL_from_OE(OE):
    CL = []
    for i in range(len(OE)):
        CL.append(count_nonzero(OE[i]))
        c = Counter(CL)
    return c,CL

def get_CL_from_CE(CE):
    CL = []
    for i in range(len(CE)):
        CL.append(sum(CE[i]))
        c = Counter(CL)
    return c,CL

def CV_split_CL(fold,seed,c):
    fold = fold
    Train_indices = [[] for i in range(fold)]
    Test_indices = []
    for i in range(len(list(c.keys()))):
        random.seed(seed)
        l = list(c.values())[i]
        Total_indices = np.arange(0,l,1).tolist()
        random.shuffle(Total_indices)
        Total_train_size = 0.8*l
        for j in range(fold):
            tmp_train = np.array(Total_indices[int(Total_train_size/fold*j):int(Total_train_size/fold*(j+1))])
            for k in tmp_train:
                Train_indices[j].append(k+i*l)
        tmp_test = np.array(Total_indices[int(Total_train_size):int(l)])
        for k in tmp_test:
            Test_indices.append(k+i*l)
    return Train_indices, Test_indices

def LC_split_CL(fold,split,seed,c):
    fold = fold
    split = split
    total_split = fold*split
    Train_indices = [[] for i in range(total_split)]
    Test_indices = []
    for i in range(len(list(c.keys()))):
        random.seed(seed)
        l = list(c.values())[i]
        Total_indices = np.arange(0,l,1).tolist()
        random.shuffle(Total_indices)
        Total_train_size = 0.8*l
        for j in range(total_split):
            tmp_train = np.array(Total_indices[int(Total_train_size/total_split*j):int(Total_train_size/total_split*(j+1))])
            for k in tmp_train:
                Train_indices[j].append(k+i*l)
        tmp_test = np.array(Total_indices[int(Total_train_size):int(l)])
        for k in tmp_test:
            Test_indices.append(k+i*l)
    return Train_indices, Test_indices

AA_index = {
    'A':1,
    'C':2,
    'D':3,
    'E':4,
    'F':5,
    'G':6,
    'H':7,
    'I':8,
    'K':9,
    'L':10,
    'M':11,
    'N':12,
    'P':13,
    'Q':14,
    'R':15,
    'S':16,
    'T':17,
    'V':18,
    'W':19,
    'Y':20}

CM_idx = {
    0:[0.,0.,0.],
    1:[0.,5.04,0.730],
    2:[0.,5.48,0.595],
    3:[-1.,5.58,0.378],
    4:[-1.,5.92,0.459],
    5:[0.,6.36,1.000],
    6:[0.,4.50,0.649],
    7:[0.5,6.08,0.973],
    8:[0.,6.18,0.973],
    9:[1.,6.36,0.514],
    10:[0.,6.18,0.973],
    11:[0.,6.18,0.838],
    12:[0.,5.68,0.432],
    13:[0.,5.56,1.000],
    14:[0.,6.02,0.514],
    15:[1.,6.56,0.000],
    16:[0.,5.18,0.595],
    17:[0.,5.62,0.676],
    18:[0.,5.86,0.892],
    19:[0.,6.78,0.946],
    20:[0.,6.46,0.865]
}

def aa_seq_to_int(s):
    return [AA_index[a] for a in s]

def format_seq(seq):
    int_seq = aa_seq_to_int(seq.strip())
    return int_seq


def seqs_to_ordinal_encoding(seqs):
    maxlen = 0
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    formatted = np.stack(formatted)
    return formatted

def seqs_to_onehot(seqs):
    seqs = seqs_to_ordinal_encoding(seqs)
    X = np.zeros((seqs.shape[0], seqs.shape[1]*21), dtype=int)
    for i in range(seqs.shape[1]):
        for j in range(21):
            X[:, i*21+j] = (seqs[:, i] == j)
    return X


def count_nonzero(a):
    s = 0
    for i in a:
        if i != 0:
            s = s+1
    return s

def seqs_to_count_encoding(seqs):
    CE = []
    for s in seqs:
        CE.append(Count_AA(s))
    return np.array(CE)

def Count_AA(seq):
    Count = []
    Count.append(seq.count("A"))
    Count.append(seq.count("C"))
    Count.append(seq.count("D"))
    Count.append(seq.count("E"))
    Count.append(seq.count("F"))
    Count.append(seq.count("G"))
    Count.append(seq.count("H"))
    Count.append(seq.count("I"))
    Count.append(seq.count("K"))
    Count.append(seq.count("L"))
    Count.append(seq.count("M"))
    Count.append(seq.count("N"))
    Count.append(seq.count("P"))
    Count.append(seq.count("Q"))
    Count.append(seq.count("R"))
    Count.append(seq.count("S"))
    Count.append(seq.count("T"))
    Count.append(seq.count("V"))
    Count.append(seq.count("W"))
    Count.append(seq.count("Y"))
    Count = np.array(Count)
    return Count

def seqs_to_bag_of_AAs(seqs, beta=-0.5):
    seq_nonbi = seqs_to_ordinal_encoding(seqs)
    All_BA = []
    if beta <= 0:
        for i in range(seq_nonbi.shape[0]):
            BA = np.zeros((20,20))
            Indices = np.arange(0,seq_nonbi.shape[1],1)
            for j in range(seq_nonbi.shape[1]):
                Indices = np.delete(Indices,0)
                for k in Indices:
                    AA_I = seq_nonbi[i][j]-1
                    AA_J = seq_nonbi[i][k]-1
                    if AA_I < 0 or AA_J < 0:
                        continue
                    elif abs(k-j)==1:
                        continue
                    else:
                        BA[AA_I][AA_J] = BA[AA_I][AA_J]+1/abs(k-j)**abs(beta)
            BA = BA.flatten()
            All_BA.append(BA)
    else:
        for i in range(seq_nonbi.shape[0]):
            BA = np.zeros((20,20))
            Indices = np.arange(0,seq_nonbi.shape[1],1)
            for j in range(seq_nonbi.shape[1]):
                Indices = np.delete(Indices,0)
                for k in Indices:
                    AA_I = seq_nonbi[i][j]-1
                    AA_J = seq_nonbi[i][k]-1
                    if AA_I < 0 or AA_J < 0:
                        continue
                    elif abs(k-j)==1:
                        continue
                    else:
                        BA[AA_I][AA_J] = BA[AA_I][AA_J]+abs(k-j)**beta
            BA = BA.flatten()
            All_BA.append(BA)
    return np.array(All_BA)

def seqs_to_color_mapping(seqs):
    OE = seqs_to_ordinal_encoding(seqs)
    CM = []
    for i in range(OE.shape[0]):
        tmp = []
        for j in range(OE.shape[1]):
            tmp.append(CM_idx[OE[i][j]])
        CM.append(tmp)
    return np.array(CM)

def create_DNN(input_dim, output_dim, n_hidden_nodes, n_hidden_layers):
    
    # Establish weight initilization technique based on chosen activation
    # functions
    
    # Create first layer
    DNN_initial = Dense(n_hidden_nodes, activation = "swish", 
                        kernel_initializer = "glorot_uniform",
                        input_shape=(input_dim,))
    
    # Create final layer
    DNN_final = Dense(output_dim, activation="linear", 
                      kernel_initializer = "glorot_uniform", use_bias=False)
    
    # Create DNN model
    DNN = Sequential()
    DNN.add(DNN_initial)
    for j in range(0, n_hidden_layers-1):
        DNN.add(Dense(n_hidden_nodes, activation = "swish", 
                      kernel_initializer = "glorot_uniform"))
    DNN.add(DNN_final)
    
    return DNN

def train_DNN(X_train, X_val, X_test, Y_train, Y_val, Y_test, n_hidden_nodes, 
                      n_hidden_layers):
    
    # Standard architecture choices
    n_batch = 32
    patience = 25
    n_epoch = 10**4
    learning_rate = 0.0001 # learning rate
    es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'min', verbose = 1, 
                   patience = patience) # patience for early stopping
    
    # Choose optimizer
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    
    # Find input and output dimensions
    input_dim = np.shape(X_train)[1]
    output_dim = np.shape(Y_train)[1]
    
    # Create DNN
    DNN = create_DNN(input_dim, output_dim, n_hidden_nodes, n_hidden_layers)
    
    # Compile DNN (and choose loss)
    DNN.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Train DNN
    DNN.fit(X_train, Y_train, epochs=n_epoch, batch_size=n_batch, 
                    shuffle=True, callbacks = [es], 
                    validation_data=(X_val, Y_val))
    
    # Calculate training, validaiton, and testing loss
    train_loss = np.asarray(DNN.history.history['loss'])
    val_loss = np.asarray(DNN.history.history['val_loss'])
    
    return DNN, train_loss, val_loss


def Hyperparameters_Tuning(model, X, Y, Train_indices, Test_indices, fold, Parameter_list, Train_Rg = True):
    if model == "LRR":
        A_range = Parameter_list
        best_mean_score = 0
        best_alpha = 0
        for alpha in A_range:
            score = []
            for i in range(fold):

                fold_number = [x for x in range(fold) if x != i]
                X_train_indices  = []
                X_train_unscaled = []
                X_val_unscaled  = []
                Y_train_unscaled = []
                Y_val_unscaled = []

                for j in fold_number:
                    X_train_indices.append(Train_indices[j])
                for j in X_train_indices:
                    X_train_unscaled.append(X[j])
                    Y_train_unscaled.append(Y[j])
                for j in Train_indices[i]:
                    X_val_unscaled.append(X[j])
                    Y_val_unscaled.append(Y[j])

                X_train_unscaled = np.vstack(X_train_unscaled)
                X_val_unscaled = np.vstack(X_val_unscaled)
                Y_train_unscaled = np.vstack(Y_train_unscaled)
                Y_val_unscaled = np.vstack(Y_val_unscaled)

                Y_train_nu_unscaled = Y_train_unscaled[:,1]
                Y_val_nu_unscaled = Y_val_unscaled[:,1]


                Y_train_rg_unscaled = Y_train_unscaled[:,0]
                Y_val_rg_unscaled = Y_val_unscaled[:,0]


                scaler = MinMaxScaler()
                scaler.fit(X_train_unscaled)
                X_train = scaler.transform(X_train_unscaled)
                X_val = scaler.transform(X_val_unscaled)
                
                clf_rg = Ridge(alpha=alpha)
                if Train_Rg:
                    clf_rg.fit(X_train, Y_train_rg_unscaled)
                    score.append(clf_rg.score(X_val,Y_val_rg_unscaled))
                else:
                    clf_rg.fit(X_train, Y_train_nu_unscaled)
                    score.append(clf_rg.score(X_val,Y_val_nu_unscaled))
            mean_score = sum(score)/(fold+1)
            print('alpha:',alpha,' score:',mean_score)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_alpha = alpha

        print('best_alpha',best_alpha,' best_mean_score',best_mean_score)
        return best_alpha
    elif model == "SVR":
        try:
            C_range = Parameter_list[0]
            G_range = Parameter_list[1]
            E_range = Parameter_list[2]
        except:
            print('You should give three sublists of parameters in SVR')
        best_mean_score = 0
        best_alpha = 0
        for C in C_range:
            for gamma in G_range:
                for eps in E_range:
                    score = []
                    for i in range(fold):

                        fold_number = [x for x in range(fold) if x != i]
                        X_train_indices  = []
                        X_train_unscaled = []
                        X_val_unscaled  = []
                        Y_train_unscaled = []
                        Y_val_unscaled = []

                        for j in fold_number:
                            X_train_indices.append(Train_indices[j])
                        for j in X_train_indices:
                            X_train_unscaled.append(X[j])
                            Y_train_unscaled.append(Y[j])
                        for j in Train_indices[i]:
                            X_val_unscaled.append(X[j])
                            Y_val_unscaled.append(Y[j])

                        X_train_unscaled = np.vstack(X_train_unscaled)
                        X_val_unscaled = np.vstack(X_val_unscaled)
                        Y_train_unscaled = np.vstack(Y_train_unscaled)
                        Y_val_unscaled = np.vstack(Y_val_unscaled)

                        Y_train_rg_unscaled = Y_train_unscaled[:,0]
                        Y_val_rg_unscaled = Y_val_unscaled[:,0]

                        scaler = MinMaxScaler()
                        scaler.fit(X_train_unscaled)
                        X_train = scaler.transform(X_train_unscaled)
                        X_val = scaler.transform(X_val_unscaled)
                        svr_rbf = SVR(kernel="rbf", C = C, gamma = gamma, epsilon = eps)
                        
                        if Train_Rg:
                            svr_rbf.fit(X_train, Y_train_rg_unscaled)
                            score.append(svr_rbf.score(X_val,Y_val_rg_unscaled))
                        else:
                            svr_rbf.fit(X_train, Y_train_nu_unscaled)
                            score.append(svr_rbf.score(X_val,Y_val_nu_unscaled))
                    mean_score = sum(score)/(fold+1)
                    print('C:',C,'Gamma:',gamma,'eps:',eps,' score:',mean_score)
                    if mean_score > best_mean_score:
                        best_mean_score = mean_score
                        best_C = C
                        best_gamma = gamma
                        best_epsilon = eps

        print('best_C:',best_C,' best_gamma:',best_gamma,' best_epsilon:',best_epsilon,' best_mean_score:',best_mean_score)
        return best_C, best_gamma, best_epsilon
    
    elif model == "KRR":
        try:
            A_range = Parameter_list[0]
            G_range = Parameter_list[1]
        except:
            print('You should give two sublists of parameters in KRR')
        best_mean_score = 0
        best_alpha = 0
        for alpha in A_range:
            for gamma in G_range:
                score = []
                for i in range(fold):

                    fold_number = [x for x in range(fold) if x != i]
                    X_train_indices  = []
                    X_train_unscaled = []
                    X_val_unscaled  = []
                    Y_train_unscaled = []
                    Y_val_unscaled = []

                    for j in fold_number:
                        X_train_indices.append(Train_indices[j])
                    for j in X_train_indices:
                        X_train_unscaled.append(X[j])
                        Y_train_unscaled.append(Y[j])
                    for j in Train_indices[i]:
                        X_val_unscaled.append(X[j])
                        Y_val_unscaled.append(Y[j])

                    X_train_unscaled = np.vstack(X_train_unscaled)
                    X_val_unscaled = np.vstack(X_val_unscaled)
                    Y_train_unscaled = np.vstack(Y_train_unscaled)
                    Y_val_unscaled = np.vstack(Y_val_unscaled)

                    Y_train_rg_unscaled = Y_train_unscaled[:,0]
                    Y_val_rg_unscaled = Y_val_unscaled[:,0]
                    
                    scaler = MinMaxScaler()
                    scaler.fit(X_train_unscaled)
                    X_train = scaler.transform(X_train_unscaled)
                    X_val = scaler.transform(X_val_unscaled)

                    krr = KernelRidge(kernel='rbf',alpha=alpha,gamma=gamma)
                    if Train_Rg:
                        krr.fit(X_train, Y_train_rg_unscaled)
                        score.append(krr.score(X_val,Y_val_rg_unscaled))
                    else:
                        krr.fit(X_train, Y_train_nu_unscaled)
                        score.append(krr.score(X_val,Y_val_nu_unscaled))
                        
                mean_score = sum(score)/(fold+1)
                print('alpha:',alpha,'gamma:',gamma,' score:',mean_score)
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_alpha = alpha
                    best_gamma = gamma

        print(' best_alpha:',best_alpha,' best_gamma:',best_gamma,' best_mean_score:',best_mean_score)
        return best_alpha, best_gamma

    elif model == "GPR":
        A_range = Parameter_list
        L = 1e5
        best_mean_score = 0
        best_alpha = 0
        best_length = 0
        for alpha in A_range:
            score = []
            for i in range(fold):

                fold_number = [x for x in range(fold) if x != i]
                X_train_indices  = []
                X_train_unscaled = []
                X_val_unscaled  = []
                Y_train_unscaled = []
                Y_val_unscaled = []

                for j in fold_number:
                    X_train_indices.append(Train_indices[j])
                for j in X_train_indices:
                    X_train_unscaled.append(X[j])
                    Y_train_unscaled.append(Y[j])
                for j in Train_indices[i]:
                    X_val_unscaled.append(X[j])
                    Y_val_unscaled.append(Y[j])

                X_train_unscaled = np.vstack(X_train_unscaled)
                X_val_unscaled = np.vstack(X_val_unscaled)
                Y_train_unscaled = np.vstack(Y_train_unscaled)
                Y_val_unscaled = np.vstack(Y_val_unscaled)

                Y_train_nu_unscaled = Y_train_unscaled[:,1]
                Y_val_nu_unscaled = Y_val_unscaled[:,1]


                Y_train_rg_unscaled = Y_train_unscaled[:,0]
                Y_val_rg_unscaled = Y_val_unscaled[:,0]


                scaler = MinMaxScaler()
                scaler.fit(X_train_unscaled)
                X_train = scaler.transform(X_train_unscaled)
                X_val = scaler.transform(X_val_unscaled)
                kernel = RBF(length_scale=L,length_scale_bounds=(1e-30, 1e30))
                gpr = GaussianProcessRegressor(kernel = kernel,random_state = 0,alpha = alpha)
                if Train_Rg:
                    gpr.fit(X_train, Y_train_rg_unscaled)
                    score.append(gpr.score(X_val,Y_val_rg_unscaled))
                else:
                    gpr.fit(X_train, Y_train_nu_unscaled)
                    score.append(gpr.score(X_val,Y_val_nu_unscaled))
            mean_score = sum(score)/(fold+1)
            print('length',L,'alpha:',alpha,' score:',mean_score)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_alpha = alpha
                best_length = L

        print('best_length:',best_length,' best_alpha:',best_alpha,' best_mean_score:',best_mean_score)
        return best_alpha, best_length
    
def Learning_curve(model, X, Y, Train_indices, Test_indices, fold, split, Train_Rg = True):
    Total_train_score = []
    Total_val_score = []
    Total_train_loss = []
    Total_val_loss = []
    Total_train_score = []
    Total_val_score = []
    Total_train_size = []
    Total_val_size = []
    for i in range(split):
        train_loss = []
        val_loss = []
        train_size = []
        val_size = []
        train_score = []
        val_score = []
        for j in range(fold):
            fold_number = [x for x in range(fold) if x != j]
            val_number = [j]
            X_train_indices  = []
            X_val_indices = []
            X_train_unscaled = []
            X_val_unscaled  = []
            Y_train_unscaled = []
            Y_val_unscaled = []

            for k in fold_number:
                for m in range(i+1):
                    tmp = k*split+m
                    X_train_indices.append(Train_indices[tmp])            
            for k in X_train_indices:
                X_train_unscaled.append(X[k])
                Y_train_unscaled.append(Y[k])
            for k in val_number:
                for m in range(i+1):
                    tmp = k*split+m
                    X_val_indices.append(Train_indices[k])
            for k in X_val_indices:
                X_val_unscaled.append(X[k])
                Y_val_unscaled.append(Y[k])

            X_train_unscaled = np.vstack(X_train_unscaled)
            X_val_unscaled = np.vstack(X_val_unscaled)
            Y_train_unscaled = np.vstack(Y_train_unscaled)
            Y_val_unscaled = np.vstack(Y_val_unscaled)

            Y_train_rg_unscaled = Y_train_unscaled[:,0]
            Y_val_rg_unscaled = Y_val_unscaled[:,0]
            
            Y_train_nu_unscaled = Y_train_unscaled[:,1]
            Y_val_nu_unscaled = Y_val_unscaled[:,1]

            scaler = MinMaxScaler()
            scaler.fit(X_train_unscaled)
            X_train = scaler.transform(X_train_unscaled)
            X_val = scaler.transform(X_val_unscaled)
            if Train_Rg:
                model.fit(X_train, Y_train_rg_unscaled)
                Y_train_pred = model.predict(X_train)
                Y_val_pred = model.predict(X_val)
                train_loss.append(MSE(Y_train_rg_unscaled, Y_train_pred))
                val_loss.append(MSE(Y_val_rg_unscaled, Y_val_pred))
                train_score.append(model.score(X_train,Y_train_rg_unscaled))
#                 val_score.append(model.score(X_val,Y_val_rg_unscaled))
            else:
                model.fit(X_train, Y_train_nu_unscaled)
                Y_train_pred = model.predict(X_train)
                Y_val_pred = model.predict(X_val)
                train_loss.append(MSE(Y_train_nu_unscaled, Y_train_pred))
                val_loss.append(MSE(Y_val_nu_unscaled, Y_val_pred))
                train_score.append(model.score(X_train,Y_train_nu_unscaled))
                val_score.append(model.score(X_val,Y_val_nu_unscaled))
            train_size.append(X_train.shape[0])
            val_size.append(X_val.shape[0])
        Total_train_loss.append(train_loss)
        Total_val_loss.append(val_loss)
        Total_train_size.append(train_size)
        Total_val_size.append(train_size)
        Total_train_score.append(train_score)
        Total_val_score.append(val_score)
    return [Total_train_loss, Total_val_loss, Total_train_size, Total_val_size, Total_train_score, Total_val_score]

def extrapolation_test_classical_regression(X, Y, Train_indices, Test_indices, model, forward = True):
    CL_range = [i for i in range(30,161,10)]
    train_size = [(x-29)*48 for x in CL_range]
    CL_range_rev = [i for i in range(192,35,-12)]
    train_size_rev = [(200-i)*48 for i in CL_range_rev]
    
    if forward:
        
        BA_rmse_list_CL_EX = []
        BA_score_list_CL_EX = []
        for i in range((160-30)//10+1):
            tmp_train_idx = []
            X_train_unscaled = []
            X_test_unscaled = []
            Y_train_unscaled = []
            Y_test_unscaled = []

            for j in range(len(Train_indices)):
                tmp_train_idx.append(Train_indices[j][0:i*10+1])
                
            tmp_train_idx = np.array(tmp_train_idx).flatten()
            
            for j in tmp_train_idx:
                X_train_unscaled.append(X[j])
                Y_train_unscaled.append(Y[j])
                
            tmp = int((i*10+30)*1.2-30)
            tmp_test_idx = Test_indices[12*tmp:12*(tmp+1)]
            
            for j in tmp_test_idx:
                X_test_unscaled.append(X[j])
                Y_test_unscaled.append(Y[j])

            Y_train_Rg = np.array(Y_train_unscaled)[:,0]
            Y_test_Rg = np.array(Y_test_unscaled)[:,0]

            scaler = MinMaxScaler()
            scaler.fit(X_train_unscaled)
            X_train = scaler.transform(X_train_unscaled)
            X_test = scaler.transform(X_test_unscaled)
            model.fit(X_train,Y_train_Rg)
            Y_pred_Rg = model.predict(X_test)
            BA_rmse_list_CL_EX.append(RMSE(Y_test_Rg,Y_pred_Rg))
            BA_score_list_CL_EX.append(model.score(X_test,Y_test_Rg))
            
        return [train_size,BA_rmse_list_CL_EX, BA_score_list_CL_EX]
        
    else:
        
        BA_rev_rmse_list_CL_EX = []
        BA_rev_score_list_CL_EX = []
        for i in range((192-29)//12+1):

            tmp_train_idx = []

            X_train_unscaled = []
            X_test_unscaled = []
            Y_train_unscaled = []
            Y_test_unscaled = []

            for j in range(len(Train_indices)):
                tmp_train_idx.append(Train_indices[j][200-30-12*i:200-30+1])
                
            tmp_train_idx = np.array(tmp_train_idx).flatten()
            
            for j in tmp_train_idx:
                X_train_unscaled.append(X[j])
                Y_train_unscaled.append(Y[j])
                
            tmp = int((-i*12+192)/1.2-30)
            tmp_test_idx = Test_indices[12*tmp:12*(tmp+1)]
            
            for j in tmp_test_idx:
                X_test_unscaled.append(X[j])
                Y_test_unscaled.append(Y[j])

            Y_train_Rg = np.array(Y_train_unscaled)[:,0]
            Y_test_Rg = np.array(Y_test_unscaled)[:,0]

            scaler = MinMaxScaler()
            scaler.fit(X_train_unscaled)
            X_train = scaler.transform(X_train_unscaled)
            X_test = scaler.transform(X_test_unscaled)

            model.fit(X_train,Y_train_Rg)
            Y_pred_Rg = model.predict(X_test)
            BA_rev_rmse_list_CL_EX.append(RMSE(Y_test_Rg,Y_pred_Rg))
            BA_rev_score_list_CL_EX.append(model.score(X_test,Y_test_Rg))
            
        return [train_size_rev, BA_rev_rmse_list_CL_EX, BA_rev_score_list_CL_EX]

def extrapolation_test_DNN(X, Y, Train_indices, Test_indices, model, forward = True):
    CL_range = [i for i in range(30,161,10)]
    train_size = [(x-29)*48 for x in CL_range]
    CL_range_rev = [i for i in range(192,35,-12)]
    train_size_rev = [(200-i)*48 for i in CL_range_rev]
    
    if forward:
        
        BA_rmse_list_CL_EX_DNN = []
        BA_score_list_CL_EX_DNN = []
        for i in range((160-30)//10+1):

            tmp_train_idx = []

            X_train_unscaled = []
            X_test_unscaled = []
            Y_train_unscaled = []
            Y_test_unscaled = []

            for j in range(len(Train_indices)):
                tmp_train_idx.append(Train_indices[j][0:i*10+1])
            tmp_train_idx = np.array(tmp_train_idx).flatten()
            for j in tmp_train_idx:
                X_train_unscaled.append(X[j])
                Y_train_unscaled.append(Y[j])
            tmp = int((i*10+30)*1.2-30)
            tmp_test_idx = Test_indices[12*tmp:12*(tmp+1)]
            for j in tmp_test_idx:
                X_test_unscaled.append(X[j])
                Y_test_unscaled.append(Y[j])

            Y_train_rg = np.array(Y_train_unscaled)[:,0]
            Y_test_rg = np.array(Y_test_unscaled)[:,0]

            scaler = MinMaxScaler()
            scaler.fit(X_train_unscaled)
            X_train = scaler.transform(X_train_unscaled)
            X_test = scaler.transform(X_test_unscaled)

            # Standard architecture choices
            n_batch = 32
            patience = 25
            n_hidden_nodes = 100
            n_hidden_layers = 2
            n_epoch = 10**4
            learning_rate = 0.0001 # learning rate
            tf.random.set_seed(10)
            es = EarlyStopping(monitor = 'val_loss', 
                           mode = 'min', verbose = 1, 
                           patience = patience,restore_best_weights=True) # patience for early stopping

            # Choose optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Find input and output dimensions
            input_dim = np.shape(X_train)[1]
            output_dim = 1

            # Create DNN
            DNN = create_DNN(input_dim, output_dim, n_hidden_nodes, n_hidden_layers)

            # Compile DNN (and choose loss)
            DNN.compile(optimizer=optimizer, loss='mean_squared_error')
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {i+1} ...')

            # Train DNN
            DNN.fit(X_train, Y_train_rg, epochs=n_epoch, batch_size=n_batch, 
                            shuffle=True, callbacks = [es], 
                            validation_split=0.25)
            Y_pred_rg = DNN.predict(X_test)
            Y_pred_rg = np.squeeze(Y_pred_rg)
            BA_rmse_list_CL_EX_DNN.append(RMSE(Y_test_rg,Y_pred_rg))
            BA_score_list_CL_EX_DNN.append(coeff_determination(Y_test_rg,Y_pred_rg))
            
        return [train_size, BA_rmse_list_CL_EX_DNN, BA_score_list_CL_EX_DNN]
        
    else:
        BA_rev_rmse_list_CL_EX_DNN = []
        BA_rev_score_list_CL_EX_DNN = []
        for i in range((160-30)//10+1):

            tmp_train_idx = []

            X_train_unscaled = []
            X_test_unscaled = []
            Y_train_unscaled = []
            Y_test_unscaled = []

            for j in range(len(Train_indices)):
                tmp_train_idx.append(Train_indices[j][0:i*10+1])
            tmp_train_idx = np.array(tmp_train_idx).flatten()
            for j in tmp_train_idx:
                X_train_unscaled.append(X[j])
                Y_train_unscaled.append(Y[j])
            tmp = int((i*10+30)*1.2-30)
            tmp_test_idx = Test_indices[12*tmp:12*(tmp+1)]
            for j in tmp_test_idx:
                X_test_unscaled.append(X[j])
                Y_test_unscaled.append(Y[j])

            Y_train_rg = np.array(Y_train_unscaled)[:,0]
            Y_test_rg = np.array(Y_test_unscaled)[:,0]

            scaler = MinMaxScaler()
            scaler.fit(X_train_unscaled)
            X_train = scaler.transform(X_train_unscaled)
            X_test = scaler.transform(X_test_unscaled)

            # Standard architecture choices
            n_batch = 32
            patience = 25
            n_hidden_nodes = 100
            n_hidden_layers = 2
            n_epoch = 10**4
            learning_rate = 0.0001 # learning rate
            tf.random.set_seed(10)
            es = EarlyStopping(monitor = 'val_loss', 
                           mode = 'min', verbose = 1, 
                           patience = patience,restore_best_weights=True) # patience for early stopping

            # Choose optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Find input and output dimensions
            input_dim = np.shape(X_train)[1]
            output_dim = 1

            # Create DNN
            DNN = create_DNN(input_dim, output_dim, n_hidden_nodes, n_hidden_layers)

            # Compile DNN (and choose loss)
            DNN.compile(optimizer=optimizer, loss='mean_squared_error')
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {i+1} ...')


            # Train DNN
            DNN.fit(X_train, Y_train_rg, epochs=n_epoch, batch_size=n_batch, 
                            shuffle=True, callbacks = [es], 
                            validation_split=0.25)
            Y_pred_rg = DNN.predict(X_test)
            Y_pred_rg = np.squeeze(Y_pred_rg)
            BA_rev_rmse_list_CL_EX_DNN.append(RMSE(Y_test_rg,Y_pred_rg))
            BA_rev_score_list_CL_EX_DNN.append(coeff_determination(Y_test_rg,Y_pred_rg))
            
        return [train_size_rev, BA_rev_rmse_list_CL_EX_DNN, BA_rev_score_list_CL_EX_DNN]