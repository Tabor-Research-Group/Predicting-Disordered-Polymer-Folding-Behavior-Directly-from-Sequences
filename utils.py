import numpy as np

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

def get_CL_OHE(seqs_nonbi):
    CL = []
    for i in range(len(seqs_nonbi)):
        CL.append(count_nonzero(seqs_nonbi[i]))
        c = Counter(CL)
    return c,CL

def get_CL_OHA(seqs):
    CL = []
    for i in range(len(seqs)):
        CL.append(sum(seqs[i]))
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
    7:[0..5,6.08,0.973],
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


def seqeuences_to_ordinal_encoding(seqs):
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
    seqs = format_batch_seqs(seqs)
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

def seqs_to_bag_of_AAs(seqs):
    seq_nonbi = seqeuences_to_ordinal_encoding(seqs)
    All_BA = []
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
                    BA[AA_I][AA_J] = BA[AA_I][AA_J]+1/sqrt(abs(k-j))
        BA = BA.flatten()
        All_BA.append(BA)
    return All_BA

def seqs_to_color_mapping(seqs):
    OE = seqeuences_to_ordinal_encoding(seqs)
    CM = np.zeros(OE.shape)
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            CM[i][j] = CM_idx[OE[i][j]]
    return CM

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


def Hyperparameters_Tuning(model, X, Y, Train_indices, Test_indices, fold, split):
    if model == "LRR":
        A_range = [x/200 for x in range(1,200,1)]
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
                clf_rg.fit(X_train, Y_train_rg_unscaled)
                score.append(clf_rg.score(X_val,Y_val_rg_unscaled))
            mean_score = sum(score)/(fold+1)
            print('alpha:',alpha,' score:',mean_score)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_alpha = alpha

        print('best_alpha',best_alpha,' best_mean_score',best_mean_score)
        return best_alpha
    elif model == "SVR":
        C_range = [1,10,100,1000]
        G_range = [0.001,0.01,0.1,1]
        E_range = [0.0001,0.001,0.01]
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
                        svr_rbf.fit(X_train, Y_train_rg_unscaled)
                        score.append(svr_rbf.score(X_val,Y_val_rg_unscaled))
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
        A_range = [1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]
        G_range = [1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4]
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
                    krr.fit(X_train, Y_train_rg_unscaled)
                    score.append(krr.score(X_val,Y_val_rg_unscaled))
                mean_score = sum(score)/(fold+1)
                print('alpha:',alpha,'gamma:',gamma,' score:',mean_score)
                if mean_score > best_mean_score:
                    best_mean_score = mean_score
                    best_alpha = alpha
                    best_gamma = gamma

        print(' best_alpha:',best_alpha,' best_gamma:',best_gamma,' best_mean_score:',best_mean_score)
        return best_alpha, best_gamma

    elif model == "GPR":
        A_range = [5,6,7,8,9,10,11,12,13,14,15]
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
                gpr.fit(X_train, Y_train_rg_unscaled)
                score.append(gpr.score(X_val,Y_val_rg_unscaled))
            mean_score = sum(score)/(fold+1)
            print('length',L,'alpha:',alpha,' score:',mean_score)
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_alpha = alpha
                best_length = L

        print('best_length:',best_length,' best_alpha:',best_alpha,' best_mean_score:',best_mean_score)
        return best_alpha, best_length
    
def Learning_curve(model, X, Y, fold, split):
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

            scaler = MinMaxScaler()
            scaler.fit(X_train_unscaled)
            X_train = scaler.transform(X_train_unscaled)
            X_val = scaler.transform(X_val_unscaled)

            model.fit(X_train, Y_train_rg_unscaled)
            Y_train_pred = model.predict(X_train)
            Y_val_pred = model.predict(X_val)
            train_loss.append(MSE(Y_train_rg_unscaled, Y_train_pred))
            val_loss.append(MSE(Y_val_rg_unscaled, Y_val_pred))
            train_score.append(model.score(X_train,Y_train_rg_unscaled))
            val_score.append(model.score(X_val,Y_val_rg_unscaled))
            train_size.append(X_train.shape[0])
            val_size.append(X_val.shape[0])
        Total_train_loss.append(train_loss)
        Total_val_loss.append(val_loss)
        Total_train_size.append(train_size)
        Total_val_size.append(train_size)
        Total_train_score.append(train_score)
        Total_val_score.append(val_score)
    return [Total_train_loss, Total_val_loss, Total_train_size, Total_val_size, Total_train_score, Total_val_score]
