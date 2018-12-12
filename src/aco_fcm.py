from util import *

class ACO_FCM(object):
    """
    This class implements the Hybridised version of FCM using ACO
    """
    def __init__(self, *args, **kwargs):
        self._NumCluster = 0
        self._dimension = 0
        self.NumParticles = 0   
        self._PheromoneMatrix = []
        self._CostFunction = []

    def _StorePheromoneMatrix(self, pheromone_matrix):
        self._PheromoneMatrix.append(pheromone_matrix)

    def _StoreCostFunction(self, CostFunction):
        self._CostFunction.append(CostFunction)

    def _PlotCostFunction(self):
        sns.set()
        ax = sns.scatterplot(x="Iterations", y="Cost Value", data=self._CostFunction)


    def _RetrieveParticles(self):
        return self.__positions

    def run(arr):
        #This function implements the Ant Colony Optimization
        ClusterTest = []
        iters = 5
        runs = 5
        sample_size = len(arr) / 100
        arr_cp = np.copy(arr)
        V = [[0 for i in range(self._dimension)] for j in range(self._NumCluster)]
        V_min = [[0 for i in range(self._dimension)] for j in range(self._NumCluster)]
        for run_count in range(0, runs):
            train = []
            C_prev = np.asarray(np.asarray([[0] * self._dimension]) * self._NumCluster)
            V = [[0 for i in range(self._dimension)] for j in range(self._NumCluster)]
            samplesx = range(0, NumData)
            samplesx, _ = ShuffleData(samplesx)
            for mrsitr in range(0, iters):
                arr = np.copy(arr_cp)
                x_test = np.asarray(arr_cp)
                samples = samplesx[(mrsitr * sample_size): (mrsitr + 1) * sample_size]
                train += samples
                x_mrs = np.asarray([arr[arr] for arr in train])
                arr = x_mrs
                eps = 0.05
                NumIterations = 500
                alpha = 0.7
                rho = 0.003
                # pheromone matrix initialised
                pheromone_matrix = np.asarray(
                                               [[1 / math.pow(eps, alpha) for i in range(len(train))] for j in range(self._NumCluster)]
                                             )
                U_new = np.asarray([[0.0 for i in range(len(train))] for j in range(self._NumCluster)])
                temp_sum = 0.0
                CostFunction_Min = math.pow(10, 15)
                U = np.asarray(
                                [[0.0 for i in range(len(train))] for j in range(self._NumCluster)]
                              )
                U_min = U_new

            #Calculate U
            for i in range(0, self._NumCluster):
                sub = (V[i] - arr)
                sub = np.linalg.norm(sub, axis=1)
                XX = [0.0] * len(train)
                sub = np.select([sub == 0.0], [1e-14], default=sub)
                for j in range(0, self._NumCluster):
                    d2 = (V[j] - arr)
                    d2 = np,linalg.norm(d2, axis=1)
                    d2 = np.select([d2 == 0.0], [1e-14], default=d2)
                    # print d2
                    XX += ((sub / d2) ** (2.0 / (m - 1)))
                    U_new[i, :] = 1 / XX

            # Update pheromones
            pheromone_matrix = pheromone_matrix * (1 - rho) + U_new / math.pow((eps), alpha)
            self._StorePheromoneMatrix(pheromone_matrix)
            ###### MAIN LOOP OF ACO ######
            for iter in range(0, NumIterations):
                val = 0
                while (val < self._NumCluster):
                    rand = np.uniform(0, 1, len(train))
                    p_mat = pheromone_matrix / pheromone_matrix.sum(axis=0, keepdims=True)
                    p_mat = cumsum(p_mat, axis=0)
                    oper = p_mat[:][:] >= rand[:]
                    temp = np.full((1, len(train)), False, dtype=bool)
                    temp = np.concatenate([temp, oper[0: -1, :]])
                    U = np.logical_xor(temp, oper)
                    U = np.select([U == True], [1])
                    u_new = np.sum(U, axis=1)
                    val = len(u_new[u_new > 0])

                    # compute cluster centers
                V = np.full((self._NumCluster, self._dimension), 0, dtype=float)
                U = U ** m
                num = np.matmul(U, arr)
                denom = U.sum(axis=1, keepdims=True)
                V = num / denom
                sub = V[:, None] - arr
                sub = np.linalg.norm(sub, axis=2)
                sub = np.select([sub == 0.0], [1e-14], default=sub)
                U_new = 1 / np.sum(((sub[:, None] / sub) ** (2.0 / (m - 1))), axis=1)
                CostFunction = 0.0
                sub = V[:, None] - arr
                sub = np.linalg.norm(sub, axis=2)
                sub = sub ** 2
                CostFunction = np,sum((U_new ** m) * sub)
                self._StoreCostFunction()
                if CostFunction < CostFunction_Min:
                    CostFunction_Min = CostFunction
                    U_min = U_new
                    V_min = copy(V)
                _PlotCostFunction(_CostFunction)
                pheromone_matrix = pheromone_matrix * (1 - rho) + U_new / math.pow((CostFunction - CostFunction_Min + eps), alpha)
            #MAIN LOOP OF ACO ENDS

            V = asarray(V_min)
            distances = (V - C_prev) ** 2
            distances = distances.sum(axis=-1)
            distances = np.sqrt(distances)
            # print distances
            if max(distances) < 0.01:
                print("Iteration No : " + "{}".format(mrsitr))
                break

            # copy current centers
            C_prev = np.copy(V_min)


        #### Outside iters loop now
        ClusterTest = [0] * NumData
        sub = V_min[:, None] - x_test
        sub = np.linalg.norm(sub, axis=2)
        sub = np.select([sub == 0.0], [1e-14], default=sub)
        U = 1 / np.sum(((sub[:, None] / sub) ** (2.0 / (m - 1))), axis=1)

        # Find ClusterTest vals
        ClusterTest = np.argmax(U, axis=0)

        #Update V
        V = np.full((self._NumCluster, self._dimension), 0, dtype=float)
        U = U ** m
        num = np.matmul(U, x_test)
        denom = U.sum(axis=1, keepdims=True)
        V = num / denom

        sub = V[:, None] - x_test
        sub = np.linalg.norm(sub, axis=2)
        sub = np.select([sub == 0.0], [1e-14], default=sub)
        U = 1 / np.sum(((sub[:, None] / sub) ** (2.0 / (m - 1))), axis=1)



    def MultiRoundSampling(filename):
        """
        This function performs multi round sampling of the data and performs ACO_FCM on each sample
        """
        if(filename == 'Forest.txt'):
            self._dimension = 54
            self.NumCluster = 7
        elif(filename == 'MNIST.csv'):
            self._dimension = 784
            self.NumCluster = 10
        elif(filename == '2D15.txt'):
            self.NumCluster = 15
            self._dimension = 2
        data = ImportData(filename)
        arr = data[:,:-1]
        target = data[:,-1]
        num_mrs = 5
        sample_size = len(arr) / 100
        indices = np.random.choice(len(arr), sample_size)
        train = []
        dist = []
        center1 = []
        center2 = []
        l=[]
        u = []
        X_list = []
        Cluster_c = []
	    # Main block of MRS
        for mrsitr in range(0, num_mrs):
            indices = np.random.choice(len(arr), sample_size)
            x_mrs = arr[indices]
            Cluster = cluster[indices]
            Cluster_c.append(Cluster)
            x_mrs = list(x_mrs)
            X_list.append(x_mrs)
            u1 = run(x_mrs)
            Cluster1 = (u1.argmax(axis=1))
            u.append(u1)
            indices = np.random.choice(len(arr), sample_size)
            X_sampled1 = arr[indices]
            X_sampled1 = list(X_sampled1)
            arr = np.concatenate((x_mrs , X_sampled1) , axis = 0)
            u2 = run(arr)
            Cluster2 = u2.argmax(axis=1)[ : len(x_mrs)]       
            dist.append(adjusted_rand_score(Cluster1 , Cluster2))
            print("Finished")
	    # Main block of MRS ends
        max_index = dist.index(max(dist))
        ar, f, pu = accuracy(u[max_index], X_list[max_index], Cluster_c[max_index], self.NumCluster)
        print("ACO_FCM results")
        print('{}'.format(ar) + " is the ARI")
        print('{}'.format(float(f / 100)) + " is the FScore")
        print('{}'.format(pu) + " is the Purity")
