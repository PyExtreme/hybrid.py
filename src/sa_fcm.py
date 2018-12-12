from util import *

class GSA_FCM(object):

    def __init__(self, *args, **kwargs):
        self.__PBest = 0
        self.__GBest = 0
        self.__loc = []
        self._NumCluster = 0
        self._dimension = 0
        self.NumAgents = 0   

    def _UpdateGBest(self, num):
        self.__GBest = num


    def _location(self, agents):
        self.__positions.append([list(i) for i in agents])

    def _RetrieveAgents(self):
        return self.__positions



    def run(arr):

        """
        This function performs hybridisation of FCM using GSA
        """

        self.NumAgents = 7
        lb = -1000
        ub = 10000
        agents = np.random.uniform(lb, ub, (self.NumAgents, self._dimension))
        U = np.random.random((len(arr), len(agents)))
        m_fcm = 2
        self._NumCluster = 7
        agents = np.asarray(agents)
        velocity = np.array([[0 for k in range(self._dimension)] for i in range(self.NumAgents)])
        for ct in range(20):
            csi = np.random.random((self.NumAgents, self._dimension))
            eps = np.random.random((1, self.NumAgents))[0]
            cluster = []
            for i in range(len(arr)):
                p_mat = []
                for j in range(len(agents)):
                    p_mat.append(np.linalg.norm(arr[i] - agents[j]))
                    cluster.append(p_mat)
                    cluster = np.asarray(cluster)
		    # Main block of GSA
                    for i in range(len(arr)):
                        for j in range(len(agents)):
                            U[i][j] = (1 / (sum((cluster[i][j] / cluster[i, :]) ** (2 / (m_fcm - 1)))))
                            dist = cluster.transpose()
                            U_t = U.transpose()
                            first_tem = np.power(U_t, m_fcm)
                            fit = np.sum(first_tem * dist, axis = 1).reshape(len(agents), 1)
                            m = np.array([(fit[x] - min(fit)) /(max(fit) - min(fit)) for x in range(len(agents))])
                            M = np.array([i / sum(m) for i in m])
                            G = 3 / exp(0.01 * ct)
                            a = np.array([sum([eps[j] * G * M[j] * (agents[j] - agents[i]) / (np.linalg.norm(agents[i] - agents[j]) + 0.001)
                                for j in range(self.NumAgents)]) for i in range(self.NumAgents)])
                            velocity = csi * velocity + a
                            agents += velocity
                            agents = np.clip(agents, lb, ub)
                            dist = distance.cdist(arr, agents)
                            for i in range(len(arr)):
                                for j in range((cl)):
                                    U[i][j] = (1 / (sum((dist[i][j] / dist[i, :]) ** (2 / (m_fcm - 1)))))
        return U
    
    def MultiRoundSampling(filename):
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
        mrs = 5
        sample_size = 581
        samplex = np.random.choice(len(arr), sample_size)
        Sample_mrs = []
        dist = []
        center1 = []
        center2 = []
        l=[]
        u = []
        X_list = []
        Cluster_c = []
	# Main block of MRS
        for mrsitr in range(0, mrs):
            samplex = np.random.choice(len(arr), sample_size)
            X_sampled = arr[samplex]
            Cluster = cluster[samplex]
            Cluster_c.append(Cluster)
            X_sampled = list(X_sampled)
            X_list.append(X_sampled)
            u1 = run(X_sampled)
            Cluster1 = (u1.argmax(axis=1))
            #print(np.where(u1[max_index] == 1))
            u.append(u1)
            samplex = np.random.choice(len(arr), sample_size)
            X_sampled1 = arr[samplex]
            X_sampled1 = list(X_sampled1)
            X = np.concatenate((X_sampled , X_sampled1) , axis = 0)
            u2 = run(X)
            Cluster2 = u2.argmax(axis=1)[ : len(X_sampled)]       
            dist.append(adjusted_rand_score(Cluster1 , Cluster2))
            print("Finished")
        max_index = dist.index(max(dist))
        ar, f, pu = accuracy(u[max_index], X_list[max_index], Cluster_c[max_index], 7)
        print("GSA_FCM results")
        print('{}'.format(ar) + " is the ARI")
        print('{}'.format(float(f / 100)) + " is the FScore")
        print('{}'.format(pu) + " is the Purity")
