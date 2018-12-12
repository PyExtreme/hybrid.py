from util import *


class PSO_FCM(object):
    """
    This class implements the Hybridised version of FCM using PSO
    """
    def __init__(self, *args, **kwargs):
        self.__PBest = 0
        self.__GBest = 0
        self.__loc = []
        self._NumCluster = 0
        self._dimension = 0
        self.NumParticles = 0	

    def _UpdateGBest(self, num):
        self.__GBest = num

    def _location(self, particles):
        self.__positions.append([list(i) for i in particles])

    def _RetrieveParticles(self):
        return self.__positions

    def run(filename):
	    #This function implements the Particle Swarm optimization
        U = np.random.random((len(arr), self._NumCluster))
        self.NumParticles = 15
        lb = -9999
        ub = 9999
        m_fcm = 2
	# Initialising Particles of Swarm
        particles = []
        for i in range(self.NumParticles):
            agents = np.random.uniform(lb, ub, (self.NumCluster, self.dimension))
            particles.append(agents)
        particles = np.array(particles)
        velocity = np.array([[[0.0 for k in range(self.dimension)] for j in range(self.NumCluster)] for i in range(len(particles))])
        c1 = 1
        c2 = 1
        w  = 0.5
        fitness = []
        for i in range(len(particles)):
            dist = distance.cdist(particles[i], arr)
            d_p = np.sum(np.power(np.sum(np.power(dist, 1 / (1 - m_fcm)), axis = 0), 1 / (1 - m_fcm)))
            fitness.append(d_p)
        self.__PBest = particles[np.array([fitness[x] for x in range(len(particles))]).argmax()]
        self.__GBest = self.__PBest
        G_ind = np.array([fitness[x] for x in range(len(particles))]).argmax()
	    #Main block of PSO starts
        NumIterations = 50
        for ct in range(NumIterations):
            fitness = []
            r1 = np.random.random((self.NumCluster, self.dimension))
            r2 = np.random.random((self.NumCluster, self.dimension))
            for i in range(len(particles)):
                velocity[i] = w * velocity[i] + c1 * r1 * (self.__PBest - particles[i]) + c2 * r2 * (self.__GBest - particles[i])
                particles[i] += velocity[i]
                particles[i] = np.clip(np.clip(particles[i], lb, ub) - (particles[i] - np.clip(particles[i], lb, ub)), lb, ub)
            for i in range(len(particles)):
                dist = distance.cdist(particles[i], data)
                d_p = np.sum(np.power(np.sum(np.power(dist, 1 / (1 - m_fcm)), axis = 0), 1 / (1 - m_fcm)))
                fitness.append(d_p)
            location = self._location(oarticles)
            self.__PBest = particles[np.array([fitness[x] for x in range(len(particles))]).argmax()]
            self.__loc = np.array([fitness[x] for x in range(len(particles))]).argmax()
            if fitness[p_ind] > fitness[G_ind]:
                self.__GBest = _UpdateGBest(self.__PBest)
                G_ind = self.__loc
            dist = distance.cdist(data, particles[G_ind])
            for i in range(len(arr)):
                for j in range((self.NumCluster)):
                    U[i][j] = (1 / (sum((dist[i][j] / dist[i, :]) ** (2 / (m_fcm - 1)))))
        return U
	# PSO ends

    def MultiRoundSampling(filename):
        """
        This function performs multi round sampling of the data and performs PSO_FCM on each sample
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
        samplex = np.random.choice(len(arr), sample_size)
        Sample_mrs = []
        dist = []
        center1 = []
        center2 = []
        l=[]
        u = []
        X_list = []
        Cluster_c = []
        for mrsitr in range(0, num_mrs):
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
        ar, f, pu = accuracy(u[max_index], X_list[max_index], Cluster_c[max_index], self.NumCluster)
        print("PSO_FCM results")
        print('{}'.format(ar) + " is the ARI")
        print('{}'.format(float(f / 100)) + " is the FScore")
        print('{}'.format(pu) + " is the Purity")
