import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import functools

VAR_INIT = 0.01
NRG_INIT = 0.2 # changed
NRG_EVOLVE = 12

class CortexNode():
    def __init__(self, center=None, var=None, evolved=True, level=0, window_mode=False):
        self.center = center
        self.var = var
        self.evolved = evolved
        self.level = level
        
        self.window_mode = window_mode

        if (var is None) and (center is not None):
            self.var = np.identity(center.shape[-1]) * VAR_INIT
        
        self.cumvar = None if center is None else self.var.copy()
        self.pc = 1
        self.nrg = NRG_INIT

        self.children = []
        
        # print('node created: ', self.center, self.level, self.var)

    def create_spine(self, x):
        if self.window_mode: x = x[0]
        self.children.append(CortexNode(
            center=x,
            var=None if (self.var is None) or (self.window_mode) else self.var/2,
            evolved=False,
            level=self.level+1,
            window_mode=self.window_mode
        ))

    def energy_loss(self):
        for i in range(len(self.children)):
            self.children[i].nrg *= 0.99
        self.children = [c for c in self.children if (c.nrg>0.1)]

    def update(self, x):
        if self.window_mode: x = x[0]
        
        m = min(self.pc, 500) # CHANGED
        self.center = ((self.center * m) + x) / (m + 1)

        diff = self.center - x
        diff = np.tile(diff, (diff.shape[-1],1))
        diff = diff * diff.T

        self.cumvar = ((self.cumvar * m) + diff) / (m + 1)
        self.var = (0.9 * self.var) + (0.1 * self.cumvar)

        self.pc += 1

    def distance(self, x):
        if self.window_mode: x = x[0]
        return mahalanobis(self.center, x, inv(self.var))

    def probability(self, x):
        if self.window_mode: x = x[0]
        return multivariate_normal.pdf(x, self.center, self.var, allow_singular=True)
    
    def get_distances(self, x):
        if self.window_mode:
            nc = [c for c in self.children]
            # return len(self.children)
            if len(x)==0: return 0
            if len(self.children)==0: return 0
            dist = np.array([c.distance(x) for c in self.children])
            return np.min(dist) + self.children[np.argmin(dist)].get_distances(x[1:])
        else:
            clusters = self.get_clusters()
            if clusters[0][0] is None: return 0
            return np.min([mahalanobis(center, x, inv(var)) for (center, var, _, _, _) in self.get_clusters()])
    

    def train(self, x, energy=NRG_INIT):
        self.energy_loss()
        if len(self.children)>0:
            dist = np.array([c.distance(x) for c in self.children])
            prob = np.array([c.probability(x)*c.pc/self.pc for c in self.children])

            inrange = np.where(dist<3)[0]
            inrange_n = [i for i in inrange if self.children[i].evolved]
            inrange_s = [i for i in inrange if not self.children[i].evolved]

            nrgs = np.array([c.nrg for c in self.children])

            if len(inrange_n)>0:
                i = inrange_n[prob[inrange_n].argmax()]
                self.children[i].update(x)
                self.children[i].nrg += energy

                d = energy
                if len(nrgs) > 1:
                    d = (nrgs[i] - np.max(nrgs[np.arange(len(nrgs))!=i])) / np.sum(nrgs)
                
                if self.window_mode:
                    level_limit = 3
                else:
                    level_limit = 1
                
                if (d>0) and (self.level<level_limit): # LEVEL LIMIT HERE
                    if self.window_mode:
                        if len(x)==1: return
                        self.children[i].train(x[1:], energy=d)
                    else:
                        self.children[i].train(x, energy=d)
                
                return

            if len(inrange_s)>0:
                i = inrange_s[prob[inrange_s].argmax()]
                self.children[i].update(x)
                self.children[i].nrg += energy

                spine_nrg = np.sum([c.nrg for c in self.children if not c.evolved])
                if spine_nrg>NRG_EVOLVE:
                    s = np.argmax([0 if c.evolved else c.nrg for c in self.children])
                    self.children[s].evolved = True

                return

        self.create_spine(x)

    def fit(self, X, y=None):
        for x in X:
            self.train(x)
            self.pc += 1
        return self



    
    def get_clusters(self):
        if len([c for c in self.children if c.evolved])<=(0 if self.level>0 else 0):
            return [(self.center, self.var, self.evolved, self.nrg, self.level)]
        else:
            return functools.reduce(lambda x, y: x+y, [c.get_clusters() for c in self.children if c.evolved])
    
    def predict(self, X):
        clusters = self.get_clusters()
        dists = [multivariate_normal(c,r,allow_singular=True) for c,r,_,_,_ in clusters]
        dists = np.array([d.pdf(X) for d in dists]).T
        return np.argmax(dists, axis=1)


    def traverse(self, parent=0, id=0):
        if self.evolved and self.level>0: print(self.level, parent, id)
        total = 1
        for i,c in enumerate([c for c in self.children if c.evolved]): total += c.traverse(parent=id, id=id+total)
        return total