#!/usr/bin/python3
import numpy as np

class discreteHMM(object):

    def __init__(self, s, e):
        """
        Returns a discrete hmm. Randomly produces parameters T (transition matrix) and E (emission matrix).

        Args:
            s (int): number of states
            e (int): number of emissions categories

        Returns:
            Hmm with attributes
            T (s by s numpy array): specifies transition probabilities. A right stochastic matrix.
            E (s by e numpy array): specifies emission probabilities given states
            pi (s element numpy array): specifies initial distribution
        """

        # T
        temp = np.random.rand(s, s)
        self.T = temp / np.sum(temp, axis=1)[:, np.newaxis]

        # E
        self.E = np.random.rand(s, e)
        self.E /= np.sum(self.E, axis=1)[:, np.newaxis]

        # pi
        self.pi = np.random.rand(s)
        self.pi /= np.sum(self.pi)



    def fitSupervised(self, s, X):
        """
        Given a number of states and a string of discrete sequences, estimate a discrete HMM 
        transition matrix and emission model

        Args:
            s (int): number of states
            X (list): a list of 2d numpy arrays specifying sequences of state values in the first row
                    and emission values in the second row. Sequences can vary in length. Assumes 
        
        Returns:
            Hmm with attributes
            T (s by s numpy array): specifies transition probabilities
            E (s by e numpy array): specifies emission probabilities given states
            pi (s element numpy array): specifies initial distribution

        """
        pass
        


    def fitUnsupervised(self, s, Y):
        """
        fit:
            fitUnsupervised: given a number of states and a list of emission sequences,
                             runs the Baum-Welch algorithm to fit a discrete hmm. 

        Args:
            s (int): number of states
            Y (list): a list of 1d numpy arrays specifying sequences

        Returns:
            Hmm with attributes
            T (s by s numpy array): specifies transition probabilities
            E (s by e numpy array): specifies emission probabilities given states
        """
        pass



    def simulateHMM(self, l):
        """
        hmmDraws: given a sequence length, generates a random sequence of
                    states and a random sequence of resulting emissions from the hmm

        Args:
            l (int): length of sequence to generate

        Returns:
            X (1 by l numpy array): random sequence of states produced by the hmm 
            Y (1 by l numpy array): random sequence of emissions produced by the hmm
        """
        s, e = self.E.shape

        X = np.zeros((l)).astype(int);   Y = np.zeros((l)).astype(int)
        S = np.arange(s);                E = np.arange(e)

        # Start
        X[0] = np.random.choice(S, p=self.pi)

        # X
        # TODO: vectorize
        for i in range(1, l):
            X[i] = np.random.choice(S, p=self.T[X[i-1],:])

        # Y
        for i in range(l):
            Y[i] = np.random.choice(E, p=self.E[X[i],:])

        return X, Y
    

    def forward(self, Y):
        """
        forward: Given a sequence of Y emissions, filter the hmm for P(X_t+1|e_1:t+1)

        Args:
            Y (1 by len(Y) numpy array): emission values

        Returns
            P (s by len(Y) numpy array): P(X_t+1|e_1:t+1) for each state and time 

        """
        s, _ = self.E.shape
        P = np.zeros((s, len(Y)))
        # p = np.ones((s, 1)) / s
        p = self.pi.reshape((s, 1))

        # observation matrix * transition matrix * state probability matrix
        for i in range(len(Y)):
            O = np.diag(self.E[:, Y[i]])
            p = np.matmul( np.matmul(O, self.T), p)
            p /= np.sum(p)
            P[:, i] = p[:, 0]

        return(P)


    def backward(self, Y):
        """
        backward: Given a sequence of Y emissions and an index k < len(Y), estimate P(e_k+1:t|X_k) for
                  each state and time combination.

        Args:
            Y (1 by len(Y) numpy array): emission values

        Returns
            P (s by len(Y) numpy array): P(e_k+1:t|X_k) for each state k and time t

        """
        s, _ = self.E.shape
        P = np.zeros((s, len(Y)))
        p = np.ones((s, 1))

        # observation matrix * transition matrix * state probability matrix
        for i in range(len(Y)-1, -1, -1):
            O = np.diag(self.E[:, Y[i]])
            p = np.matmul(O, np.matmul(np.transpose(self.T), p))
            p /= np.sum(p)
            P[:, i] = p[:, 0]

        return(P)
    



    def smooth(self, Y):
        """
        smooth: Perform forward-backward smoothing on the sequence of emission values Y

        Args:
            Y (1 by len(Y) numpy array): emission values

        Returns:
            P (#states by len(Y) numpy array): smoothed probabilities for each time-state combination

        """
        A = self.forward(Y);  B = self.backward(Y)
        P = A*B
        # Normalize
        P /= np.sum(P, axis=0)
        return(P)

    

    def viterbi(self, Y):
        """
        viterbi: Run the viterbi algorithm on the sequence of Y emission values to determine the most likely 
                sequence of states
        
        Args:
            Y (1 dimensional numpy array): emission values

        Returns:
            X (1 by len(Y) numpy array): most likely state sequence

        """
        s, _ = self.E.shape
        links = np.zeros((s, len(Y)))
        vals = np.zeros((s, len(Y)))
        O = np.diag(self.E[:, Y[0]])
        P = self.pi.T

        X = np.zeros((len(Y))).astype(int)
        vals[:, 0] = np.matmul(O, self.pi.T)
        links[:, 0] = np.argmax(P)

        # Forward pass
        for i in range(1, len(Y)):
            P = vals[:, i-1]
            O = np.diag(self.E[:, Y[i]])
            temp = np.matmul(self.T, O)
            temp = temp.T * np.tile(P.T, reps=(s, 1))
            vals[:, i] = np.max(temp, axis=1)
            links[:, i] = np.argmax(temp, axis=1)
        
        # Backward pass
        X[len(Y)-1] = np.argmax(vals[:, len(Y)-1])
        for i in range(len(Y)-2, -1, -1):
            X[i] = links[X[i+1], i]

        return(X)


if __name__=="__main__":
 
    trials = 100
    vAttempts = np.zeros(trials)
    smoothAttempts = np.zeros(trials)

    for i in range(trials):
        # Create HMM
        hmm = discreteHMM(s=2, e=5)
        # Generate a sequence of states and emissions drawn from the HMM
        X, Y = hmm.simulateHMM(l=20)
        # Get the most likely states according to smoothing
        smoothStates = hmm.smooth(Y)
        smoothStates = np.argmax(smoothStates, axis=0)
        # Get the most likely state sequence according to Viterbi
        vStates = hmm.viterbi(Y)
        # Calculate how many states each predicted correctly and store
        percCorrect = np.sum(X==smoothStates) / len(smoothStates)
        vCorrect = np.sum(X==vStates) / len(X)
        vAttempts[i] = vCorrect
        smoothAttempts[i] = percCorrect

    print("Average #of states predicted correctly by Viterbi: \n"  + str(np.mean(vAttempts)))
    print("Average #of states predicted correctly by Smoothing: \n"  + str(np.mean(smoothAttempts)))

    # print("pi: " + str(hmm.pi) + "\n")
    # print("T: " + str(hmm.T) + "\n")
    # print("E: " + str(hmm.E) + "\n")    