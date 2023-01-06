from Morphogen import *


class SpikyNeuralNet(object):
    def __init__(self, dim=8, num_sig=4, num_dif=16, max_range_dif=15, num_func=12, gene=np.array([])):
        # Creates and develops a multi-cellular system
        self.neural_array = CellArray(dim, num_sig, num_dif, max_range_dif, num_func, gene)
        self.neural_array.develop()
        self.gene = self.neural_array.gene

        # Synapse weight matrix
        self.synapses = self.neural_array.conn_matrix.T

        # XY coordinates (within dim x dim) of sensory (IR & Grad), motor and energy neurons
        self.pos_neurons = self.neural_array.pos_neurons

        # Setup parameters and state variables
        self.dim = dim              # dim x dim neuronal array
        self.N = dim*dim            # number of neurons
        self.dt = .125              # simulation time step (msec)
        # LIF properties
        self.Vm = np.ones(self.N)*.9  # membrane potential (V) for each neuron
        self.tau_m = 15             # time constant (msec)
        self.tau_ref = 4            # refractory period (msec)
        self.tau_psc = 5            # post synaptic current filter time constant
        self.Vth = 1                # spike threshold (V)
        # Currents
        self.I = np.zeros(self.N)
        # Array of times of each neuron's last spike event
        self.last_spike = np.zeros(self.N) - 1000*self.tau_ref
        self.spiked = []
        # Index used in update() for tracking time
        self.t = 0

        self.I_ext = []


    # simple implementation of spike-timing dependent plasticity based on exponential decay
    def stdp(self):
        for spiked in np.arange(self.N)[self.spiked]:
            post = np.arange(self.N)[np.nonzero(self.synapses[spiked, :])] # indices for the postsynaptic neurons
            post = post[self.last_spike[post]-self.t != 0.]
            self.synapses[spiked, post] -= .5*np.exp((self.last_spike[post]-self.t)/self.tau_m)

            pre = np.arange(self.N)[np.nonzero(self.synapses[:, spiked])]  # indices for the presynaptic neurons
            pre = pre[self.last_spike[pre]-self.t != 0.]
            self.synapses[pre, spiked] += .5*np.exp((self.last_spike[pre]-self.t)/self.tau_m)

        '''
        for jj in range(self.N):
            if self.last_spike[jj] == self.t:
                for ii in range(self.N):

                    if self.synapses[ii, jj] != 0.:
                        diff = self.t - self.last_spike[ii]
                        if diff != 0.:
                            self.synapses[ii, jj] += .5 * np.exp(-diff/self.tau_m)

                    if self.synapses[jj, ii] != 0.:
                        diff = self.last_spike[ii] - self.t
                        if diff != 0.:
                            self.synapses[jj, ii] -= .5 * np.exp(diff/self.tau_m)
        '''

    # Synapse current model
    def Isyn(self, last_t):
        '''t is an array of times since each neuron's last spike event'''
        last_t[np.nonzero(last_t < 0)] = 0
        return last_t * np.exp(-last_t/self.tau_psc)


    # Updates the network with spiking dynamics and with current values of Epuck's sensors
    def update(self, IR_vals, Grad_sens, energy):   # analog values of the IR and gradient Epuck's sensors, in order
        self.t += self.dt

        active = np.nonzero(self.t > self.last_spike + self.tau_ref)
        self.Vm[active] += (-self.Vm[active] + self.I[active]) / self.tau_m * self.dt

        self.spiked = np.nonzero(self.Vm > np.random.normal(self.Vth, .1))  # added noisy threshold for avoiding locked oscillations
        self.last_spike[self.spiked] = self.t

        self.I_ext = np.zeros(self.N)            # externally applied stimulus

        for ii in range(8):
            # infrared proximity neurons
            pos_ir = int(self.dim * self.pos_neurons[ii, 0] + self.pos_neurons[ii, 1])
            self.I_ext[pos_ir] += (1 - IR_vals[ii]) * 121  # 121 mA == intensity current that generates maximum firing rate

            # gradient neurons
            pos_grad = int(self.dim * self.pos_neurons[ii + 8, 0] + self.pos_neurons[ii + 8, 1])
            self.I_ext[pos_grad] += Grad_sens[ii] * 121

        # energy neuron
        pos_ene = int(self.dim * self.pos_neurons[16, 0] + self.pos_neurons[16, 1])
        self.I_ext[pos_ene] += (1 - energy) * 121

        self.I = self.I_ext + self.synapses.dot(self.Isyn(self.t - self.last_spike))

        self.stdp()


        ## WITH NOISE #####################################################
        '''
        for ii in range(8):
            # infrared proximity neurons
            noise = 0
            if IR_vals[ii] < 1.:  # if senses something, add noise
                noise = np.random.normal(0, .05)
            pos_ir = int(self.dim*self.pos_neurons[ii,0] + self.pos_neurons[ii,1])
            self.I_ext[pos_ir] += ((1 - IR_vals[ii]) + noise)*121   # 121 mA == intensity current that generates maximum firing rate

            # gradient neurons
            noise = 0
            if Grad_sens[ii] < 1.:
                noise = np.random.normal(0, .05)
            pos_grad = int(self.dim*self.pos_neurons[ii+8,0] + self.pos_neurons[ii+8,1])
            self.I_ext[pos_grad] += ((1 - Grad_sens[ii]) + noise)*121
        
        # energy neuron
        pos_ene = int(self.dim*self.pos_neurons[16,0] + self.pos_neurons[16,1])
        self.I_ext[pos_ene] += ((1 - energy) + np.random.normal(0, .05))*121

        self.I = self.I_ext + self.synapses.dot(self.Isyn(self.t - self.last_spike))
        '''
        #####################################################################

    # Reads membrane potential of both motoneurons
    def read_motoneurons(self):
        pos_left = int(self.dim*self.pos_neurons[17,0] + self.pos_neurons[17,1])
        left_motoneuron = self.Vm[pos_left]

        pos_right = int(self.dim*self.pos_neurons[18,0] + self.pos_neurons[18,1])
        right_motoneuron = self.Vm[pos_right]
        return np.array([left_motoneuron, right_motoneuron])
