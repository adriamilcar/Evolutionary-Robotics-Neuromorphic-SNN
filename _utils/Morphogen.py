import numpy as np


# Custom binary-to-integer converter (for gene decoding).
def bin_int_conv(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


# Custom Gray_code-to-binary converter (for gene decoding).
def gray_bin_conv(gray):
    out = np.zeros(gray.size)
    out[0] = gray[0]
    for bit in range(1, gray.size):
        out[bit] = out[bit-1] != gray[bit]
    return out.astype(int).tolist()


# Custom integer-to-binary converter (for intensity_signals matching with expression table).
def int_bin_conv(integer):
    return np.array([int(digit) for digit in bin(integer)[2:]])


# Does element-wise integer-to-binary conversion adding extra 0's until having 4 bits per number.
def signals_convert_bits(sig_int, max_range):
    v = np.array([])
    for ii in range(sig_int.size):
        temp = int_bin_conv(int(sig_int[ii]))
        while temp.size < np.floor(np.log2(max_range)+1):  # N bits for encoding signal intensities "max_range"
            temp = np.insert(temp, 0, 0)
        v = np.append(v, temp)
    return v


class CellArray(object):
    def __init__(self, dim, num_sig, num_dif, max_range, num_func, gene=np.array([])):
        self.conn_matrix = np.zeros((dim * dim) * (dim * dim)).reshape((dim * dim, dim * dim))  # connectivity matrix
        self.dim = dim                 # cell lattice of dim x dim dimensions
        self.num_sig = num_sig         # number of types of signals
        self.num_dif = num_dif         # number of cells that are diffusers
        self.max_range = max_range     # range of diffusers
        self.num_func = num_func       # number of functions (e.g. connectivity patterns) to implement
        self.express_table = np.array([])  # expression table for cells' differentiation
        self.pos_dif = np.zeros(num_dif * 3).reshape((num_dif, 3))  # position and type of diffusers
        self.pos_neurons = np.zeros((19,2))
        self.w_mean = 0                # mean within Gaussian(mean, sd), generator of initial weights
        self.w_sd = 0                  # st. dev. within Gaussian(mean, sd), generator of initial weights
        self.gene = gene

        self.bits_dif_pos = int(2 * np.floor(np.log2(self.dim - 1) + 1))  # number of bits necessary to encode diffusers XY position
        self.bits_num_sig = int(np.floor(np.log2(self.num_sig - 1) + 1))  # number of bits necessary to encode the type of signals
        self.bits_range   = int(np.floor(np.log2(self.max_range) + 1))    # number of bits necessary to encode the range of diffusion

        # create a 2D lattice (dim x dim) of Cells
        self.cell_array = np.empty((dim, dim), dtype=object)
        for ii in range(dim):
            for jj in range(dim):
                self.cell_array[ii, jj] = Cell(num_sig, num_func, max_range)

    # We generate a random gene for a (dim x dim) cell array with "num_dif" diffusers,
    # "num_sig" types of signals (of diffusers), "num func" different functions (incoming connectivity patterns).
    # Gene contains expression table (entries cell differentiation), and diffusers' (Gray) positions within dim x dim.
    # np.floor(np.log2(X)+1) == "number of bits necessary to encode X".
    def generate_random_gene(self):
        bits_expression_table = self.num_func * self.bits_range * self.num_sig
        bits_diffusers = self.num_dif * (self.bits_dif_pos + self.bits_num_sig)
        bits_pos_neurons = 19*self.bits_dif_pos  # XY positions of the 16 sensory neurons (8 IR + 8 Grad), 2 motoneurons and 1 energy-sensitive neuron
        gene_length = bits_expression_table + bits_diffusers + bits_pos_neurons
        gene = np.random.choice(np.array([0, 1]), gene_length)

        w_mean = abs(np.random.normal(3, .5))    # the mean of the prob. distr. that will generate the initial weights of the SNN
        w_sd = abs(np.random.normal(0, .3))      # the st. dev. of the prob. distr. that will generate the initial weights of the SNN
        gene = np.append(gene, [w_mean, w_sd])
        return gene

    def decode_gene(self):
        self.express_table = np.reshape(self.gene[:self.num_func * self.bits_range * self.num_sig],
                                        (self.num_func, self.bits_range * self.num_sig))
        for ii in range(self.dim):
            for jj in range(self.dim):
                self.cell_array[ii, jj].express_table = self.express_table  # we pass express_table to all Cells in array

        diffusers = np.reshape(self.gene[self.num_func * self.bits_range * self.num_sig:-(19*self.bits_dif_pos)-2],
                               (self.num_dif, self.bits_dif_pos + self.bits_num_sig))
        for ii in range(self.pos_dif.shape[0]):  # decode X and Y Gray-code coordinates
            self.pos_dif[ii, 0] = bin_int_conv(gray_bin_conv(diffusers[ii, 0:self.bits_dif_pos // 2]))  # Y pos
            self.pos_dif[ii, 1] = bin_int_conv(gray_bin_conv(diffusers[ii, self.bits_dif_pos // 2:-self.bits_num_sig]))  # X pos
            self.pos_dif[ii, 2] = bin_int_conv(gray_bin_conv(diffusers[ii, -self.bits_num_sig:]))  # signal type

        # decode XY Gray-code coordinates of sensory (IR & Grad), motor and energy neurons
        pos_neurons = np.reshape(self.gene[-(19*self.bits_dif_pos)-2:-2], (19, self.bits_dif_pos))
        for ii in range(self.pos_neurons.shape[0]):
            self.pos_neurons[ii, 0] = bin_int_conv(gray_bin_conv(pos_neurons[ii, 0:self.bits_dif_pos // 2]))  # Y pos
            self.pos_neurons[ii, 1] = bin_int_conv(gray_bin_conv(pos_neurons[ii, self.bits_dif_pos // 2:]))   # X pos

        self.w_mean = self.gene[-2]
        self.w_sd = self.gene[-1]

    # Randomly checks neighbouring cells and updates current cell by neighbour_intensity - 1,
    # from the corresponding signal type
    def read_neighbours(self, ii, jj, s_t):
        activate = 0
        intensity = 0
        rand = np.arange(4)  # indices for randomizing the checking order of neighbouring cells with the loop below
        np.random.shuffle(rand)

        for kk in rand:
            if kk == 0 and ii > 0 and activate == 0:               # check northern cell and update current cell
                if self.cell_array[ii - 1, jj].s_type[s_t] == 1:
                    activate = 1
                    intensity = np.amax([self.cell_array[ii - 1, jj].s_intens[s_t] - 1, 0])

            elif kk == 1 and ii < self.dim - 1 and activate == 0:  # check southern cell and update current cell
                if self.cell_array[ii + 1, jj].s_type[s_t] == 1:
                    activate = 1
                    intensity = np.amax([self.cell_array[ii + 1, jj].s_intens[s_t] - 1, 0])

            elif kk == 2 and jj > 0 and activate == 0:             # check western cell and update current cell
                if self.cell_array[ii, jj - 1].s_type[s_t] == 1:
                    activate = 1
                    intensity = np.amax([self.cell_array[ii, jj - 1].s_intens[s_t] - 1, 0])

            elif kk == 3 and jj < self.dim - 1 and activate == 0:  # check eastern cell and update current cell
                if self.cell_array[ii, jj + 1].s_type[s_t] == 1:
                    activate = 1
                    intensity = np.amax([self.cell_array[ii, jj + 1].s_intens[s_t] - 1, 0])

        return (activate, intensity)

    def diffusion_phase(self):
        for ii in range(self.pos_dif.shape[0]):  # set the diffusers
            self.cell_array[int(self.pos_dif[ii, 0]), int(self.pos_dif[ii, 1])].propagate(int(self.pos_dif[ii, 2]), self.max_range)

        # propagate the signals
        for step in range(self.max_range)[::-1]:  # "max_range" steps of development while chemicals are diffusing through the lattice
            for s_t in range(self.num_sig):       # for each type of signal
                for ii in range(self.cell_array.shape[0]):
                    for jj in range(self.cell_array.shape[1]):
                        if self.cell_array[ii, jj].s_type[s_t] == 0:         # if current cell hasn't propagated signal s_t yet
                            act, s_i = self.read_neighbours(ii, jj, s_t)     # check if neighbours are propagating signal s_t
                            if act == 1 and s_i == step:
                                self.cell_array[ii, jj].propagate(s_t, s_i)  # and update current cell correspondingly
                                # for on-line development:
                                # self.cell_array[ii,jj].match()
                                # self.build_function(ii, jj, self.cell_array[ii,jj].function)

    def expression_phase(self):  # each cell expresses its function selected from "Cell.match()"
        for ii in range(self.cell_array.shape[0]):
            for jj in range(self.cell_array.shape[1]):
                self.cell_array[ii, jj].match()

    def develop(self):
        # if no inherited gene, create a new random gene
        if self.gene.size == 0:
            self.gene = self.generate_random_gene()

        # now the phenotype is developed
        self.decode_gene()
        self.diffusion_phase()
        self.expression_phase()

        # build the final connectivity matrix
        for ii in range(self.cell_array.shape[0]):
            for jj in range(self.cell_array.shape[1]):
                self.build_function(ii, jj, self.cell_array[ii, jj].function)

    # build local connectivity pattern with the neighbouring cells depending of assigned functionality
    def build_function(self, ii, jj, func):
        if func == 0 or func == 6:
            self.build_k_4(ii, jj, 1)

        elif func == 1 or func == 7:
            self.build_k_4(ii, jj, 1)
            self.build_diagonal_4(ii, jj)

        elif func == 2 or func == 8:
            self.build_k_4(ii, jj, 1)
            self.build_k_4(ii, jj, 2)
            self.build_diagonal_4(ii, jj)

        elif func == 3 or func == 9:
            self.build_k_4(ii, jj, 1)
            self.build_k_4(ii, jj, 2)

        elif func == 4 or func == 10:
            self.build_k_4(ii, jj, 1)
            self.build_k_4(ii, jj, 2)
            self.build_k_4(ii, jj, 3)

    # build the connections with the closests "cross" neighouring cells (north, south, east, west) in "k" layer distance
    def build_k_4(self, ii, jj, k):
        if ii > k - 1:         # north
            self.conn_matrix[self.dim * (ii - k) + jj, self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii - k, jj].function > 5:
                self.conn_matrix[self.dim * (ii - k) + jj, self.dim * ii + jj] *= -1

        if ii < self.dim - k:  # south
            self.conn_matrix[self.dim * (ii + k) + jj, self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii + k, jj].function > 5:
                self.conn_matrix[self.dim * (ii + k) + jj, self.dim * ii + jj] *= -1

        if jj > k - 1:         # west
            self.conn_matrix[self.dim * ii + (jj - k), self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii, jj - k].function > 5:
                self.conn_matrix[self.dim * ii + (jj - k), self.dim * ii + jj] *= -1

        if jj < self.dim - k:  # east
            self.conn_matrix[self.dim * ii + (jj + k), self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii, jj + k].function > 5:
                self.conn_matrix[self.dim * ii + (jj + k), self.dim * ii + jj] *= -1

    # build the connections with the closests DIAGONAL neighouring cells in first layer distance
    def build_diagonal_4(self, ii, jj):
        if ii > 0 and jj > 0:                        # north-west
            self.conn_matrix[self.dim * (ii - 1) + (jj - 1), self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii - 1, jj - 1].function > 5:
                self.conn_matrix[self.dim * (ii - 1) + (jj - 1), self.dim * ii + jj] *= -1

        if ii < self.dim - 1 and jj > 0:             # south-west
            self.conn_matrix[self.dim * (ii + 1) + (jj - 1), self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii + 1, jj - 1].function > 5:
                self.conn_matrix[self.dim * (ii + 1) + (jj - 1), self.dim * ii + jj] *= -1

        if ii < self.dim - 1 and jj < self.dim - 1:  # south-east
            self.conn_matrix[self.dim * (ii + 1) + (jj + 1), self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii + 1, jj + 1].function > 5:
                self.conn_matrix[self.dim * (ii + 1) + (jj + 1), self.dim * ii + jj] *= -1

        if ii > 0 and jj < self.dim - 1:             # north-east
            self.conn_matrix[self.dim * (ii - 1) + (jj + 1), self.dim * ii + jj] = abs(np.random.normal(self.w_mean, self.w_sd))
            if self.cell_array[ii - 1, jj + 1].function > 5:
                self.conn_matrix[self.dim * (ii - 1) + (jj + 1), self.dim * ii + jj] *= -1


class Cell(object):
    def __init__(self, num_sig, num_func, max_range):
        self.num_sig = num_sig
        self.num_func = num_func
        self.max_range = max_range
        self.express_table = np.array([])
        self.s_type = np.zeros(self.num_sig)  # 0 or 1 if there is or there's no signal of type 'index'
        self.s_intens = np.zeros(self.num_sig)  # internal concentrations of different types of signals, 0 - max_range
        self.function = -1  # number of function (e.g. connectivity pattern) that should be implemented for current cell
        # if (self.function < 6) the cell is excitatory, else it's inhibitory

    # select 'best' index's function among entries within the expression table
    def match(self):
        distances = np.zeros(self.num_func)
        for ii in range(self.num_func):
            signals_bits = signals_convert_bits(self.s_intens, self.max_range)
            distances[ii] = np.count_nonzero(self.express_table[ii, :] != signals_bits)  # bit-wise hamming distance
        self.function = np.random.choice(np.flatnonzero(distances == distances.min()))
        # select random entry (index) among the ones with minimum hamming distance

    # set flags and intensities for received types of signals
    def propagate(self, s_type, s_int):
        self.s_type[s_type] = 1
        self.s_intens[s_type] = s_int



