import os
os.environ['KERAS_BACKEND'] = 'theano'
from Robots import *
import SpikingNN as snn
# from reinforcement_learning import q_learning
from robotSounds import SOUNDS, soundEvent, PITCH
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


CHEM_DICT = {}
CHEM_DICT['CHEMA'] = 0
CHEM_DICT['CHEMB'] = 1
CHEM_DICT['CHEMC'] = 2
CHEM_DICT['CHEMD'] = 3


class GradSensorPlus(object):
    """ Gradient Sensor used by EPuck."""

    def __init__(self, ngrad=1, name="grad", maxdist=3.):
        """Init Gradient Sensor."""
        self.name = name  # name has to be the kind of chemical it detects
        # self.ind = CHEM_DICT[name[:5]]

        self.ngrad = ngrad
        self.maxd = maxdist
        if (ngrad < 4):
            m, da = (1 + ngrad) % 2, np.pi / (2 + ngrad)
        elif (ngrad == 4):
            m, da = (1 + ngrad) % 2, np.pi / ngrad
        else:
            m, da = (1 + ngrad) % 2, np.pi / (ngrad - 1)
        self.GradAngles = [k * da - ((ngrad - m) / 2) * da - m * da / 2 for k in range(ngrad)]
        self.GradValues = [0 for i in range(ngrad)]

    def update(self, pos, angle, centers=[], extremes=0):
        """Update passing agnet pos, angle and list of positions of gradient emmiters."""
        sensors = range(self.ngrad)
        if (extremes):
            sensors = [0, self.ngrad - 1]

        if (len(centers) == 0): return
        for k in sensors:
            v = vrotate((1, 0), angle + self.GradAngles[k])
            vals = [0 for i in range(len(centers))]
            for i, cl in enumerate(centers):
                c = cl[0]
                vc = (c[0] - pos[0], c[1] - pos[1])
                d = dist(pos, c)
                if (d > self.maxd):
                    d = self.maxd
                vals[i] += ((self.maxd - d) / self.maxd) * (1 - abs(vangle(v, vc)) / np.pi)
            self.GradValues[k] = (1 - max(vals))  # *cl[1][self.ind]


# Sigmoid
def sigmoid(x, k=10., th=0.5):
    return 1. / (1. + np.e ** (-k * (x - th)))


class defaultAgent(Epuck):
    def __init__(self, position=(0, 0), angle=np.pi / 2, r=0.3, bHorizontal=False, frontIR=6, nother=0, nrewsensors=2,
                 RGB=(200, 20, 50), nvs=2, bodyType='circle', categoryBits=0x0001, name='epuck_prey', maskBits=0x0009):
        Epuck.__init__(self, position=position, angle=angle, nother=nother, nrewsensors=nrewsensors, r=r,
                       frontIR=frontIR, categoryBits=categoryBits, name=name, maskBits=maskBits)
        self.VS = VisualSensor(nvs, 180)
        self.energy = 1.
        self.energy_decay = .0001  # This is a parameter to tune
        self.body.userData['food'] = []
        self.userData["RGB"] = RGB
        self.GradSensors = []
        for chem in CHEM_DICT.keys():
            self.GradSensors.append(GradSensorPlus(ngrad=2, name=chem, maxdist=3))
        self.userData['chem'] = [.0, .1, .5, .7]

        self.log = False
        self.energy_log = []

    def setLog(self, flag=True):
        self.log = flag

    def logData(self):
        if self.log:
            self.energy_log.append(self.energy)

    def updateVariables(self):

        # self.energy=np.clip(self.energy*self.energy_decay-abs(self.motors[0]+self.motors[1])*.001,0,1)
        self.energy = np.clip(self.energy-self.energy_decay-abs(self.motors[0]**2+self.motors[1]**2)*.0001, 0,1)

        if len(self.body.userData['food']) > 0:
            self.energy += (1. - self.energy) * self.body.userData['food'].pop()

        self.energyWarning = sigmoid(1. - self.energy, 40, .95) * 2

        Epuck.update(self)
        self.VS.update(self.body.position, self.body.angle, self.r)

    def update(self):
        self.updateVariables()
        self.motors = 2 * np.random.random_sample((2)) - 1
        self.logData()


# modified class to implement a SNN and evolution
class EvoPrey(defaultAgent):
    def __init__(self, ghostMode=False, gene=np.array([]), position=(0, 0), angle=np.pi / 2, r=0.46, bHorizontal=False,
                 frontIR=8, nother=0, nrewsensors=2, RGB=(200, 20, 50), nvs=2, bodyType='circle'):
        catBits = 0x0001
        if ghostMode:
            catBits = 0x0010

        defaultAgent.__init__(self, position=position, angle=angle, r=r, bHorizontal=bHorizontal, frontIR=frontIR,
                              nother=nother, nrewsensors=nrewsensors, RGB=RGB, nvs=nvs, bodyType=bodyType,
                              categoryBits=catBits)
        if ghostMode:
            self.body.userData["ignore"] = 1.0

        self.GradSensors = []
        self.GradSensors.append(GradSensorPlus(ngrad=8, maxdist=3))
        self.Gs = []
        self.motors = [0, 0]
        self.max_speed = 1

        self.time = 0  # lifetime
        self.food = 0  # amount of accumulated food through lifetime
        self.offspring = 0  # number of offspring generated through lifetime
        self.reproduce = False  # ready to reproduce or not

        self.SNN = snn.SpikyNeuralNet(gene=gene)  # create the neural controller
        self.gene = self.SNN.gene
        # self.energy_decay = .005
        self.energy_decay = 0
        # self.energy_decay += np.count_nonzero(self.SNN.synapses)*.00001  # extra decay due to brain's size (num. synapses)

    # creates genes with mutation probability of 2% for each bit
    def mutatedGene(self):
        mean = abs(self.gene[-2] + np.random.choice([np.random.normal(0, .2), 0], p=[.01, .99]))
        sd = abs(self.gene[-1] + np.random.choice([np.random.normal(0, .05), 0], p=[.01, .99]))

        mutated_gene = np.resize(self.gene, self.gene.size - 2)
        mutated_gene[np.random.rand(*mutated_gene.shape) < .01] -= 1
        mutated_gene **= 2

        mutated_gene = np.append(mutated_gene, [mean, sd])
        return mutated_gene

    def updateVariables(self):
        if len(self.body.userData['food']) > 0:
            self.food += self.body.userData['food'][-1]

        super(EvoPrey, self).updateVariables()
        # self.energy=np.clip(self.energy*self.energy_decay-abs(self.motors[0]+self.motors[1])*.001,0,1)

        # if Epuck has lived enough time and still has some energy left (parameters to tune)
        if self.time % 1100 == 0 and self.energy > .2:
            self.reproduce = True
            self.offspring += 1
        else:
            self.reproduce = False

        self.Gs = []
        for ii in range(len(self.GradSensors)):
            for jj in range(len(self.GradSensors[ii].GradValues)):
                self.Gs.append(self.GradSensors[ii].GradValues[jj])

        for _ in range(16):  # accelerate de activity of the SNN so that 1 time_step in the program == 1 ms in the SNN
            self.SNN.update(self.IR.IRValues, self.Gs, self.energy)
        self.motors = np.clip(self.SNN.read_motoneurons(), -self.max_speed, self.max_speed)

        ## WITH NOISE ##
        '''
        motors = self.SNN.read_motoneurons()
        self.motors = np.clip(motors + [np.random.normal(0, abs(motors[0]*.1)), np.random.normal(0, abs(motors[1]*.1))],
                              -self.max_speed, self.max_speed)
        '''
        #############################

    def update(self):
        self.time += 1
        self.updateVariables()
        self.logData()

