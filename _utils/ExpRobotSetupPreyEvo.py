import numpy as np
import Box2D
import Box2DWorld
from Box2DWorld import (world, arm, createBox, createCircle, createTri, createRope,
                        myCreateRevoluteJoint, myCreateDistanceJoint,
                        myCreateLinearJoint, collisions, RayCastCallback)

from VectorFigUtils import vnorm, dist
from Robots import Epuck
import random

from predatorPrey import EvoPrey
#from PredictivePrey_tests import predictivePrey2 as predictivePrey

# put some walls independant of the screen; beacuse screen is defined in PyGame
def addWalls(pos, dx=3, dh=0, h=2.8, th=0, bHoriz=True, bVert=True, damping = 0): 
    """ Also defined locally in ExpSetupDualCartPole!!! """
    x, y = pos
    wl = 0.2
    yh = (5 + 1) / 2.0
    if(bHoriz):
        createBox((x, y - 1 - dh + th), w=h + dh + wl + th, h=wl, bDynamic=False, damping=damping, name="wall_top")
        createBox((x, y + 5 + dh + th), w=h + dh + wl + th, h=wl, bDynamic=False, damping=damping, name="wall_bottom")

    if(bVert):
        createBox((x - dx - wl, y + yh - 1 + dh / 2 + th), w=wl, h=h + dh, bDynamic=False, damping=damping, friction=0, name="wall_left")
        createBox((x + dx + wl, y + yh - 1 + dh / 2 + th), w=wl, h=h + dh, bDynamic=False, damping=damping, friction=0, name="wall_right")

def addStandardWall(pos,L,W,name,angle=0):
    createBox(pos, w=W, h=L, bDynamic=False, damping=-1, name=name,angle=angle)


def addReward(who, pos=(0,0), vel=(0,0), reward_type=0, bDynamic=False, bCollideNoOne=False):
    if(reward_type == 0):
        name, rew = "reward", 0.27
    else:
        name, rew = "reward_small", 0.2

    r=.3 #random.random()*.3
    R=int(160-80*(r/.3))
    G=int(120+120*(r/.3))

    obj = createCircle(position=pos, bDynamic=bDynamic, bCollideNoOne=bCollideNoOne, density=5, damping=0, friction=0, name=name, r=.2, maskBits=0x0019)

    obj.userData["RGB"] = [R,G,0] # ideally add some noise
    obj.userData["ignore"] = 1.0
    obj.userData["visible"] = True
    obj.userData["name"] = name
    obj.userData["worth"] = .2 #rew  #all rewards equally energetic, for now
    obj.userData['chem'] = [1.,1.,1.,1.] # ideally add some noise and dependence on energy worth
    obj.linearVelocity = vel
    who.objs.append(obj)


# *****************************************************************
# Experimental Setup Epuck Preys
# *****************************************************************

class ExpSetupPreys(object):
    """Exp setup class with two epucks and two reward sites."""

    def __init__(self, ghostMode=True, n=50, debug=False, n_obstacles=40, n_visual_sensors=10,field_size=6.,n_rewards=100, reactive=True,
                 genes=np.array([]), reset=False, objs=[]):
        """Create the two epucks, two rewards and walls."""
        global bDebug

        bDebug = debug
        print ("-------------------------------------------------")

        th = .2
        positions = [(-3, 2 + th), (3, 2 + th)]
        L_wall=field_size
        self.L_wall=L_wall
        W_wall=.1
        area=L_wall/2.-W_wall
        positions=[]
        angles=[]
        for rob in range(n):
            x=1+int(random.random()*4.)/4.*2*L_wall*.8-L_wall*.8
            y=1+int(random.random()*4.)/4.*2*L_wall*.8-L_wall*.8
            positions.append([x,y]*np.random.normal(1, .2, 2))
            angles.append(random.random()*2 * np.pi)

        pos=[[-field_size/2., -field_size/2.],
             [-field_size/2., field_size/2.],
             [field_size/2., -field_size/2.],
             [field_size/2., field_size/2.],
             [0., 0.]]
        pos_ii = np.random.choice(range(5))

        self.n = n
        self.n_rew = n_rewards
        self.genes_reset = genes

        if self.genes_reset.size == 0:
            self.epucks = [EvoPrey(ghostMode=ghostMode, position=pos[pos_ii], angle=angles[0], frontIR=8, nother=1, nrewsensors=2, nvs=n_visual_sensors) for i in range(n)]
        else:
            self.epucks = [EvoPrey(ghostMode=ghostMode, gene=self.genes_reset[i], position=pos[pos_ii], angle=angles[0], frontIR=8, nother=1, nrewsensors=2, nvs=n_visual_sensors) for i in range(n)]
            self.genes_reset = np.array([])


        self.deaths_lifetime = []                        # store the age at which robots die (fitness-related)
        self.acc_food = []                               # store accumulated food through lifetime of ePuck's once they die
        self.obs_avoid = [[] for _ in range(self.n)]     # store fitness values regarding obstacle avoidance in each iteration
        self.previous_pos = [[0.,0.] for _ in range(self.n)]
        self.fitness_oa = []                             # final fitness for obstacle avoidance
        self.offspring = []                              # store number of offspring through lifetime
        self.behav_dis = np.zeros((self.n, self.n))

        #self.epucks = []
        #i=0
        #self.epucks.append(predictivePrey(position=positions[i], angle=angles[i], frontIR=6, nother=1, nrewsensors=2,nvs=n_visual_sensors))
        #i=1
        #self.epucks.append(reactivePrey(position=positions[i], angle=angles[i], frontIR=6, nother=1, nrewsensors=2,nvs=n_visual_sensors))
                
        #addWalls((0, 0), dx=0, dh=0.1, h=10, th=th)

        self.objs = objs

        if not reset:
        #self.objs = []
            addStandardWall((-L_wall, 0), L_wall, W_wall, 'wall_W')
            addStandardWall((L_wall, 0), L_wall, W_wall, 'wall_E')
            addStandardWall((0, L_wall), W_wall, L_wall, 'wall_N')
            addStandardWall((0, -L_wall), W_wall, L_wall, 'wall_S')

        for obs in range(n_obstacles):
            x = 1 + int(random.random() * 5.) / 5. * 2 * L_wall - L_wall
            y = 1 + int(random.random() * 5.) / 5. * 2 * L_wall - L_wall
            w = .2  # int(random.random()*5.)/5.*0.5
            l = int(random.random() * 5.) / 5. * 2
            ang = int(random.random() > .5) * np.pi / 2.  # *2*np.pi
            # print ang
            addStandardWall((x, y), w, l, 'wall_' + str(obs).zfill(2), angle=ang)

        # addReward(self, pos=(0, 4 + th), vel=(0, 0), bDynamic=True, bCollideNoOne=False)
        for n in range(n_rewards):
            addReward(self, pos=(random.uniform(-1, 1) * self.L_wall, random.uniform(-1, 1) * self.L_wall),
                        vel=(0, 0), bDynamic=True, bCollideNoOne=False)
        #else:
            #self.objs = objs


    def update(self,i):
        #if random.random()<1: #(.95+.9*2**(-i)):
            #addReward(self, pos=(random.uniform(-1,1)*self.L_wall, random.uniform(-1,1)*self.L_wall), vel=(0, 0), bDynamic=True, bCollideNoOne=False)

        death = False
        """Update of epucks positions and gradient sensors: other and reward."""
        for ii, e in enumerate(self.epucks):
            #if len(e.body.userData['food']) > 0:  # add reward each time epuck eats one (maintaining fixed the #food)
                #addReward(self, pos=(random.uniform(-1, 1) * self.L_wall, random.uniform(-1, 1) * self.L_wall), vel=(0, 0), bDynamic=True, bCollideNoOne=False)

            if i == 3:  # avoid spurious consumption of food in the beginning of experiment
                e.food = 0

            if (not death and e.energy == 0) or i==1000: # or e.time == 500:  # either if the robot wastes its energy or gets "too old", it dies
                print 'Dead!'
                death = True

                self.deaths_lifetime.append(e.time)   # append age of death
                self.acc_food.append(e.food)          # append accumulated food value (fitness) through ePuck's lifetime at death
                self.fitness_oa.append(np.mean(self.obs_avoid[ii]))  # append accumulated fitness regarding obstacle avoidance
                self.offspring.append(e.offspring)    # append number of offspring at death


                if self.genes_reset.size == 0:
                    self.genes_reset = e.gene
                else:
                    self.genes_reset = np.vstack((self.genes_reset, e.gene))

                Box2DWorld.TODESTROY.append(e.body)
                e.body.userData['name']+='_destroy'

            else:
                if e.reproduce:
                    print 'Reproduced!'
                    self.epucks.append(EvoPrey(gene=e.mutatedGene(), position=e.getPosition(), angle=e.getAngle()))

                e.update()
                pos = e.getPosition()

                # append obstacle avoidance fitness in each iteration
                motors_mean = 0
                if self.previous_pos[ii] != list(pos):  #temporal solution for problem of ePuck stuck in wall backwards
                    motors_mean = np.mean(np.absolute(e.motors)/e.max_speed)
                motors_diff = 1 - (abs(e.motors[0]/e.max_speed - e.motors[1]/e.max_speed)/2.)**.5
                max_sensor = np.amin(e.IR.IRValues)
                fit_oa = motors_mean * motors_diff * max_sensor
                self.obs_avoid[ii].append(fit_oa)

                self.previous_pos[ii] = list(pos)


                # Consume food:


                for g in e.GradSensors:
                    centers = [[o.position,o.userData['chem']] for o in self.objs]#[o.getPosition(),o.userData['chem']] for o in self.epucks if o != e] + [[o.position(),o.userData['chem']] for o in self.objs]
                    g.update(pos, e.getAngle(), centers)
                    if(g.name == "other"):
                        centers = [o.getPosition() for o in self.epucks if o != e]
                        g.update(pos, e.getAngle(), centers)
                    elif('reward' in g.name ):
                        centers = [o.position for o in self.objs[:1]]
                        g.update(pos, e.getAngle(), centers)
                        centers = [o.position for o in self.objs[-1:]]
                        g.update(pos, e.getAngle(), centers, extremes=1)

        if not death:
            self.deaths_lifetime.append(np.nan)
            self.acc_food.append(np.nan)
            self.fitness_oa.append(np.nan)
            self.offspring.append(np.nan)

        # update behavioral distances
        self.behavior_dis()


        print len(self.objs)


    def setMotors(self, epuck=0, motors=[10, 10]):
        self.epucks[epuck].motors = motors


    # computes behavioral diversity based on hamming distance of sensorimotor states among ePucks
    def behavior_dis(self):
        states = [[] for _ in range(self.n)]
        for ii, e in enumerate(self.epucks):  # get all binarized sensorimotor states
            sens_state = np.append(e.IR.IRValues, e.Gs)
            sens_state = np.append(sens_state, e.energy)
            sens_state = np.piecewise(sens_state, [sens_state <= .5, sens_state > .5], [0, 1])
            motor_state = np.array(e.motors)
            motor_state = np.piecewise(motor_state, [motor_state <= 0, motor_state > 0], [0, 1])
            states[ii] = np.append(sens_state, motor_state)

        for ii in range(self.n):
            for jj in range(self.n):
                if ii < jj:
                    self.behav_dis[ii,jj] += np.count_nonzero(np.array(states[ii]) != np.array(states[jj]))


    # creates new genes with fitness-adapted mutation probability
    def mutatedGene(self, gene, rate):
        prob = [rate, 1.-rate]
        mean = abs(gene[-2] + np.random.choice([np.random.uniform(-1, 1), 0], p=prob))
        sd = abs(gene[-1] + np.random.choice([np.random.uniform(-.2, .2), 0], p=prob))

        mutated_gene = np.resize(gene, gene.size-2)
        mutated_gene[np.random.rand(*mutated_gene.shape) < rate] -= 1
        mutated_gene **= 2

        mutated_gene = np.append(mutated_gene, [mean, sd])
        return mutated_gene


    # one-point crossover
    def crossover(self, gene1, gene2, rate):
        if np.random.random() < rate:
            cross_point = np.random.randint(0, gene1.size)
            temp1 = gene1[cross_point:]
            temp2 = gene2[cross_point:]
            gene1[cross_point:] = temp2
            gene2[cross_point:] = temp1
        return [gene1, gene2]


    #roulette wheel selection and reproduction for the Adaptive Genetic Algorithm (with Incremental Evolution)
    def roulette_wheel(self):
        #behavioral diversity
        for ii in range(self.n):
            for jj in range(self.n):
                if ii > jj:
                    self.behav_dis[ii,jj] = self.behav_dis[jj,ii]

        behav_div = np.zeros(self.n)
        for ii in range(self.n):
            behav_div[ii] = np.mean(self.behav_dis[ii,:])
        behav_div /= np.amax(behav_div)

        genes = [[] for _ in range(self.n)]

        fitness_oa = np.array(self.fitness_oa)[~np.isnan(self.fitness_oa)]
        fitness_oa /= np.amax(fitness_oa)

        if self.n_rew == 0:
            fitness = np.minimum(behav_div, fitness_oa)
        else:
            fitness_rew = np.array(self.acc_food)[~np.isnan(self.acc_food)]
            fitness_rew /= np.amax(fitness_rew)

            # multi-objective fitness --> (fit_oa + fit_rew) - abs(fit_oa - fit_rew) = 2*min([fit_oa, fit_rew]) == min([fit_oa, fit_rew])
            # based on normalized (relative) subfitness by the maximum fitness individual in each generation
            fitness = np.minimum(behav_div, fitness_rew, fitness_oa)


        for ii in range(self.n//2):
            index1 = np.random.choice(range(self.n), p=fitness/fitness.sum())  # normalized fitness to [0.,1.]
            index2 = np.random.choice(range(self.n), p=fitness/fitness.sum())

            index_max = index1
            if fitness[index2] > fitness[index1]:
                index_max = index2

            cross_rate = 1.
            if fitness[index_max] >= np.mean(fitness):
                cross_rate = 1.*(np.amax(fitness) - fitness[index_max]) / (np.amax(fitness) - np.mean(fitness))
            genes_crossed = self.crossover(self.genes_reset[index1], self.genes_reset[index2], cross_rate)

            mut_rate = .5
            if fitness[index1] >= np.mean(fitness):  # adaptive fitness-dependent mutation rate
                mut_rate = .5*(np.amax(fitness) - fitness[index1]) / (np.amax(fitness) - np.mean(fitness))
            genes[ii] = self.mutatedGene(genes_crossed[0], rate=mut_rate)

            mut_rate = .5
            if fitness[index2] >= np.mean(fitness):  # adaptive fitness-dependent mutation rate
                mut_rate = .5 * (np.amax(fitness) - fitness[index2]) / (np.amax(fitness) - np.mean(fitness))
            genes[ii+self.n//2] = self.mutatedGene(genes_crossed[1], rate=mut_rate)

        return np.array(genes)


    # incremental evolution (n_obstacles)
    def reset(self, generation):
        n_obs = 0
        ghostMode = True
        if generation <= 500:    # obstacle avoidance + reward seeking, individual-based
            if generation%50 == 0:
                n_obs = generation/50
        elif generation <= 1000:  # obstacle avoidance + reward seeking, group-based
            ghostMode = False

        genes = self.roulette_wheel()
        for o in self.objs:
            o.userData["name"] += "_destroy"

        self.__init__(ghostMode=ghostMode, n_obstacles=n_obs, field_size=self.L_wall, genes=genes, reset=True, objs=self.objs)



# *****************************************************************
# Experimental Setup MultiAgent
# *****************************************************************

class ExpSetupMultiAgent(object):
    """Exp setup class with two epucks and two reward sites."""

    def __init__(self, n=1, debug=False):
        """Create the two epucks, two rewards and walls."""
        global bDebug
        bDebug = debug
        print ("-------------------------------------------------")
        th = .2

        positions = [ (random.uniform(-5,5), random.uniform(-1,3)) for i in range(n)]

        angles = [random.uniform(0,2*np.pi) for i in range(n)]
        self.epucks = [Epuck(position=positions[i], angle=angles[i], frontIR=0, nother=2, nrewsensors=4) for i in range(n)]

        self.objs = []
        addReward(self, pos=(0, 4 + th), vel=(0, 0), bDynamic=False, bCollideNoOne=True)
        addReward(self, pos=(0, 0 + th), vel=(0, 0), reward_type=1, bDynamic=False, bCollideNoOne=True)

    def update(self):
        """Update of epucks positions and gradient sensors: other and reward."""
        for e in self.epucks:
            e.update()
            pos = e.getPosition()

            for g in e.GradSensors:
                if(g.name == "other"):
                    centers = [o.getPosition() for o in self.epucks if o != e]
                    g.update(pos, e.getAngle(), centers)
                elif(g.name == "reward"):
                    centers = [o.position for o in self.objs[:1]]
                    g.update(pos, e.getAngle(), centers)
                    centers = [o.position for o in self.objs[-1:]]
                    g.update(pos, e.getAngle(), centers, extremes=1)



    def setMotors(self, epuck=0, motors=[10, 10]):
        self.epucks[epuck].motors = motors




