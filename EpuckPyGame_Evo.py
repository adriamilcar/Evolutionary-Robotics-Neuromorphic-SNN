#!/usr/bin/env python

import sys
sys.path.append('./_utils/')
import numpy as np
import pygame
import pygame.surfarray as surfarray
from pygame.locals import *


import PyGameUtils
import Box2DWorld 
from ExpRobotSetupPreyEvo import ExpSetupPreys


box2dWH = (PyGameUtils.SCREEN_WIDTH, PyGameUtils.SCREEN_HEIGHT)

#***************************
#PYGAME initialization
#***************************
field_size=12.
screenSize=1000


def saveLogs(exp):
    for i,e in enumerate(exp.epucks):
        if 'loss' in e.__dict__.keys():
            np.save('2loss_epuck'+str(i)+'.npy',e.loss)
        if 'energy_log' in e.__dict__.keys():
            np.save('2ener_epuck'+str(i)+'.npy',e.energy_log)
        if 'net' in e.__dict__.keys():
            if 'save' in e.net.__dict__.keys():
                e.net.save()

def getPPM(field_size=5.,screenSize=640):
    def_field_size=5.
    scale=field_size/def_field_size
    ppm_default=screenSize*65./640.
    return int(ppm_default/scale)

ppm=getPPM(field_size,screenSize)
pygame.init()
pygame.mixer.quit()

PyGameUtils.setScreenSize(screenSize,screenSize,ppm=ppm,center=True)
box2dWH = (PyGameUtils.SCREEN_WIDTH, PyGameUtils.SCREEN_HEIGHT)

#flags = HWSURFACE |FULLSCREEN | DOUBLEBUF | RESIZABLE
flags = HWSURFACE | DOUBLEBUF | RESIZABLE
#screen = pygame.display.set_mode(box2dWH, flags, 8)
screen = pygame.display.set_mode(box2dWH, flags, 32)
screen.set_alpha(None)
surfarray.use_arraytype('numpy')

pygame.display.set_caption('Epuck Simulation')
clock=pygame.time.Clock()

exp = ExpSetupPreys(debug = True,field_size=field_size,reactive=False)
for e in exp.epucks:
    e.setLog()


population = []
deaths_lifetime = []
mean_death = []
acc_food = []
mean_food = []
fitness_oa = []   # ePuck's fitness in obstacle avoidance
mean_fit_oa = []
offspring = []

i=0
generations = 1
running=True

V = [np.array([]) for _ in range(600/10)]
v_counter = 0

while running and generations <= 600:
  try:

    if generations % 10 == 0:
        if V[v_counter].size == 0:
            V[v_counter] = np.array(exp.epucks[0].SNN.Vm)
        else:
            V[v_counter] = np.column_stack((V[v_counter], exp.epucks[0].SNN.Vm))

            if V[v_counter].shape[1] == 500:
                v_counter += 1


    i+=1
    # Check the event queue

    for event in pygame.event.get():
        if(event.type!=pygame.KEYDOWN): continue

        if(event.key== pygame.K_LEFT): exp.setMotors(motors=[-1,1])
        if(event.key== pygame.K_RIGHT): exp.setMotors(motors=[1,-1])
        if(event.key== pygame.K_UP): exp.setMotors(motors=[1,1])
        if(event.key== pygame.K_DOWN): exp.setMotors(motors=[-1,-1])
        if(event.key== pygame.K_SPACE): exp.setMotors(motors=[0,0])

        if event.type==pygame.QUIT or event.key==pygame.K_ESCAPE:
            # The user closed the window or pressed escape
            running=False


    screen.fill((0,0,0,0))
    #PyGameUtils.draw_contacts(screen,exp)

    flag=Box2DWorld.destroy([exp.objs,exp.epucks])

    PyGameUtils.draw_world(screen)
    exp.update(i)
    Box2DWorld.step()

    population.append(len(exp.epucks))

    if not running:
        deaths_lifetime += exp.deaths_lifetime
        mean_death.append(np.nan)
        acc_food += exp.acc_food
        mean_food.append(np.nan)
        fitness_oa += exp.fitness_oa
        mean_fit_oa.append(np.nan)
        offspring += exp.offspring

    elif len(exp.epucks) == 0:  # if they are all dead, end/reset the experiment(with best ePucks)
        deaths_lifetime += exp.deaths_lifetime
        offspring += exp.offspring
        acc_food += exp.acc_food
        fitness_oa += exp.fitness_oa


        extra = np.zeros(49) # for 50 ePucks
        extra.fill(np.nan)

        mean_death += extra.tolist()
        mean_death.append(np.nanmean(exp.deaths_lifetime))

        mean_food += extra.tolist()
        mean_food.append(np.nanmean(exp.acc_food))

        mean_fit_oa += extra.tolist()
        mean_fit_oa.append(np.nanmean(exp.fitness_oa))

        if i <= 1100:
            exp.reset(generations)
            generations += 1
            for e in exp.epucks:
                e.setLog()
            i = 0
        else:
            running = False
    else:
        mean_death.append(np.nan)
        mean_food.append(np.nan)
        mean_fit_oa.append(np.nan)


    #PyGameUtils.draw_salient(screen, exp)

    pygame.display.flip()              # Flip the screen and try to keep at the target FPS
    clock.tick(Box2DWorld.TARGET_FPS)
    pygame.display.set_caption("FPS: {:6.3}{}".format(clock.get_fps(), " "*5))
  except KeyboardInterrupt:
    saveLogs(exp)
    print ('error 0')
    break

pygame.quit()
print('Done!')


#### DRAW PLOTS #########################################
import matplotlib.pyplot as plt
'''
plt.scatter(range(len(deaths_lifetime)), deaths_lifetime, c='r')
plt.title("ePuck's Deaths")
plt.xlabel("Time (#iterations)")
plt.xlim((0, len(deaths_lifetime)-1))
plt.ylabel("Age")
plt.plot(np.arange(len(deaths_lifetime))[~np.isnan(mean_death)], np.array(mean_death)[~np.isnan(mean_death)],
         label="Mean", marker='o', linewidth=3)
plt.legend()
plt.show()

plt.scatter(range(len(acc_food)), acc_food, c='g')
plt.title("ePuck's Accumulated food through lifetime")
plt.xlabel("Generation")
plt.xlim((0, len(acc_food)-1))
plt.xticks(np.arange(0, len(acc_food), step=1040), range(generations+1))
plt.ylabel("Food value")
plt.plot(np.arange(len(acc_food))[~np.isnan(mean_food)], np.array(mean_food)[~np.isnan(mean_food)],
         label="mean", marker='o', linewidth=3)
plt.legend()
plt.show()
'''

plt.scatter(range(len(fitness_oa)), fitness_oa, c='r')
plt.title("ePuck's fitness on obstacle avoidance")
plt.xlabel("Generation")
plt.xlim((0, len(fitness_oa)-1))
plt.xticks(np.arange(0, len(fitness_oa), step=1040), range(generations+1))
plt.ylim((0, 1))
plt.ylabel("Fitness")
plt.plot(np.arange(len(fitness_oa))[~np.isnan(mean_fit_oa)], np.array(mean_fit_oa)[~np.isnan(mean_fit_oa)],
         label="mean", marker='o', linewidth=3)
plt.legend()
plt.show()

## https://stackoverflow.com/questions/12957582/matplotlib-plot-yerr-xerr-as-shaded-region-rather-than-error-bars

for jj in range(v_counter):
    for ii in range(V[jj].shape[0]):
        plt.plot(range(V[jj].shape[1]), V[jj][ii, :])
    plt.title("Membrane potential time series of Robot[0] (" + str(jj*10) + "th generation)")
    plt.xlabel("Time (#iterations)")
    plt.xlim((0, V[jj].shape[1] - 1))
    plt.ylabel("Vm")
    plt.show()


'''
plt.scatter(range(t), offspring, c='b')
plt.title("Number of offspring through lifetime")
plt.xlabel("Time (#iterations)")
plt.xlim((0, t))
plt.ylabel("#offspring (lifetime)")
plt.show()

plt.plot(range(t), population)
plt.title("Population")
plt.xlabel("Time (#iterations)")
plt.xlim((0, t))
plt.ylabel("Number of ePucks")
plt.show()
'''
##########################################################