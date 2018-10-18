#!/usr/bin/env python

import glib
import sys

from thymioII import ThymioII

speed = 500
check_freq = 10
prox_threshold = 500
big_angle = 90
small_angle = 30

turn_decider = [

    # Far left
    lambda robot: robot.turn_right(small_angle),
        
    # Middle left
    lambda robot: robot.turn_right(big_angle),

    # Middle
    lambda robot: robot.u_turn(),

    # Middle right
    lambda robot: robot.turn_left(big_angle),

    # Far right
    lambda robot: robot.turn_left(small_angle)
]

def check_prox(robot):
    prox_sensors = robot.get('prox.horizontal')
    print(prox_sensors)
    closest_sensor = max(range(5), key=prox_sensors.__getitem__)

    if prox_sensors[closest_sensor] > prox_threshold:
        turn_decider[closest_sensor](robot)

    glib.timeout_add(check_freq, lambda: check_prox(robot))

def main(name='thymio-II'):
    
    robot = ThymioII(name)

    check_prox(robot)
    robot.move_forward(speed)
    robot.run()
        
if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError:
        main()
