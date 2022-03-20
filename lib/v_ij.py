import math


def magnitude_of_flows(population_density,  radius, f_home = 1):
    # population_density: \rho_{pop} 
    # radius: r_j

    # magitude of the flows: mu_j
    mu = population_density * radius * radius * f_home
    return mu
'''
def average_daily_trips(population_density, area, radius, distance, f_max = 1, f_min, f_home = 1):
    # population_density: \rho_{pop} 
    # radius: r_j
    # area: A_i
    # distance: the distance between the origin i and visit destination r_{ij}

    # magitude of the flows: mu_j
    mu = magnitude_of_flows(population_density,  radius, f_home)


    # averagDailyTrip: V_{ij}
    e = 2.718281
    averageDailyTrips  = mu * area / (distance * distance) * (math.log(e, f_max/ f_min))

    return averageDailyTrips
'''


#def total_average_daily_trips()
 # based on matrix computation