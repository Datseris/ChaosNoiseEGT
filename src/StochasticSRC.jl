"""
Functions used to define the stochastic pairwise comparison process
"""

using DrWatson

include(srcdir("PayoffMatrix.jl"))


################################################################################
#                      PAIRWISE COMPARISON PROCESS functions                   #
################################################################################

"""
    picktype_randomly(num_types) → Int64
Randomly picks a type among the different types present in the population
Generates and returns a random number from the indexable collection `1:num_types`
"""
function pick_type_randomly(amounts)
    #Random number used to randomly pick types
    randomnumber_decision = rand()

    limit_1 = amounts[1]
    limit_2 = amounts[1] + amounts[2]
    limit_3 = amounts[1] + amounts[2] + amounts[3]

    if randomnumber_decision <= limit_1
        random_type = 1
    elseif randomnumber_decision > limit_1 && randomnumber_decision <= limit_2
        random_type = 2
    elseif randomnumber_decision > limit_2 && randomnumber_decision <= limit_3
        random_type = 3
    else
        random_type = 4
    end
    return random_type
end

"""
    fitness_values(amounts) → Vector{Float64}
`amounts`: Vector{Int64} contains the amount of individuals of each type in the population
Calculates the fitness values of each type present in the population
"""
function fitness_values(amounts)
    fitness_vals = payoff_matrix*amounts
    return fitness_vals
end

"""
    replacement_probability(fi,fj,B) → Float64
`fi`: fitness of player i
`fi`: fitness of player j
`B`: selection intensity coefficient
Calculates the probability for i to replace j
"""
function replacement_probability(fi,fj,B)
    exponent = -B*(fi-fj)
    deno = 1 + exp(exponent)
    probability = 1.0/deno
    return probability
end

"""
    replacement_decision_update(Player_i,Player_j,amounts,probability) → Vector{Float64}
`Player_i`: type of player i
`Player_i`: type of player j
`amounts`: Vector{Int64} contains the amount of individuals of each type in the population
`B`: selection intensity coefficient
`probability`: probability for i to replace j
Updates the `amounts` vector after one replacement decision
"""
function replacement_decision_update(Player_i,Player_j,amounts,probability,populationsize)
    step_size = 1.0/populationsize

    #Random number used to take the replacement decision
    randomnumber_decision = rand()

    if randomnumber_decision <= probability
        #i replaces j
        amounts[Player_i] += step_size
        amounts[Player_j] -= step_size
    end
    return amounts
end

"""
    step(amounts,B) → Vector{Float64}
`amounts`: Vector{Int64} contains the amount of individuals of each type in the population
`B`: selection intensity coefficient
Generates a step in the process, returns the updated `amounts` vector
"""
function step(amounts,B,populationsize)

    Player_i = pick_type_randomly(amounts)
    Player_j = pick_type_randomly(amounts)

    #If the two randomly chosen individuals are of the same type,
    # there is no need for replacement
    #If A replaces A, the composition of the population doesn't change
    if Player_i != Player_j
        fit_vals = fitness_values(amounts)
        fi = fit_vals[Player_i]
        fj = fit_vals[Player_j]
        probability = replacement_probability(fi,fj,B)
        amounts_update = replacement_decision_update(Player_i,Player_j,amounts,probability,populationsize)
    else
        amounts_update = amounts
    end
    return amounts_update
end

################################################################################
#                              RUN PROCESS functions                           #
################################################################################

"""
    reach_boundary(amounts) → Boolean
If any of the quantities in `amounts` reaches the boundary, it returns true.
i.e. when any quantity in `amounts` reaches zero (the process should be stopped)
"""
function reach_boundary(amounts,populationsize)
    stop = false
    for i in amounts
        if i < (1/populationsize)
            stop = true
        end
    end
    return stop
end

"""
    run_process(initial_amounts, Beta, time_steps) → Vector{Any}
`Beta`: selection intensity coefficient
It runs the pairwise comparison process step for `time_steps` amount of times.
It returns an array of arrays with the data of the evolution of the process
"""
function run_full_simulation(initial_condition, Beta, populationsize, time_steps)
    data_process = Array{Float64}(undef,time_steps,4)

    #Initialise first entry with initial conditions
    data_process[1,:] = initial_condition

    cut = time_steps
    for i in 2:time_steps
        #Current step only depends on previous step
        step_data = step(data_process[i-1,:], Beta, populationsize)

        #If any type reaches the boundary, stop the process
        if reach_boundary(step_data,populationsize) == true
            data_process[i,:] = step_data
            cut = i
            @goto escape_label
        end
        data_process[i,:] = step_data
    end
    #Break label
    @label escape_label
    return data_process[1:cut,:]
end

################################################################################
#                          SET PARAMETERS functions                            #
################################################################################

"""
    set_timesteps_FD(populationsize) → Int64
It sets the amount of time steps, such that the number of generations is fixed for
all population sizes
The constant factor is approximated, such that the time steps are enough to visualise
the attractor and for the running time of the quantifiers doesn't explode
"""
function set_timesteps_FD(populationsize)
    constant_factor = 500.0/3.0
    tau = round(Int, populationsize*constant_factor)
    return tau
end

"""
    set_timesteps_others(populationsize, beta) → Int64
It sets the amount of time steps, such that the number of generations is fixed for
all population sizes
The constant factor is approximated, such that the time steps are enough to visualise
the attractor and for the running time of the quantifiers doesn't explode
    Outlier cases B=0 or B>=1000
"""
function set_timesteps_others(populationsize, beta)
    if typeof(populationsize) == String
        populationsize = parse(Int64, populationsize)
        beta = parse(Float64, beta)
    end

    if beta == 0
        #Rescale according to population size
        constant_factor = 1000.0
        ans2 = round(Int, populationsize*constant_factor)
    else
        #Rescale according to population size
        constant_factor = 1000.0
        ans1 = round(Int, populationsize*constant_factor)
        #Rescale according to beta
        if beta >= 1000
            #Limit for rescaling
            # if beta is very large, it has same timesteps as beta=100
            # for num timesteps to not be too small
            beta = 100
        end
        beta_factor = beta/5.0
        ans2 = round(Int, ans1/beta_factor)
    end
    return ans2
end