using DrWatson
@quickactivate
using StateSpaceSets
using Random: Xoshiro
using LinearAlgebra: mul!

# demographic type
struct DemographicModel{D, R}
    β::Float64
    percentages::Vector{Float64}
    populationsize::Int
    payoff_matrix::Matrix{Float64}
    fitness::Vector{Float64}
    rng::R
    increment::Float64
end

# game rule, helper functions
function one_interaction!(demographic)
    i, j = two_random_species(demographic)
    # If the two randomly chosen individuals are of the same type,
    # there is no need for replacement, so we don't alter anything
    if i ≠ j
        mul!(demographic.fitness, demographic.payoff_matrix, demographic.percentages)
        fi = demographic.fitness[i]
        fj = demographic.fitness[j]
        probability = replacement_probability(fi,fj,demographic.β)
        pairwise_comparison_process!(demographic, i, j, probability)
    end
    return
end

function two_random_species(demographic)
    r = rand(demographic.rng)
    limit_1 = demographic.percentages[1]
    limit_2 = demographic.percentages[1] + demographic.percentages[2]
    limit_3 = demographic.percentages[1] + demographic.percentages[2] + demographic.percentages[3]
    if r <= limit_1
        random_type = 1
    elseif r > limit_1 && r <= limit_2
        random_type = 2
    elseif r > limit_2 && r <= limit_3
        random_type = 3
    else
        random_type = 4
    end
    r = rand(demographic.rng)
    if r <= limit_1
        random_type2 = 1
    elseif r > limit_1 && r <= limit_2
        random_type2 = 2
    elseif r > limit_2 && r <= limit_3
        random_type2 = 3
    else
        random_type2 = 4
    end
    return random_type, random_type2
end


function pairwise_comparison_process!(demographic, i, j, probability)
    # random number used to take the replacement decision
    if rand(demographic.rng) <= probability
        demographic.percentages[i] += demographic.increment
        demographic.percentages[j] -= demographic.increment
    end
    return
end

replacement_probability(fi,fj,β) = 1/(1 + exp(-β*(fi-fj)))

# run the agent based model simulation using the provided `β, N (populationsize)`
# for `total_steps`. The simulation stops pre-emptively if any of the species
# goes extinct and sets the remaining simulation steps to the model last state.
# Return the trajectory and the break time of the simulation
# (in units of sampling_time).
function run_simulation(;
        # important variables
        β = 0.1,
        total_steps = 1000,
        populationsize = 1000,
        # rest can be left alone in most cases
        ic = ones(4), # initial condition
        payoff_matrix = [
            0.0 -0.6 0.0 1.0
            1.0 0.0 0.0 -0.5
            -1.05 -0.2 0.0 1.75
            0.5 -0.1 0.1 0.0
        ],
        seed = rand(Int),
        sampling_frequency = 10, # how often to sample, like a `Δt`, normalized for N and β
        sampling_time = ceil(Int, (2(populationsize/β))/sampling_frequency),
    )
    rng = Xoshiro(seed)
    ic = ic .+ 0.01randn(rng, length(ic))
    demographic = DemographicModel{length(ic), typeof(rng)}(
        β, ic/(sum(ic)), populationsize, payoff_matrix,
        payoff_matrix*ic/(sum(ic)), rng, 1/populationsize,
    )

    run_simulation(demographic, sampling_time, total_steps)
end

# internal function: you probably don't want to call this!
function run_simulation(demographic::DemographicModel{D}, sampling_time, total_steps) where {D}
    traj = StateSpaceSet([zero(SVector{D, Float64}) for _ in 1:total_steps])
    traj[1] = demographic.percentages # store initial condition
    ibreak = total_steps
    for i in 2:total_steps
        # simulate
        for _ in 1:sampling_time
            one_interaction!(demographic)
        end
        # and update only once every sampling time
        traj[i] = demographic.percentages

        if reach_boundary(demographic) # exit early if need be
            for k in (i+1):total_steps
                traj[k] = demographic.percentages
            end
            ibreak = i
            break
        end
    end
    return traj, ibreak
end

# Termination condition
reach_boundary(demographic) = any(<(demographic.increment), demographic.percentages)

# Simulating valid trajectories, i.e., just `run_simulation` is not enough.
# The problem is that as β gets smaller, a species will reach 0 population.
# This happens faster for smaller β and smaller N.
# So we need to run lots of simulations where we sample many trajectories that are
# are valid, i.e., they have been simulated up to a requested "natural time"
# (characteristic oscillation) without reaching the extinction condition.
# The function below does this, and returns the trajectories and their total time.

# It works by simulating at most `max_tries`. From these, it keeps the `max_keep` tries
# that are the longest. Trajectories can be at most `total_steps`, so when
# `max_keep` trajectories with `total_steps` each has been simulated, the function terminates
# early. For small β, this is never the case, and all `max_tries` will typically be exhausted.
function obtain_valid_trajectories(;
    threshold = 500, total_steps = 5000, max_tries = 200, max_keep = 10, kwargs...)
    trajs = typeof(StateSpaceSet{4, Float64}())[]
    times = Int[]
    k = 0
    while k < max_tries
        # Generate a store a simulation no matter what
        X, t = run_simulation(; total_steps, kwargs...)
        push!(trajs, X)
        push!(times, t)
        # if we have stored more than what we want to return, eliminate
        # the simulation that is of shortest total time
        if length(times) > max_keep
            idx = argmin(times)
            deleteat!(trajs, idx)
            deleteat!(times, idx)
        end
        k += 1
        if length(times) == max_keep
            all(==(total_steps), times) && break # stop if all are successful (unlikely...)
        end
    end
    # Truncate
    trajs = [traj[1:times[i]] for (i, traj) in enumerate(trajs)]
    # Filter trajectories and times by threshold
    valid_idx = findall(>=(threshold), times)
    trajs = trajs[valid_idx]
    times = times[valid_idx]
    return trajs, times
end