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

# Run demographic function
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
        sampling_frequency = 10, # how often to sample, like a `Δt`
        # note the factor 3 here, which provides an equivalence per step to
        # the simulations.
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
