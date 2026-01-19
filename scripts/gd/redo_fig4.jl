include("agent_based_model_definition.jl")
include("deterministic_system_definition.jl")

β = 5.0
total_steps = 5000
sampling_frequency = 1
seed = 50
populationsize = 2_000
X, term = run_simulation(; β = 10, total_steps = total_steps, populationsize, sampling_frequency, seed)


# plot
include("theme.jl")

# %%

fig, axs = axesgrid(2,1; sharex = true, xlabels = "sampled simulation step")
for j in 1:4
    lines!(axs[1], X[:, j])
end

# add deterministic system
using LinearAlgebra: normalize
pcpds = deterministic_pcp_system(; β)
u0 = rand(Xoshiro(42), 4)

sf = 1*sampling_frequency
Δt = 2/β/sampling_frequency
Y, t = trajectory(pcpds, total_steps*Δt, u0/sum(u0); Δt)

for j in 1:4
    lines!(axs[2], 1:length(Y), Y[:, j])
end

figuretitle!(fig, "proof of sampling equivalence")
wsave(plotsdir("gd", "sampling_equivalence"), fig)

fig

# this figure holds the proof


# %% Okay, now a problem is that many sims reach a 0 population for one species,
# so we need to run lots of simulations where we sample many trajectories that are
# are valid, and store them in memory. Let's say 100 runs are enough!
# at each β we therefore run simulations until we can store trajectories

function obtain_valid_trajectories(;
    threshold = 500, total_steps = 5000, max_tries = 200, max_keep = 5, kwargs...)
    trajs = typeof(StateSpaceSet{4, Float64}())[]
    times = Int[]
    k = 0
    while k < max_tries
        X, t = run_simulation(; total_steps, kwargs...)
        push!(trajs, X)
        push!(times, t)
        if length(times) > max_keep
            idx = argmin(times)
            deleteat!(trajs, idx)
            deleteat!(times, idx)
        end
        k += 1
        if length(times) == 10
            all(==(total_steps), times) && break # stop if all are successful (impossible...)
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

# function obtain_valid(; total_steps = 5000, max_tries = 1000, kwargs...)
#     traj = nothing
#     maxt = k = 0
#     while k < max_tries
#         X, t = run_simulation(;total_steps, kwargs...)
#         if t == total_steps
#             traj = X
#             break
#         elseif t > maxt
#             maxt = t
#             traj = X
#         end
#         k += 1
#     end
#     return traj, maxt
# end

βs = 10 .^ range(-0.5, 3; length = 21)

total_steps = 5000
sampling_frequency = 1
N = 15_000
max_keep = 5
using ProgressMeter
using Statistics

@showprogress Threads.@threads for β in βs
    trajectories, maxt = obtain_valid_trajectories(; β, max_keep, total_steps, sampling_frequency, populationsize = N)
    # for each trajectory calculate the fractal dim, σ, and LZ complexity
    sigma, fractal_dim, lempel_ziv = Float64[], Float64[], Float64[]
    for (j, X) in enumerate(trajectories)
        push!(fractal_dim, grassberger_proccacia_dim(X; show_progress = false))
        x = X[:, 1]
        push!(sigma, std(x))
        push!(lempel_ziv, lempel_ziv_complexity(x))
    end
    data = @strdict sigma fractal_dim lempel_ziv maxt trajectories
    wsave(datadir("gd", "stochastic", savename("stochastic", @dict(β, N), "jld2")), data)
end


# %% Then plot

fig, axs = axesgrid(3, 1; xlabels = "β", sharex = true, ylabels = ["σ", "Δ", "LZ"])

Ns = [5000, 10000, 15000, 20000]

for (j, N) in enumerate(Ns)
    # Load and aggregate data across β values
    sigma_means = Float64[]
    sigma_stds = Float64[]
    fractal_dim_means = Float64[]
    fractal_dim_stds = Float64[]
    lempel_ziv_means = Float64[]
    lempel_ziv_stds = Float64[]

    for β in βs
        data = load(datadir("gd", "stochastic", savename("stochastic", @dict(β, N), "jld2")))
        push!(sigma_means, mean(data["sigma"]))
        push!(sigma_stds, std(data["sigma"]))
        push!(fractal_dim_means, mean(data["fractal_dim"]))
        push!(fractal_dim_stds, std(data["fractal_dim"]))
        push!(lempel_ziv_means, mean(data["lempel_ziv"]))
        push!(lempel_ziv_stds, std(data["lempel_ziv"]))
    end

    replace!.((sigma_stds, fractal_dim_stds, lempel_ziv_stds), NaN => 0)
    # Plot
    band!(axs[1], βs, sigma_means .- sigma_stds, sigma_means .+ sigma_stds, alpha = 0.1, color = COLORS[j])
    lines!(axs[1], βs, sigma_means, color = COLORS[j], linewidth = 3)
    band!(axs[2], βs, fractal_dim_means .- fractal_dim_stds, fractal_dim_means .+ fractal_dim_stds, alpha = 0.1, color = COLORS[j])
    lines!(axs[2], βs, fractal_dim_means, color = COLORS[j], linewidth = 3)
    band!(axs[3], βs, lempel_ziv_means .- lempel_ziv_stds, lempel_ziv_means .+ lempel_ziv_stds; alpha = 0.1, color = COLORS[j])
    lines!(axs[3], βs, lempel_ziv_means, color = COLORS[j], linewidth = 3, label = "N = $(N)")
end

axs[end].xscale = log10
axs[1].xscale = log10
axs[2].xscale = log10

Legend(fig[0, 1], axs[3], nbanks = 2, tellwidth = false)

wsave(plotsdir("gd", "stochastic_metrics"), fig)
fig