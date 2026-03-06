include("agent_based_model_definition.jl")
include("deterministic_system_definition.jl")
include("theme.jl")

# %% proof of sampling equivalence

fig, axs = axesgrid(2,1; sharex = true, xlabels = "sampled simulation step")
axs[1].title = "proof of sampling equivalence for determ. and stoch."

# stochastic system
β = 5.0
total_steps = 5000
sampling_frequency = 1
seed = 50
populationsize = 2_000
X, term = run_simulation(; β = 10, total_steps = total_steps, populationsize, sampling_frequency, seed)

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

# this figure holds the proof. You can create it for any β or N and it will be the same.
# The `sampling_frequency` has been tuned for that.

# %% Simulate many trajectories (re-run this block for various `N`)
βs = 10 .^ range(-0.5, 3; length = 21)
total_steps = 10000 # total amount of data points in the timeseries; must be long enough for LZ calculation
sampling_frequency = 1 # let this at 1 like before; it samples about 10 points per oscillation
N = 15_000
max_keep = 10 # how many timeseries to try and record in total
max_tries = 1000 # how many stochastic timeseries to simulate to reach the `max_keep` goal.

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