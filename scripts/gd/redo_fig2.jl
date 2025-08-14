include("deterministic_system_definition.jl")

using Statistics: std, mean

# re-do fig 2
pcpds = deterministic_pcp_system()
βs = 10 .^ range(-2, 3; length = 51)

λs = fill(zeros(dimension(pcpds)), length(βs))
fractal_dim_corr = fill(0.0, length(βs))
fractal_dim_takens = fill(0.0, length(βs))
σs = fill(0.0, length(βs))
sample_entropy = fill(0.0, length(βs))
permutation_entropy = fill(0.0, length(βs))
lempel_ziv = fill(0.0, length(βs))

# Simple parallelism over threads:
using ChunkSplitters: chunks
using Base.Threads: nthreads, @threads
n = nthreads()
systems = [deepcopy(pcpds) for _ in 1:n-1]
pushfirst!(systems, pcpds)

@time Threads.@threads for (i, chunk) in enumerate(chunks(eachindex(βs); n=n))
    local ds = systems[i]
    for j in chunk # indices of beta parameter and all containers
        # update system parameter
        β = βs[j]
        set_parameter!(ds, :β, β)
        # TODO: Crucial!!! We must scale the integration time with beta!
        # At least for very small beta the integration time of only 25,000 units is not enough!!!
        τ = 2/β # TODO: should we use this?
        τ = 1.0 # for now I ignore this timescale
        # Execute in parallel the estimation of various quantifiers
        λs[j] = lyapunovspectrum(ds, 20_000; Δt = τ)
        X, tvec = trajectory(ds, 25000.0*τ; Δt = τ)
        fractal_dim_corr[j] = grassberger_proccacia_dim(X; show_progress = false)
        fractal_dim_takens[j] = takens_best_estimate_dim(X, 0.05)
        σs[j] = std(X[:, 1])
        X, tvec = trajectory(ds, 2500.0*τ; Δt = 0.1τ)
        x = X[:, 1]
        sample_entropy[j] = complexity_normalized(SampleEntropy(x), x)
        permutation_entropy[j] = entropy_normalized(OrdinalPatterns(m = 4), x)
        lempel_ziv[j] = lempel_ziv_complexity(x)
    end
end

# %% save data

data = @strdict βs λs σs sample_entropy permutation_entropy fractal_dim_corr fractal_dim_takens lempel_ziv

wsave(datadir("gd", "deterministic_system.jld2"), data)

# %% plot
include("theme.jl")

fig, axs = axesgrid(8, 1; sharex = true, xlabels = "β", ylabels = ["λ1", "λ<", "Δ_C", "Δ_T", "σ", "sample", "pe4", "c_lz"])
scatterlines!(axs[1], βs, getindex.(λs, 1); color = "red")
for index in 2:4
    scatterlines!(axs[2], βs, getindex.(λs, index); color = "gray$(index)0")
end
ylims!(axs[2], -10, 2)
scatterlines!(axs[3], βs, fractal_dim_corr)
scatterlines!(axs[4], βs, fractal_dim_takens)
scatterlines!(axs[5], βs, σs)
scatterlines!(axs[6], βs, sample_entropy)
scatterlines!(axs[7], βs, permutation_entropy)
scatterlines!(axs[8], βs, lempel_ziv)
resize!(fig, 600, 800)

for ax in axs; ax.xscale = log10; end

display(fig)

wsave(plotsdir("gd", "deterministic_quantifiers"), fig)