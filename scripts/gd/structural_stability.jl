include("deterministic_system_definition.jl")

using Statistics: std, mean

# re-do fig 2
pcpds = deterministic_pcp_system()
βs = [0.05, 5, 500]

original_payoff_matrix = [
    0.0 -0.6 0.0 1.0
    1.0 0.0 0.0 -0.5
    -1.05 -0.2 0.0 1.75
    0.5 -0.1 0.1 0.0
]

σ = 0.1 # chosen standard deviation for normal distribution perturbing non-zero parameters.
N = 10 # random samples for each β value

function obtain_quantifiers(ds)
    β = current_parameter(ds, :β)
    τ = 2/β # use system timescale. Should it be at most some value for large beta?
    λ1 = lyapunovspectrum(ds, 20_000; Δt = τ)[1]
    X, tvec = trajectory(ds, 10000.0*τ; Δt = τ)
    fractal_dim_corr = grassberger_proccacia_dim(X; show_progress = false)
    σ = std(X[:, 1])
    X, tvec = trajectory(ds, 1000.0*τ; Δt = τ/10)
    x = X[:, 1]
    lempel_ziv = lempel_ziv_complexity(x)
    return λ1, fractal_dim_corr, σ, lempel_ziv
end

# now run the above function for the original pay off matrix,
# and then repeat for each beta values `N` times

original_quantities = map(βs) do β
    set_parameter!(pcpds, :β, β)
    obtain_quantifiers(pcpds)
end

# todo...