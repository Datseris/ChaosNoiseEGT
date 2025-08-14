using DrWatson
@quickactivate

using DynamicalSystems
using LinearAlgebra: mul!
using OrdinaryDiffEqVerner: Vern9
# This is some internal magic for pre-allocating the payoff matrix multiplication
import DynamicalSystems.DynamicalSystemsBase.ForwardDiff: Dual

mutable struct DeterministicPCPConfig{T, FD}
    β::T
    payoff_matrix::Matrix{T}
    f::Vector{T} # pre-initialized container to store fitness values
    fdual::FD
end

DeterministicPCPConfig(β, m) = DeterministicPCPConfig(β, m, zeros(eltype(m), size(m, 1)), nothing)
DeterministicPCPConfig(β, m, fd) = DeterministicPCPConfig(β, m, zeros(eltype(m), size(m, 1)), fd)

"""
        dynamic_rule_PCP!(du, u, p::DeterministicPCPConfig, t)

Dynamic rule of the deterministic system.
General deterministic description of the Pairwise Comparison Process
It consists of a set of n ordinary differential equations.
n is defined by the dimension of the nxn payoff matrix
The dynamic rule is defined in-place to allow for arbitrarily-large state space.

The parameter container `p` must be an instance of `DeterministicPCPConfig`
that contains the `β` value and the payoff matrix as its two fields.
The size of the payoff matrix must match the length of `u`.
If not, a memory leak will silently close Julia.
"""
function dynamic_rule_PCP!(du, u, p::DeterministicPCPConfig, t)
    (; β, payoff_matrix) = p
    n = length(u)
    f = eltype(u) <: Dual ? p.fdual : p.f

    # update fitness values
    mul!(f, payoff_matrix, u)

    # equations of motion
    @inbounds for i in 1:n
        inner_term = zero(eltype(u))
        for j in 1:n
            if i != j
                element_term = u[j]*tanh((β/2.0)*(f[i]-f[j]))
                inner_term += element_term
            end
            du[i] = u[i]*inner_term
        end
    end

    return nothing
end

function deterministic_pcp_system(;
        β = 1.0, payoff_matrix = [
            0.0 -0.6 0.0 1.0
            1.0 0.0 0.0 -0.5
            -1.05 -0.2 0.0 1.75
            0.5 -0.1 0.1 0.0
        ],
        u0 = fill(0.25, size(payoff_matrix, 1)),
    )
    p = DeterministicPCPConfig(β, payoff_matrix)
    ds = CoupledODEs(dynamic_rule_PCP!, u0, p; diffeq=(alg = Vern9(), abstol = 1.0e-9,reltol = 1.0e-9))
    # preinitialize dual container for in-place matrix mul
    J = jacobian(ds)
    fdummy = fill(first(J.cfg.duals[2]), size(payoff_matrix, 2))
    p = DeterministicPCPConfig(β, payoff_matrix, fdummy)
    ds = CoupledODEs(dynamic_rule_PCP!, u0, p; diffeq=(alg = Vern9(), abstol = 1.0e-9,reltol = 1.0e-9))
    return ds
end

function lempel_ziv_complexity(x)
    m = mean(x)
    y = map(v -> v < m ? 0 : 1, x)
    return complexity_normalized(LempelZiv76(), y)
end