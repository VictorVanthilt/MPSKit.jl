using MPSKit
using Test

using TestEnv;
TestEnv.activate("MPSKit");

include("setup.jl")

psi = FiniteMPS((ℂ^2)^10, ℂ^4)
H = transverse_field_ising(; g=0.1)

envs = environments(psi, H)

ψ, envs, δ = find_groundstate(psi, H)
@show E = sum(expectation_value(ψ, H))
dt = 0.1
for t in 0:dt:10
    if t == 0
        ψ2 = ψ
    else
        ψ2, envs = timestep!(ψ2, H, dt, TDVP(), envs)
    end
    @show dot(ψ, ψ2) exp(-im * t * E)
end
ψ2, envs = timestep!(ψ, H, 0.1, TDVP(), envs)

H_mpo = make_time_mpo(H, 0.1, TaylorCluster{2}())

# Union type tests
