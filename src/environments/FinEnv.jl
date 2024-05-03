"
    FinEnv keeps track of the environments for FiniteMPS / WindowMPS
    It automatically checks if the queried environment is still correctly cached and if not - recalculates

    if above is set to nothing, above === below.

    opp can be a vector of nothing, in which case it'll just be the overlap
"
struct FinEnv{A,B,C,D} <: Cache
    above::A
    opp::B # the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{D}
    rightenvs::Vector{D}
end

VectorInterface.scalartype(::Type{<:FinEnv{A,B,C,D}}) where {A,B,C,D} = scalartype(D)

"""
    FinEnv(ψ::AbstractFiniteMPS, O::Union{InfiniteMPO,MPOHamiltonian})

Initialize environments.
"""
function FinEnv(ψ::AbstractFiniteMPS,
                O::AbstractMPO,
                above::Union{Nothing,AbstractFiniteMPS})
    # Initialize left environment tensors
    GL = map(0:length(ψ)) do i
        Vbot = left_virtualspace(ψ, i)
        Vmpo = left_virtualspace(O, i)
        Vtop = isnothing(above) ? Vbot : left_virtualspace(above, i)
        return BlockTensorMap(undef, scalartype(ψ), Vbot ⊗ Vmpo' ← Vtop)
    end

    # Initialize right environment tensors
    GR = map(0:length(ψ)) do i
        Vbot = right_virtualspace(ψ, i)
        Vmpo = right_virtualspace(O, i)
        Vtop = isnothing(above) ? Vbot : right_virtualspace(above, i)
        return BlockTensorMap(undef, scalartype(ψ), Vtop ⊗ Vmpo' ← Vbot)
    end

    # Initialize dependency vectors
    left_deps = fill!(similar(ψ.AL), similar(ψ.AL[1]))
    right_deps = fill!(similar(ψ.AR), similar(ψ.AR[1]))

    return FinEnv(above, O, left_deps, right_deps, GL, GR)
end

function FinEnv(ψ::AbstractFiniteMPS, O::Vector{Nothing},
                above::Union{Nothing,AbstractFiniteMPS})
    # Initialize left environment tensors
    GL = map(0:length(ψ)) do i
        Vbot = left_virtualspace(ψ, i)
        Vtop = isnothing(above) ? Vbot : left_virtualspace(above, i)
        return TensorMap(undef, scalartype(ψ), Vbot ← Vtop)
    end

    # Initialize right environment tensors
    GR = map(0:length(ψ)) do i
        Vbot = right_virtualspace(ψ, i)
        Vtop = isnothing(above) ? Vbot : right_virtualspace(above, i)
        return TensorMap(undef, scalartype(ψ), Vtop ← Vbot)
    end

    # Initialize dependency vectors
    left_deps = fill!(similar(ψ.AL), similar(ψ.AL[1]))
    right_deps = fill!(similar(ψ.AR), similar(ψ.AR[1]))

    return FinEnv(ψ, O, left_deps, right_deps, GL, GR)
end

function environments(below, t::Tuple, args...; kwargs...)
    return environments(below, t[1], t[2], args...; kwargs...)
end

function environments(ψ::FiniteMPS, O::Union{InfiniteMPO,MPOHamiltonian}, top=nothing)
    envs = FinEnv(ψ, O, top)

    # left boundary: [1 0 0]
    I = CartesianIndex(1, 1, 1)
    GL = envs.leftenvs[1]
    GL[I] = isometry(storagetype(GL), BlockTensorKit.getsubspace(space(GL), I))

    # right boundary: [1 0 0]' (InfiniteMPO) or [0 0 1]' (MPOHamiltonian)
    GR = envs.rightenvs[end]
    I = O isa InfiniteMPO ? CartesianIndex(1, 1, 1) : CartesianIndex(1, lastindex(GR, 2), 1)
    GR[I] = isometry(storagetype(GR), BlockTensorKit.getsubspace(space(GR), I))

    return envs
end

function environments(state::WindowMPS,
                      O::Union{SparseMPO,MPOHamiltonian,DenseMPO},
                      above=nothing;
                      lenvs=environments(state.left_gs, O, above),
                      renvs=environments(state.right_gs, O, above))
    (O isa MPOHamiltonian && (isnothing(above) || above === state)) ||
        throw(ArgumentError("MPOHamiltonian requires top and bottom states to be equal."))
    envs = FinEnv(state, O, above)

    # left boundary: extract from left_envs
    envs.leftenvs[1] = leftenv(lenvs, 1, state.left_gs)

    # right boundary: extract from right_envs
    envs.rightenvs[end] = rightenv(renvs, length(state), state.right_gs)

    return envs
end

function environments(below::S, above::S) where {S<:Union{FiniteMPS,WindowMPS}}
    S isa WindowMPS &&
        (above.left_gs == below.left_gs || throw(ArgumentError("left gs differs")))
    S isa WindowMPS &&
        (above.right_gs == below.right_gs || throw(ArgumentError("right gs differs")))

    opp = fill(nothing, length(below))
    envs = FinEnv(below, opp, above)
    envs.leftenvs[1] = l_LL(above)
    envs.rightenvs[end] = r_RR(above)

    return envs
end

function environments(state::Union{FiniteMPS,WindowMPS}, opp::ProjectionOperator)
    @plansor leftstart[-1; -2 -3 -4] := l_LL(opp.ket)[-3; -4] * l_LL(opp.ket)[-1; -2]
    @plansor rightstart[-1; -2 -3 -4] := r_RR(opp.ket)[-1; -2] * r_RR(opp.ket)[-3; -4]
    return environments(state, fill(nothing, length(state)), state, leftstart, rightstart)
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::FinEnv, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(cache::FinEnv, ind, ψ)
    0 <= ind <= length(ψ) || throw(BoundsError(cache, ind))
    a = findlast(i -> ψ.AR[i] !== cache.rdependencies[i], (ind + 1):length(ψ))

    if !isnothing(a) # we need to recalculate
        for j in (a + ind):-1:(ind + 1)
            above = isnothing(cache.above) ? ψ.AR[j] : cache.above.AR[j]
            cache.rightenvs[j] = TransferMatrix(above, cache.opp[j], ψ.AR[j]) *
                                 cache.rightenvs[j + 1]
            cache.rdependencies[j] = ψ.AR[j]
        end
    end

    return cache.rightenvs[ind + 1]
end

function leftenv(cache::FinEnv, ind, ψ)
    0 <= ind <= length(ψ) || throw(BoundsError(cache, ind))
    a = findfirst(i -> ψ.AL[i] !== cache.ldependencies[i], 1:(ind - 1))

    if !isnothing(a) # we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(cache.above) ? ψ.AL[j] : cache.above.AL[j]
            cache.leftenvs[j + 1] = cache.leftenvs[j] *
                                    TransferMatrix(above, cache.opp[j], ψ.AL[j])
            cache.ldependencies[j] = ψ.AL[j]
        end
    end

    return cache.leftenvs[ind]
end
