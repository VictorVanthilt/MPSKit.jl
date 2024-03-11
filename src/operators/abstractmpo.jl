# MPO Tensor types
# ----------------

const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S}
# const SparseMPOTensor{T<:MPOTensor} = BlockTensorMap{S,2,2,T,4} where {S}
# const AbstractMPOTensor{S} = Union{SparseMPOTensor{<:MPOTensor{S}},MPOTensor{S}} where {S}
# VICTOR CHANGES
const SparseMPOTensor = BlockTensorMap{S,2,2,E,4} where {S,E<:Number}
const AbstractMPOTensor{S} = Union{SparseMPOTensor,MPOTensor{S}} where {S}

left_virtualspace(O::AbstractMPOTensor) = space(O, 1)
right_virtualspace(O::AbstractMPOTensor) = space(O, 4)
physicalspace(O::AbstractMPOTensor) = space(O, 2)

left_virtualsize(O::MPOTensor) = 1
left_virtualsize(O::SparseMPOTensor) = size(O, 1)
right_virtualsize(O::MPOTensor) = 1
right_virtualsize(O::SparseMPOTensor) = size(O, 4)

function ismpoidentity(O::MPOTensor{S}; tol=eps(real(scalartype(O)))^3 / 4) where {S}
    O isa BraidingTensor && return true
    τ = TensorKit.BraidingTensor{S,storagetype(O)}(space(O, 2), space(O, 1))
    return space(O) == space(τ) && isapprox(O, τ; atol=tol)
end

# MPO types
# ---------

abstract type AbstractMPO end
abstract type AbstractFiniteMPO <: AbstractMPO end
abstract type AbstractInfiniteMPO <: AbstractMPO end

left_virtualspace(O::AbstractMPO, i::Int) = left_virtualspace(O[i])
right_virtualspace(O::AbstractMPO, i::Int) = right_virtualspace(O[i])
physicalspace(O::AbstractMPO, i::Int) = physicalspace(O[i])
physicalspace(H::AbstractMPO) = ProductSpace(ntuple(i -> physicalspace(H[i]), length(H)))

TensorKit.space(O::AbstractMPO, i::Int) = physicalspace(O, i)

left_virtualsize(H::AbstractMPO, i::Int) = left_virtualsize(H[i])
right_virtualsize(H::AbstractMPO, i::Int) = right_virtualsize(H[i])

TensorKit.spacetype(::Union{O,Type{O}}) where {O<:AbstractMPO} = spacetype(eltype(O))
TensorKit.sectortype(::Union{O,Type{O}}) where {O<:AbstractMPO} = sectortype(eltype(O))
function TensorKit.storagetype(::Union{O,Type{O}}) where {O<:AbstractMPO}
    return storagetype(eltype(O))
end

VectorInterface.scalartype(::Type{O}) where {O<:AbstractMPO} = scalartype(eltype(O))

Base.eltype(O::AbstractMPO) = eltype(parent(O))
Base.length(O::AbstractMPO) = length(parent(O))

Base.iterate(O::AbstractMPO, args...) = iterate(parent(O), args...)

Base.getindex(O::AbstractMPO, i) = getindex(parent(O), i)
Base.setindex!(O::AbstractMPO, v, i) = setindex!(parent(O), v, i)
Base.lastindex(O::AbstractMPO) = lastindex(parent(O))
Base.checkbounds(::Type{Bool}, O::AbstractMPO, args...) = checkbounds(parent(O), args...)

Base.similar(O::MPO, args...) where {MPO<:AbstractMPO} = MPO(similar(parent(O), args...))
Base.copy(O::MPO) where {MPO<:AbstractMPO} = MPO(copy(parent(O)))
Base.deepcopy(O::MPO) where {MPO<:AbstractMPO} = MPO(deepcopy(parent(O)))
Base.repeat(O::MPO, args...) where {MPO<:AbstractMPO} = MPO(repeat(parent(O), args...))

# Utility
# -------

"""
    _deduce_spaces(data::AbstractArray{Union{T,E},3}) where {T<:MPOTensor,E<:Number} -> physicalspaces, virtualspaces

Given an array representation of an MPO, deduce its spaces.
"""
function _deduce_spaces(data::AbstractArray{Union{T,E},3}) where {T<:MPOTensor,E<:Number}
    S = spacetype(T)
    L, left_virtualsz, right_virtualsz = size(data)
    @assert left_virtualsz == right_virtualsz "MPOs should be square"
    virtualspaces = PeriodicArray([Vector{Union{Missing,S}}(missing, left_virtualsz)
                                   for _ in 1:L])
    physicalspaces = Vector{Union{Missing,S}}(missing, L)

    isused = falses(size(data))
    ischanged = true
    while ischanged
        ischanged = false

        for I in CartesianIndices(data)
            isused[I] && continue # skip information that is already known
            i, j, k = I.I
            if data[I] isa T
                P = physicalspace(data[I])
                Vₗ = left_virtualspace(data[I])
                Vᵣ = dual(right_virtualspace(data[I]))

                if ismissing(physicalspaces[i])
                    physicalspaces[i] = P
                else
                    P == physicalspaces[i] ||
                        throw(ArgumentError("physical space mismatch at $(I.I)"))
                end

                if ismissing(virtualspaces[i][j])
                    virtualspaces[i][j] = Vₗ
                else
                    Vₗ == virtualspaces[i][j] ||
                        throw(ArgumentError("left virtual space mismatch at $(I.I)"))
                end

                if ismissing(virtualspaces[i + 1][k])
                    virtualspaces[i + 1][k] = Vᵣ
                else
                    Vᵣ == virtualspaces[i + 1][k] ||
                        throw(ArgumentError("right virtual space mismatch at $(I.I)"))
                end
                isused[I] = true

                # not necessarily changed, but tensors are only checked first time around
                # so something should definitely have changed
                ischanged = true

            elseif !iszero(data[I])
                ismissing(virtualspaces[i][j]) && ismissing(virtualspaces[i + 1][k]) &&
                    continue

                if ismissing(virtualspaces[i][j])
                    # left space can be deduced from right
                    Vᵣ = virtualspaces[i + 1][k]
                    Vₗ = Vᵣ
                    virtualspaces[i][j] = Vₗ
                    ischanged = true
                elseif ismissing(virtualspaces[i + 1][k])
                    # right space can be deduced from left
                    Vₗ = virtualspaces[i][j]
                    Vᵣ = Vₗ
                    virtualspaces[i + 1][k] = Vᵣ
                    ischanged = true
                else
                    # both spaces are assigned
                    Vₗ = virtualspaces[i][j]
                    Vᵣ = virtualspaces[i + 1][k]
                    Vₗ == Vᵣ ||
                        throw(ArgumentError("virtual space mismatch at $(I.I)"))
                end

                # check that braiding is possible
                _can_unambiguously_braid(Vₗ) ||
                    throw(ArgumentError("Ambiguous braiding operator at $(I.I)"))

                isused[I] = true
            end
        end
    end

    # check that all spaces are assigned
    for i in 1:L
        ismissing(physicalspaces[i]) &&
            throw(ArgumentError("Physical space at $i is not assigned"))
        replace!(virtualspaces[i], missing => oneunit(S)) # this should not cause problems?
    end

    return physicalspaces, virtualspaces
end

function _normalize_mpotypes(data::AbstractArray{<:Any,3})
    tensortypes = Set()
    scalartypes = Set()
    for x in data
        if x isa AbstractTensorMap
            if numin(x) == numout(x) == 2
                push!(tensortypes, typeof(x))
            elseif numin(x) == numout(x) == 1
                push!(tensortypes, typeof(add_util_leg(x)))
            else
                throw(ArgumentError("$(typeof(x)) is not an MPO tensor or a single-site tensor"))
            end
        elseif x isa Number
            push!(scalartypes, typeof(x))
        else
            ismissing(x) ||
                throw(ArgumentError("data should only contain mpo tensors or scalars"))
        end
    end
    @show T = Union{tensortypes...}
    E = promote_type(scalartype(T), scalartypes...)

    # convert data
    data′ = similar(data, Union{E,T})
    for (i, x) in enumerate(data)
        if x isa AbstractTensorMap
            if numin(x) == numout(x) == 2
                data′[i] = convert(T, x)
            elseif numin(x) == numout(x) == 1
                data′[i] = convert(T, add_util_leg(x))
            else
                error("this should not happen")
            end
        elseif x isa Number
            data′[i] = convert(E, x)
        else
            data′[i] = zero(E)
        end
    end
    return data′
end
