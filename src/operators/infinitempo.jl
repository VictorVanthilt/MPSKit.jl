"
    SparseMPO - used to represent both time evolution mpos and hamiltonians
"

struct InfiniteMPO{T<:AbstractMPOTensor} <: AbstractInfiniteMPO
    data::PeriodicVector{T}

    # regular constructor
    function InfiniteMPO{T}(data::PeriodicVector{T}) where {T<:AbstractMPOTensor}
        return new{T}(data)
    end

    # constructor with guaranteed space checks
    function InfiniteMPO(data::PeriodicVector{T}) where {T<:AbstractMPOTensor}
        for i in eachindex(data)
            Vₗ = left_virtualspace(data[i])
            Vᵣ = dual(right_virtualspace(data[i - 1]))
            Vₗ == Vᵣ ||
                throw(SpaceMismatch("Incompatible virtual spaces at $i:\n$Vₗ ≠ $Vᵣ"))
            space(data[i], 2) == dual(space(data[i], 3)) ||
                throw(TensorKit.SpaceMismatch("Incompatible physical spaces at $i"))
        end
        return new{T}(data)
    end
end

const SparseMPO{T<:SparseMPOTensor} = InfiniteMPO{T}
const DenseMPO{T<:MPOTensor} = InfiniteMPO{T}

# Constructors
# ------------

function InfiniteMPO(data::AbstractVector{<:AbstractMPOTensor})
    return InfiniteMPO(PeriodicArray(data))
end

function InfiniteMPO(data::AbstractArray{Union{T,E},3}) where {T<:MPOTensor,E<:Number}
    @assert scalartype(T) == E "scalar type should match mpo scalartype"
    L = size(data, 1)

    # deduce spaces from tensors
    S = spacetype(T)
    physicalspaces, virtualspaces = _deduce_spaces(data)

    # construct blocktensors
    τtype = TensorKit.BraidingTensor{S,TensorKit.storagetype(T)}
    ttype = Union{T,τtype}

    Ws = map(1:L) do i
        Vₗ = SumSpace(virtualspaces[i]...)
        Vᵣ = SumSpace(virtualspaces[i + 1]...)
        P = SumSpace(physicalspaces[i])
        tdst = BlockTensorMap{scalartype(T),S,2,2}(undef, Vₗ ⊗ P, P ⊗ Vᵣ)
        for j in axes(data, 2), k in axes(data, 3)
            if data[i, j, k] isa E
                iszero(data[i, j, k]) && continue
                τ = τtype(domain(BlockTensorKit.getsubspace(space(tdst), j, 1, 1, k))...)
                if isone(data[i, j, k])
                    tdst[j, 1, 1, k] = τ
                else
                    tdst[j, 1, 1, k] = scale!(τ, data[i, j, k])
                end
            else
                if ismpoidentity(data[i, j, k])
                    tdst[j, 1, 1, k] = τtype(domain(BlockTensorKit.getsubspace(space(tdst),
                                                                               j, 1, 1, k))...)
                else
                    tdst[j, 1, 1, k] = data[i, j, k]
                end
            end
        end
        return tdst
    end

    return InfiniteMPO(Ws)
end

# Attempt to deduce eltype information for non-strictly typed data
InfiniteMPO(data::AbstractArray{<:Any,3}) = InfiniteMPO(_normalize_mpotypes(data))

# Properties
# ----------
Base.parent(O::InfiniteMPO) = O.data

# promotion and conversion
# ------------------------

function Base.promote_rule(::Type{InfiniteMPO{T₁}}, ::Type{InfiniteMPO{T₂}}) where {T₁,T₂}
    return InfiniteMPO{promote_type(T₁, T₂)}
end

function Base.convert(::Type{InfiniteMPO{T₁}}, x::InfiniteMPO{T₂}) where {T₁,T₂}
    T₁ === T₂ && return x
    return InfiniteMPO{T₁}(map(Base.Fix1(convert, T₁), x.data))
end
function Base.convert(::Type{DenseMPO}, x::InfiniteMPO)
    data = map(Base.Fix1(convert, MPOTensor{spacetype(x)}), parent(x))
    # filter out zero blocks -> necessary to avoid problems with environments (?)
    for i in 1:length(x)
        U, S, V, = tsvd!(transpose(data[i], ((3, 1, 2), (4,)));
                         trunc=truncbelow(eps(real(scalartype(x)))^(4 / 5)))
        data[i] = transpose(U, ((2, 3), (1, 4)))
        @plansor data[i + 1][-1 -2; -3 -4] := S[-1; 1] * V[1; 2] * data[i + 1][2 -2; -3 -4]

        U′, S′, V′, = tsvd!(transpose(data[i], ((1,), (3, 4, 2)));
                            trunc=truncbelow(eps(real(scalartype(x)))^(4 / 5)))
        data[i] = transpose(V′, ((1, 4), (2, 3)))
        @plansor data[i - 1][-1 -2; -3 -4] := data[i - 1][-1 -2; -3 1] * U′[1; 2] *
                                              S′[2; -4]
    end

    return InfiniteMPO(data)
end

function Base.convert(::Type{InfiniteMPO}, ψ::InfiniteMPS)
    return InfiniteMPO(map(ψ.AL) do al
                           @plansor tt[-1 -2; -3 -4] := al[-1 -2 1; 2] * τ[-3 2; -4 1]
                       end)
end

function Base.convert(::Type{InfiniteMPS}, O::InfiniteMPO)
    return InfiniteMPS(map(parent(O)) do o
                           @plansor A[-1 -2 -3; -4] := o[-1 -2; 1 2] * τ[1 2; -4 -3]
                       end)
end

# Linear Algebra
# --------------

function Base.:*(O::InfiniteMPO, ψ::InfiniteMPS)
    length(O) == length(ψ) ||
        throw(ArgumentError("Period mismatch: $(length(O)) ≠ $(length(ψ))"))
    Fs = PeriodicArray(fuser.(scalartype(ψ), left_virtualspace.(ψ.AL),
                              left_virtualspace.(parent(O))))
    AL = map(1:length(O)) do i
        @plansor t[-1 -2; -3] := ψ.AL[i][1 2; 3] * O[i][4 -2; 2 5] * Fs[i][-1; 1 4] *
                                 conj(Fs[i + 1][-3; 3 5])
    end
    return InfiniteMPS(AL)
end

Base.:(*)(O₁::InfiniteMPO, O₂::InfiniteMPO) = *(promote(O₁, O₂)...)
function Base.:*(O₁::InfiniteMPO{T}, O₂::InfiniteMPO{T}) where {T<:AbstractMPOTensor}
    length(O₁) == length(O₂) ||
        throw(ArgumentError("Period mismatch: $(length(O₁)) ≠ $(length(O₂))"))

    S = spacetype(T)
    E = scalartype(T)
    Fs = PeriodicArray(fuser.(E, left_virtualspace.(parent(O₁)),
                              left_virtualspace.(parent(O₂))))

    O = map(1:length(O₁), parent(O₁), parent(O₂)) do i, o₁, o₂
        @plansor o[-1 -2; -3 -4] := Fs[i][-1; 1 2] * o₁[1 5; -3 3] * o₂[2 -2; 5 4] *
                                    conj(Fs[i + 1][-4; 3 4])
        if T <: SparseMPOTensor
            # restore sparsity -> when both factors are braidingtensors, we know that the
            # result can again be represented as a braidingtensor
            cinds = CartesianIndices((size(o₁, 1), size(o₂, 1)))
            for j in axes(o, 1), k in axes(o, 4)
                rowinds = cinds[j]
                colinds = cinds[k]
                Ia = CartesianIndex(rowinds[1], 1, 1, colinds[1])
                Ib = CartesianIndex(rowinds[2], 1, 1, colinds[2])
                if (haskey(o₁, Ia) && ismpoidentity(o₁[Ia])) &&
                   (haskey(o₂, Ib) && ismpoidentity(o₂[Ib]))
                    V = getsubspace(space(o), CartesianIndex(j, 1, 1, k))
                    o[j, 1, 1, k] = BraidingTensor{S,Matrix{E}}(V[2], V[1])
                end
            end
        end
        return o
    end

    return InfiniteMPO(O)
end

function Base.conj(O::InfiniteMPO)
    data′ = map(parent(O)) do o
        @plansor o′[-1 -2; -3 -4] := conj(o[-1 -3; -2 -4])
    end
    return InfiniteMPO(data′)
end

function TensorKit.dot(O₁::InfiniteMPO, O₂::InfiniteMPO)
    length(O₁) == length(O₂) ||
        throw(ArgumentError("Period mismatch: $(length(O₁)) ≠ $(length(O₂))"))
    return TensorKit.dot(convert.(InfiniteMPS, (O₁, O₂))...)
end

function TensorKit.dot(ψ₁::InfiniteMPS, O::InfiniteMPO, ψ₂::InfiniteMPS;
                       krylovdim=30)
    length(ψ₁) == length(O) == length(ψ₂) ||
        throw(ArgumentError("Period mismatch: $(length(ψ₁)) ≠ $(length(O)) ≠ $(length(ψ₂))"))

    T = TransferMatrix(ψ₂.AL, parent(O), ψ₁.AL)
    ρ₀ = similar(O[1], _firstspace(ψ₁.AL[1]) ⊗ _firstspace(O[1]) ← _firstspace(ψ₂.AL[1]))
    vals, _, info = eigsolve(T, ρ₀, 1, :LM, Arnoldi(; krylovdim=krylovdim))
    info.converged == 0 && @warn "dot mps not converged"
    return first(vals)
end

# Utility
# -------

Base.eltype(::Type{InfiniteMPO{T}}) where {T} = T

"""
    remove_orphans(mpo::SparseMPO)

Prune all branches of the finite state machine that do not contribute. Additionally, attempt
to compact the representation as much as possible.
"""
function remove_orphans!(O::SparseMPO; tol=eps(real(scalartype(O)))^(3 / 4))
    # drop zeros
    for slice in parent(O)
        for (key, val) in nonzero_pairs(slice)
            norm(val) < tol && delete!(slice, key)
        end
    end

    # drop dead starts/ends
    changed = true
    while changed
        changed = false
        for i in 1:length(O)
            # slice empty columns on right or empty rows on left
            mask = filter(1:size(O[i], 4)) do j
                return j ∈ getindex.(nonzero_keys(O[i]), 1) ||
                       j ∈ getindex.(nonzero_keys(O[i + 1]), 4)
            end
            changed |= length(mask) == size(O[i], 4)
            O[i] = O[i][:, :, :, mask]
            O[i + 1] = O[i + 1][mask, :, :, :]
        end
    end

    return O
end

"""
    add_physical_charge(O::Union{SparseMPO{S},MPOHamiltonion{S}},
                        auxspaces::AbstractVector{I}) where {S,I}

create an operator which passes an auxiliary space.
"""
function add_physical_charge(O::InfiniteMPO, charges::AbstractVector{I}) where {I<:Sector}
    length(charges) == length(O) || throw(ArgumentError("unmatching lengths"))
    sectortype(S) == I ||
        throw(ArgumentError("Unmatching chargetypes $sectortype(S) and $I"))

    auxspaces = map(c -> Vect[I](c => 1), charges)
    O_new = copy(O)
    O_new.pspaces .= fuse.(O.pspaces, auxspaces)

    for i in eachindex(O.pspaces)
        F = unitary(O_new.pspaces[i] ← O.pspaces[i] ⊗ auxspaces[i])
        for (j, k) in MPSKit.opkeys(O_new[i])
            @plansor begin
                O_new[i][j, k][-1 -2; -3 -4] := F[-2; 1 2] * O[i][j, k][-1 1; 4 3] *
                                                τ[3 2; 5 -4] * conj(F[-3; 4 5])
            end
        end
    end

    return O_new
end

Base.isone(x::SparseMPOTensor, I::Vararg{Int,4}) = isone(x.scalars[I...])
function Base.iszero(x::SparseMPOTensor, I::Vararg{Int,4})
    return iszero(x.scalars[I...]) && !haskey(x.tensors.data.data, CartesianIndex(I))
end
