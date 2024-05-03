"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian

    really just a sparsempo, with some garantuees on its structure
"

struct MPOHamiltonian{T<:SparseMPOTensor} <: AbstractMPO
    data::PeriodicVector{T}

    # regular constructor
    function MPOHamiltonian{T}(data::PeriodicVector{T}) where {T<:SparseMPOTensor}
        return new{T}(data)
    end

    # constructor with guaranteed space checks and structure checks
    function MPOHamiltonian(data::PeriodicVector{T}) where {T<:SparseMPOTensor}
        for i in eachindex(data)
            Vₗ = left_virtualspace(data[i])
            Vᵣ = dual(right_virtualspace(data[i - 1]))
            Vₗ == Vᵣ ||
                throw(SpaceMismatch("Incompatible virtual spaces at $i:\n$Vₗ ≠ $Vᵣ"))
            space(data[i], 2) == dual(space(data[i], 3)) ||
                throw(TensorKit.SpaceMismatch("Incompatible physical spaces at $i"))
            isjordanstructure(data[i]) ||
                throw(ArgumentError("MPOHamiltonian should be in Jordan form ($i)"))
        end
        return new{T}(data)
    end
end

# Constructors
# ------------

function MPOHamiltonian(data::AbstractVector{<:SparseMPOTensor})
    return MPOHamiltonian(PeriodicArray(data))
end

function MPOHamiltonian(data::AbstractArray{Union{T,E},3}) where {T<:MPOTensor,E<:Number}
    @assert scalartype(T) == E "scalar type should match mpo scalartype"
    L = size(data, 1)

    # deduce spaces from tensors
    S = spacetype(T)
    physicalspaces, virtualspaces = _deduce_spaces(data)

    # construct blocktensors
    τtype = BraidingTensor{E,S}
    ttype = Union{T,τtype}

    Ws = map(1:L) do i
        Vₗ = SumSpace(virtualspaces[i]...)
        Vᵣ = SumSpace(virtualspaces[i + 1]...)
        P = SumSpace(physicalspaces[i])
        tdst = BlockTensorMap{E,S,2,2}(undef, Vₗ ⊗ P, P ⊗ Vᵣ)
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
    return MPOHamiltonian(Ws)
end

# Attempt to deduce eltype information for non-strictly typed data
MPOHamiltonian(data::AbstractArray{<:Any,3}) = MPOHamiltonian(_normalize_mpotypes(data))

# Construct from local operators
function MPOHamiltonian(local_operator::TensorMap{E,S,N,N}) where {E,S,N}
    return MPOHamiltonian(decompose_localmpo(add_util_leg(local_operator)))
end
function MPOHamiltonian(local_mpo::Vector{O}) where {O<:MPOTensor}
    allequal(physicalspace.(local_mpo)) ||
        throw(ArgumentError("all physical spaces should be equal"))
    S = spacetype(O)
    V₀ = oneunit(S)
    P = physicalspace(local_mpo[1])

    τ = BraidingTensor{scalartype(O),S}(P, V₀)
    ttype = Union{O,typeof(τ)}

    Vₗ = push!(left_virtualspace.(local_mpo), dual(right_virtualspace(local_mpo[end])))
    Vᵣ = pushfirst!(dual.(right_virtualspace.(local_mpo)), left_virtualspace(local_mpo[1]))
    W = BlockTensorMap{scalartype(O),S,2,2}(undef, SumSpace(Vₗ) ⊗ P, P ⊗ SumSpace(Vᵣ))

    W[1, 1, 1, 1] = τ
    W[end, 1, 1, end] = τ
    for (i, o) in enumerate(local_mpo)
        W[i, 1, 1, i + 1] = o
    end

    return MPOHamiltonian([W])
end

# Converters
# ----------
# TODO: this assumes Finite mpo hamiltonian, split off these cases.
function Base.convert(::Type{T}, H::MPOHamiltonian) where {T<:TensorMap}
    N = length(H)
    # add trivial tensors to remove left and right trivial leg.
    V_left = left_virtualspace(H, 1)
    U_left = similar(H[1], V_left)'
    U_left[1] = fill!(U_left[1], one(scalartype(H)))

    V_right = right_virtualspace(H, length(H))
    U_right = similar(H[end], V_right')
    U_right[end] = fill!(U_right[end], one(scalartype(H)))

    tensors = [U_left, H.data..., U_right]
    indices = [[i, -i, -(i + N), i + 1] for i in 1:N]
    pushfirst!(indices, [1])
    push!(indices, [N + 1])

    O = ncon(tensors, indices)

    return permute(O[1], (ntuple(identity, N), ntuple(i -> i + N, N)))
end

# Properties
# ----------

function Base.getproperty(H::MPOHamiltonian, sym::Symbol)
    if sym === :A
        return map(h -> h[2:(end - 1), 1, 1, 2:(end - 1)], H.data)
    elseif sym === :B
        return map(h -> h[2:(end - 1), 1, 1, end], H.data)
    elseif sym === :C
        return map(h -> h[1, 1, 1, 2:(end - 1)], H.data)
    elseif sym === :D
        return map(h -> h[1, 1, 1, end], H.data)
    else
        return getfield(H, sym)
    end
end

Base.eltype(::Type{MPOHamiltonian{T}}) where {T} = T
Base.parent(H::MPOHamiltonian) = H.data

function Base.show(io::IO, ::MIME"text/plain", H::Union{MPOHamiltonian,SparseMPO})
    typestr = H isa MPOHamiltonian ? "MPOHamiltonian" : "SparseMPO"
    print(io, "$(length(H))-periodic $(typestr):")
    print(io, " ⋯ ")
    Base.join(io, physicalspace(H), " ⊗ ")
    println(io, " ⋯")
    foreach(((i, W),) -> println(io, " W[$i] = ", W), enumerate(H.data))
    return nothing
end

"
checks if ham[:,i,i] = 1 for every i
"
function Base.isone(H::MPOHamiltonian, i::Int)
    I = CartesianIndex(i, 1, 1, i)
    return all(x -> haskey(x, I) && ismpoidentity(x[I]), parent(H))
end

function Base.iszero(H::MPOHamiltonian, i::Int)
    I = CartesianIndex(i, 1, 1, i)
    return any(x -> !haskey(x, I), parent(H))
end

# addition / substraction
Base.:-(H::MPOHamiltonian) = -one(scalartype(H)) * H
function Base.:+(H::MPOHamiltonian, λs::AbstractVector{<:Number})
    check_length(H, λs)
    H′ = deepcopy(H)
    foreach(H′.data, λs) do h, λ
        D = h[1, 1, 1, end]
        return h[1, 1, 1, end] = add(D,
                                     BraidingTensor{spacetype(H),storagetype(H)}(domain(D)...),
                                     λ)
    end
    return H′
end

Base.:-(e::AbstractVector, a::MPOHamiltonian) = (-a) + e
Base.:+(e::AbstractVector, a::MPOHamiltonian) = a + e
Base.:-(a::MPOHamiltonian, e::AbstractVector) = a + (-e)

# Base.:+(a::H1, b::H2) where {H1<:MPOHamiltonian,H2<:MPOHamiltonian} = +(promote(a, b)...)
function Base.:+(a::MPOHamiltonian{T}, b::MPOHamiltonian{T}) where {T}
    check_length(a, b)

    # @assert sanitycheck(a) "a is not a valid hamiltonian"
    # @assert sanitycheck(b) "b is not a valid hamiltonian"

    Hnew = map(a.data, b.data) do h1, h2
        Vₗ₁ = left_virtualspace(h1)
        Vₗ₂ = left_virtualspace(h2)
        @assert Vₗ₁[1] == Vₗ₂[1] && Vₗ₁[end] == Vₗ₂[end] "trivial spaces should match"
        Vₗ = Vₗ₁[1:(end - 1)] ⊕ Vₗ₂[2:end]

        Vᵣ₁ = right_virtualspace(h1)
        Vᵣ₂ = right_virtualspace(h2)
        @assert Vᵣ₁[1] == Vᵣ₂[1] && Vᵣ₁[end] == Vᵣ₂[end] "trivial spaces should match"
        Vᵣ = Vᵣ₁[1:(end - 1)] ⊕ Vᵣ₂[2:end]

        Wnew = T(undef, Vₗ ⊗ space(h1, 2), space(h1, 3)' ⊗ Vᵣ')

        # add blocks from first hamiltonian
        for (I, O) in nonzero_pairs(h1)
            if I[1] == 1
                if I[4] == 1
                    # 1 block
                    Wnew[I] = O
                elseif I[4] == size(h1, 4)
                    # D block
                    Wnew[1, 1, 1, end] = O
                else
                    # C block
                    Wnew[I] = O
                end
            elseif I[4] == size(h1, 4)
                if I[1] == size(h1, 1)
                    # 1 block
                    Wnew[I[1], 1, 1, end] = O
                else
                    # B block
                    Wnew[I[1], 1, 1, end] = O
                end
            else
                # A block
                Wnew[I] = O
            end
        end

        # add blocks from second hamiltonian
        for (I, O) in nonzero_pairs(h2)
            if I[1] == 1
                if I[4] == 1
                    # 1 block - already done
                elseif I[4] == size(h2, 4)
                    # D block
                    Wnew[1, 1, 1, end] += O
                else
                    # C block
                    shift = CartesianIndex(0, 0, 0, size(h1, 4) - 2)
                    Wnew[I + shift] = O
                end
            elseif I[4] == size(h2, 4)
                if I[1] == size(h1, 1)
                    # 1 block - already done
                else
                    # B block
                    shift = CartesianIndex(size(h1, 1) - 2, 0, 0, size(h1, 4) - 2)
                    Wnew[I + shift] = O
                end
            else
                # A block
                shift = CartesianIndex(size(h1, 1) - 2, 0, 0, size(h1, 4) - 2)
                Wnew[I + shift] = O
            end
        end
        return Wnew
    end

    return MPOHamiltonian(Hnew)
end

Base.:-(a::MPOHamiltonian, b::MPOHamiltonian) = a + (-b)

#multiplication
Base.:*(λ::Number, H::MPOHamiltonian) = H * λ
function Base.:*(H::MPOHamiltonian, λ::Number)
    Hλ = copy(H)
    foreach(Hλ.data) do h
        # multiply scalar with start of every interaction
        # this avoids double counting
        # 2:end to avoid multiplying the top left and bottom right corners
        return rmul!(h[1, 1, 1, 2:end], λ)
    end
    return Hλ
end

function Base.:*(a::H1, b::H2) where {H1<:MPOHamiltonian,H2<:MPOHamiltonian}
    a′, b′ = promote(a, b)
    return *(a′, b′)
end
function Base.:*(b::H, a::H) where {H<:MPOHamiltonian}
    T = eltype(b.data)
    S = spacetype(T)
    check_length(b, a)

    E = promote_type(scalartype(b), scalartype(a))
    Fs = PeriodicArray(fuser.(E, left_virtualspace.(parent(a)),
                              left_virtualspace.(parent(b))))
    C = similar(b.data)
    for i in 1:length(b)
        C[i] = T(undef,
                 space(Fs[i], 1) ⊗ physicalspace(b, i) ←
                 physicalspace(a, i) ⊗ space(Fs[i + 1], 1))
        @plansor C[i][-1 -2; -3 -4] = Fs[i][-1; 1 2] * a[i][1 5; -3 3] * b[i][2 -2; 5 4] *
                                      conj(Fs[i + 1][-4; 3 4])
        if eltype(H) <: SparseMPOTensor
            # restore sparsity -> when both factors are braidingtensors, we know that the
            # result can again be represented as a braidingtensor
            cinds = CartesianIndices((size(a[i], 1), size(b[i], 1)))
            for j in axes(C[i], 1), k in axes(C[i], 4)
                rowinds = cinds[j]
                colinds = cinds[k]
                Ia = CartesianIndex(rowinds[1], 1, 1, colinds[1])
                Ib = CartesianIndex(rowinds[2], 1, 1, colinds[2])
                if (haskey(a[i], Ia) && a[i][Ia] isa TensorKit.BraidingTensor) &&
                   (haskey(b[i], Ib) && b[i][Ib] isa TensorKit.BraidingTensor)
                    V = getsubspace(space(C[i]), CartesianIndex(j, 1, 1, k))
                    C[i][j, 1, 1, k] = TensorKit.BraidingTensor{S,Matrix{E}}(V[2], V[1])
                end
            end
        end
    end
    return MPOHamiltonian(C)
end

function Base.:(^)(a::MPOHamiltonian, n::Int)
    n >= 1 || throw(DomainError(n, "n should be a positive integer"))
    return Base.power_by_squaring(a, n)
end

function Base.:*(H::MPOHamiltonian, mps::FiniteMPS)
    length(H) == length(mps) || throw(ArgumentError("dimension mismatch"))

    # extract mps tensors
    A = [mps.AC[1]; mps.AR[2:end]]

    # extract mpo tensors -> put boundary conditions in place
    O = copy(H.data)

    V_left = left_virtualspace(H, 0)
    U_left = similar(H[1], V_left, oneunit(V_left))'
    U_left[1] = fill!(U_left[1], one(scalartype(H)))
    @plansor O[1][-1 -2; -3 -4] := U_left[-1; 1] * O[1][1 -2; -3 -4]

    V_right = right_virtualspace(H, length(H))
    U_right = similar(H[end], V_right', oneunit(V_right))
    U_right[end] = fill!(U_right[end], one(scalartype(H)))
    @plansor O[end][-1 -2; -3 -4] := O[end][-1 -2; -3 1] * U_right[1; -4]

    E = promote_type(scalartype(H), scalartype(A))

    local Fᵣ # trick to make Fᵣ defined in the loop
    for i in 1:length(mps)
        Fₗ = i != 1 ? Fᵣ : fuser(E, left_virtualspace(A[i]), left_virtualspace(O[i]))
        Fᵣ = fuser(E, right_virtualspace(A[i])', right_virtualspace(O[i])')
        @plansor A[i][-1 -2; -3] := Fₗ[-1; 1 3] * A[i][1 2; 4] *
                                    O[i][3 -2; 2 5] *
                                    conj(Fᵣ[-3; 4 5])
    end

    # use changebonds to get rid of non-full-rank MPS
    mps′ = FiniteMPS(A)
    return changebonds!(mps′, SvdCut(); normalize=false)
end

Base.repeat(x::MPOHamiltonian, n::Int) = MPOHamiltonian(repeat(x.data, n))
function Base.conj(a::MPOHamiltonian)
    return MPOHamiltonian(map(a.data) do x
                              @plansor x′[-1 -2; -3 -4] := conj(x[-1 -3; -2 -4])
                          end)
end

Base.convert(::Type{DenseMPO}, H::MPOHamiltonian) = convert(DenseMPO, convert(SparseMPO, H))
Base.convert(::Type{SparseMPO}, H::MPOHamiltonian{T}) where {T} = InfiniteMPO{T}(H.data)

Base.:*(H::MPOHamiltonian, mps::InfiniteMPS) = convert(DenseMPO, H) * mps

function add_physical_charge(O::MPOHamiltonian, charges::AbstractVector)
    return MPOHamiltonian(add_physical_charge(O.data, charges))
end

# promotion and conversion
# ------------------------
function Base.promote_rule(::Type{MPOHamiltonian{T₁}},
                           ::Type{MPOHamiltonian{T₂}}) where {T₁,T₂}
    return MPOHamiltonian{promote_type(T₁, T₂)}
end

function Base.convert(::Type{MPOHamiltonian{T}}, x::MPOHamiltonian) where {T}
    typeof(x) == MPOHamiltonian{T} && return x
    return MPOHamiltonian(convert.(T, x.data))
end

# Utility
# -------

function isjordanstructure(O::SparseMPOTensor)
    # check for identity blocks
    ismpoidentity(O[1, 1, 1, 1]) || return false
    ismpoidentity(O[end, 1, 1, end]) || return false
    # check upper triangular
    for I in nonzero_keys(O)
        I[1] <= I[4] || return false
    end
    return true
end
isjordanstructure(O::MPOHamiltonian) = all(isjordanstructure, parent(O))

Base.copy(H::MPOHamiltonian{T}) where {T} = MPOHamiltonian{T}(map(copy, H.data))
