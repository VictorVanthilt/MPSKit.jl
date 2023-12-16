const AbstractBlockOrTensorMap{S,N₁,N₂} = Union{BlockTensorMap{S,N₁,N₂},
                                                AbstractTensorMap{S,N₁,N₂}} where {S,N₁,N₂}

function _transpose_front(t::AbstractTensorMap) # make TensorMap{S,N₁+N₂-1,1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return transpose(t, ((I1..., reverse(Base.tail(I2))...), (I2[1],)))
end
function _transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return transpose(t, ((I1[1],), (I2..., reverse(Base.tail(I1))...)))
end
function _transpose_as(t1::AbstractTensorMap,
                       t2::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    I1 = (TensorKit.codomainind(t1)..., reverse(TensorKit.domainind(t1))...)

    A = ntuple(x -> I1[x], N1)
    B = ntuple(x -> I1[x + N1], N2)

    return transpose(t1, (A, B))
end

_firstspace(t::AbstractTensorMap) = space(t, 1)
_firstspace(t::BlockTensorMap) = space(t, 1)
_lastspace(t::AbstractTensorMap) = space(t, numind(t))
_lastspace(t::BlockTensorMap) = space(t, numind(t))

#given a hamiltonian with unit legs on the side, decompose it using svds to form a "localmpo"
function decompose_localmpo(inpmpo::AbstractTensorMap{PS,N,N},
                            trunc=truncbelow(Defaults.tol)) where {PS,N}
    N == 2 && return [inpmpo]

    leftind = (N + 1, 1, 2)
    rightind = (ntuple(x -> x + N + 1, N - 1)..., reverse(ntuple(x -> x + 2, N - 2))...)
    U, S, V = tsvd(transpose(inpmpo, (leftind, rightind)); trunc=trunc)

    A = transpose(U * S, ((2, 3), (1, 4)))
    B = transpose(V,
                  ((1, reverse(ntuple(x -> x + N, N - 2))...), ntuple(x -> x + 1, N - 1)))
    return [A; decompose_localmpo(B)]
end

# given a state with util legs on the side, decompose using svds to form an array of mpstensors
function decompose_localmps(state::AbstractTensorMap{PS,N,1},
                            trunc=truncbelow(Defaults.tol)) where {PS,N}
    N == 2 && return [state]

    leftind = (1, 2)
    rightind = reverse(ntuple(x -> x + 2, N - 1))
    U, S, V = tsvd(transpose(state, (leftind, rightind)); trunc=trunc)

    A = U * S
    B = _transpose_front(V)
    return [A; decompose_localmps(B)]
end

function add_util_leg(tensor::AbstractTensorMap{S,N1,N2}) where {S,N1,N2}
    ou = oneunit(_firstspace(tensor))

    util_front = isomorphism(storagetype(tensor), ou * codomain(tensor), codomain(tensor))
    util_back = isomorphism(storagetype(tensor), domain(tensor), domain(tensor) * ou)

    return util_front * tensor * util_back
end

function union_split(a::AbstractArray)
    T = reduce((a, b) -> Union{a,b}, typeof.(a))
    nA = similar(a, T)
    return copy!(nA, a)
end
union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type) = (x,)

function _embedders(spaces)
    totalspace = reduce(⊕, spaces)

    maps = [isometry(totalspace, first(spaces))]
    restmap = leftnull(first(maps))

    for sp in spaces[2:end]
        cm = isometry(domain(restmap), sp)

        push!(maps, restmap * cm)
        restmap = restmap * leftnull(cm)
    end

    return maps
end

function _can_unambiguously_braid(sp::VectorSpace)
    s = sectortype(sp)

    BraidingStyle(s) isa SymmetricBraiding && return true

    # if it's not symmetric, then we are only really garantueed that this is possible when only one irrep occurs - the trivial one
    for sect in sectors(sp)
        sect == one(sect) || return false
    end
    return true
end

#needed this; perhaps move to tensorkit?
TensorKit.fuse(f::T) where {T<:VectorSpace} = f

function inplace_add!(a::Union{AbstractTensorMap,Nothing},
                      b::Union{AbstractTensorMap,Nothing})
    isnothing(a) && isnothing(b) && return nothing
    isnothing(a) && return b
    isnothing(b) && return a
    return axpy!(true, a, b)
end

#=
map every element in the tensormap to dfun(E)
allows us to create random tensormaps for any storagetype
=#
function fill_data!(a::TensorMap, dfun)
    for (k, v) in blocks(a)
        map!(x -> dfun(typeof(x)), v, v)
    end

    return a
end
randomize!(a::TensorMap) = fill_data!(a, randn)
function randomize!(a::BlockTensorMap)
    for i in eachindex(a)
        a[i] = randomize!(a[i])
    end
    return a
end

function safe_xlogx(t::AbstractTensorMap, eps=eps(real(scalartype(t))))
    (U, S, V) = tsvd(t; alg=SVD(), trunc=truncbelow(eps))
    return U * S * log(S) * V
end

"""
    tensorexpr(name::Symbol, ind_out, [ind_in])

Generates expressions for use within [`@tensor`](@ref TensorOperations.@tensor) environments of the form `name[ind_out...; ind_in]`.
"""
tensorexpr(name::Symbol, inds) = Expr(:ref, name, inds...)
function tensorexpr(name::Symbol, indout, indin)
    return Expr(:typed_vcat, name, Expr(:row, indout...), Expr(:row, indin...))
end

backendsymbol(::Type{TensorOperations.Backend{T}}) where {T} = T

# check all elements are equal -> only defined in 1.8+
@static if !isdefined(Base, :allequal)
    allequal(itr) = isempty(itr) ? true : all(isequal(first(itr)), itr)
end

function between(x1, x, x2)
    @assert x1 <= x2 "x1 should be smaller than  or equal to x2"
    x < x1 && return x1
    x > x2 && return x2
    return x
end
function fuser(::Type{T}, V1::S, V2::S) where {T<:Number,S<:IndexSpace}
    return isomorphism(Matrix{T}, fuse(V1 ⊗ V2), V1 ⊗ V2)
end

function fuser(::Type{T}, V1::SumSpace{S}, V2::SumSpace{S}) where {T<:Number,S<:IndexSpace}
    W = fuse(V1, V2)
    TT = tensormaptype(S, 1, 2, T)
    F = BlockTensorMap{S,1,2,TT,3}(undef, W ← V1 ⊗ V2)
    for I in CartesianIndices(F)
        V = getsubspace(space(F), I)
        if I[1] == I[2] + (I[3] - 1) * size(F, 2)
            F[I] = isomorphism(storagetype(F), V)
        end
    end

    return F
end

"""
    check_length(A, B)

Check if the length of two arrays match, throwing a `DimensionMismatch` if not.
"""
function check_length(A, B)
    length(A) == length(B) ||
        throw(DimensionMismatch("Lengths should match: $(length(A)) ≠ $(length(B))"))
    return nothing
end