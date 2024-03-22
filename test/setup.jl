# Planar stuff
# ----------------------------

using Test, TestExtras
using MPSKit
using MPSKit: _transpose_tail, _transpose_front, SparseMPOTensor
using TensorKit, BlockTensorKit
using TensorKit: PlanarTrivial, ℙ
using BlockTensorKit

# using TensorOperations

force_planar(V::Union{CartesianSpace,ComplexSpace}) = ℙ^dim(V)
force_planar(V::ProductSpace) = mapreduce(force_planar, ⊗, V.spaces)
force_planar(V::SumSpace) = SumSpace(map(force_planar, V.spaces))
function force_planar(x::TensorMap)
    cod = force_planar(codomain(x))
    dom = force_planar(domain(x))
    t = TensorMap(undef, scalartype(x), cod ← dom)
    copyto!(blocks(t)[PlanarTrivial()], convert(Array, x))
    return t
end
function force_planar(x::TensorKit.BraidingTensor)
    V1 = force_planar(x.V1)
    V2 = force_planar(x.V2)
    return TensorKit.BraidingTensor{typeof(V1),storagetype(x)}(V1, V2, x.adjoint)
end
function force_planar(x::BlockTensorMap{E,S,N₁,N₂}) where {E,S,N₁,N₂}
    planar_tensors = Dict(I => force_planar(V) for (I, V) in nonzero_pairs(x))
    ttypes = Union{unique(typeof(t) for t in values(planar_tensors))...}
    cod = force_planar(codomain(x))
    dom = force_planar(domain(x))
    tdst = BlockTensorMap{E,spacetype(ttypes),N₁,N₂}(undef, cod, dom)
    for (I, v) in planar_tensors
        tdst[I] = v
    end
    @assert norm(x) ≈ norm(tdst)
    return tdst
end
function force_planar(x::InfiniteMPO)
    return InfiniteMPO(force_planar.(parent(x)))
end
function force_planar(x::MPOHamiltonian)
    return MPOHamiltonian(force_planar.(parent(x)))
end
force_planar(x) = x

# Toy models
# ----------
using LinearAlgebra: Diagonal

function transverse_field_ising(; g=1.0)
    X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
    E = TensorMap(ComplexF64[1 0; 0 1], ℂ^2 ← ℂ^2)
    H = Z ⊗ Z + (g / 2) * (X ⊗ E + E ⊗ X)
    return MPOHamiltonian(H)
end

function heisenberg_XXX(::Type{SU2Irrep}; spin=1)
    H = TensorMap(ones, ComplexF64, SU2Space(spin => 1)^2 ← SU2Space(spin => 1)^2)
    for (c, b) in blocks(H)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - spin * (spin + 1)
    end
    return MPOHamiltonian(H * 4)
end

function heisenberg_XXX(; spin=1)
    H = TensorMap(ones, ComplexF64, SU2Space(spin => 1)^2 ← SU2Space(spin => 1)^2)
    for (c, b) in blocks(H)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - spin * (spin + 1)
    end
    A = convert(Array, H)
    d = convert(Int, 2 * spin + 1)
    H′ = TensorMap(A, (ℂ^d)^2 ← (ℂ^d)^2)
    return MPOHamiltonian(H′)
end

function bilinear_biquadratic_model(::Type{SU2Irrep}; θ=atan(1 / 3))
    H1 = TensorMap(ones, ComplexF64, SU2Space(1 => 1)^2 ← SU2Space(1 => 1)^2)
    for (c, b) in blocks(H1)
        S = (dim(c) - 1) / 2
        b .= S * (S + 1) / 2 - 1 * (1 + 1)
    end
    H2 = H1 * H1
    H = cos(θ) * H1 + sin(θ) * H2
    return MPOHamiltonian(H)
end

function classical_ising()
    β = log(1 + sqrt(2)) / 2
    t = [exp(β) exp(-β); exp(-β) exp(β)]
    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors
    O = zeros(ComplexF64, (2, 2, 2, 2))
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]
    return InfiniteMPO([TensorMap(o, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2)])
end

function sixvertex(; a=1.0, b=1.0, c=1.0)
    d = ComplexF64[a 0 0 0
                   0 c b 0
                   0 b c 0
                   0 0 0 a]
    return InfiniteMPO([permute(TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), ((1, 2), (4, 3)))])
end
