println("
-----------------
|   Operators   |
-----------------
")

include("setup.jl")

pspaces = (ℂ^4, Rep[U₁](0 => 2), Rep[SU₂](1 => 1))
vspaces = (ℂ^10, Rep[U₁]((0 => 20)), Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 1))

@testset "Finite MPOHamiltonian" begin
    # simple single-site operator
    h1 = TensorMap(rand, ComplexF64, ℂ^2, ℂ^2)
    h1 += h1'
    normalize!(h1)
    H1 = MPOHamiltonian(h1)

    @test convert(TensorMap, H1) ≈ h1

    α = randn(ComplexF64)
    @test convert(TensorMap, H1 * α) ≈ h1 * α

    @test convert(TensorMap, H1 + H1) ≈ h1 + h1
    @test convert(TensorMap, H1 - H1) ≈ h1 - h1

    @test convert(TensorMap, H1 + 1.4 * H1) ≈ 2.4 * h1

    # single-site operator acting on 3 sites
    E = id(Matrix{ComplexF64}, ℂ^2)
    H3 = repeat(copy(H1), 3)
    h3 = h1 ⊗ E ⊗ E + E ⊗ h1 ⊗ E + E ⊗ E ⊗ h1
    @test convert(TensorMap, H3) ≈ h3

    @test convert(TensorMap, H3 * α) ≈ h3 * α
    @test convert(TensorMap, H3 + H3) ≈ h3 + h3
    @test convert(TensorMap, H3 - H3) ≈ h3 - h3

    # slightly more complicated 2-site operator
    h2 = TensorMap(rand, ComplexF64, (ℂ^2)^2, (ℂ^2)^2)
    h2 += h2'
    normalize!(h2)

    H2 = repeat(MPOHamiltonian(h2), 2)
    @test convert(TensorMap, H2) ≈ h2

    @test convert(TensorMap, H2 * α) ≈ h2 * α
    @test convert(TensorMap, H2 + H2) ≈ h2 + h2
    @test convert(TensorMap, H2 - H2) ≈ h2 - h2
    @show H2
    @show -H2
    @show repeat(H1, 2)
    @test convert(TensorMap, H2 + repeat(H1, 2)) ≈ h2 + h1 ⊗ E + E ⊗ h1
end

@testset "MPOHamiltonian $(sectortype(pspace))" for (pspace, Dspace) in
                                                    zip(pspaces, vspaces)
    #generate a 1-2-3 body interaction
    n = TensorMap(rand, ComplexF64, pspace, pspace)
    n += n'
    nn = TensorMap(rand, ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    nnn = TensorMap(rand, ComplexF64, pspace * pspace * pspace, pspace * pspace * pspace)
    nnn += nnn'

    #can you pass in a proper mpo?
    identity = complex(isomorphism(oneunit(pspace) * pspace, pspace * oneunit(pspace)))
    mpoified = MPSKit.decompose_localmpo(MPSKit.add_util_leg(nnn))
    d3 = Array{Union{Missing,typeof(identity)},3}(missing, 1, 4, 4)
    d3[1, 1, 1] = identity
    d3[1, end, end] = identity
    d3[1, 1, 2] = mpoified[1]
    d3[1, 2, 3] = mpoified[2]
    d3[1, 3, 4] = mpoified[3]
    h1 = MPOHamiltonian(d3)

    #¢an you pass in the actual hamiltonian?
    h2 = MPOHamiltonian(nn)

    #can you generate a hamiltonian using only onsite interactions?
    d1 = Array{Any,3}(missing, 2, 3, 3)
    d1[1, 1, 1] = 1
    d1[1, end, end] = 1
    d1[1, 1, 2] = n
    d1[1, 2, end] = n
    d1[2, 1, 1] = 1
    d1[2, end, end] = 1
    d1[2, 1, 2] = n
    d1[2, 2, end] = n
    h3 = MPOHamiltonian(d1)

    #make a teststate to measure expectation values for
    ts1 = InfiniteMPS([pspace], [Dspace])
    ts2 = InfiniteMPS([pspace, pspace], [Dspace, Dspace])

    e1 = expectation_value(ts1, h1)
    e2 = expectation_value(ts1, h2)

    h1 = 2 * h1 - [1]
    @test e1[1] * 2 - 1 ≈ expectation_value(ts1, h1)[1] atol = 1e-10

    h1 = h1 + h2

    @test e1[1] * 2 + e2[1] - 1 ≈ expectation_value(ts1, h1)[1] atol = 1e-10

    h1 = repeat(h1, 2)

    e1 = expectation_value(ts2, h1)
    e3 = expectation_value(ts2, h3)

    @test e1 + e3 ≈ expectation_value(ts2, h1 + h3) atol = 1e-10

    h4 = h1 + h3
    h4 = h4 * h4
    @test real(sum(expectation_value(ts2, h4))) >= 0
end

@testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin=1))
    physical_space = physicalspace(ham, 1)[1]
    ou = oneunit(physical_space)

    ts = InfiniteMPS([physical_space], [ou ⊕ physical_space])

    W = convert(DenseMPO, make_time_mpo(ham, 1im * 0.5, WII()))

    @test abs(dot(W * (W * ts), (W * W) * ts)) ≈ 1.0 atol = 1e-10
end
