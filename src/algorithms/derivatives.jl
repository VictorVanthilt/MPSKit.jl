# Given a state and it's environments, we can act on it

"""
    Draft operators
"""
struct MPO_∂∂C{L,R}
    leftenv::L
    rightenv::R
end

struct MPO_∂∂AC{O,L,R}
    o::O
    leftenv::L
    rightenv::R
end

struct MPO_∂∂AC2{O,L,R}
    o1::O
    o2::O
    leftenv::L
    rightenv::R
end

Base.:*(h::Union{MPO_∂∂C,MPO_∂∂AC,MPO_∂∂AC2}, v) = h(v);

(h::MPO_∂∂C)(x) = ∂C(x, h.leftenv, h.rightenv)
(h::MPO_∂∂AC)(x) = ∂AC(x, h.o, h.leftenv, h.rightenv)
(h::MPO_∂∂AC2)(x) = ∂AC2(x, h.o1, h.o2, h.leftenv, h.rightenv)

# draft operator constructors
function ∂∂C(pos::Int, mps, ::AbstractMPO, envs)
    return MPO_∂∂C(leftenv(envs, pos + 1, mps), rightenv(envs, pos, mps))
end
function ∂∂C(col::Int, mps, ::MPOMultiline, envs)
    return MPO_∂∂C(leftenv(envs, col + 1, mps), rightenv(envs, col, mps))
end
function ∂∂C(row::Int, col::Int, mps, ::MPOMultiline, envs)
    return MPO_∂∂C(leftenv(envs, row, col + 1, mps), rightenv(envs, row, col, mps))
end

function ∂∂AC(pos::Int, mps, O::AbstractMPO, envs)
    return MPO_∂∂AC(O[pos], leftenv(envs, pos, mps), rightenv(envs, pos, mps))
end
function ∂∂AC(row::Int, col::Int, mps, O::MPOMultiline, envs)
    return MPO_∂∂AC(O[row, col], leftenv(envs, row, col, mps),
                    rightenv(envs, row, col, mps))
end
function ∂∂AC(col::Int, mps, O::MPOMultiline, envs)
    return MPO_∂∂AC(O[:, col], leftenv(envs, col, mps), rightenv(envs, col, mps))
end

function ∂∂AC2(pos::Int, mps, O::AbstractMPO, envs)
    return MPO_∂∂AC2(O[pos], O[pos + 1],
                     leftenv(envs, pos, mps), rightenv(envs, pos + 1, mps))
end
function ∂∂AC2(col::Int, mps, O::MPOMultiline, envs)
    return MPO_∂∂AC2(O[:, col], O[:, col + 1],
                     leftenv(envs, col, mps), rightenv(envs, col + 1, mps))
end
function ∂∂AC2(row::Int, col::Int, mps, O::MPOMultiline, envs)
    return MPO_∂∂AC2(O[row, col], O[row, col + 1],
                     leftenv(envs, row, col, mps), rightenv(envs, row, col + 1, mps))
end

# allow calling them with CartesianIndices
∂∂C(pos::CartesianIndex, mps, O, envs) = ∂∂C(Tuple(pos)..., mps, O, envs)
∂∂AC(pos::CartesianIndex, mps, O, envs) = ∂∂AC(Tuple(pos)..., mps, O, envs)
∂∂AC2(pos::CartesianIndex, mps, O, envs) = ∂∂AC2(Tuple(pos)..., mps, O, envs)

"""
    ∂C(x::MPSBondTensor, GL::AbstractMPSTensor, rightenv::AbstractMPSTensor)
    
Compute the action of the zero-site derivative on a vector `x`.
"""
function ∂C(x::MPSBondTensor, GL::AbstractMPSTensor, GR::AbstractMPSTensor)
    return ∂C(x, GL, GR, Defaults.get_backend(), Defaults.get_allocator())
end
@generated function ∂C(x::MPSBondTensor, GL::AbstractMPSTensor, GR::AbstractMPSTensor,
                       backend::B, allocator::A) where {B,A}
    ex = :(@plansor y[-1; -2] := GL[-1 3; 1] * x[1; 2] * GR[2 3; -2])
    B <: TensorOperations.Backend && insert!(ex.args, 3, :(backend = $(backendsymbol(B))))
    A <: TensorOperations.Backend && insert!(ex.args, 3, :(allocator = $(backendsymbol(A))))
    return quote
        @no_escape $ex
        return convert(typeof(x), y)
    end
end

"""
    ∂AC(x::AbstractMPSTensor, O::AbstractMPOTensor, GL::AbstractMPSTensor, GR::AbstractMPSTensor)

Compute the action of the one-site derivative on a vector `x`.
"""
function ∂AC(x::AbstractMPSTensor, O::AbstractMPOTensor,
              GL::AbstractMPSTensor, GR::AbstractMPSTensor)
    return ∂AC(x, O, GL, GR, Defaults.get_backend(), Defaults.get_allocator())
end
@generated function ∂AC(x::AbstractMPSTensor, O::AbstractMPOTensor,
                        GL::AbstractMPSTensor, GR::AbstractMPSTensor,
                        backend::B, allocator::A) where {B,A}
    ex = :(@plansor y[-1 -2; -3] := GL[-1 2; 1] * x[1 3; 4] * O[2 -2; 3 5] * GR[4 5; -3])
    B <: TensorOperations.Backend && insert!(ex.args, 3, :(backend = $(backendsymbol(B))))
    A <: TensorOperations.Backend && insert!(ex.args, 3, :(allocator = $(backendsymbol(A))))
    return quote
        @no_escape $ex
        return convert(typeof(x), y)
    end
end
function ∂AC(x::RecursiveVec, O, GL, GR)
    return RecursiveVec(circshift(map(∂AC, x.vecs, O, GL, GR), 1))
end
function ∂AC(x::MPSTensor, ::Nothing, GL, GR)
    return _transpose_front(GL * _transpose_tail(x * GR))
end

"""
    ∂AC2(x::AbstractMPSTensor, O₁::AbstractMPOTensor, O₂::AbstractMPOTensor,
          GL::AbstractMPSTensor, GR::AbstractMPSTensor)

Compute the action of the two-site derivative on a vector `x`.
"""
function ∂AC2(x::AbstractMPOTensor, O₁::AbstractMPOTensor, O₂::AbstractMPOTensor,
              GL::AbstractMPSTensor, GR::AbstractMPSTensor)
    @plansor y[-1 -2; -3 -4] := x[6 5; 1 3] * O₁[7 -2; 5 4] * O₂[4 -4; 3 2] *
                                GL[-1 7; 6] * GR[1 2; -3]
    return convert(typeof(x), y)
end
function ∂AC2(x::RecursiveVec, O₁, O₂, GL, GR)
    return RecursiveVec(circshift(map(∂AC2, x.vecs, O₁, O₂, GL, GR)), 1)
end
function ∂AC2(x::MPOTensor, ::Nothing, ::Nothing, GL, GR)
    @plansor y[-1 -2; -3 -4] := x[1 -2; 2 -4] * GL[-1; 1] * GR[2; -3]
end

#downproject for approximate
function c_proj(pos, below, envs::FinEnv)
    return ∂C(envs.above.CR[pos], leftenv(envs, pos + 1, below), rightenv(envs, pos, below))
end

function c_proj(row, col, below, envs::PerMPOInfEnv)
    return ∂C(envs.above.CR[row, col],
              leftenv(envs, row, col + 1, below),
              rightenv(envs, row, col, below))
end

function ac_proj(pos, below, envs)
    le = leftenv(envs, pos, below)
    re = rightenv(envs, pos, below)

    return ∂AC(envs.above.AC[pos], envs.opp[pos], le, re)
end
function ac_proj(row, col, below, envs::PerMPOInfEnv)
    return ∂AC(envs.above.AC[row, col],
               envs.opp[row, col],
               leftenv(envs, row, col, below),
               rightenv(envs, row, col, below))
end
function ac2_proj(pos, below, envs)
    le = leftenv(envs, pos, below)
    re = rightenv(envs, pos + 1, below)

    return ∂AC2(envs.above.AC[pos] * _transpose_tail(envs.above.AR[pos + 1]),
                envs.opp[pos],
                envs.opp[pos + 1],
                le,
                re)
end
function ac2_proj(row, col, below, envs::PerMPOInfEnv)
    @plansor ac2[-1 -2; -3 -4] := envs.above.AC[row, col][-1 -2; 1] *
                                  envs.above.AR[row, col + 1][1 -4; -3]
    return ∂AC2(ac2, leftenv(envs, row, col + 1, below),
                rightenv(envs, row, col + 1, below))
end

function ∂∂C(pos::Int, mps, opp::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂C(pos, mps, h, e), opp.opps, cache.envs),
                             opp.coeffs)
end

function ∂∂AC(pos::Int, mps, opp::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC(pos, mps, h, e), opp.opps,
                                       cache.envs), opp.coeffs)
end

function ∂∂AC2(pos::Int, mps, opp::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC2(pos, mps, h, e), opp.opps,
                                       cache.envs), opp.coeffs)
end

struct AC_EffProj{A,L}
    a1::A
    le::L
    re::L
end
struct AC2_EffProj{A,L}
    a1::A
    a2::A
    le::L
    re::L
end
Base.:*(h::Union{AC_EffProj,AC2_EffProj}, v) = h(v);

function (h::AC_EffProj)(x::MPSTensor)
    @plansor v[-1; -2 -3 -4] := h.le[4; -1 -2 5] * h.a1[5 2; 1] * h.re[1; -3 -4 3] *
                                conj(x[4 2; 3])
    @plansor y[-1 -2; -3] := conj(v[1; 2 5 6]) * h.le[-1; 1 2 4] * h.a1[4 -2; 3] *
                             h.re[3; 5 6 -3]
end
function (h::AC2_EffProj)(x::MPOTensor)
    @plansor v[-1; -2 -3 -4] := h.le[6; -1 -2 7] *
                                h.a1[7 4; 5] *
                                h.a2[5 2; 1] *
                                h.re[1; -3 -4 3] *
                                conj(x[6 4; 3 2])
    @plansor y[-1 -2; -3 -4] := conj(v[2; 3 5 6]) *
                                h.le[-1; 2 3 4] *
                                h.a1[4 -2; 7] *
                                h.a2[7 -4; 1] *
                                h.re[1; 5 6 -3]
end

function ∂∂AC(pos::Int, state, opp::ProjectionOperator, env)
    return AC_EffProj(opp.ket.AC[pos], leftenv(env, pos, state), rightenv(env, pos, state))
end;
function ∂∂AC2(pos::Int, state, opp::ProjectionOperator, env)
    return AC2_EffProj(opp.ket.AC[pos],
                       opp.ket.AR[pos + 1],
                       leftenv(env, pos, state),
                       rightenv(env, pos + 1, state))
end;
