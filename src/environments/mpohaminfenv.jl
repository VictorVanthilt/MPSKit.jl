"
    This object manages the hamiltonian environments for an InfiniteMPS
"
mutable struct MPOHamInfEnv{H<:MPOHamiltonian,V,S<:InfiniteMPS,A} <: AbstractInfEnv
    opp::H

    dependency::S
    solver::A

    lw::PeriodicArray{V,1}
    rw::PeriodicArray{V,1}

    lock::ReentrantLock
end

function Base.copy(p::MPOHamInfEnv)
    return MPOHamInfEnv(p.opp, p.dependency, p.solver, copy(p.lw), copy(p.rw))
end

function gen_lw_rw(st::InfiniteMPS, ham::Union{SparseMPO,MPOHamiltonian})
    lw = PeriodicArray(map(st.AL, ham) do al, h
                           V_mps = SumSpace(_firstspace(al))
                           V_mpo = space(h, 1)
                           return BlockTensorMap(undef, scalartype(st),
                                                 V_mps ⊗ V_mpo' ← V_mps)
                       end)
    rw = PeriodicArray(map(st.AR, ham) do ar, h
                           V_mps = SumSpace(_lastspace(ar)')
                           V_mpo = space(h, 4)
                           return BlockTensorMap(undef, scalartype(st),
                                                 V_mps ⊗ V_mpo' ← V_mps)
                       end)

    randomize!.(lw)
    randomize!.(rw)

    return lw, rw
end

#randomly initialize envs
function environments(st::InfiniteMPS, ham::MPOHamiltonian, above=nothing;
                      solver=Defaults.linearsolver)
    (isnothing(above) || above === st) ||
        throw(ArgumentError("MPOHamiltonian requires top and bottom states to be equal."))
    lw, rw = gen_lw_rw(st, ham)
    envs = MPOHamInfEnv(ham, similar(st), solver, lw, rw, ReentrantLock())
    return recalculate!(envs, st)
end

function leftenv(envs::MPOHamInfEnv, pos::Int, state)
    check_recalculate!(envs, state)
    return envs.lw[pos]
end

function rightenv(envs::MPOHamInfEnv, pos::Int, state)
    check_recalculate!(envs, state)
    return envs.rw[pos]
end

function recalculate!(envs::MPOHamInfEnv, nstate; tol=envs.solver.tol)
    sameDspace = reduce(&, _lastspace.(envs.lw) .== _firstspace.(nstate.CR))

    if !sameDspace
        envs.lw, envs.rw = gen_lw_rw(nstate, envs.opp)
    end

    solver = envs.solver
    solver = solver.tol == tol ? solver : @set solver.tol = tol
    if Threads.nthreads() > 1
        @sync begin
            Threads.@spawn calclw!(envs.lw, nstate, envs.opp; solver)
            Threads.@spawn calcrw!(envs.rw, nstate, envs.opp; solver)
        end
    else
        calclw!(envs.lw, nstate, envs.opp; solver)
        calcrw!(envs.rw, nstate, envs.opp; solver)
    end

    envs.dependency = nstate
    envs.solver = solver

    return envs
end

function calclw!(fixpoints, st::InfiniteMPS, H::MPOHamiltonian;
                 solver=Defaults.linearsolver)
    check_length(st, H)
    len = length(st)

    # the start element
    leftutil = similar(st.AL[1], left_virtualspace(H, 1)[1])
    fill_data!(leftutil, one)

    ρₗ = get!(fixpoints[1], CartesianIndex(1, 1, 1))
    @plansor ρₗ[-1 -2; -3] = l_LL(st)[-1; -3] * conj(leftutil[-2])

    (len > 1) && left_cyclethrough!(1, fixpoints, H, st)

    for i in 2:left_virtualsize(H, 1)
        prev = copy(fixpoints[1][1, i, 1]) # use as initial guess in linsolve
        fixpoints[1][1, i, 1] = zerovector!(fixpoints[1][1, i, 1])
        left_cyclethrough!(i, fixpoints, H, st)
        if isone(H, i) # identity matrices; do the hacky renormalization
            tm = regularize(TransferMatrix(st.AL, st.AL), l_LL(st), r_LL(st))
            fixpoints[1][1, i, 1], convhist = linsolve(flip(tm), fixpoints[1][1, i, 1],
                                                       prev, solver, 1, -1)
            convhist.converged == 0 && @info "calclw failed to converge" convhist

            (len > 1) && left_cyclethrough!(i, fixpoints, H, st)

            # go through the unitcell, again subtracting fixpoints
            for 🥔 in 1:len
                fp = get!(fixpoints[🥔], CartesianIndex(1, i, 1))
                @plansor fp[-1 -2; -3] -= fp[1 -2; 2] * r_LL(st, 🥔 - 1)[2; 1] *
                                          l_LL(st, 🥔)[-1; -3]
            end
        else
            if !iszero(H, i)
                diag = map(b -> b[i, 1, 1, i], parent(H))
                tm = TransferMatrix(st.AL, diag, st.AL)
                fixpoints[1][1, i, 1], convhist = linsolve(flip(tm), fixpoints[1][1, i, 1],
                                                           prev, solver, 1, -1)
                convhist.converged == 0 &&
                    @info "calclw failed to converge $(convhist.normres)"
            end
            (len > 1) && left_cyclethrough!(i, fixpoints, H, st)
        end
    end
    return fixpoints
end

function calcrw!(fixpoints, st::InfiniteMPS, ham::MPOHamiltonian;
                 solver=Defaults.linearsolver)
    check_length(st, ham)
    len = length(st)
    odim = right_virtualsize(ham, len)

    # the start element
    rightutil = similar(st.AL[1], right_virtualspace(ham, len)[end])
    fill_data!(rightutil, one)
    ρᵣ = get!(fixpoints[end], CartesianIndex(1, odim, 1))
    @plansor ρᵣ[-1 -2; -3] = r_RR(st)[-1; -3] * conj(rightutil[-2])

    (len > 1) && right_cyclethrough!(odim, fixpoints, ham, st) #populate other sites

    for i in (odim - 1):-1:1
        prev = copy(fixpoints[end][1, i, 1]) # use as initial guess in linsolve
        fixpoints[end][1, i, 1] = zerovector!(fixpoints[end][1, i, 1])

        right_cyclethrough!(i, fixpoints, ham, st)

        if isone(ham, i) # identity matrices; do the hacky renormalization
            tm = regularize(TransferMatrix(st.AR, st.AR), l_RR(st), r_RR(st))
            fixpoints[end][1, i, 1], convhist = linsolve(tm, fixpoints[end][1, i, 1], prev,
                                                         solver, 1, -1)
            convhist.converged == 0 && @info "calcrw failed to converge $(convhist.normres)"
            len > 1 && right_cyclethrough!(i, fixpoints, ham, st)

            #go through the unitcell, again subtracting fixpoints
            for 🥔 in 1:len
                fp = get!(fixpoints[🥔], CartesianIndex(1, i, 1))
                @plansor fp[-1 -2; -3] -= fp[1 -2; 2] * l_RR(st, 🥔 + 1)[2; 1] *
                                          r_RR(st, 🥔)[-1; -3]
            end
        else
            if !iszero(ham, i)
                diag = map(b -> b[i, 1, 1, i], parent(ham))
                tm = TransferMatrix(st.AR, diag, st.AR)
                fixpoints[end][1, i, 1], convhist = linsolve(tm, fixpoints[end][1, i, 1],
                                                             prev, solver, 1, -1)
                convhist.converged == 0 &&
                    @info "calcrw failed to converge $(convhist.normres)"
            end

            (len > 1) && right_cyclethrough!(i, fixpoints, ham, st)
        end
    end
    return fixpoints
end
"""
    left_cyclethrough!(index::Int, fp, H, st)

This function computes all fixpoints at layer index, using the fixpoints at previous layers.
"""
function left_cyclethrough!(index::Int, fp::PeriodicArray{T,1}, H, st) where {T}
    for i in 1:length(fp)
        fp[i + 1][1, index, 1] = zerovector!(fp[i + 1][1, index, 1])
        transfer = TransferMatrix(st.AL[i], H[i][1:index, 1, 1, index], st.AL[i])
        fp[i + 1][1, index, 1] = fp[i][1, 1:index, 1] * transfer
    end
    return nothing
end

function right_cyclethrough!(index::Int, fp::PeriodicArray{T,1}, H, st) where {T}
    for i in reverse(1:length(fp))
        fp[i - 1][1, index, 1] = zerovector!(fp[i - 1][index])
        transfer = TransferMatrix(st.AR[i], H[i][index, 1, 1, index:end], st.AR[i])
        fp[i - 1][1, index, 1] = transfer * fp[i][1, index:end, 1]
    end
    return nothing
end
