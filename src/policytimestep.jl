# TODO: better names for the functions

function updatecoeffs!{T<:Real}(coeff0, coeff1, coeff2, rhs, model, v, t::T, x,
                                a::T, Δτ::T, Δx::T)
    # TODO: should we just remove this dispatch and pass in ones(n-2)*a instead?
    # Updates
    # coeffn = values in linear system
    # rhs    = value function at previous timestep + f at current timestep

    # Input
    # model  = HJBOneDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = vector of x-values
    # a      = policy-values on interior
    # Δτ     = time-step size
    # Δx     = spacial step length
    taux = Δτ/Δx
    htaux2 = 0.5*Δτ/Δx^2

    n = length(coeff0)
    for j = 2:n-1
        @inbounds begin
            bval = model.b(t,x[j],a)
            sval2 = model.σ(t,x[j],a)^2
            coeff1[j] = -(sval2*htaux2 + max(bval,0.)*taux)
            coeff2[j-1] = -(sval2*htaux2 - min(bval,0.)*taux)
            coeff0[j] = 1.-coeff1[j]-coeff2[j-1]
            rhs[j] = v[j] + Δτ*model.f(t,x[j],a)
        end
    end

end

function policytimestep(model::HJBOneDim,
                        v, avals, x, Δx, Δτ, ti::Int)
    taux = Δτ/Δx
    htaux2 = 0.5*Δτ/Δx^2
    t = (ti-1)*Δτ
    n = length(v)

    # TODO: redo this thing
    newind = ones(Int, n)
    @inbounds vnew = fill(-maxintfloat(typeof(v[1])), n)

    ind12 = zeros(Bool, length(vnew))

    coeff0 = ones(v)   # v_i
    coeff1 = zeros(n-1) # v_{i+1} # TODO: type stability
    coeff2 = zeros(n-1) # v_{i-1} # TODO: type stability
    rhs = zeros(v)
    # Dirichlet conditions
    if model.bcond[1] == true
        @inbounds rhs[1] = model.Dfun[1](t)
    end
    if model.bcond[2] == true
        @inbounds rhs[end] = model.Dfun[2](t)
    end

    for i = 1:length(avals)
        @inbounds begin
            a = avals[i]
            vold = vnew

            updatecoeffs!(coeff0, coeff1, coeff2, rhs, model, v, t, x, a, Δτ, Δx)
            Mat = spdiagm((coeff2, coeff0, coeff1), -1:1, n, n)
            # Move to sparse(I,J,K) instead?
            if model.bcond[1] == false
                bval = taux*model.b(t,x[1],a)
                sval2 = htaux2*model.σ(t,x[1],a)^2
                Mat[1,1:3] = [1.-(sval2-bval), -(bval-2*sval2), -sval2]
                rhs[1] = v[1] + Δτ*model.f(t,x[1],a)
            end

            if model.bcond[2] == false
                bval = taux*model.b(t,x[end],a)
                sval2 = htaux2*model.σ(t,x[end],a)^2
                Mat[end,end-2:end] = [-sval2, bval+2*sval2, 1.-(sval2+bval)]
                rhs[end] = v[end] + Δτ*model.f(t,x[end],a)
            end

            # TODO: Use Krylov solver for high-dimensional PDEs
            vnew = Mat\rhs

            ind12[:] = vold .> vnew
            vnew[ind12] = vold[ind12]
            newind[!ind12] = i
        end
    end

    @inbounds pol = avals[newind]

    return vnew, pol
end


function timeloopconstant(model::HJBOneDim, K::Int, N::Int,
                          Δτ, vinit, avals, x, Δx)
    # Pass v and pol by reference?
    v = zeros(K+1, N+1)
    pol = zeros(K+1, N)

    @inbounds v[:,end] = vinit # We use forward time t instead of backward time τ

    for j = N:-1:1
        @inbounds begin
            # t = (j-1)*Δτ
            # TODO: pass v-column, pol-column by reference?
            v[:,j], pol[:,j] = policytimestep(model, v[:,j+1],
                                              avals, x, Δx, Δτ, j)
        end
    end

    return v, pol
end
