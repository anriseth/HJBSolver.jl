using ForwardDiff, Optim

function updateinteriorsystem!{T<:Real}(I, J, V, rhs, model, v, t::T, x,
                                        a1::Vector{T}, a2::Vector{T}, Δt::T, Δx::Vector{T})
    # Updates:
    # rhs    = value function at previous timestep + f at current timestep
    # I,J,V  = vectors for creating sparse system matrix (see ?sparse)

    # Input
    # model  = HJBTwoDim object
    # v      = value function at previous timestep
    # t      = value of (forward) time
    # x      = tuple of vectors of x-values
    # a1     = policy-values on interior, element 1
    # a2     = policy-values on interior, element 2
    # Δt     = time-step size
    # Δx     = spacial step length
    taux = Δt ./Δx
    htaux2 = 0.5*Δt ./ Δx.^2
    K = [length(xi) for xi in x]

    counter = 0

    for i = 2:K[1]-1, j = 2:K[2]-1
        @inbounds begin
            idxi = K[2]*(i-1) + j
            idxj1f = idxi + K[2]; idxj1b = idxi - K[2]
            idxj2f = idxi + 1;    idxj2b = idxi - 1
            xij = [x[1][i], x[2][j]]
            aij = [a1[idxi], a2[idxi]]

            bval = model.b(t,xij,aij)
            sval2 = model.σ(t,xij,aij).^2
            coeff1f = -(sval2[1]*htaux2[1] + max(bval[1],0.)*taux[1])
            coeff1b = -(sval2[1]*htaux2[1] - min(bval[1],0.)*taux[1])
            coeff2f = -(sval2[2]*htaux2[2] + max(bval[2],0.)*taux[2])
            coeff2b = -(sval2[2]*htaux2[2] - min(bval[2],0.)*taux[2])
            coeff0 = 1.0-(coeff1f+coeff1b + coeff2f+coeff2b)

            # TODO: does it make a performance difference what order I put these in?
            counter = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
            counter = setIJV!(I,J,V,idxi,idxj1f,coeff1f, counter)
            counter = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
            counter = setIJV!(I,J,V,idxi,idxj2f,coeff2f, counter)
            counter = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)

            rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
        end
    end
    @assert counter == length(V)
end

function updateboundarysystem!{T<:Real}(I, J, V, rhs, model, v, t::T, x,
                                        a1::Vector{T}, a2::Vector{T}, Δt::T,
                                        Δx::Vector{T})
    #==
    1.  Dirichlet condition for x = 0
    2.  For x_1 = 0, solve 1D PDE as if it was only for product 2
    2.1 For x_2 = xmax[2], approximate differential operators with backward difference
    3.  For x_2 = 0, solve 1D PDE as if it was only for product 2
    3.1 For x_1 = xmax[1], approximate differential operators with backward difference
    4.  For x_2 = xmax[2], x_1 interior, approximate x_2 differential operators with backward difference
    5.  For x_1 = xmax[1], x_2 interior, approximate x_1 differential operators with backward difference
    6.  For x = xmax, approximate all operators with backward difference
    ==#
    taux = Δt ./Δx
    htaux2 = 0.5*Δt ./ Δx.^2
    K = [length(xi) for xi in x]

    # 1. Dirichlet condition for x = 0
    counter = 0

    counter = setIJV!(I,J,V, 1, 1, 1.0, counter)
    rhs[1] = 0.

    # 2. 1D PDE on x_1 = 0 boundary
    let
        i = 1
        for j = 2:K[2]-1
            @inbounds begin
                xij = [x[1][i], x[2][j]]
                idxi = j
                idxj2f = idxi + 1; idxj2b = idxi - 1

                aij = [a1[idxi], a2[idxi]]

                bval = model.b(t,xij,aij)
                sval2 = model.σ(t,xij,aij).^2
                coeff2f = -(sval2[2]*htaux2[2] + max(bval[2],0.)*taux[2])
                coeff2b = -(sval2[2]*htaux2[2] - min(bval[2],0.)*taux[2])
                coeff0 = 1.0-(coeff2f+coeff2b)

                counter = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
                counter = setIJV!(I,J,V,idxi,idxj2f,coeff2f, counter)
                counter = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)

                rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
            end
        end
    end

    # 2.1 1D PDE approximation at boundary x_1 = 0, x_2 = xmax[2]
    let
        i = 1; j = K[2]
        xij = [x[1][i], x[2][j]]
        idxi = j
        idxj2b = idxi - 1; idxj2bb = idxi - 2

        aij = [a1[idxi], a2[idxi]]

        bval = model.b(t,xij,aij)
        sval2 = model.σ(t,xij,aij).^2
        coeff2b = 2*sval2[2]*htaux2[2] + bval[2]*taux[2]
        coeff2bb = -sval2[2]*htaux2[2]
        coeff0 = 1.0-(coeff2b+coeff2bb)

        counter = setIJV!(I,J,V,idxi,idxi, coeff0, counter)
        counter = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)
        counter = setIJV!(I,J,V,idxi,idxj2bb,coeff2bb, counter)

        rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
    end

    # 3. 1D PDE on x_2 = 0 boundary
    let
        j = 1
        for i = 2:K[1]-1
            @inbounds begin
                xij = [x[1][i], x[2][j]]
                idxi = K[2]*(i-1) + j
                idxj1f = idxi + K[2]; idxj1b = idxi - K[2]

                aij = [a1[idxi], a2[idxi]]

                bval = model.b(t,xij,aij)
                sval2 = model.σ(t,xij,aij).^2
                coeff1f = -(sval2[1]*htaux2[1] + max(bval[1],0.)*taux[1])
                coeff1b = -(sval2[1]*htaux2[1] - min(bval[1],0.)*taux[1])
                coeff0 = 1.0-(coeff1f+coeff1b)

                counter = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
                counter = setIJV!(I,J,V,idxi,idxj1f,coeff1f, counter)
                counter = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)

                rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
            end
        end
    end

    # 3.1 1D PDE approximation at boundary x_2 = 0, x_1 = xmax[1]
    let
        j = 1; i = K[1]
        xij = [x[1][i], x[2][j]]
        idxi = K[2]*(i-1) + j
        idxj1b = idxi - K[2]; idxj1bb = idxi - 2K[2]

        aij = [a1[idxi], a2[idxi]]

        bval = model.b(t,xij,aij)
        sval2 = model.σ(t,xij,aij).^2
        coeff1b = 2*sval2[1]*htaux2[1] + bval[1]*taux[1]
        coeff1bb = -sval2[1]*htaux2[1]
        coeff0 = 1.0-(coeff1b+coeff1bb)

        counter = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
        counter = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
        counter = setIJV!(I,J,V,idxi,idxj1bb,coeff1bb, counter)

        rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
    end

    # 4. For x_2 = xmax[2], x_1 interior, approximate x_2 differential operators with backward difference
    let
        j = K[2]
        for i = 2:K[1]-1
            @inbounds begin
                xij       = [x[1][i], x[2][j]]
                idxi      = K[2]*(i-1) + j
                idxj1f    = idxi + K[2] ; idxj1b = idxi - K[2]
                idxj2b    = idxi - 1    ; idxj2bb = idxi - 2

                aij       = [a1[idxi], a2[idxi]]

                bval      = model.b(t,xij,aij)
                sval2     = model.σ(t,xij,aij).^2
                coeff1f   = -(sval2[1]*htaux2[1] + max(bval[1],0.)*taux[1])
                coeff1b   = -(sval2[1]*htaux2[1] - min(bval[1],0.)*taux[1])
                coeff2b   = 2*sval2[2]*htaux2[2] + bval[2]*taux[2]
                coeff2bb  = -sval2[2]*htaux2[2]
                coeff0    = 1.0-(coeff1f+coeff1b + coeff2b+coeff2bb)

                counter   = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
                counter   = setIJV!(I,J,V,idxi,idxj1f,coeff1f, counter)
                counter   = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
                counter   = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)
                counter   = setIJV!(I,J,V,idxi,idxj2bb,coeff2bb, counter)

                rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
            end
        end
    end

# TODO: why is this not indented? Emacs ...?
# 5. For x_1 = xmax[1], x_2 interior, approximate x_1 differential operators with backward difference
let
    i = K[1]
    for j = 2:K[2]-1
        @inbounds begin
            xij       = [x[1][i], x[2][j]]
            idxi      = K[2]*(i-1) + j
            idxj1b    = idxi - K[2]; idxj1bb = idxi - 2K[2]
            idxj2f    = idxi + 1;    idxj2b = idxi - 1

            aij       = [a1[idxi], a2[idxi]]

            bval      = model.b(t,xij,aij)
            sval2     = model.σ(t,xij,aij).^2

            coeff1b   = 2*sval2[1]*htaux2[1] + bval[1]*taux[1]
            coeff1bb  = -sval2[1]*htaux2[1]
            coeff2f   = -(sval2[2]*htaux2[2] + max(bval[2],0.)*taux[2])
            coeff2b   = -(sval2[2]*htaux2[2] - min(bval[2],0.)*taux[2])
            coeff0    = 1.0-(coeff1b+coeff1bb + coeff2f+coeff2b)

            counter   = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
            counter   = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
            counter   = setIJV!(I,J,V,idxi,idxj1bb,coeff1bb, counter)
            counter   = setIJV!(I,J,V,idxi,idxj2f,coeff2f, counter)
            counter   = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)

            rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
        end
    end
end

# 6. For x = xmax, approximate all operators with backward difference
let
    i = K[1]; j = K[2]
    xij       = [x[1][i], x[2][j]]
    idxi      = K[2]*(i-1) + j
    idxj1b    = idxi - K[2]; idxj1bb = idxi - 2K[2]
    idxj2b    = idxi - 1; idxj2bb = idxi - 2

    aij       = [a1[idxi], a2[idxi]]

    bval      = model.b(t,xij,aij)
    sval2     = model.σ(t,xij,aij).^2

    coeff1b   = 2*sval2[1]*htaux2[1] + bval[1]*taux[1]
    coeff1bb  = -sval2[1]*htaux2[1]
    coeff2b   = 2*sval2[2]*htaux2[2] + bval[2]*taux[2]
    coeff2bb  = -sval2[2]*htaux2[2]
    coeff0    = 1.0-(coeff1b+coeff1bb + coeff2b+coeff2bb)

    counter   = setIJV!(I,J,V,idxi,idxi,coeff0, counter)
    counter   = setIJV!(I,J,V,idxi,idxj1b,coeff1b, counter)
    counter   = setIJV!(I,J,V,idxi,idxj1bb,coeff1bb, counter)
    counter   = setIJV!(I,J,V,idxi,idxj2b,coeff2b, counter)
    counter   = setIJV!(I,J,V,idxi,idxj2bb,coeff2bb, counter)

    rhs[idxi] = v[idxi] + Δt*model.f(t,xij,aij)
end
@assert counter == length(V)
end



function updateinteriorpol!(pol1, pol2, v, model::HJBTwoDim, t,
                            x::Tuple, Δx::Vector;
                            tol=1e-3)
    # Loops over each interior x value and optimises the control
    @assert size(pol1) == size(pol2)

    K = [length(xi) for xi in x]
    invdx = 1.0 ./ Δx
    hdx2 = 0.5 ./ Δx.^2

    function hamiltonian(a, i::Int, j::Int)
        # Evaluate Hamiltonian with value a at (t,x_{i,j}), indices starting at 1
        # coeffn = values in linear system
        # e.g. coeff0  is the coefficient in front of v at x_{i,j}
        #      coeff1f is the coefficient in front of v at x_{i+1,j}
        #      coeff2b is the coefficient in front of v at x_{i,j-1}
        @inbounds begin
            idxi = K[2]*(i-1) + j
            idxj1f = idxi + K[2]; idxj1b = idxi - K[2]
            idxj2f = idxi + 1;    idxj2b = idxi - 1

            xij = [x[1][i], x[2][j]]
            bval = model.b(t,xij,a)
            sval2 = model.σ(t,xij,a).^2
            coeff1f = sval2[1]*hdx2[1] + max(bval[1],0.)*invdx[1]
            coeff1b = sval2[1]*hdx2[1] - min(bval[1],0.)*invdx[1]
            coeff2f = sval2[2]*hdx2[2] + max(bval[2],0.)*invdx[2]
            coeff2b = sval2[2]*hdx2[2] - min(bval[2],0.)*invdx[2]
            return (coeff1f*(v[idxi]-v[idxj1f]) + coeff1b*(v[idxi]-v[idxj1b]) +
                    coeff2f*(v[idxi]-v[idxj2f]) + coeff2b*(v[idxi]-v[idxj2b]) -
                    model.f(t,xij,a))
        end
    end

    for i = 2:K[1]-1, j = 2:K[2]-1
        @inbounds begin
            idxi = K[2]*(i-1) + j
            objective(a) = hamiltonian(a, i, j)
            g!(x, out) = ForwardDiff.gradient!(out, objective, x)
            diffobj = DifferentiableFunction(objective, g!)
            res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                           model.amin, model.amax, Fminbox(),
                           optimizer = LBFGS)
            # TODO: add optimizer options?
            pol1[idxi], pol2[idxi] = res.minimum
        end
    end

end

function updateboundarypol!(pol1, pol2, v, model::HJBTwoDim, t,
                            x::Tuple, Δx::Vector;
                            tol=1e-3)
    #==
    1.  For x = 0, pol1 and pol 2 is NaN
    2.  For x_1 = 0, pol1 = NaN, pol2 normal
    2.1 For x_2 = xmax[2], pol1 = NaN, pol2 backward difference
    3.  For x_2 = 0, pol1 normal, pol2 = NaN
    3.1 For x_1 = xmax[1], pol1 backward difference, pol2 = NaN
    4.  For x_2 = xmax[2], pol1 normal, pol2 backward difference
    5.  For x_1 = xmax[1], pol1 backward difference, pol2 normal
    6.  For x = xmax, pol1 and pol 2 backward difference

    # TODO: how to deal with pol-values that are set to NaN? Optim dies
    ==#

    # Loops over each interior x value and optimises the control
    @assert size(pol1) == size(pol2)

    K = [length(xi) for xi in x]
    invdx = 1.0 ./ Δx
    hdx2 = 0.5 ./ Δx.^2

    # 1.  For x = 0, pol1 and pol 2 is NaN
    # pol1[1] = NaN; pol2[1] = NaN

    # 2.  For x_1 = 0, pol1 = NaN, pol2 normal
    i = 1
    for j = 2:K[2]-1
        #@show (i,j)
        xij = [x[1][i], x[2][j]]
        idxi = K[2]*(i-1) + j
        idxj2f = idxi + 1; idxj2b = idxi - 1

        function objective(a)
            bval = model.b(t,xij,a)
            sval2 = model.σ(t,xij,a).^2

            coeff2f = sval2[2]*hdx2[2] + max(bval[2],0.)*invdx[2]
            coeff2b = sval2[2]*hdx2[2] - min(bval[2],0.)*invdx[2]
            coeff0 = -(coeff2f+coeff2b)

            return -(coeff0*v[idxi] + coeff2f*v[idxj2f] + coeff2b*v[idxj2b] +
                     model.f(t,xij,a))
        end

        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)
        res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                       model.amin, model.amax, Fminbox(),
                       optimizer = LBFGS)#, optimizer_o=OptimizationOptions(show_trace=true, extended_trace=true))
        #pol1[idxi] = NaN
        # pol2[idxi] = res.minimum[2]
        pol1[idxi], pol2[idxi] = res.minimum
    end

    # 2.1 For x_2 = xmax[2], pol1 = NaN, pol2 backward difference
    let
        const i = 1; const j = K[2]
        xij = [x[1][i], x[2][j]]
        idxi = K[2]*(i-1) + j
        idxj2b = idxi - 1; idxj2bb = idxi - 2

        function objective(a)
            bval = model.b(t,xij,a)
            sval2 = model.σ(t,xij,a).^2
            coeff2b = -(2*sval2[2]*hdx2[2] + bval[2]*invdx[2])
            coeff2bb = sval2[2]*hdx2[2]
            coeff0 = -(coeff2b+coeff2bb)

            return -(coeff0*v[idxi] + coeff2b*v[idxj2b] + coeff2bb*v[idxj2bb] +
                     model.f(t,xij,a))
        end

        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)
        res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                       model.amin, model.amax, Fminbox(),
                       optimizer = LBFGS)
        #pol1[idxi] = NaN
        #pol2[idxi] = res.minimum[2]
        pol1[idxi], pol2[idxi] = res.minimum
    end

    # 3.  For x_2 = 0, pol1 normal, pol2 = NaN
    j = 1
    for i = 2:K[1]-1
        xij = [x[1][i], x[2][j]]
        idxi = K[2]*(i-1) + j
        idxj1f = idxi + K[2]; idxj1b = idxi - K[2]

        function objective(a)
            bval = model.b(t,xij,a)
            sval2 = model.σ(t,xij,a).^2

            coeff1f = sval2[1]*hdx2[1] + max(bval[1],0.)*invdx[1]
            coeff1b = sval2[1]*hdx2[1] - min(bval[1],0.)*invdx[1]
            coeff0 = -(coeff1f+coeff1b)

            return -(coeff0*v[idxi] + coeff1f*v[idxj1f] + coeff1b*v[idxj1b] +
                     model.f(t,xij,a))
        end

        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)
        res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                       model.amin, model.amax, Fminbox(),
                       optimizer = LBFGS)
        # pol1[idxi] = res.minimum[1]
        # pol2[idxi] = NaN
        pol1[idxi], pol2[idxi] = res.minimum
    end

    # 3.1 For x_1 = xmax[1], pol1 backward difference, pol2 = NaN
    let
        const i = K[1]; const j = 1
        xij = [x[1][i], x[2][j]]
        idxi = K[2]*(i-1) + j
        idxj1b = idxi - K[2]; idxj1bb = idxi - 2K[2]

        function objective(a)
            bval = model.b(t,xij,a)
            sval2 = model.σ(t,xij,a).^2
            coeff1b = -(2*sval2[1]*hdx2[1] + bval[1]*invdx[1])
            coeff1bb = sval2[1]*hdx2[1]
            coeff0 = -(coeff1b+coeff1bb)

            return -(coeff0*v[idxi] + coeff1b*v[idxj1b] + coeff1bb*v[idxj1bb] +
                     model.f(t,xij,a))
        end

        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)
        res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                       model.amin, model.amax, Fminbox(),
                       optimizer = LBFGS)
        # pol1[idxi] = res.minimum[1]
        # pol2[idxi] = NaN
        pol1[idxi], pol2[idxi] = res.minimum
    end

    # 4.  For x_2 = xmax[2], pol1 normal, pol2 backward difference
    let
        const j = K[2]
        for i = 2:K[1]-1
            xij       = [x[1][i], x[2][j]]
            idxi      = K[2]*(i-1) + j
            idxj1f    = idxi + K[2] ; idxj1b = idxi - K[2]
            idxj2b    = idxi - 1    ; idxj2bb = idxi - 2

            function objective(a)
                bval      = model.b(t,xij,a)
                sval2     = model.σ(t,xij,a).^2
                coeff1f   = sval2[1]*hdx2[1] + max(bval[1],0.)*invdx[1]
                coeff1b   = sval2[1]*hdx2[1] - min(bval[1],0.)*invdx[1]
                coeff2b   = -(2*sval2[2]*hdx2[2] + bval[2]*invdx[2])
                coeff2bb  = sval2[2]*hdx2[2]
                coeff0    = -(coeff1f+coeff1b + coeff2b+coeff2bb)

                return -(coeff0*v[idxi] + coeff1f*v[idxj1f] + coeff1b*v[idxj1b] +
                         coeff2b*v[idxj2b] + coeff2bb*v[idxj2bb] +
                         model.f(t,xij,a))
            end
            g!(x, out) = ForwardDiff.gradient!(out, objective, x)
            diffobj = DifferentiableFunction(objective, g!)
            res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                           model.amin, model.amax, Fminbox(),
                           optimizer = LBFGS)
            pol1[idxi], pol2[idxi] = res.minimum
        end
    end

# 5.  For x_1 = xmax[1], pol1 backward difference, pol2 normal
let
    const i = K[1]
    for j = 2:K[2]-1
        xij       = [x[1][i], x[2][j]]
        idxi      = K[2]*(i-1) + j
        idxj1b    = idxi - K[2]; idxj1bb = idxi - 2K[2]
        idxj2f    = idxi + 1;    idxj2b = idxi - 1


        function objective(a)
            bval      = model.b(t,xij,a)
            sval2     = model.σ(t,xij,a).^2

            coeff1b   = -(2*sval2[1]*hdx2[1] + bval[1]*invdx[1])
            coeff1bb  = sval2[1]*hdx2[1]
            coeff2f   = sval2[2]*hdx2[2] + max(bval[2],0.)*invdx[2]
            coeff2b   = sval2[2]*hdx2[2] - min(bval[2],0.)*invdx[2]
            coeff0    = -(coeff1b+coeff1bb + coeff2f+coeff2b)

            return -(coeff0*v[idxi] + coeff1b*v[idxj1b] + coeff1bb*v[idxj1bb] +
                     coeff2f*v[idxj2f] + coeff2b*v[idxj2b] +
                     model.f(t,xij,a))
        end

        g!(x, out) = ForwardDiff.gradient!(out, objective, x)
        diffobj = DifferentiableFunction(objective, g!)
        res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                       model.amin, model.amax, Fminbox(),
                       optimizer = LBFGS)
        pol1[idxi], pol2[idxi] = res.minimum
    end
end
# 6.  For x = xmax, pol1 and pol 2 backward difference
let
    i = K[1]; j = K[2]
    xij       = [x[1][i], x[2][j]]
    idxi      = K[2]*(i-1) + j
    idxj1b    = idxi - K[2]; idxj1bb = idxi - 2K[2]
    idxj2b    = idxi - 1; idxj2bb = idxi - 2

    function objective(a)
        bval      = model.b(t,xij,a)
        sval2     = model.σ(t,xij,a).^2

        coeff1b   = -(2*sval2[1]*hdx2[1] + bval[1]*invdx[1])
        coeff1bb  = sval2[1]*hdx2[1]
        coeff2b   = -(2*sval2[2]*hdx2[2] + bval[2]*invdx[2])
        coeff2bb  = sval2[2]*hdx2[2]
        coeff0    = -(coeff1b+coeff1bb + coeff2b+coeff2bb)
        return -(coeff0*v[idxi] + coeff1b*v[idxj1b] + coeff1bb*v[idxj1bb] +
                 coeff2b*v[idxj2b] + coeff2bb*v[idxj2bb] +
                 model.f(t,xij,a))
    end

    g!(x, out) = ForwardDiff.gradient!(out, objective, x)
    diffobj = DifferentiableFunction(objective, g!)
    res = optimize(diffobj, [pol1[idxi], pol2[idxi]],
                   model.amin, model.amax, Fminbox(),
                   optimizer = LBFGS)
    pol1[idxi], pol2[idxi] = res.minimum
end
end

function policynewtonupdate_boundary{T<:Real}(model::HJBTwoDim{T},
                                     v, a1, a2, x::Tuple{Vector{T},Vector{T}},
                                     Δx::Vector{T}, Δt, ti::Int;
                                     tol = 1e-3,
                                     scale = 1.0,
                                     maxpolicyiter::Int = 10)
    # v  = value function at previous time-step
    # an = policy function at previous time-step / initial guess for update
    t = (ti-1)*Δt
    @show t
    n = length(v)
    K = [length(xi) for xi in x]
    @assert length(a1) == n && length(a2) == n

    # Elements in sparse system matrix (n\times n) size
    interiornnz = 5*prod(K-2)
    boundarynnz = 8*sum(K-2) + 12
    totnnz = interiornnz + boundarynnz
    Ii = zeros(Int, interiornnz); Ji = zeros(Ii); Vi = zeros(T, interiornnz)
    Ib = zeros(Int, boundarynnz); Jb = zeros(Ib); Vb = zeros(T, boundarynnz)

    rhs = zeros(v)

    # TODO: copy or pass reference?
    pol1 = copy(a1)
    pol2 = copy(a2)
    vnew = copy(v)

    for k in 0:maxpolicyiter
        # TODO: create updatepol! that calls both functions
        updateboundarypol!(pol1, pol2, vnew, model, t, x, Δx)
        updateinteriorpol!(pol1, pol2, vnew, model, t, x, Δx)
        # TODO: create updatesystem! that calls both?
        updateboundarysystem!(Ib,Jb,Vb, rhs, model, v, t, x, pol1, pol2, Δt, Δx)
        updateinteriorsystem!(Ii,Ji,Vi, rhs, model, v, t, x, pol1, pol2, Δt, Δx)

        Mat = sparse([Ib;Ii],[Jb;Ji],[Vb;Vi],n,n,(x,y)->error("Each index should be unique"))

        # TODO: Use Krylov solver for high-dimensional PDEs?
        vold = vnew
        vnew = Mat\rhs

        vchange = maximum(abs(vnew-vold)./max(1.,abs(vnew)))
        if vchange < Δt*tol && k>0
            break
        end
    end
    # TODO: do we need this one?
    updateboundarypol!(pol1, pol2, vnew, model, t, x, Δx)
    updateinteriorpol!(pol1, pol2, vnew, model, t, x, Δx)

    return vnew, pol1, pol2
end

function timeloopiteration_boundary(model::HJBTwoDim, K::Vector{Int}, N::Int,
                           Δt, vinit, x::Tuple, Δx::Vector)
    # TODO: Pass v and pol by reference?
    v = zeros(length(vinit), N+1)
    # No policy at t = T
    pol1 = zeros(length(vinit), N)
    pol2 = zeros(length(vinit), N)
    pol = (pol1, pol2)

    @inbounds v[:,N+1] = vinit

    # initial guess for control
    pol1init = fill(0.5*(model.amax[1]+model.amin[1]), prod(K))
    pol2init = fill(0.5*(model.amax[2]+model.amin[2]), prod(K))
    @inbounds v[:,N], pol1[:,N], pol2[:,N] = policynewtonupdate_boundary(model, v[:,N+1], pol1init, pol2init,
                                                                x, Δx, Δt, N)

    for j = N-1:-1:1
        # t = (j-1)*Δt
        # TODO: pass v-column, pol-columns by reference?
        @inbounds (v[:,j], pol1[:,j],
                   pol2[:,j]) = policynewtonupdate_boundary(model, v[:,j+1], pol1[:,j+1], pol2[:,j+1],
                                                   x, Δx, Δt, j)
    end

    return v, pol
end

function solve_boundary{T1<:Real}(model::HJBTwoDim{T1}, K::Vector{Int}, N::Int)
    # K   = number of points in each direction of the space domain
    # N+1 = number of points in time domain
    x1 = linspace(model.xmin[1], model.xmax[1], K[1])
    x2 = linspace(model.xmin[2], model.xmax[2], K[2])
    x = (collect(x1), collect(x2))
    Δx = (model.xmax-model.xmin)./(K-1)
    Δt = model.T/N

    vinit = zeros(T1, prod(K))
    for i = 1:K[1], j = 1:K[2]
        @inbounds begin
            idx = K[2]*(i-1)+j
            xij = [x[1][i], x[2][j]]
            vinit[idx] = model.g(xij)
        end
    end

    v, pol = timeloopiteration_boundary(model, K, N, Δt, vinit, x, Δx)
    return v, pol
end

solve_boundary(model::HJBTwoDim, K::Int, N::Int) = solve_boundary(model, [K,K], N)
