export HJBOneDim

abstract AbstractHJB

type HJBOneDim{T1<:Real} <: AbstractHJB
    b::Function # Drift term
    σ::Function # Volatility term
    f::Function # "Gain" function
    g::Function # Terminal value
    T::T1     # Terminal time
    amin::T1  # Control minimum value
    amax::T1  # Control maximum value
    xmin::T1  # Domain minimum value
    xmax::T1  # Domain maximum value
    # bcond = (true, false) means Dirichlet condition on x=x_{min},
    #         but no condition on x=x_{max}
    bcond::Tuple{Bool,Bool}
    Dfun::Tuple{Function,Function} # Dirichlet condition on boundaries
end
