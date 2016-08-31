export HJBOneDim, HJBTwoDim

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


type HJBTwoDim{T1<:Real} <: AbstractHJB
    b::Function      # Drift term
    σ::Function      # Volatility term
    f::Function      # "Gain" function
    g::Function      # Terminal value
    T::T1            # Terminal time
    amin::Vector{T1} # Control minimum value
    amax::Vector{T1} # Control maximum value
    xmin::Vector{T1} # Domain minimum value
    xmax::Vector{T1} # Domain maximum value
    Dbound::Function # Dirichlet condition on x-boundary
    # TODO: assert T>0, amin < amax, xmin < xmax
    # TODO: assert that the vectors are of size 2
    # TODO: make sure b and sigma return a length-2 vector
end
