#This is basically F81, but with arb state size and linear prop ops
#IndependentDiscreteDiffusion = "Independent Jumps", as in every time a mutation event happens, you jump to a new state independent of the current state.
struct IndependentDiscreteDiffusion{K, T <: Real} <: DiscreteStateProcess
    r::T
    π::SVector{K, T}

    function IndependentDiscreteDiffusion{T}(r::T, π::SVector{K, T}) where {K, T <: Real}
        r > 0 || throw(ArgumentError("r must be positive"))
        sum(π) > 0 || throw(ArgumentError("sum of π must be positive"))
        all(≥(0), π) || throw(ArgumentError("elements of π must be non-negative"))
        return new{K, T}(r, π ./ sum(π))
    end
end

"""
    IndependentDiscreteDiffusion(r::Real, π::AbstractVector{<: Real})

Create a discrete diffusion process with independent jumps.

The new state after a state transition is independent of the current state.  The
transition probability matrix at time t is

    P(t) = exp(r Q t),

where Q is a rate matrix with equilibrium distribution π.
"""
function IndependentDiscreteDiffusion(r::Real, π::SVector{K, <: Real}) where K
    T = promote_type(typeof(r), eltype(π))
    return IndependentDiscreteDiffusion{T}(convert(T, r), convert(SVector{K, T}, π))
end

eq_dist(model::IndependentDiscreteDiffusion) = Categorical(model.π)

function forward(process::IndependentDiscreteDiffusion, x_s::AbstractArray, s::Real, t::Real)
    #println("Size of x_s in forward: ", size(x_s))
    (;r, π) = process
    pow = exp(-r * (t - s))
    c1 = (1 - pow) .* π
    c2 = pow .+ c1
    return CategoricalVariables([@. c1 * (1 - x) + c2 * x for x in x_s])
end

function backward(process::IndependentDiscreteDiffusion, x_t::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    pow = exp(-r * (t - s))
    c1 = (1 - pow) .* π
    return [pow * x .+ x'c1 for x in x_t]
end

_sampleforward(rng::AbstractRNG, process::IndependentDiscreteDiffusion, t::Real, x::AbstractArray) =
    sample(rng, forward(process, x, 0, t))

function _endpoint_conditioned_sample(rng::AbstractRNG, process::IndependentDiscreteDiffusion, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    #println("f3A")
    prior = forward(process, x_0, 0, s)
    #println("Size of of prior: ", size(prior))
    #println(" prior: ", prior)


    likelihood = backward(process, x_t, s, t)
    #println("f3B")

    temp = sample(rng, combine(prior, likelihood))
    #println("f3C")

    return temp
end


struct PiQDiffusion{K, T <: Real} <: DiscreteStateProcess
    r::T
    π::SVector{K, T}
    function PiQDiffusion(r, π::SVector{K, T}) where {K, T <: Real}
        new{K,T}(r, π)
    end
end


# # Define the function to convert the tensor to an array of SVectors
# function tensor_to_svector_array(tensor::Array{Float64, 3})
#     # Reshape the tensor to a 2D array where each column is a vector of length 4
#     reshaped_tensor = reshape(tensor, size(tensor, 1), :)

#     # Convert each column to an SVector{4, Float64} and reshape the result into a 5x12 array
#     svector_array = reshape(SVector{4, Float64}.(eachcol(reshaped_tensor)), size(tensor, 2), size(tensor, 3))

#     return svector_array
# end

# Handle case when x_s is dense? like with the one-hot training data x_0
function forward(process::PiQDiffusion, x_s::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    #cum_noise =  a*(b^t-b^s)
    #pow = exp(-cum_noise)
    cum_noise = r * (t - s)
    pow = exp(-cum_noise)
    return pow*x_s+(1-pow)*π.*sum(x_s, dims = 1)
end

# Handle case when x_s is dense? like with the one-hot training data x_0
function backward(process::PiQDiffusion, x_t::AbstractArray, s::Real, t::Real)
    (;r, π) = process
    cum_noise =  r*(t-s)
    pow = exp(-cum_noise)
    return pow*x_t.+(1-pow).*sum(π.*x_t, dims=1)
end

function _endpoint_conditioned_distrubution(process::PiQDiffusion, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    prior = forward(process, x_0, 0, s)
    likelihood = backward(process, x_t, s, t)
    temp = prior.*likelihood
    return temp ./ sum(temp, dims=1)
end


function _endpoint_conditioned_sample(rng::AbstractRNG, process::PiQDiffusion, s::Real, t::Real, x_0::AbstractArray, x_t::AbstractArray)
    return my_sample(rng, _endpoint_conditioned_distrubution(process, s, t, x_0, x_t))
end


# function WTF(rng::AbstractRNG, process::PiQDiffusion, t::Real, x::AbstractArray)
#     return my_sample(rng, forward(process, x, 0, t))
# end
_sampleforward(rng::AbstractRNG, process::PiQDiffusion, t::Real, x::AbstractArray) = my_sample(rng, forward(process, x, 0, t))

eq_dist(P::PiQDiffusion) = Categorical(P.π)

function my_sample(rng::AbstractRNG, X)
    x = similar(X)
    d1, d2, d3 = size(X)
    for i=1:d2 
        for j=1:d3
            k = randcat(rng, X[:,i,j])
            x[:,i,j] = onehotsvec(d1, k)
        end
    end
    return x
end



struct MultiPiQDiffusion{K, T <: Real} <: DiscreteStateProcess
    # Assuming noise schedule a*b^t*log(b) from this paper https://arxiv.org/pdf/2205.14987
    #a::Real
    #b::Real
    tree
    π::SVector{K, T}
    function MultiPiQDiffusion(tree, π::SVector{K, T}) where {K, T <: Real}
        new{K,T}(tree, π)
    end
end

#Speed up
function forward(process::MultiPiQDiffusion, x_s::AbstractArray, s::Real, t::Real)
    (;tree, π) = process
    no_event = ones(length(π))
    transitions = zeros(size(x_s))
    for node in tree
        transitions[Int.(node[1]):Int.(node[2]), :, :] .+= (1-exp(-node[3]*(t-s)))*no_event[Int.(node[1])]*(π[Int.(node[1]):Int.(node[2])]/sum(π[Int.(node[1]):Int.(node[2])])).*sum(x_s[Int.(node[1]):Int.(node[2]), :, :], dims = 1)
        no_event[Int.(node[1]):Int.(node[2])] .*= exp(-node[3]*(t-s))
    end
    return no_event.*x_s+transitions
end

#Speed up
function backward(process::MultiPiQDiffusion, x_t::AbstractArray, s::Real, t::Real)
    (;tree, π) = process
    no_event = ones(length(π))
    transitions = zeros(size(x_t))

    #pow*x_t.+(1-pow).*sum(π.*x_t, dims=1)
    for node in tree
        transitions[Int.(node[1]):Int.(node[2]), :, :] .+= (1-exp(-node[3]*(t-s)))*no_event[Int.(node[1])].*sum((π[Int.(node[1]):Int.(node[2])]/sum(π[Int.(node[1]):Int.(node[2])])).*x_t[Int.(node[1]):Int.(node[2]), :, :], dims = 1)
        no_event[Int.(node[1]):Int.(node[2])] .*= exp(-node[3]*(t-s))
    end
    return no_event.*x_t+transitions
end

#_sampleforward(rng::AbstractRNG, process::PiQDiffusion, t::Real, x::AbstractArray) =
#    sample(rng, forward(process, x, 0, t))