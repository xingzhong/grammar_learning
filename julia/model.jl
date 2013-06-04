using Distributions
using PatternDispatch

type State
  symbol::String
  left::Union(Distribution, ())
  right::Union(Distribution, ())
  init::Real
  State(symbol, left) = new(symbol, left, (), 0.0)
  State(symbol, left, init) = new(symbol, left, (), init)
  State(symbol, left, right) = new(symbol, left, right, 0.0)
  State(symbol, left, right, init) = new(symbol, left, right, init)
end

type Trans
  from::State
  to::State
  prob::Real
  Trans(from, to) = new(from, to, 0.0)
  Trans(from, to, prob) = new(from, to, prob)
end

@pattern singlePdf(state::State, x::Real) = begin 
  if state.left != () && state.right != ()
    return (0.0, 0.0)
  else
    return (Pdf(state.left, x), Pdf(state.right, x))
  end
end
@pattern Pdf((), x::Real) = 0.0
@pattern Pdf(dist::Distribution, x::Real) = pdf(dist, x)

model = Array(Trans, 0)
states = Array(State, 0)
A = State("A", Normal(-1.0, 1.0), Normal(1.0, 1.0))
B = State("B", Normal(0.0, 1.0), ())
C = State("C", (), Normal(1.0, 2.0))
push!(states, A)
push!(states, B)
push!(states, C)
push!(model, Trans(A,B,0.5))
push!(model, Trans(A,A,0.5))
push!(model, Trans(B,A,0.3))
push!(model, Trans(B,B,0.4))
push!(model, Trans(B,C,0.3))
push!(model, Trans(C,B,0.5))
push!(model, Trans(C,C,0.5))
map(x->println("$(x.from.symbol) -> $(x.to.symbol) [$(x.prob)]"), model)
println()
seq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
single = map(x->map(y->singlePdf(y, x), states), seq)
map(x->map(println, x), single)
double = { (seq[i], seq[i+1]) for i:length(seq)-1 }
map(println, double)
println()
