x = 1
α = 2 # \alpha + TAB key

var"log(1+θ)" = 10

x = 1
for i in 1:10
    x += 1
end
println(x)

function wrong!(a::Array)
    a = ones(size(a))
end

function right!(a::Array)
    a[:] = ones(size(a))
end

using Random
v = rand(2, 2)
println("v : ", v)

wrong!(v)
println("wrong : ", v)

right!(v)
println("right : ", v)

foo(a,b) = sum(a) + b

println(foo.(Ref([1,2]),[3,4,5]))
println(foo.(([1,2],), [3,4,5]))
println(foo.([[1,2]], [3,4,5]))
