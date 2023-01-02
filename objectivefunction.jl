using DataFrames
using CSV
using StatsModels 
using LinearAlgebra
using Random
using Plots
##2a
oil_wells = CSV.read("/Users/dhondupdolma/Desktop/Data/oil-well-drilling-costs.csv", DataFrame)

fo= @formula(Cost ~ 1+ Depth) 
h(x)=.5*(A*x-b)' * (A*x -b) # h is the objective function
A= modelmatrix(fo.rhs, oil_wells)  # 16 by 2 model matrix A
b = vec(modelmatrix(fo.lhs, oil_wells)) # response vector b
##2b
∇f(x)= A'*A*x - A'*b
x =[-2200; .5] 
∇f(x) #gradient of the function h

##2c
∇2h= A'*A # hessian 
∇2h
##2e
function newton_method(∇f, x) # ∇n is gradient at x 
    i = 1
    α=1
    k =10000
    while norm(∇f(x)) > 1e-1
        p = -inv(∇2h)*(∇f(x))  # steepest descent direction
        x = x + α*p  # the new iterate
        i % k == 0 && println("iteration ", i, ". x = ", x)
        i += 1  
    end
    return x # returns a vector x 
end

##2f
x =[-2200; .5]
∇h(x)= A'*A*x - A'*b
new1 = newton_method(∇h,x) 
println(new1)
# newton_method to find the mimizer of the function
##2g
function backtracking_line_search(f, ∇f, x, p)
    α = 1    # initial step length = 1 required for newton-like methods
    rho = 0.75
    c = 1e-4
    x1 = x[1,1]
    x2= x[2,1]
    x12 = x + α*p
    
    while (f.(x12[1,1], x12[2,1])) > (f.(x1,x2) + c*α*transpose(∇f)*p)
        # sufficient decrease not met, reduce step size
        α = rho * α
        x12 = x + α*p  
    end
    return α
end

function newton_method1( f, ∇f, ∇2h::Matrix, x::Vector; α=1, ϵ=1e-1, k=1000) # ∇n is gradient at x 
    i = 1
    while norm(∇f) > ϵ 
        p = -inv(∇2h) * (∇f)  # steepest descent direction
        # fixed step size for now
        α = backtracking_line_search(f, ∇f, x, p)
        x = x + α*p  # the new iterate
        y = x[1,1]
        z = x[2,1]
       # ∇f1 = 2y - 2 - 200y*(2z - 2(y^2))
      #  ∇f2 =  200z - 200(y^2)
        ∇f1 = -400*y*(z-y^2)-2(1-y)
        ∇f2 =  200(z-y^2)
        ∇f = [∇f1; ∇f2]
        ∇2h=[-400(z-y^2)+800y^2 -400y; -400y 200]
        i % k == 0 && println("iteration ", i, ". x = ", x)
        i += 1
    end
    return x
end


y=1.2#-1.2
z=1.2#1
x = [y;z]
#hes = [-400*z+1200*y^1+2 -400*y;-400*y 200]
hes=[-400(z-y^2)+800y^2 -400y; -400y 200]
∇f1 = -400*y*(z-y^2)-2(1-y)
∇f2 =  200(z-y^2)
∇j = [∇f1; ∇f2]
g(y,z) = 100*(z −(y)^2)^2 +(1−y)^2 
new2= newton_method1(g, ∇j,hes,  x, k=1000)
println(new2)



