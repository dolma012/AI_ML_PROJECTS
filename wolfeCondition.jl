using DataFrames, CSV, LinearAlgebra
using GLM, StatsModels, Plots
using ForwardDiff
# use the data on cherry trees
Trees = CSV.read("/Users/dhondupdolma/Desktop/Data/cherry-trees.csv", DataFrame)
using Symbolics
fo = @formula(Volume ~ 1 + Girth)
A = modelmatrix(fo.rhs, Trees)      # the predictors
print(A)
b = vec(modelmatrix(fo.lhs, Trees)) # the response
[1.0 8.3; 1.0 8.6; 1.0 8.8; 1.0 10.5; 1.0 10.7; 1.0 10.8; 1.0 11.0; 1.0 11.0; 1.0 11.1; 1.0 11.2; 1.0 11.3; 1.0 11.4; 1.0 11.4; 1.0 11.7; 1.0 12.0; 1.0 12.9; 1.0 12.9; 1.0 13.3; 1.0 13.7; 1.0 13.8; 1.0 14.0; 1.0 14.2; 1.0 14.5; 1.0 16.0; 1.0 16.3; 1.0 17.3; 1.0 17.5; 1.0 17.9; 1.0 18.0; 1.0 18.0; 1.0 20.6]


function line_search(f, ∇f, x, p)
    α = 1    # initial step length = 1 required for newton-like methods
    rho = 0.75
    c = .01
    while f(x + α*p) > f(x) + c*α*transpose(∇f(x))*p
        # sufficient decrease not met, reduce step size
        α = rho * α
    end
    return α
end

function hub(u; M=1) #takes in a parameter u which is a scalar value supposedly a residual 
    sum =0
    if abs(u) <= M #check whether the absolute value of the residual is less than or equal to M
        return u^2 # square the residual
    else 
        return M*(2*abs(u)-M)
    end
end

function hubsum(x; M=1) #takes in a parameter x which is a vector 
    sum = 0
    for i in 1:31
        checkhub = A[i,:]'*x-b[i] # residual of each data point 
        sum = sum +  hub(checkhub) # pass the residual into the hub function and add the huberdized residual to total sum
         # return the sum of the huberdized residual
    end   
    return sum 
end 

function ∇h(x; M=1)  #gradient of the residual function 
    j = 1
    sum =0
    for k in 1:31 #loop for the gradient with respect to x1
        u = A[k,:]'*x-b[k] #residual of each data point 
        if abs(u) <= M #check whether the absolute value of the residual is less than or equal to 1 
            sum = sum + 2*(A[k,j]*x[1] + A[k,j+1]*x[2] - b[k])*A[k,j]# #gradient of the residual of with respect to x1
        else 
            sum = sum + 2*M*A[k,j]*sign(u) #if abs(u ) > 1 gradient of the residual of with respect to x1
        end
    end
    sum1=0
    y =1
    for t in 1:31 #loop for the gradient with respect to x2
        u1 = A[t,:]'*x-b[t]   #the residual of each data point 
        if abs(u1) <= M #check whether the absolute value of the residual is less than or equal to 1 
            sum1 = sum1 + 2*(A[t,y]*x[1] + A[t,y+1]*x[2] - b[t])*A[t,y+1] #gradient of the residual of with respect to x2
        else 
            sum1 = sum1 + 2*M*A[t,y+1]*sign(u1) #if abs(u ) > 1 gradient of the residual of with respect to x2
        end
    end
    return [sum; sum1] #returns a vector of the gradient  
    
end

function steepest_descent(f, ∇f, x; ϵ=1e-1, k=100)
    i = 1
    α =1   
    while norm(∇f(x)) > ϵ
        # compute the direction
        B = -I       # steepest descent
        p = B * ∇f(x)
        # determine the step size
        α = line_search(f, ∇f, x, p)
        x = x + α*p  # the new iterate

        if i % k == 0
            println("updatex:", x + α*p )
            println("iteration ", i, ": step size = ", α,"p", p ,", x = ", x)
        end
        i += 1
    end
    return x
end


x_ls = steepest_descent(hubsum, ∇h,[1,1])

println(x_ls)
a = A[:,2]

h(x) = x_ls[1]+ x_ls[2]*x  #x is suppose to be the girth 
scatter(a, b, label="", xlabel="Girth", ylabel="Volume") #data points 
plot!(h, label = "residual sum of squared", legend = :bottomright)