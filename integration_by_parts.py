from sympy import * 

x=symbols('x') 
y=symbols('y') 

def f(x,y): 
    u = Function('u') 
    return u(x)  

def g(x,y): 
    p = Function('p') 
    return diff(diff(p(x),x),x) 

def IPs(n,a,b,r): 
    if n==1: 
        ans = a * integrate(b,r) - integrate(diff(a,r)*integrate(b,r),r) 
        return ans 
    else: 
        ans = IPs(n-1,a,integrate(b,r),r) - integrate(IPs(n-1,diff(a,r),integrate(b,r),r),r) 
        return ans 

dummy=IPs(2,f(x,y),g(x,y),x) 
init_printing()
pprint(dummy)
