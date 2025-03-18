import numpy as np
import matplotlib.pyplot as plt
import time

from numpy import exp, log, sqrt
from scipy.special import erf
from scipy.linalg import solve_banded

def bsexact(sigma: float, R: float, K: float, T: float, s: float):
    d1 = (log(s/K)+(R+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1-sigma*sqrt(T)
    F = 0.5*s*(1+erf(d1/sqrt(2)))-exp(-R*T)*K*0.5*(1+erf(d2/sqrt(2)))
    return F



def compute_option_matrix(j: int, k: int, r: float, maturity: float, strike: float, 
                           volatility: float, gamma_param: float, 
                           price_range: tuple, is_implicit: bool):
    """
    Compute option pricing matrix using finite difference method
    
    Parameters:
    - j: Number of spatial points
    - k: Number of time points
    - r: Risk-free rate
    - maturity: Total time to maturity
    - strike: Strike price
    - volatility: Volatility parameter
    - gamma_param: Constant elasticity of variance (CEV) parameter
    - price_range: Tuple of (min_price, max_price)
    - is_implicit: Boolean to choose implicit or explicit method
    """
    # Time discretization
    time_points, time_step = np.linspace(0, maturity, k, retstep=True)
    
    # Spatial discretization
    price_points, price_step = np.linspace(price_range[0], price_range[1], j, retstep=True)

    # Compute coefficient matrices
    price_factor = price_points**(2 * gamma_param)
    alpha_coeff = 0.5 * volatility**2 * price_factor * (time_step / (price_step**2))
    beta_coeff = r * price_points * (time_step / (2 * price_step))

    if is_implicit:  # Implicit method coefficients
        lower_diag = -(alpha_coeff - beta_coeff)
        main_diag = 1 + r * time_step + 2 * alpha_coeff
        upper_diag = -(alpha_coeff + beta_coeff)

        # Create banded matrix for efficient solving
        matrix_banded = np.zeros((3, j))
        matrix_banded[0, 1:] = upper_diag[:-1]
        matrix_banded[1, :] = main_diag
        matrix_banded[2, :-1] = lower_diag[1:]
    else:  # Explicit method coefficients
        lower_diag = alpha_coeff - beta_coeff
        main_diag = 1 - r * time_step - 2 * alpha_coeff
        upper_diag = alpha_coeff + beta_coeff

        # Full matrix for explicit method
        full_matrix = np.zeros((j, j))
        np.fill_diagonal(full_matrix, main_diag)
        np.fill_diagonal(full_matrix[:-1, 1:], upper_diag[:-1])
        np.fill_diagonal(full_matrix[1:, :-1], lower_diag[1:])

    # Initial option values (payoff function)
    option_values = np.maximum(price_points - strike, 0)
    auxiliary_vector = np.zeros(j-2)

    # Backward time stepping
    for n in range(len(time_points) - 1, -1, -1):
        # Boundary conditions
        option_values[0] = 0
        option_values[-1] = price_range[1] - strike * np.exp(-r * (maturity - time_points[n]))

        if is_implicit:
            # Solve implicit system
            rhs = option_values[1:-1].copy()
            rhs[0] -= lower_diag[1] * option_values[0]
            rhs[-1] -= upper_diag[-2] * option_values[-1]
            
            option_values[1:-1] = solve_banded((1, 1), matrix_banded[:, 1:-1], rhs)
        else:
            # Explicit method calculations
            option_values[1:-1] = (
                np.dot(full_matrix[1:-1, 1:-1], option_values[1:-1]) + 
                auxiliary_vector +
                full_matrix[1:-1, 0] * option_values[0] + 
                full_matrix[1:-1, -1] * option_values[-1]
            )

    return option_values, price_points

def cev_option_pricing(strike, risk_free_rate, volatility, maturity, 
                        gamma, num_time_steps, num_price_steps, method):
    """
    Wrapper function for CEV option pricing
    
    Matches the original interface while using the advanced matrix computation method
    """
    # Set price range
    min_price = 1e-5
    max_price = 1.5 * strike

    # Determine method (implicit or explicit)
    is_implicit = method == 'implicit'

    # Compute option values
    option_values, price_points = compute_option_matrix(
        j=num_price_steps+1,
        k=num_time_steps+1,
        r=risk_free_rate,
        maturity=maturity,
        strike=strike,
        volatility=volatility,
        gamma_param=gamma,
        price_range=(min_price, max_price),
        is_implicit=is_implicit
    )

    # Reconstruct full solution matrix
    full_solution = np.zeros((num_price_steps+1, num_time_steps+1))
    full_solution[:, -1] = option_values

    return price_points, option_values, full_solution


# Parameters
K = 15       # Strike price
r = 0.1      # Risk-free rate
sigma = 0.25 # Volatility
T = 0.5      # Time to maturity
M = 100      # Time steps
N = 100      # Asset price steps

# w is dummy variable

fig1, axs1 = plt.subplots(1, 5, figsize=(10, 5))
S1, V1, V_over_time = cev_option_pricing(K, r, sigma, T, 0.8, 50, 50, 'explicit')
S2, V2, w = cev_option_pricing(K, r, sigma, T, 0.8, 100, 100, 'explicit')
S10, V10, w = cev_option_pricing(K, r, sigma, T, 0.8, 50, 100, 'explicit')
S9, V9, w  = cev_option_pricing(K, r, sigma, T, 0.8, 100, 50, 'explicit')
Stest , Vtest, wtest = cev_option_pricing(K, r, sigma, T, 1, 100, 50, 'explicit')


axs1[0].plot(S1, V1, label='M=50, N=50')
axs1[0].set_title('Accuracy (M=50, N=50)')
axs1[0].set_xlabel('Asset Price (S)')
axs1[0].set_ylabel('Option Value (V)')
axs1[0].grid(True)

axs1[1].plot(S2, V2, label='M=100, N=100')
axs1[1].set_title('Accuracy (M=100, N=100)')
axs1[1].set_xlabel('Asset Price (S)')
axs1[1].grid(True)

axs1[2].plot(S10, V10, label='M=50, N=100')
axs1[2].set_title('Accuracy (M=50, N=100)')
axs1[2].set_xlabel('Asset Price (S)')
axs1[2].grid(True)

axs1[3].plot(S9, V9, label='M=100, N=50')
axs1[3].set_title('Accuracy (M=100, N=50)')
axs1[3].set_xlabel('Asset Price (S)')
axs1[3].grid(True)

axs1[4].plot(Stest, Vtest, label='M=100, N=50')
axs1[4].set_title('Accuracy (M=100, N=50), gamma = 1')
axs1[4].set_xlabel('Asset Price (S)')
axs1[4].grid(True)
plt.tight_layout()
plt.show()


#Explicit
path_counts = [10, 25, 50, 75, 80]
errors = []
for n in path_counts:
    Stest, Vtest, wtest = cev_option_pricing(K, r, sigma, T, 1, 400, n, 'explicit')
    exact_values = np.array([bsexact(sigma, r, K, T, s_i) for s_i in Stest])
    error = np.abs(Vtest - exact_values)
    errors.append(np.mean(error))


plt.figure(figsize=(10, 6))
plt.plot(path_counts, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Asset price steps (M)')
plt.ylabel('Mean Error')
plt.title('Mean Error of CEV Option Pricing vs. Asset price steps : N = 400')
plt.yscale('log')  # Log scale for better visualization of errors
plt.grid(True)
plt.show()


time_steps = [30, 50, 100, 200, 500]
errors = []
for t in time_steps:
    Stest, Vtest, wtest = cev_option_pricing(K, r, sigma, T, 1, t, 50, 'explicit')
    exact_values = np.array([bsexact(sigma, r, K, T, s_i) for s_i in Stest])
    error = np.abs(Vtest - exact_values)
    errors.append(np.mean(error))
    
plt.figure(figsize=(10, 6))
plt.plot(time_steps, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Time steps (N)')
plt.ylabel('Mean Error')
plt.title('Mean Error of CEV Option Pricing vs. Time steps : M = 50')
plt.yscale('log')  # Log scale for better visualization of errors
plt.grid(True)
plt.show()



#Implicit
path_counts = [10, 25, 50, 75, 80]
errors = []
for n in path_counts:
    Stest, Vtest, wtest = cev_option_pricing(K, r, sigma, T, 1, 100, n, 'implicit')
    print(Vtest)
    exact_values = np.array([bsexact(sigma, r, K, T, s_i) for s_i in Stest])
    error = np.abs(Vtest - exact_values)
    errors.append(np.mean(error))


plt.figure(figsize=(10, 6))
plt.plot(path_counts, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Asset price steps (M)')
plt.ylabel('Mean Error')
plt.title('Mean Error of CEV Option Pricing vs. Asset price steps : N = 200')
plt.yscale('log')  # Log scale for better visualization of errors
plt.grid(True)
plt.show()


time_steps = [30, 50, 100, 200, 500]
errors = []
for t in time_steps:
    Stest, Vtest, wtest = cev_option_pricing(K, r, sigma, T, 0.99, t, 50, 'implicit')
    exact_values = np.array([bsexact(sigma, r, K, T, s_i) for s_i in Stest])
    error = np.abs(Vtest - exact_values)
    errors.append(np.mean(error))
    
plt.figure(figsize=(10, 6))
plt.plot(time_steps, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Time steps (N)')
plt.ylabel('Mean Error')
plt.title('Mean Error of CEV Option Pricing vs. Time steps : M = 50')
plt.yscale('log')  # Log scale for better visualization of errors
plt.grid(True)
plt.show()

''''
[cev_option_pricing(K, r, sigma, T, 1, 100, n, 'explicit') for n in path_counts])
errors = Vtest - bsexact(sigma, r, K, T, Stest)
slope = np.round(np.polyfit(np.log(np.absolute(path_counts)),np.log(np.absolute(errors)),1)[0],2)

# Plot sample error as function of number of paths
plt.figure()
plt.loglog(path_counts, np.absolute(errors), label=f"error, slope = {slope}")
#plt.axhline(y=bsexact(sigma, r, K, T, S0), color='black', label="Exact Price (gamma=1)")
plt.xlabel("Number of Paths")
plt.ylabel("Error")
plt.legend()
plt.title("Sample Error as Function of Number of Paths")
plt.show()
'''

# Experiment 2: Stability

M_values = [75, 90, 200, 800]      
N_values = [100, 400]
fig, ax = plt.subplots(len(N_values),figsize=(10, 5))  
for i, N in enumerate(N_values):
    for M in M_values:
        S, V , w = cev_option_pricing(K, r, sigma, T, 0.8, M, N, 'explicit')  
        ax[i].plot(S, V, label=f'M={M}, N={N} ')
    ax[i].set_title('Stability with Different M Values (Explicit Method)')
    ax[i].set_xlabel('Asset Price (S)')
    ax[i].set_ylabel('Option Value (V)')
    ax[i].grid(True)
    ax[i].legend()

# Customize the plot


plt.tight_layout()
plt.show()

# Experiment 3: Complexity (Time comparison)
# Combinations of M and N
M_values = [50, 100]
N_values = [50, 100]


fig, axs = plt.subplots(len(M_values), len(N_values), figsize=(12, 10))


for i, M in enumerate(M_values):
    for j, N in enumerate(N_values):
        # Compute explicit and implicit solutions
        S_explicit, V_explicit, w = cev_option_pricing(K, r, sigma, T, 0.8, M, N, 'explicit')
        S_implicit, V_implicit, w = cev_option_pricing(K, r, sigma, T, 0.8, M, N, 'implicit')

        axs[i, j].plot(S_explicit, V_explicit, label=f'Explicit, M={M}, N={N}', linestyle='--')
        axs[i, j].plot(S_implicit, V_implicit, label=f'Implicit, M={M}, N={N}', linestyle='-')
        
        axs[i, j].set_title(f'Comparison for M={M}, N={N}')
        axs[i, j].set_xlabel('Asset Price (S)')
        axs[i, j].set_ylabel('Option Value (V)')
        axs[i, j].grid(True)
        axs[i, j].legend()

plt.tight_layout()
plt.show()

# Experiment 4: Varying gamma
fig4, axs4 = plt.subplots(1, 3, figsize=(15, 5))
S6, V6, w = cev_option_pricing(K, r, sigma, T, 0.2, 200, 50, 'explicit')  # gamma = 0.6
S7, V7, w = cev_option_pricing(K, r, sigma, T, 0.5, 200, 50, 'explicit')  # gamma = 0.8
S8, V8, w = cev_option_pricing(K, r, sigma, T, 1.0, 200, 50, 'explicit')  # gamma = 1.0

axs4[0].plot(S6, V6, label='gamma = 0.2')
axs4[0].set_title('Varying gamma (0.2), 400 time steps, 50 asset price steps')
axs4[0].set_xlabel('Asset Price (S)')
axs4[0].set_ylabel('Option Value (V)')
axs4[0].grid(True)

axs4[1].plot(S7, V7, label='gamma = 0.5')
axs4[1].set_title('Varying gamma (0.5), 400 time steps, 50 asset price steps')
axs4[1].set_xlabel('Asset Price (S)')
axs4[1].grid(True)

axs4[2].plot(S8, V8, label='gamma = 1.0')
axs4[2].set_title('Varying gamma (1.0), 400 time steps, 50 asset price steps')
axs4[2].set_xlabel('Asset Price (S)')
axs4[2].grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

# Use imshow to display the heatmap of option values
plt.imshow(V_over_time, extent=[0, T, S.min(), S.max()], aspect='auto', origin='lower', cmap='jet')

# Add color bar to represent the scale of option values
plt.colorbar(label="Option Value (V)")

# Add labels and title
plt.xlabel("Time")
plt.ylabel("Asset Price (S)")
plt.title("Option Value Heatmap")

plt.show()







#Convergence

# Parameters
K = 15       # Strike price
r = 0.1      # Risk-free rate
sigma = 0.25 # Volatility
T = 0.5      # Time to maturity
M = 100      # Time steps
N = 50      # Asset price steps

def convergence_study():
    K = 15       # Strike price
    r = 0.1      # Risk-free rate
    sigma = 0.25 # Volatility
    T = 0.5      # Time to maturity
    M = 400      # Time steps
    N = 50      # Asset price steps
    gamma = 1
    exact_values = []
    max_errors_explicit = []
    max_errors_implicit = []
    num_steps = np.arange(10, 61, 5)  # Asset price steps (N)

    for N in num_steps:
        S, V_explicit, _ = cev_option_pricing(K, r, sigma, T, gamma, M, N, "explicit")
        S, V_implicit, _ = cev_option_pricing(K, r, sigma, T, gamma, 50, N, "implicit")

        exact = np.array([bsexact(sigma, r, K, T, s) for s in S])
        exact_values.append(exact)

        max_error_explicit = np.mean(np.abs(V_explicit - exact))
        max_error_implicit = np.mean(np.abs(V_implicit - exact))

        max_errors_explicit.append(max_error_explicit)
        max_errors_implicit.append(max_error_implicit)

    # Plotting convergence
    plt.figure(figsize=(8, 6))
    plt.plot(num_steps, max_errors_explicit, '-o', label='Explicit Method')
    plt.plot(num_steps, max_errors_implicit, '-o', label='Implicit Method')
    plt.yscale('log')
    plt.xlabel('Number of Asset Price Steps (N)')
    plt.ylabel('Maximum Absolute Error')
    plt.title('Convergence of Explicit and Implicit Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the convergence study
convergence_study()

