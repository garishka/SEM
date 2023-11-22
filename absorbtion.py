import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl

cm = 1 / 2.54
width = 14 * cm
height = 10 * cm

# LaTeX глупости, които даже нз дали така работят
preamble = [r'\usepackage[utf8]{inputenc}',
    r'\usepackage[bulgarian]{babel}',
    r"\usepackage{amsmath}",
    r'\usepackage{siunitx}',
    r'\usepackage{emoji}']

mpl.use("pgf")
LaTeX = {
"text.usetex": True,
"font.family": "CMU Serif",
    "pgf.preamble": "\n".join(line for line in preamble),
    "pgf.rcfonts": True,
    "pgf.texsystem": "lualatex"}
plt.rcParams.update(LaTeX)

# взети от графиката
wave_numbers = np.asarray([15141.51038, 15141.4693, 15141.22241, 15141.06219, 15140.94197, 15140.71562, 15140.51406, 15140.41803])
peak_t_val = np.asarray([-0.01125, -0.01038, -0.00548, -0.00239, -0.00030, 0.00366, 0.00699, 0.00856])

# channel 1 data (t, X)
x_data1, y_data1 = np.genfromtxt('ch1.txt', unpack=True)
# channel 2 data (t, (X-Y)/Y)
x_data2, y_data2 = np.genfromtxt('ch2.txt', unpack=True)

# времена на min, =0, max на тока на channel 1
t = [-0.01375, -0.01147, 0.01043]
# съответстващ ток
current = 21.5
i = [current - 2, current, current + 2]


# Akaike Information Criterion за неизвестно σ
def aic(data, model_data, num_params):
    n = len(data)
    data = np.asarray(data)
    model_data = np.asarray(model_data)
    log_l = np.log(np.sum((data - model_data) ** 2) / n)
    return -log_l + 2 * num_params / n


def results(x, y, deg):
    model_params, cov_matrix = np.polyfit(x, y, deg, cov=True)
    model_data = np.polyval(model_params, x)
    model_aic = aic(y, model_data,(deg+1))
    results_log = f"Results for the {deg}-degree polynomial fit of the data\n" \
                  f"=======================================================\n" \
                  f"Model parameter, highest power first: {model_params}\n" \
                  f"Covariant matrix of the parameters: {cov_matrix}\n" \
                  f"Model AIC: {model_aic}\n" \
                  f"======================================================\n" \
                  f"Note: the coefficients are calculated with numpy's polyfit.\n"
    return results_log, model_params


# extracting the results for the different fits
res_linear, params_linear = results(peak_t_val, wave_numbers, 1)
res_quad, params_quad = results(peak_t_val, wave_numbers, 2)
res_cubic, params_cubic = results(peak_t_val, wave_numbers, 3)

with open("results.txt","w") as file:
    file.write(res_linear)
    file.write(" \n")
    file.write(res_quad)
    file.write(" \n")
    file.write(res_cubic)

# creating more points at which the fit will be calculated, needed for the plots
t_data = np.linspace(peak_t_val[0], peak_t_val[-1], 100)
k_linear = np.polyval(params_linear, t_data)
k_quad = np.polyval(params_quad, t_data)
k_cubic = np.polyval(params_cubic, t_data)

# creating plot, the size is defined by the properties of my LaTeX document setup
fig, ax = plt.subplots(figsize=(218.62206/72, 3*218.62206/(4*72)))

# за 2 задача
k = np.polyval(params_quad, x_data2)
#ax.set_xlabel("k, cm$^{-1}$")
#ax.set_ylabel("Интензитет, отн.ед.")
#ax.plot(k, y_data2)

# за 3 задача
params_i, cov_i = curve_fit(lambda x, a, b: a * x + b, t, i)
i_data = np.polyval(params_i, peak_t_val)
delta_k = [wave_numbers[i+1] - wave_numbers[i] for i in range(len(wave_numbers)-1)]
delta_i = [i_data[i+1]-i_data[i] for i in range(len(i_data)-1)]
I_avg = [(i_data[i+1] + i_data[i])/2 for i in range(len(i_data)-1)]
dkdi = [delta_k[i]/delta_i[i] for i in range(len(delta_k))]
sigma_k = [-0.02 * delta_k[i] / delta_i[i] for i in range(len(dkdi))]

#ax.set_xlabel("$I_{avg}$, mA")
#ax.set_ylabel("$\Delta k/ \Delta I$, cm$^{-1}/mA$")
#plt.errorbar(I_avg, dkdi, yerr=sigma_k, fmt='.', capsize=3, barsabove=True)

# 4 задача
#ax.set_xlabel("$t$, s")
#ax.set_ylabel("$k$, cm$^{-1}$")
#ax.scatter(peak_t_val, wave_numbers, marker=".", color="black")
#ax.plot(t_data, k_linear, label="linear", color="red", linewidth=1)
#ax.plot(t_data, k_quad, label="quadratic", color="green", linewidth=1.5, linestyle="-.")
#ax.plot(t_data, k_cubic, label="cubic", color="gold", linewidth=1, linestyle=":")

plt.tight_layout(pad=0.1)
#ax.legend(loc='best', facecolor="white", edgecolor="white")
#plt.savefig("zad4.pdf")