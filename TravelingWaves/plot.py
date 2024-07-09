import numpy as np
import matplotlib.pyplot as plt
import os
import sys

plot_params = {
  #'backend': 'pdf',
  # 'lines.marker' : 'x',
  'scatter.marker' : 'x',
  'lines.markersize' : 4,
  'lines.linewidth' : 1,
  'axes.labelsize': 16,
  # 'textfontsize': 12,
  'font.size' : 16,
  'legend.fontsize': 16,
  'xtick.labelsize': 14,
  'ytick.labelsize': 14,
  'text.usetex': True,
  'figure.figsize': [9,6],
  'axes.grid': True
}

plt.rcParams.update(plot_params)


if len(sys.argv) > 1:

  filename = sys.argv[1]

  if os.path.exists(filename):
    data = np.loadtxt(filename, np.float64)
    data_unique = np.unique(data, axis=0)
    data_unique = np.array(sorted(data_unique, key=lambda x : x[0]))
    x = data_unique[:, 0]
    u_sol = data_unique[:, 1]
    T_sol = data_unique[:, 2]
    lambda_sol = data_unique[:, 3]

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.scatter(x, u_sol, label=r"$u$", color='blue')
    ax.scatter(x, T_sol, label=r"$T$", color='red')
    ax.scatter(x, lambda_sol, label=r"$\lambda$", color='green')


    # Plot of limit solutions for the detonation case. Uncomment, if needed.
    #===============================================================#
    '''

    path_to_solution_files = os.path.split(filename)[0]
    u_limit_path = os.path.join(path_to_solution_files, 'solution_u_limit.txt')
    T_limit_path = os.path.join(path_to_solution_files, 'solution_T_limit.txt')
    lambda_limit_path = os.path.join(path_to_solution_files, 'solution_lambda_limit.txt')

    if os.path.exists(u_limit_path):
      u_limit = np.loadtxt(u_limit_path, np.float64)
      ax.plot(u_limit[:, 0], u_limit[:, 1], label=r"$u_{\mathrm{lim}}$", color='blue')
      ax.plot([0, x[-1]], [u_sol[-1], u_sol[-1]], color='blue')
    else:
      print("No such file:", u_limit_path)

    if os.path.exists(T_limit_path):
      T_limit = np.loadtxt(T_limit_path, np.float64)
      ax.plot(T_limit[:, 0], T_limit[:, 1], label=r"$T_{\mathrm{lim}}$", color='red')
      ax.plot([0, x[-1]], [T_sol[-1], T_sol[-1]], color='red')
    else:
      print("No such file:", T_limit_path)

    if os.path.exists(lambda_limit_path):
      lambda_limit = np.loadtxt(lambda_limit_path, np.float64)
      ax.plot(lambda_limit[:, 0], lambda_limit[:, 1], label=r"$\lambda_{\mathrm{lim}}$", color='green')
      ax.plot([0, x[-1]], [lambda_sol[-1], lambda_sol[-1]], color='green')
    else:
      print("No such file:", lambda_limit_path)


    '''
    #===============================================================#
    

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$u, T, \lambda$")
    ax.legend()

    # plt.savefig("fast_deflagration_delta_0.01.png", bbox_inches='tight', dpi=500)
    # plt.savefig('slow_deflagration_delta_0.01.png', bbox_inches='tight', dpi=500)
    # plt.savefig('detonation_delta_0.01.png', bbox_inches='tight', dpi=500)

    plt.show()
  else:
    print("No such file:", filename)
