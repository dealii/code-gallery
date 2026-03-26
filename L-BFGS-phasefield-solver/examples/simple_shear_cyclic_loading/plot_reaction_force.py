import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick


time1, forceX1, forceY1 = np.loadtxt('Reaction_force.hist',
                                              delimiter='\t', unpack=True)                                                    

#time2, forceX2, forceY2 = np.loadtxt('Reaction_force_timestep_10-5.hist',
#                                              delimiter='\t', unpack=True)      

font = {'size': 18}
matplotlib.rc('font', **font)

labels = []
labels.append("Loading")
labels.append("Unloading")
labels.append("Rev. loading")
labels.append("Rev. unloading")

symbols = ['-k', '--*m', '-.b^', '--k', '-rD', 'bx']

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(time1[:16], forceX1[:16], symbols[0],
         linewidth=2.0, label=labels[0], fillstyle='none', markersize=8)
plt.plot(time1[15:], forceX1[15:], symbols[1],
         linewidth=2.0, label=labels[1], fillstyle='none', markersize=8)                 
#plt.plot(time2, forceX2, symbols[1],
#         linewidth=2.0, label=labels[1], fillstyle='none', markersize=8)   
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

plt.xlabel('Pseudo time')
plt.ylabel('Reaction force (kN)', multialignment='center')
plt.xlim(0,0.03)
# plt.xticks(np.arange(0, 6, step=1))
#plt.ylim(0.0, 1.0)
# plt.legend( bbox_to_anchor=(2.4, 1.1),prop={'size':10})
plt.grid()
plt.legend(fontsize=16)
# ax.set_aspect('equal')
# plt.subplots_adjust(top = 0.95, bottom = 0.125,
#                     right = 0.95, left = 0.125, hspace = 0, wspace = 0)
plt.savefig("Reaction_force_history_time.eps", bbox_inches='tight', pad_inches=0.1)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(time1[:16], forceX1[:16], symbols[0],
         linewidth=2.0, label=labels[0], fillstyle='none', markersize=8)
plt.plot(30.0e-3 - time1[15:], forceX1[15:], symbols[1],
         linewidth=2.0, label=labels[1], fillstyle='none', markersize=8)         
#plt.plot(time2, forceX2, symbols[1],
#         linewidth=2.0, label=labels[1], fillstyle='none', markersize=8)   
# plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 3))
# ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

plt.xlabel('Displacement (mm)')
plt.ylabel('Reaction force (kN)', multialignment='center')
plt.xlim(0.0,0.02)
#plt.xticks(np.arange(-0.02, 0.021, step=0.01))
#plt.ylim(0.0, 1.0)
# plt.legend( bbox_to_anchor=(2.4, 1.1),prop={'size':10})
plt.grid()
plt.legend(fontsize=16)
# ax.set_aspect('equal')
# plt.subplots_adjust(top = 0.95, bottom = 0.125,
#                     right = 0.95, left = 0.125, hspace = 0, wspace = 0)
plt.savefig("Reaction_force_history_disp.eps", bbox_inches='tight', pad_inches=0.1)
