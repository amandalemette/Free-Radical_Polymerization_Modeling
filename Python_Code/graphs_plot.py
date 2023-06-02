import matplotlib.pyplot as plt
import numpy as np

def plotgraphs(t,X,PDI,Mn,Mw):
    
    plt.figure(0,figsize=(8,6))
    
    exp_data = np.loadtxt('X_artigo_exp.txt',delimiter = ',')
    mod_data = np.loadtxt('X_artigo_mod.txt',delimiter = ',')    
    t_exp = exp_data[:,0]
    X_exp = exp_data[:,1]
    t_mod = mod_data[:,0]
    X_mod = mod_data[:,1]

    tam_font = 20

    plt.plot(t, X, color = 'darkblue', linewidth=3)
    plt.plot(t_exp,X_exp,'+' ,color = 'darkred', markersize=10,markeredgewidth=3)
    plt.plot(t_mod,X_mod, '--' , color = 'darkgreen', linewidth = 3.0)
    axes = plt.gca()
    #axes.set_xlim([0.0,1])
    axes.set_ylim([0.0,1.0])
    plt.ylabel('X [adm]', fontsize=tam_font)
    plt.xlabel('t [min]', fontsize=tam_font)
    axes.tick_params(axis="x", labelsize=tam_font)
    axes.tick_params(axis="y", labelsize=tam_font)
    plt.legend(('Proposed Model','Experimental data$^{[10]}$', 'Literature Model$^{[10]}$'), fontsize=tam_font-2)
    plt.savefig('Conversion.png',bbox_inches = 'tight',dpi = 600)

    plt.figure(1,figsize=(8,6))
    PDI[0] = PDI[1]
    plt.plot(X, PDI, color = 'darkblue', linewidth = 3.0)
    axes = plt.gca()
    axes.set_xlim([0.0,1.0])
    axes.set_ylim([0.0,15])
    plt.ylabel('Dispersity [adm]', fontsize=tam_font)
    plt.xlabel('X [adm]', fontsize=tam_font)
    plt.annotate('(b)',xy=(0.05,13.5), fontsize=tam_font)
    axes.tick_params(axis="x", labelsize=tam_font)
    axes.tick_params(axis="y", labelsize=tam_font)
    plt.savefig('PDI.png',bbox_inches = 'tight',dpi = 600)

    plt.figure(2,figsize=(8,6))
    plt.semilogy(X, Mn, color = 'green', linewidth = 3.0)
    plt.semilogy(X, Mw, color = 'lightblue', linewidth = 3.0)
    axes = plt.gca()
    axes.set_ylim([1.e5,1.e7])
    axes.set_xlim([0,1])
    plt.ylabel('Molecular Weight [g/mol]', fontsize=tam_font)
    plt.xlabel('X [adm]', fontsize=tam_font)
    plt.legend(('$M_n$','$M_w$'), fontsize=tam_font-2)
    plt.annotate('(a)',xy=(0.05,6.3e6), fontsize=tam_font)
    axes.tick_params(axis="x", labelsize=tam_font)
    axes.tick_params(axis="y", labelsize=tam_font)
    plt.savefig('MolecularWeight.png',bbox_inches = 'tight',dpi = 600)

    plt.show()