import sys
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from variable_conversions import HBARC  # noqa
import my_plotting as myplt  # noqa

analytic_style = {'ls': '-', 'lw': 2}
# sim_style = {'ls':'-'}
sim_style = {'marker': 'o', 's': 60.0, 'lw': .25, 'edgecolors': 'k'}
sim_list = [10, 12, 14, 16, 18, 20]
analytic_list = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
filter_criteria = 'abs(phi - 3.141592653589793/4.) < 1.e-2'

cmap = myplt.get_cmap(len(analytic_list), 'copper')
# cmap = myplt.get_cmap(len(sim_list), 'cividis')
# cmap = myplt.get_cmap(len(regulator_list), 'nipy_spectral')

mpl.rcParams['text.usetex'] = True

dpi = 150
fig, ax = plt.subplot_mosaic([['e', 'rhoB', 'ux', 'cbar'],
                              ['pixx', 'Rey', 'pietaeta', 'cbar']],
                             width_ratios=[1, 1, 1, 0.1],
                             figsize=2.0 * np.array([7 * 3 / 2, 7]),
                             constrained_layout=True)


def get_reynolds_number(df, t_squared):
    df['u0'] = np.sqrt(1 + df['ux']**2 + df['uy']**2)
    df['pitx'] = (df['pixx'] * df['ux'] + df['pixy'] * df['uy']) / df['u0']
    df['pity'] = (df['pixy'] * df['ux'] + df['piyy'] * df['uy']) / df['u0']
    df['pitt'] = (df['pitx'] * df['ux'] + df['pity'] * df['uy']) / df['u0']
    df['pizz'] = -(df['pixx'] + df['piyy'] - df['pitt']) / t_squared
    df['p'] = df['e'] / 3  # Pressure

    df['pi_norm'] = np.sqrt(df['pitt']**2 + df['pixx']**2 + df['piyy']**2
                            + (t_squared * df['pizz'])**2
                            + 2 * (df['pixy']**2 - df['pitx']**2
                                   - df['pity']**2))
    df['reynolds'] = df['pi_norm'] / df['p']
    return df


def read_sol(analytic_sol_folder):
    for ii, t in enumerate(analytic_list):
        inp_path = os.path.join(analytic_sol_folder,
                                'tau=' + f'{t:.2f}' + '.txt')
        df = pd.read_table(
            inp_path,
            names=[
                'x',
                'y',
                'eta',
                'e',
                'rhoB',
                'rhoS',
                'rhoQ',
                'ux',
                'uy',
                'ueta',
                'Bulk',
                'pixx',
                'pixy',
                'pixeta',
                'piyy',
                'pyeta',
                'pietaeta'],
            sep=" ",
            engine='python',
            header=1)

        df['r'] = np.sqrt(df['x']**2 + df['y']**2)
        df['phi'] = np.arctan2(df['y'], df['x'])
        df = get_reynolds_number(df, float(t)**2)

        df = df.query(filter_criteria)

        ana_color = cmap(ii)
        ax['e'].plot(df['r'], df['e'], color=ana_color, **analytic_style)
        ax['rhoB'].plot(df['r'], df['rhoB'], color=ana_color, **analytic_style)
        ax['ux'].plot(df['r'], df['ux'], color=ana_color, **analytic_style)
        ax['pixx'].plot(df['r'], -df['pixx'],
                        color=ana_color, **analytic_style)
        ax['Rey'].plot(
            df['r'],
            df['reynolds'],
            color=ana_color,
            **analytic_style)
        ax['pietaeta'].plot(
            df['r'],
            df['pietaeta'],
            color=ana_color,
            **analytic_style)


def read_sim(sim_result_folder):
    dt = .001
    for ii, t in enumerate(sim_list):
        col_names = [
            'id',
            't',
            'x',
            'y',
            'p',
            'T',
            'muB',
            'muS',
            'muQ',
            'e',
            'rhoB',
            'rhoS',
            'rhoQ',
            's',
            's_smoothed',
            's_specific',
            'sigma',
            'spec_s',
            'stauRelax',
            'bigTheta',
            '??',
            '??2',
            'pi00',
            'pixx',
            'piyy',
            'pixy',
            't2pi33',
            'v1',
            'v2',
            'gamma',
            'frz',
            'eos']
        inp_path = os.path.join(
            sim_result_folder,
            f'system_state_{t}.dat')
        print(inp_path)
        df = pd.read_table(inp_path,
                           names=col_names, sep=' ', header=0)
        df['ux'] = df.loc[:, 'v1'] * df.loc[:, 'gamma']
        df['uy'] = df.loc[:, 'v2'] * df.loc[:, 'gamma']
        df['r'] = np.sqrt(df['x']**2 + df['y']**2)
        df['phi'] = np.arctan2(df['y'], df['x'])
        df['e'] = df['e'] / 1000  # convert to GeV/fm^3

        df['pixx'] = df['pixx'] * HBARC  # convert to GeV/fm^3
        df['pixy'] = df['pixy'] * HBARC  # convert to GeV/fm^3
        df['piyy'] = df['piyy'] * HBARC  # convert to GeV/fm^3

        df['t2pi33'] = df['t2pi33'] * HBARC  # convert to GeV/fm^3

        df = get_reynolds_number(df, float(t)**2)
        df_query = df.query(filter_criteria)

        stride = 1
        offset = 0.0 * ii
        width = 50 * (len(sim_list) - ii) / \
            len(sim_list) + 20 - 17.5
        localstyle = {'facecolors': cmap(ii)}
        print(width)
        ax['e'].scatter(
            df_query['r'].to_numpy()[::stride] + offset,
            df_query['e'].to_numpy()[::stride],
            **localstyle,
            **sim_style)
        ax['rhoB'].scatter(
            df_query['r'].to_numpy()[::stride] + offset,
            df_query['rhoB'].to_numpy()[::stride],
            **localstyle,
            **sim_style)
        ax['ux'].scatter(
            df_query['r'].to_numpy()[::stride] + offset,
            df_query['ux'].to_numpy()[::stride],
            **localstyle,
            **sim_style)
        ax['pixx'].scatter(
            df_query['r'].to_numpy()[::stride] + offset,
            -df_query['pixx'].to_numpy()[::stride],
            **localstyle,
            **sim_style)
        ax['Rey'].scatter(
            df_query['r'].to_numpy()[::stride] + offset,
            df_query['reynolds'].to_numpy()[::stride],
            **localstyle,
            **sim_style)
        ax['pietaeta'].scatter(
            df_query['r'].to_numpy()[::stride] + offset,
            df_query['t2pi33'].to_numpy()[::stride] / float(t)**2,
            **localstyle,
            **sim_style)


def beautify():
    # fig.set_tight_layout(True)
    ylabels = {'e': r'$\varepsilon$ (GeV/fm$^3$)',
               'rhoB': r'$n_B$ (fm$^{-3}$)',
               'ux': r'$u^x$',
               'pixx': r'$-\pi^{xx}$ (GeV/fm$^3$)',
               'Rey': r'$\mathcal{R}^{-1}$',
               # 'pixy':r'$\pi^{xy}$ (GeV/fm$^3$)',
               'pietaeta': r'$\pi^{\eta \eta}$ (GeV/fm$^3$)'}
    for key in ax.keys():
        if key == 'cbar':
            continue
        myplt.costumize_axis(ax=ax[key],
                             x_title=r'$r$ (fm)',
                             y_title=ylabels[key])
        ax[key].set_xlim(0, 4.5)

    myplt.costumize_axis(ax=ax['cbar'], x_title='', y_title='')

    phi_list = np.arange(0, len(sim_list)) + .5

    # deal with the colorbar
    boundaries = np.arange(0, len(sim_list) + 1)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(sim_list))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=phi_list, label=r'$\tau$ (fm/c)',
                        cax=ax['cbar'],
                        boundaries=boundaries)

    cbar.set_ticklabels(["{0:4.2f}".format(float(reg))
                        for reg in analytic_list])
    ax['e'].plot([], [], **analytic_style, label='Analytic', color=cmap(0))
    ax['e'].scatter([], [], **sim_style, label='Simulation', color=cmap(0))
    ax['e'].legend(loc='upper right', frameon=False, fontsize=18)
    ax['e'].text(3.5, 0.205, 'EoS 2', fontsize=18, bbox={'boxstyle': 'round',
                                                         'facecolor': 'white'})

    # ax['ux'].set_ylim(1E-7, 2.)
    # ax['ux'].set_ylim(0, 1.3)
    # ax['e'].set_ylim(1E-7, 12)
    # ax['e'].set_ylim(1E-7, .25)
    # ax['Rey'].set_ylim(0, 3.1)
    # ax['pixx'].set_ylim(1E-7, .0275)
    # ax['pietaeta'].set_ylim(1E-7, .01)
    # ax['rhoB'].set_ylim(1E-7, .6)

    ax['e'].set_ylim(bottom=1e-2)
    ax['rhoB'].set_ylim(bottom=1e-2)
    ax['pixx'].set_ylim(bottom=1e-7)
    ax['pietaeta'].set_ylim(bottom=1e-7)

    ax['e'].set_yscale('log')
    ax['rhoB'].set_yscale('log')
    ax['pietaeta'].set_yscale('log')
    ax['pixx'].set_yscale('log')


if __name__ == '__main__':
    analytical_folder = sys.argv[1]
    simulation_folder = sys.argv[2]
    read_sol(analytical_folder)
    read_sim(simulation_folder)
    beautify()

    if (len(sys.argv) < 4):
        fig.savefig('gubser_with_regulator.pdf')
    else:
        fig.savefig(sys.argv[3])
