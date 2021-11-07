import numpy as np
import time
from subprocess import Popen, PIPE
import networkx as nx
import random


def f1(points, i=None):
    if len(points.shape) == 1:
        points = np.array([points])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (x - z) ** 2 + (2*y + z) ** 2 + (4*x - 2*y + z) ** 2 + x + y


def f2(points, i=None):
    if len(points.shape) == 1:
        points = np.array([points])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return (x - 1) ** 2 + (y - 1) ** 2 + 100 * (y - x**2) ** 2 + 100 * (z - y**2) ** 2


def shrink(points):
    points[1:, :] = (points[1:, :] + points[0, :]) / 2
    return points


def nelder_mead(f, points, tol_f=1e-5, max_iter=100, i=None):
    stop = False
    steps = 0
    t = time.time()
    while not stop:
        steps += 1
        fun_val = f(points, i)
        order = np.argsort(fun_val)

        points = points[order, :]
        fun_ordered = fun_val[order]

        xm = np.mean(points[:-1, :], axis=0)    # mean point
        xn = points[-1, :]      # worst point
        xr = xn + 2 * (xm - xn)         # reflection point
        yr = f(xr, i)

        if yr < fun_ordered[0]:         # better than previous best
            xe = xn + 3 * (xm - xn)     # expansion point
            ye = f(xe, i)
            if ye < yr:
                points[-1, :] = xe
            else:
                points[-1, :] = xr

        elif yr < fun_ordered[-2]:      # better than lousy point
            points[-1, :] = xr

        elif yr < fun_ordered[-1]:      # better than worst point
            xc = xn + 3/2 * (xm - xn)   # outside contracted point
            yc = f(xc, i)
            if yc < yr:
                points[-1, :] = xc
            else:
                points = shrink(points)

        else:                           # worse than worst point
            xc = xn + 1/2 * (xm - xn)   # inside contracted point
            yc = f(xc, i)
            if yc < yr:
                points[-1, :] = xc
            else:
                points = shrink(points)

        # stopping criteria
        f_diff = fun_ordered[-1] - fun_ordered[0]

        if i is not None:
            if steps % 50 == 0:
                print(f'Minimum function value: {fun_ordered[0]}, symplex: {points}.')

        if f_diff < tol_f:
            stop = True
            print(f'Reached tolerance in {steps} steps. Time needed: {time.time() - t}.')

        elif steps == max_iter:
            stop = True
            print(f'Reached maximum number of steps: {max_iter}, difference: {f_diff}. Time needed: {time.time() - t}.')

    fun_val = f(points, i)
    order = np.argsort(fun_val)

    points = points[order, :]

    return points[0, :]


def get_points(x0, diameter):
    n = len(x0)
    tetraeder = np.ones([n, n]) - np.diag(np.ones(n))
    tetraeder = tetraeder / np.sqrt(n - 1) * diameter
    points = np.vstack([x0, x0 + tetraeder])
    return points


def black_box(x, i):
    path_to_cygwin = "C:/cygwin64/bin"
    path_to_exe = "C:/Users/ursau/OneDrive/Documents/DataScience/MAT2/part2/hw4_win.exe"
    student_id = "63200441"

    sh = x.shape

    if len(sh) == 1:
        cmd = f"{path_to_exe} {student_id} {i} {x[0]} {x[1]} {x[2]}"
        cygwin = Popen(cmd, cwd=path_to_cygwin, stdin=PIPE, stdout=PIPE)
        return float(cygwin.communicate()[0])

    else:
        f = []
        for j in range(sh[0]):
            cmd = f"{path_to_exe} {student_id} {i} {x[j, 0]} {x[j, 1]} {x[j, 2]}"
            cygwin = Popen(cmd, cwd=path_to_cygwin, stdin=PIPE, stdout=PIPE)
            f.append(float(cygwin.communicate()[0]))
        return np.array(f)


def min_span_tree(G, T, rand_in=False, rand_out=False, max_swaps=500):
    no_change = 0
    nr_swaps = 0
    for k in range(max_swaps):
        # removing edge
        if rand_out:
            e_out = list(T.edges)[np.random.randint(0, len(T.edges))]
        else:
            sorted_edges = sorted(T.edges(data=True), key=lambda t: t[2].get('weight', 1), reverse=True)
            e_out = sorted_edges[0][:-1]
        T.remove_edge(*e_out)

        # adding edge
        conn1 = nx.node_connected_component(T, e_out[0])
        conn2 = nx.node_connected_component(T, e_out[1])

        possible_edges = [(u, v) for u in conn1 for v in conn2 if sum(abs(np.array(u) - np.array(v))) == 1]

        if rand_in:
            e_in = random.sample(possible_edges, 1)[0]
        else:
            sorted_edges = sorted(possible_edges, key=lambda t: G.get_edge_data(*t)['weight'])
            e_in = sorted_edges[0]

        if G.get_edge_data(*e_in)['weight'] < G.get_edge_data(*e_out)['weight']:
            T.add_edge(*e_in, weight=G.get_edge_data(*e_in)['weight'])
            no_change = 0
            nr_swaps += 1
        else:
            T.add_edge(*e_out, weight=G.get_edge_data(*e_out)['weight'])
            no_change += 1
            if not (rand_in or rand_out):
                break
            if no_change > 100:
                break

    return T, nr_swaps, k+1


if __name__ == '__main__':

    # NELDER-MEAD
    fn = [f1, f2]
    start_points = [np.array([0, 0, 0]), np.array([1.2, 1.2, 1.2])]
    true_min = [np.array([-1/6, -11/48, 1/6]), np.array([1, 1, 1])]

    for i in range(2):
        print(f'FUNCTION {i}')
        for diameter in [1, 2, 5, 10]:
            print(f'diameter {diameter}:')
            for max_iter in [2, 5, 10, 100, 10000]:
                points = get_points(start_points[i], diameter)

                x = nelder_mead(fn[i], points, 1e-15, max_iter)
                print(f'Distance from true minimum: {np.linalg.norm(x - true_min[i])}, '
                      f'function difference: {fn[i](x)[0] - fn[i](true_min[i])[0]}.')
                print('...............................................................................................')
            print('_______________________________________________________________________________________________')

    # # BLACK BOX OPTIMIZATION
    # for i in range(1, 4):
    #     t = time.time()
    #     points = get_points(np.array([-1, -1, -1]), 2)
    #     x = nelder_mead(black_box, points, 1e-10, 400, i)
    #
    #     print(f'Minimum of function {i}: {black_box(x, i)}, obtained at {x} . Time needed: {time.time() - t}')
    #     print('_______________________________________________________________________________________________')
    #
    # # LOCAL SEARCH STUDY
    # for p in range(3):
    #     np.random.seed(p**2)
    #     G = nx.grid_2d_graph(20, 20)
    #     weights = np.random.permutation(range(1, 761))
    #     for i, edge in enumerate(G.edges):
    #         G[edge[0]][edge[1]]['weight'] = weights[i]
    #
    #     mst_edges = list(nx.algorithms.tree.mst.minimum_spanning_edges(G))
    #     true_min_weight = sum([d['weight'] for u, v, d in mst_edges])
    #     print(f'Weight of MST calculated with Kruskal algorithm: {true_min_weight}')
    #
    #     T0 = G.copy()
    #     T0.remove_edges_from([((i, j), (i+1, j)) for i in range(20) for j in range(1, 20)])
    #     T0_weight = sum([T0.get_edge_data(u, v)['weight'] for u, v in T0.edges])
    #     print(f'Weight of starting tree: {T0_weight}')
    #
    #     for rand_in, rand_out in [(True, True), (True, False), (False, True), (False, False)]:
    #         T, nr_swaps, nr_iter = min_span_tree(G.copy(), T0.copy(), rand_in, rand_out, 5000)
    #         T_weight = sum([T.get_edge_data(u, v)['weight'] for u, v in T.edges])
    #         print(f'Approach rand_in, rand_out = {rand_in, rand_out}: weight = {T_weight}, nr iterations: {nr_iter}, '
    #               f'swaps: {nr_swaps}')
    #     print('_______________________________________________________________________________________________________')

