# Group 9 - Deepti Lakshmi Ravi, Sristi Chakraborty, Sinchana Manjula Prabhakara, Harishitha Chowdary Alapati

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
import gc
import unittest

sys.setrecursionlimit(50000)  # Need this for large graphs to avoid stack overflow

class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = defaultdict(list)  # Using defaultdict so we don't get KeyError

    def add_edge(self, u, v):
        self.adj[u].append(v)  # Undirected, so add both ways
        self.adj[v].append(u)

    def get_adjacency_list(self):
        return [self.adj[i] for i in range(self.n)]


def find_articulation_points(graph, n):
    discovery = [-1] * n  # When we first visit each vertex
    low = [-1] * n        # Earliest vertex reachable from subtree
    parent = [-1] * n     # To track DFS tree structure
    articulation_points = set()
    time_counter = [0]    # Using list so it's mutable in nested function

    def dfs(u):
        children = 0
        discovery[u] = low[u] = time_counter[0]
        time_counter[0] += 1

        for v in graph[u]:
            if discovery[v] == -1:
                children += 1
                parent[v] = u
                dfs(v)

                low[u] = min(low[u], low[v])  # Update with child's low value

                if parent[u] == -1 and children > 1:  # Root with 2+ children is AP
                    articulation_points.add(u)

                if parent[u] != -1 and low[v] >= discovery[u]:  # Non-root AP condition
                    articulation_points.add(u)

            elif v != parent[u]:  # Back edge (not going back to parent)
                low[u] = min(low[u], discovery[v])

    for i in range(n):  # Start DFS from each unvisited vertex
        if discovery[i] == -1:
            dfs(i)

    return articulation_points


def is_biconnected(graph, n):
    if n < 2:
        return True
    articulation_points = find_articulation_points(graph, n)
    return len(articulation_points) == 0  # Biconnected = no articulation points


def generate_random_connected_graph(n, m):
    if m < n - 1:
        m = n - 1  # Need at least n-1 edges to be connected
    if m > n * (n - 1) // 2:
        m = n * (n - 1) // 2  # Can't have more than this many edges

    g = Graph(n)
    edges = set()

    # First create a spanning tree to guarantee connectivity
    vertices = list(range(n))
    random.shuffle(vertices)
    for i in range(1, n):
        u = vertices[i]
        v = vertices[random.randint(0, i-1)]  # Connect to someone already in tree
        edge = tuple(sorted([u, v]))
        edges.add(edge)
        g.add_edge(u, v)

    # Then add remaining edges randomly
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            edge = tuple(sorted([u, v]))
            if edge not in edges:  # Don't add duplicate edges
                edges.add(edge)
                g.add_edge(u, v)

    return g, len(edges)


def measure_runtime(n, m, num_trials=30):
    g, actual_m = generate_random_connected_graph(n, m)
    adj_list = g.get_adjacency_list()

    times = []

    for _ in range(10):  # Warm-up runs to stabilize measurements
        find_articulation_points(adj_list, n)

    gc.collect()  # Clean up before measuring
    time.sleep(0.01)

    for _ in range(num_trials):
        start = time.perf_counter()
        find_articulation_points(adj_list, n)
        end = time.perf_counter()

        times.append((end - start) * 1e9)  # Convert to nanoseconds

    times.sort()
    if len(times) > 12:
        times = times[5:-5]  # Remove outliers (top 5 and bottom 5)

    return np.mean(times), actual_m


def run_experiments():
    test_sizes = [100, 1000, 2000, 4000, 8000, 16000]      # Values of n

    results = []

    print("Running experiments...")
    print("=" * 80)

    for n in test_sizes:
        m = 2 * n  # For a sparse graph where 2 edges per vertex
        print(f"Testing n={n:5d}, target m={m:5d}...", end=" ", flush=True)

        avg_time, actual_m = measure_runtime(n, m, num_trials=30)
        theoretical_ops = n + actual_m  # Total operations = process n vertices + m edges

        results.append({
            'n': n,
            'm': actual_m,
            'theoretical_ops': theoretical_ops,
            'experimental_time': avg_time
        })

        print(f"Done. Actual m={actual_m:5d}, Time={avg_time:10.2f} ns")

    return results


def calculate_scaling_constants(results):       # Using linear regression
    x = np.array([r['theoretical_ops'] for r in results])  # Theoretical operations
    y = np.array([r['experimental_time'] for r in results])  # Measured time

    C1, C0 = np.polyfit(x, y, 1)  # polyfit returns [slope, intercept]

    return C0, C1


def create_output_table(results, C0, C1):
    print("\n" + "=" * 110)
    print("EXPERIMENTAL RESULTS")
    print("=" * 110)
    print(f"\nScaling Model: T(n) = C0 + C1 × (n + m)")
    print(f"  Constant Overhead: C0 = {C0:,.2f} ns")
    print(f"  Per-Operation Cost: C1 = {C1:.4f} ns/operation\n")

    print(f"{'n':<8} {'m':<8} {'Theoretical':<15} {'Scaled Theory':<18} {'Experimental':<18} {'Ratio':<8}")
    print(f"{'':8} {'':8} {'Operations':<15} {'(ns)':<18} {'(ns)':<18} {'Exp/Theory':<8}")
    print("-" * 110)

    for r in results:
        theo_ops = r['theoretical_ops']
        scaled_theory = C0 + C1 * theo_ops  # Predicted theoretical values
        exp_time = r['experimental_time']  # Experimental values
        ratio = exp_time / scaled_theory

        print(f"{r['n']:<8} {r['m']:<8} {theo_ops:<15} {scaled_theory:<18.2f} {exp_time:<18.2f} {ratio:<8.4f}")

    print("=" * 110)

    # Calculate R² to see how good our fit is
    predicted = [C0 + C1 * r['theoretical_ops'] for r in results]
    actual = [r['experimental_time'] for r in results]
    residuals = [a - p for a, p in zip(actual, predicted)]
    rss = sum(r**2 for r in residuals)
    tss = sum((a - np.mean(actual))**2 for a in actual)
    r_squared = 1 - (rss / tss)

    print(f"\nGoodness of Fit: R² = {r_squared:.6f}")
    avg_ratio = np.mean([r['experimental_time'] / (C0 + C1 * r['theoretical_ops']) for r in results])
    print(f"Average Exp/Theory Ratio: {avg_ratio:.4f}")


def plot_results(results, C0, C1):
    n_values = [r['n'] for r in results]
    theoretical_ops = [r['theoretical_ops'] for r in results]
    experimental_times = [r['experimental_time'] for r in results]

    scaled_theoretical = [C0 + C1 * ops for ops in theoretical_ops]

    plt.figure(figsize=(9, 7))

    # Plot actual measurements
    plt.plot(n_values, experimental_times, 'bo-', label='Experimental Runtime',
             linewidth=2.5, markersize=10, markeredgewidth=2, markeredgecolor='darkblue')

    # Plot what theory predicts
    plt.plot(n_values, scaled_theoretical, 'r--', label='Scaled Theoretical Runtime',
             linewidth=2.5, markersize=8, marker='s', markeredgewidth=2, markeredgecolor='darkred')

    plt.xlabel('Number of Vertices (n)', fontsize=14, fontweight='bold')
    plt.ylabel('Runtime (nanoseconds)', fontsize=14, fontweight='bold')
    plt.title('Articulation Points Detection: Experimental vs Theoretical Runtime\n' +
              f'O(n + m) Complexity - Sparse Graphs (m ≈ 2n)\n' +
              f'Model: T(n) = {C0:,.0f} + {C1:.2f}(n+m) ns',
              fontsize=13, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    textstr = f'Time Complexity: O(n + m)\nFor sparse graphs: O(n)\nLinear growth confirmed\nConstant overhead: {C0:,.0f} ns'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    plt.text(0.58, 0.20, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax = plt.gca()
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig('articulation_points_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nGraph saved as 'articulation_points_analysis.png'")



# UNIT TESTS
class TestGraph(unittest.TestCase):

    def test_graph_initialization(self):
        g = Graph(5)
        self.assertEqual(g.n, 5)

    def test_add_edge(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        self.assertIn(1, g.adj[0])
        self.assertIn(0, g.adj[1])
        self.assertIn(2, g.adj[1])
        self.assertIn(1, g.adj[2])

    def test_get_adjacency_list(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)

        adj_list = g.get_adjacency_list()
        self.assertEqual(len(adj_list), 3)


class TestArticulationPoints(unittest.TestCase):

    def test_simple_chain(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        adj_list = g.get_adjacency_list()
        ap = find_articulation_points(adj_list, 4)

        self.assertEqual(ap, {1, 2})

    def test_triangle(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        adj_list = g.get_adjacency_list()
        ap = find_articulation_points(adj_list, 3)

        self.assertEqual(len(ap), 0)

    def test_star_graph(self):
        g = Graph(5)
        for i in range(1, 5):
            g.add_edge(0, i)

        adj_list = g.get_adjacency_list()
        ap = find_articulation_points(adj_list, 5)

        self.assertEqual(ap, {0})

    def test_two_triangles_connected(self):
        g = Graph(6)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(5, 3)

        g.add_edge(2, 3)

        adj_list = g.get_adjacency_list()
        ap = find_articulation_points(adj_list, 6)

        self.assertEqual(ap, {2, 3})

    def test_complete_graph(self):
        g = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(i, j)

        adj_list = g.get_adjacency_list()
        ap = find_articulation_points(adj_list, 4)

        self.assertEqual(len(ap), 0)


class TestBiconnectivity(unittest.TestCase):

    def test_biconnected_triangle(self):
        g = Graph(3)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 0)

        adj_list = g.get_adjacency_list()
        self.assertTrue(is_biconnected(adj_list, 3))

    def test_not_biconnected_chain(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

        adj_list = g.get_adjacency_list()
        self.assertFalse(is_biconnected(adj_list, 4))

    def test_biconnected_complete_graph(self):
        g = Graph(4)
        for i in range(4):
            for j in range(i + 1, 4):
                g.add_edge(i, j)

        adj_list = g.get_adjacency_list()
        self.assertTrue(is_biconnected(adj_list, 4))

    def test_biconnected_cycle(self):
        g = Graph(5)
        for i in range(5):
            g.add_edge(i, (i + 1) % 5)

        adj_list = g.get_adjacency_list()
        self.assertTrue(is_biconnected(adj_list, 5))


class TestRandomGraphGeneration(unittest.TestCase):

    def test_generates_correct_vertex_count(self):
        g, m = generate_random_connected_graph(10, 20)
        self.assertEqual(g.n, 10)

    def test_generates_correct_edge_count(self):
        n, target_m = 10, 20
        g, actual_m = generate_random_connected_graph(n, target_m)

        edge_count = sum(len(g.adj[i]) for i in range(n)) // 2
        self.assertEqual(edge_count, actual_m)
        self.assertEqual(actual_m, target_m)

    def test_minimum_edges_for_connectivity(self):
        g, m = generate_random_connected_graph(10, 5)
        self.assertGreaterEqual(m, 9)

    def test_graph_is_connected(self):
        g, m = generate_random_connected_graph(10, 15)
        adj_list = g.get_adjacency_list()

        visited = [False] * 10

        def dfs(u):
            visited[u] = True
            for v in adj_list[u]:
                if not visited[v]:
                    dfs(v)

        dfs(0)
        self.assertTrue(all(visited))



# MAIN EXECUTION
def run_unit_tests():
    print("\nRunning unit tests...", end=" ", flush=True)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestArticulationPoints))
    suite.addTests(loader.loadTestsFromTestCase(TestBiconnectivity))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomGraphGeneration))

    runner = unittest.TextTestRunner(stream=open('/dev/null', 'w'), verbosity=0)
    result = runner.run(suite)

    if result.wasSuccessful():
        print(f"All {result.testsRun} tests passed!")
    else:
        print(f"{len(result.failures) + len(result.errors)} tests failed")
        for test, traceback in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(traceback)

    return result.wasSuccessful()


def main():
    print("\n" + "=" * 80)
    print("BICONNECTIVITY AND ARTICULATION POINTS ANALYSIS")
    print("=" * 80)

    tests_passed = run_unit_tests()

    if not tests_passed:
        print("\n  Tests failed. Please review.\n")
        return

    results = run_experiments()

    C0, C1 = calculate_scaling_constants(results)

    create_output_table(results, C0, C1)

    plot_results(results, C0, C1)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
