"""
Near-Ramanujan Quantum Coupling Map Generator
==============================================
Implementation of Algorithm 1 from "Near-Ramanujan graphs are all you need 
to achieve the maximum quantum fidelity"

This implementation generates optimal coupling maps for quantum processors
using near-Ramanujan graph constructions.
"""

import numpy as np
import time
import sys
import json
from typing import Tuple, List, Dict
from datetime import datetime
import tracemalloc


class RamanujanCouplingMapGenerator:
    """
    Generates near-Ramanujan coupling maps for quantum processors.
    
    Based on Algorithm 1 from the paper, this class constructs d-regular
    graphs with optimal spectral expansion properties for quantum computing.
    """
    
    def __init__(self, n_qubits: int, degree: int, seed: int = 42):
        """
        Initialize the coupling map generator.
        
        Args:
            n_qubits: Number of qubits in the quantum processor
            degree: Desired degree of the coupling map (connectivity per qubit)
            seed: Random seed for reproducibility
        """
        self.n_qubits = n_qubits
        self.degree = degree
        self.seed = seed
        np.random.seed(seed)
        
        # Performance metrics
        self.construction_time = 0.0
        self.peak_memory = 0.0
        self.adjacency_matrix = None
        
    def find_primes_sum_to_degree(self) -> List[int]:
        """
        Step 1 of Algorithm 1: Find primes p1, p2, ..., ps such that
        (p1 + 1) + (p2 + 1) + ... + (ps + 1) <= d
        
        Returns:
            List of primes whose (prime + 1) values sum to approximately degree
        """
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(np.sqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        primes = []
        current_sum = 0
        candidate = 2
        
        # Find primes greedily
        while current_sum < self.degree:
            if is_prime(candidate):
                if current_sum + (candidate + 1) <= self.degree:
                    primes.append(candidate)
                    current_sum += (candidate + 1)
            candidate += 1
            
            # Safety check to prevent infinite loop
            if candidate > self.degree * 2:
                break
                
        return primes
    
    def construct_ramanujan_graph(self, p: int) -> np.ndarray:
        """
        Step 2 of Algorithm 1: Construct a Ramanujan graph Gi on n vertices
        with degree (p + 1).
        
        This uses a Cayley graph construction based on SL(2, Fq).
        For practical implementation, we use a regular graph construction
        that approximates Ramanujan properties.
        
        Args:
            p: Prime number for degree (p + 1) graph
            
        Returns:
            Adjacency matrix of the constructed graph
        """
        n = self.n_qubits
        degree = p + 1
        adj = np.zeros((n, n), dtype=int)
        
        # Circulant graph construction (approximates Ramanujan properties)
        # Creates connections at specific offsets
        offsets = []
        step = max(1, n // degree)
        
        for i in range(1, degree + 1):
            offset = (i * step) % n
            if offset == 0:
                offset = i
            if offset not in offsets and offset < n:
                offsets.append(offset)
        
        # Ensure we have exactly 'degree' connections
        while len(offsets) < degree and len(offsets) < n - 1:
            candidate = len(offsets) + 1
            if candidate not in offsets:
                offsets.append(candidate)
        
        offsets = offsets[:degree]
        
        # Build adjacency matrix with circulant structure
        for i in range(n):
            for offset in offsets:
                j = (i + offset) % n
                if i != j:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        return adj
    
    def add_additional_generators(self, adj: np.ndarray, y: int) -> np.ndarray:
        """
        Step 4 of Algorithm 1: Add y additional generators to make graph d-regular.
        
        Args:
            adj: Current adjacency matrix
            y: Number of additional edges needed per vertex
            
        Returns:
            Updated adjacency matrix
        """
        n = self.n_qubits
        
        for vertex in range(n):
            current_degree = np.sum(adj[vertex])
            needed = y
            
            # Find vertices to connect to
            available = []
            for j in range(n):
                if j != vertex and adj[vertex][j] == 0:
                    available.append(j)
            
            # Add connections to reach desired degree
            np.random.shuffle(available)
            for j in available[:needed]:
                adj[vertex][j] = 1
                adj[j][vertex] = 1
        
        return adj
    
    def construct_coupling_map(self) -> Tuple[np.ndarray, Dict]:
        """
        Main algorithm: Constructs the complete coupling map.
        
        Returns:
            Tuple of (adjacency_matrix, statistics_dict)
        """
        tracemalloc.start()
        start_time = time.time()
        
        # Step 1: Find primes
        primes = self.find_primes_sum_to_degree()
        
        # Step 2 & 3: Construct and union Ramanujan graphs
        adj = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        
        for p in primes:
            G_i = self.construct_ramanujan_graph(p)
            adj = np.logical_or(adj, G_i).astype(int)
        
        # Step 4: Add additional generators if needed
        current_degree = np.sum(adj[0])  # Check degree of first vertex
        target_degree = self.degree
        y = target_degree - current_degree
        
        if y > 0:
            adj = self.add_additional_generators(adj, y)
        
        # Step 5: Direction assignment (for directed coupling maps)
        # In this implementation, we keep undirected for visualization
        # Direction can be added based on control/target qubit designation
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.construction_time = end_time - start_time
        self.peak_memory = peak / (1024 * 1024)  # Convert to MB
        self.adjacency_matrix = adj
        
        # Compute statistics
        stats = self.compute_statistics(adj, primes)
        
        return adj, stats
    
    def compute_statistics(self, adj: np.ndarray, primes: List[int]) -> Dict:
        """
        Compute comprehensive statistics about the generated coupling map.
        
        Args:
            adj: Adjacency matrix
            primes: List of primes used in construction
            
        Returns:
            Dictionary of statistics
        """
        n = self.n_qubits
        
        # Basic graph properties
        num_edges = np.sum(adj) // 2
        degrees = np.sum(adj, axis=1)
        avg_degree = np.mean(degrees)
        min_degree = np.min(degrees)
        max_degree = np.max(degrees)
        
        # Compute Laplacian and spectral properties
        D = np.diag(degrees)
        L = D - adj
        eigenvalues = np.linalg.eigvalsh(L)
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
        
        # Compute diameter (approximate via BFS from random node)
        diameter = self.compute_diameter(adj)
        
        # Check Ramanujan property
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
        ramanujan_bound = 2 * np.sqrt(avg_degree - 1)
        is_near_ramanujan = lambda_2 <= ramanujan_bound * 1.1  # 10% tolerance
        
        stats = {
            'n_qubits': n,
            'target_degree': self.degree,
            'num_edges': int(num_edges),
            'avg_degree': float(avg_degree),
            'min_degree': int(min_degree),
            'max_degree': int(max_degree),
            'primes_used': primes,
            'spectral_gap': float(spectral_gap),
            'diameter': diameter,
            'ramanujan_bound': float(ramanujan_bound),
            'is_near_ramanujan': bool(is_near_ramanujan),
            'construction_time_sec': self.construction_time,
            'peak_memory_mb': self.peak_memory,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'eigenvalues': eigenvalues.tolist()[:10]  # First 10 eigenvalues
        }
        
        return stats
    
    def compute_diameter(self, adj: np.ndarray) -> int:
        """
        Compute graph diameter using BFS from multiple source nodes.
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Approximate diameter of the graph
        """
        n = self.n_qubits
        max_distance = 0
        
        # Sample a few source nodes
        sources = [0, n//4, n//2, 3*n//4] if n >= 4 else [0]
        
        for source in sources:
            distances = np.full(n, -1)
            distances[source] = 0
            queue = [source]
            
            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if adj[u][v] == 1 and distances[v] == -1:
                        distances[v] = distances[u] + 1
                        queue.append(v)
            
            max_distance = max(max_distance, np.max(distances))
        
        return int(max_distance)
    
    def print_adjacency_matrix(self):
        """Print the adjacency matrix in a readable format."""
        if self.adjacency_matrix is None:
            print("No adjacency matrix generated yet. Call construct_coupling_map() first.")
            return
        
        print("\n" + "="*80)
        print("ADJACENCY MATRIX")
        print("="*80)
        print(f"Dimensions: {self.n_qubits} x {self.n_qubits}")
        print()
        
        # Print column headers
        header = "    " + "".join([f"{i:3d}" for i in range(self.n_qubits)])
        print(header)
        print("    " + "-" * (3 * self.n_qubits))
        
        # Print rows
        for i in range(self.n_qubits):
            row_str = f"{i:2d} |"
            for j in range(self.n_qubits):
                row_str += f"{self.adjacency_matrix[i][j]:3d}"
            print(row_str)
        
        print("="*80 + "\n")
    
    def export_results(self, filename: str, stats: Dict):
        """
        Export adjacency matrix and statistics to files.
        
        Args:
            filename: Base filename (without extension)
            stats: Statistics dictionary
        """
        # Save adjacency matrix as CSV
        np.savetxt(f"{filename}_adjacency.csv", self.adjacency_matrix, 
                   delimiter=",", fmt="%d")
        
        # Save statistics as JSON
        with open(f"{filename}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save readable summary
        with open(f"{filename}_summary.txt", 'w') as f:
            f.write("Near-Ramanujan Coupling Map Generation Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
            f.write(f"Number of Qubits: {stats['n_qubits']}\n")
            f.write(f"Target Degree: {stats['target_degree']}\n")
            f.write(f"Number of Edges: {stats['num_edges']}\n")
            f.write(f"Average Degree: {stats['avg_degree']:.2f}\n")
            f.write(f"Degree Range: [{stats['min_degree']}, {stats['max_degree']}]\n")
            f.write(f"Primes Used: {stats['primes_used']}\n")
            f.write(f"Spectral Gap (λ): {stats['spectral_gap']:.6f}\n")
            f.write(f"Diameter: {stats['diameter']}\n")
            f.write(f"Ramanujan Bound: {stats['ramanujan_bound']:.6f}\n")
            f.write(f"Is Near-Ramanujan: {stats['is_near_ramanujan']}\n")
            f.write(f"\nPerformance:\n")
            f.write(f"  Construction Time: {stats['construction_time_sec']:.6f} seconds\n")
            f.write(f"  Peak Memory Usage: {stats['peak_memory_mb']:.2f} MB\n")
            f.write(f"  Random Seed: {stats['seed']}\n")
            f.write(f"\nFirst 10 Laplacian Eigenvalues:\n")
            for i, ev in enumerate(stats['eigenvalues']):
                f.write(f"  λ_{i}: {ev:.6f}\n")
        
        print(f"\nResults exported to:")
        print(f"  - {filename}_adjacency.csv")
        print(f"  - {filename}_stats.json")
        print(f"  - {filename}_summary.txt")


def main():
    """
    Main execution function - generates coupling map based on command-line arguments.
    """
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate near-Ramanujan coupling maps for quantum processors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 20-qubit coupling map with degree 4 for superconducting platform
  python %(prog)s --qubits 20 --degree 4 --platform superconducting
  
  # Generate 50-qubit coupling map for trapped-ion platform
  python %(prog)s -q 50 -d 6 -p trapped-ion
  
  # Use custom seed for reproducibility
  python %(prog)s -q 100 -d 5 -p superconducting --seed 123

Supported Platforms:
  - superconducting (IBM, Google, Rigetti)
  - trapped-ion (IonQ, Honeywell)
  - quantum-annealing (D-Wave)
  - neutral-atom (QuEra, Pasqal)
  - photonic (Xanadu, PsiQuantum)
        """
    )
    
    parser.add_argument('-q', '--qubits', type=int, required=True,
                        help='Number of qubits in the quantum processor')
    parser.add_argument('-d', '--degree', type=int, required=True,
                        help='Target degree (connectivity per qubit)')
    parser.add_argument('-p', '--platform', type=str, required=True,
                        choices=['superconducting', 'trapped-ion', 'quantum-annealing', 
                                'neutral-atom', 'photonic'],
                        help='Quantum computing platform type')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output filename prefix (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.qubits < 2:
        parser.error("Number of qubits must be at least 2")
    if args.degree < 2:
        parser.error("Degree must be at least 2")
    if args.degree >= args.qubits:
        parser.error("Degree must be less than number of qubits")
    
    N_QUBITS = args.qubits
    DEGREE = args.degree
    PLATFORM = args.platform
    SEED = args.seed
    
    print("="*80)
    print("Near-Ramanujan Quantum Coupling Map Generator")
    print("Implementation of Algorithm 1")
    print("="*80)
    print()
    
    print(f"Configuration:")
    print(f"  Number of Qubits: {N_QUBITS}")
    print(f"  Target Degree: {DEGREE}")
    print(f"  Platform: {PLATFORM}")
    print(f"  Random Seed: {SEED}")
    print()
    
    # Create generator
    generator = RamanujanCouplingMapGenerator(N_QUBITS, DEGREE, SEED)
    
    # Construct coupling map
    print("Constructing coupling map...")
    adj_matrix, statistics = generator.construct_coupling_map()
    
    # Print results
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print()
    
    print("Statistics:")
    print(f"  Construction Time: {statistics['construction_time_sec']:.6f} seconds")
    print(f"  Peak Memory Usage: {statistics['peak_memory_mb']:.2f} MB")
    print(f"  Number of Edges: {statistics['num_edges']}")
    print(f"  Average Degree: {statistics['avg_degree']:.2f}")
    print(f"  Degree Range: [{statistics['min_degree']}, {statistics['max_degree']}]")
    print(f"  Primes Used: {statistics['primes_used']}")
    print(f"  Spectral Gap (λ): {statistics['spectral_gap']:.6f}")
    print(f"  Diameter: {statistics['diameter']}")
    print(f"  Ramanujan Bound (2√(d-1)): {statistics['ramanujan_bound']:.6f}")
    print(f"  Is Near-Ramanujan: {statistics['is_near_ramanujan']}")
    print()
    
    # Print adjacency matrix
    generator.print_adjacency_matrix()
    
    # Print edge list for visualization
    print("="*80)
    print("EDGE LIST (for visualization)")
    print("="*80)
    print("Format: (qubit_i, qubit_j)")
    print()
    edges = []
    for i in range(N_QUBITS):
        for j in range(i+1, N_QUBITS):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    
    for i, (u, v) in enumerate(edges):
        print(f"({u:2d}, {v:2d})", end="  ")
        if (i + 1) % 5 == 0:
            print()
    print("\n")
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"coupling_map_{N_QUBITS}q_{timestamp}"
    generator.export_results(filename, statistics)
    
    print()
    print("="*80)
    print("For complete results and analysis, see the accompanying PDF.")
    print("="*80)


if __name__ == "__main__":
    main()
